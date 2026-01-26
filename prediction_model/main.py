from imports import *

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Local imports
from config import DB_CONFIG, OUTPUT_BASE_DIR
from database_utils import DatabaseManager
from data_processing import ElectricityDataset, preprocess_data
from model_definition import BiLSTM
from model_training import train_model
from prediction_utils import predict_next_timestep, create_prediction_plot, save_prediction_to_db, save_model_to_db
from logger import setup_logger

# Setup global logger
logger = setup_logger()

class CustomerBehaviorPipeline:
    def __init__(self, logger: logging.Logger, output_base_dir: str = f"{OUTPUT_BASE_DIR}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
        self.db_manager = DatabaseManager(db_config=DB_CONFIG, logger=logger)
        self.output_base_dir = output_base_dir
        self.logger = logger
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
            self.logger.info(f"Created output directory: {self.output_base_dir}")

    def connect_db(self):
        self.db_manager.connect()

    def close_db(self):
        self.db_manager.close()

    def fetch_customer_refs(self) -> List[int]:
        return self.db_manager.fetch_customer_refs()

    def fetch_data(self, customer_ref: int) -> "pd.DataFrame":
        return self.db_manager.fetch_data(customer_ref)

    def load_existing_model(self, cur, customer_ref: int) -> tuple["BiLSTM", float, float, datetime]:
        try:
            cur.execute("""
                SELECT model_data, mse, r2_score, last_trained_data_timestamp
                FROM customer_model
                WHERE customer_ref = %s
            """, (customer_ref,))
            result = cur.fetchone()
            if result:
                model_data, mse, r2_score, last_trained_time = result
                model = BiLSTM(input_size=9)
                buffer = io.BytesIO(model_data)
                model = torch.jit.load(buffer)
                self.logger.info(f"Loaded existing model for customer {customer_ref}")
                return model, mse, r2_score, last_trained_time
            self.logger.info(f"No existing model for customer {customer_ref}")
            return None, None, None, None
        except Exception as e:
            self.logger.error(f"Error loading model for customer {customer_ref}: {e}")
            raise

    def run(self, sequence_length: int = 192, batch_size: int = 32) -> List[Dict]:
        try:
            self.connect_db()
            cur = self.db_manager.cur
            conn = self.db_manager.conn

            customer_refs = self.fetch_customer_refs()
            results = []

            for customer_ref in customer_refs:
                self.logger.info(f"Processing customer {customer_ref}")
                df = self.fetch_data(customer_ref)
                last_known_kwh = df['import_kwh'].iloc[-1]

                # --- 1. Data length check ---
                if len(df) < sequence_length + 96:
                    self.logger.warning(f"Insufficient data for customer {customer_ref}")
                    continue

                current_max_timestamp = df['timestamp'].max()
                model, prev_mse, prev_r2, last_trained_time = self.load_existing_model(cur, customer_ref)

                # --- 2. No new data check ---
                if last_trained_time and current_max_timestamp <= last_trained_time:
                    self.logger.info(
                        f"Skipping customer {customer_ref} - No new data since last training."
                    )
                    results.append({
                        'customer_ref': customer_ref,
                        'status': 'skipped',
                        'predictions': None,
                        'last_known_kwh': last_known_kwh,
                        'plot_path': None
                    })
                    continue

                # --- 3. Flat signal check ---
                recent_window = df['import_kwh'].iloc[-(sequence_length + 96):]

                if recent_window.std() < 1e-4:
                    constant_value = recent_window.iloc[-1]

                    self.logger.info(
                        f"Customer {customer_ref}: Near-flat import_kwh detected "
                        f"(std={recent_window.std():.6f}). Skipping training."
                    )

                    predictions_abs = np.full(96, constant_value)
                    predictions_deltas = np.zeros(96)

                    last_timestamp = df['timestamp'].iloc[-1]
                    next_timestamp = last_timestamp + timedelta(minutes=15)

                    try:
                        save_prediction_to_db(
                            cur, conn,
                            customer_ref,
                            predictions_abs,
                            predictions_deltas,
                            next_timestamp,
                            self.logger
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Could not save flat prediction for customer {customer_ref}: {e}"
                        )

                    results.append({
                        'customer_ref': customer_ref,
                        'status': 'skipped',
                        'predictions': None,
                        'last_known_kwh': last_known_kwh,
                        'plot_path': None
                    })

                    continue

                # --- 4. Normal pipeline (training + prediction) ---
                scaled_data, scaler, original_import_kwh = preprocess_data(df, self.logger)

                dataset = ElectricityDataset(scaled_data, sequence_length)
                train_size = int(0.8 * len(dataset))
                val_size = len(dataset) - train_size

                train_dataset, val_dataset = torch.utils.data.random_split(
                    dataset, [train_size, val_size]
                )

                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size)

                if model is None:
                    model = BiLSTM(input_size=9)
                    self.logger.info(f"Created new Bi-LSTM model for customer {customer_ref}")

                model, best_val_loss, best_r2 = train_model(
                    model,
                    train_loader,
                    val_loader,
                    logger=self.logger
                )

                last_sequence = scaled_data[-sequence_length:]
                predictions_abs, predictions_deltas = predict_next_timestep(
                    model,
                    last_sequence,
                    scaler,
                    last_known_kwh,
                    self.logger
                )

                last_timestamp = df['timestamp'].iloc[-1]
                next_timestamp = last_timestamp + timedelta(minutes=15)

                df_plot = df.copy()
                df_plot['import_kwh'] = original_import_kwh

                plot_path = create_prediction_plot(
                    df_plot,
                    predictions_abs,
                    customer_ref,
                    sequence_length,
                    self.output_base_dir,
                    self.logger
                )

                save_prediction_to_db(
                    cur, conn,
                    customer_ref,
                    predictions_abs,
                    predictions_deltas,
                    next_timestamp,
                    self.logger
                )

                save_model_to_db(
                    cur, conn,
                    model,
                    customer_ref,
                    best_val_loss,
                    best_r2,
                    current_max_timestamp,
                    self.logger
                )

                results.append({
                    'customer_ref': customer_ref,
                    'status': 'trained',
                    'predictions': predictions_abs,
                    'last_known_kwh': last_known_kwh,
                    'plot_path': plot_path
                })
                
            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.close_db()

if __name__ == "__main__":
    pipeline = CustomerBehaviorPipeline(logger=logger)
    results = pipeline.run()
    for result in results:
        if result['status'] == 'trained':
            logger.info(
                f"Customer {result['customer_ref']}: "
                f"Predicted kWh: {result['predictions'][:3]}..., "
                f"Last known kWh: {result['last_known_kwh']}, "
                f"Plot: {result['plot_path']}"
            )
        else:
            logger.info(
                f"Customer {result['customer_ref']}: "
                f"Skipped (last_known_kwh={result['last_known_kwh']})"
            )