import io
import logging
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import DB_CONFIG, OUTPUT_DIR_PREFIX
from prediction_model.database_utils import DatabaseManager
from prediction_model.data_processing import ElectricityDataset, preprocess_data
from prediction_model.model_definition import BiLSTMQuantile
from prediction_model.model_training import train_model
from prediction_model.prediction_utils import predict_next_timestep, create_prediction_plot
from prediction_model.logger import setup_logger

# Setup global logger
logger = setup_logger()

class CustomerBehaviorPipeline:
    def __init__(self, logger: logging.Logger, output_base_dir: str = f"{OUTPUT_DIR_PREFIX}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"):
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

    def fetch_data(self, customer_ref: int, end_time: datetime = None) -> "pd.DataFrame":
        return self.db_manager.fetch_data(customer_ref, end_time=end_time)

    def load_existing_model(self, customer_ref: int) -> tuple["BiLSTMQuantile", float, float, datetime]:
        try:
            result = self.db_manager.load_model(customer_ref)
            if result:
                model_data, mse, r2_score, last_trained_time = result
                model = BiLSTMQuantile(input_size=9)
                buffer = io.BytesIO(model_data)
                model = torch.jit.load(buffer)
                self.logger.info(f"Loaded existing model for customer {customer_ref}")
                return model, mse, r2_score, last_trained_time
            self.logger.info(f"No existing model for customer {customer_ref}")
            return None, None, None, None
        except Exception as e:
            self.logger.error(f"Error loading model for customer {customer_ref}: {e}")
            raise

    def run(self, sequence_length: int = 192, batch_size: int = 32, end_time: datetime = None) -> List[Dict]:
        try:
            self.connect_db()

            customer_refs = self.fetch_customer_refs()
            results = []

            for customer_ref in customer_refs:
                self.logger.info(f"Processing customer {customer_ref}")

                # Initialize prediction variables
                predictions_abs = None
                predictions_deltas = None
                plot_path = None

                df = self.fetch_data(customer_ref, end_time=end_time)
                if df.empty:
                    self.logger.warning(f"No data for customer {customer_ref} up to {end_time}")
                    continue
                    
                last_known_kwh = df['import_kwh'].iloc[-1]

                # --- 1. Data length check ---
                if len(df) < sequence_length + 1:
                    self.logger.warning(f"Insufficient data for customer {customer_ref}")
                    results.append({
                        'customer_ref': customer_ref,
                        'status': 'skipped',
                        'predictions': None,
                        'last_known_kwh': last_known_kwh,
                        'plot_path': None
                    })
                    continue

                current_max_timestamp = df['timestamp'].max()
                model, prev_mse, prev_r2, last_trained_time = self.load_existing_model(customer_ref)

                # --- 2. Check if we need to train ---
                should_train = False
                if model is None:
                    should_train = True
                elif last_trained_time is None or (current_max_timestamp - last_trained_time).total_seconds() >= 24 * 3600:
                    should_train = True

                # If no new data at all, skip everything
                # Assuming predictions are generated up to current_max_timestamp + 15m
                # For simplicity, if we are simulating, we always want to predict if called, but skip if literally no new data
                if last_trained_time and current_max_timestamp <= last_trained_time and not should_train:
                    # check if we already predicted for this timestamp.
                    if self.db_manager.check_existing_prediction(customer_ref, current_max_timestamp):
                        self.logger.info(
                            f"Skipping customer {customer_ref} - No new data and already predicted."
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
                recent_window = df['import_kwh'].iloc[-(sequence_length + 1):]
                if recent_window.std() < 1e-4:
                    constant_value = recent_window.iloc[-1]
                    self.logger.info(
                        f"Customer {customer_ref}: Near-flat import_kwh detected "
                        f"(std={recent_window.std():.6f}). Skipping model prediction."
                    )

                    predictions_abs = np.full(1, constant_value)
                    predictions_deltas = np.zeros(1)
                    last_timestamp = df['timestamp'].iloc[-1]
                    next_timestamp = last_timestamp + timedelta(minutes=15)

                    try:
                        prediction_times = [next_timestamp]
                        data_to_insert = []
                        for i in range(1):
                            data_to_insert.append((
                                customer_ref,
                                float(predictions_deltas[i]),
                                float(predictions_abs[i]),
                                prediction_times[i]
                            ))
                        self.db_manager.save_prediction(data_to_insert, customer_ref)
                    except Exception as e:
                        self.logger.warning(
                            f"Could not save flat prediction for customer {customer_ref}: {e}"
                        )

                    results.append({
                        'customer_ref': customer_ref,
                        'status': 'skipped',
                        'predictions': predictions_abs,
                        'last_known_kwh': last_known_kwh,
                        'plot_path': None
                    })
                    continue

                # --- 4. Normal pipeline (training + prediction) ---
                scaled_data, scaler, original_import_kwh = preprocess_data(df, self.logger)
                
                best_val_loss = 0.0
                if should_train:
                    dataset = ElectricityDataset(scaled_data, sequence_length)
                    train_size = int(0.8 * len(dataset))
                    val_size = len(dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
                    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = DataLoader(val_dataset, batch_size=batch_size)

                    if model is None:
                        model = BiLSTMQuantile(input_size=9)
                        self.logger.info(f"Created new Bi-LSTM model for customer {customer_ref}")

                    model, best_val_loss, _ = train_model(model, train_loader, val_loader, logger=self.logger)
                else:
                    self.logger.info(f"Skipping training for customer {customer_ref}, using existing model.")

                last_sequence = scaled_data[-sequence_length:]
                lower_abs, median_abs, upper_abs = predict_next_timestep(model, last_sequence, scaler, last_known_kwh, self.logger)

                last_timestamp = df['timestamp'].iloc[-1]
                next_timestamp = last_timestamp + timedelta(minutes=15)

                df_plot = df.copy()
                df_plot['import_kwh'] = original_import_kwh

                plot_path = create_prediction_plot(
                    df_plot,
                    median_abs,
                    lower_abs,
                    upper_abs,
                    customer_ref,
                    sequence_length,
                    self.output_base_dir,
                    self.logger
                )

                try:
                    prediction_times = [next_timestamp]
                    data_to_insert = []
                    predictions_deltas = np.diff(np.insert(median_abs, 0, last_known_kwh))
                    for i in range(1):
                        data_to_insert.append((
                            customer_ref,
                            float(predictions_deltas[i]),
                            float(median_abs[i]),
                            prediction_times[i]
                        ))
                    self.db_manager.save_prediction(data_to_insert, customer_ref)
                except Exception as e:
                    self.logger.error(f"Error saving prediction for customer {customer_ref}: {e}")

                if should_train:
                    buffer = io.BytesIO()
                    torch.jit.save(torch.jit.script(model), buffer)
                    model_data = buffer.getvalue()
                    self.db_manager.save_model(
                        customer_ref,
                        model_data,
                        best_val_loss,
                        0.0,  # R² not computed for quantile model
                        current_max_timestamp
                    )

                predictions_abs = median_abs  # assign for results

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