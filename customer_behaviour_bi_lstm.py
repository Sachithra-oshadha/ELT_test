import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import psycopg2
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import io
import logging
from dotenv import load_dotenv
import pytz
import json

# Load environment variables
load_dotenv()
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'customer_behavior_bilstm_{now}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ElectricityDataset(Dataset):
    def __init__(self, data: np.ndarray, sequence_length: int):
        self.data = data
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return len(self.data) - self.sequence_length - 96  # Ensure we have 96 targets

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length:idx + self.sequence_length + 96, 0]  # Predict next 96 steps
        if len(y) < 96:
            raise ValueError("Not enough data to create label")
        return torch.FloatTensor(x), torch.FloatTensor(y).unsqueeze(-1)


class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2, output_size: int = 96):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                            dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take last time step, predict 96 outputs
        return out.unsqueeze(-1)  # shape: (batch, 96, 1)


class CustomerBehaviorPipeline:
    def __init__(self, output_base_dir: str = f"customer_outputs_bilstm_day_{now}"):
        self.db_config = {
            'dbname': os.getenv('DB_NAME_BILSTM_DAY'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        self.output_base_dir = output_base_dir
        self.conn = None
        self.cur = None
        if not os.path.exists(self.output_base_dir):
            os.makedirs(self.output_base_dir)
            logger.info(f"Created output directory: {self.output_base_dir}")

    def connect_db(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor()
            logger.info("Connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close_db(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def fetch_customer_refs(self) -> list[int]:
        try:
            self.cur.execute("SELECT customer_ref FROM customer")
            customer_refs = [row[0] for row in self.cur.fetchall()]
            logger.info(f"Fetched {len(customer_refs)} customer references")
            return customer_refs
        except Exception as e:
            logger.error(f"Error fetching customer references: {e}")
            raise

    def fetch_data(self, customer_ref: int) -> pd.DataFrame:
        try:
            query = """
                SELECT m.timestamp, m.import_kwh, m.avg_import_kw, m.power_factor, 
                       pm_a.inst_current AS phase_a_current, pm_a.inst_voltage AS phase_a_voltage,
                       pm_b.inst_current AS phase_b_current, pm_b.inst_voltage AS phase_b_voltage,
                       pm_c.inst_current AS phase_c_current, pm_c.inst_voltage AS phase_c_voltage
                FROM measurement m
                JOIN meter mt ON m.serial = mt.serial
                LEFT JOIN phase_measurement pm_a ON m.serial = pm_a.serial AND m.timestamp = pm_a.timestamp AND pm_a.phase = 'A'
                LEFT JOIN phase_measurement pm_b ON m.serial = pm_b.serial AND m.timestamp = pm_b.timestamp AND pm_b.phase = 'B'
                LEFT JOIN phase_measurement pm_c ON m.serial = pm_c.serial AND m.timestamp = pm_c.timestamp AND pm_c.phase = 'C'
                WHERE mt.customer_ref = %s
                ORDER BY m.timestamp
            """
            df = pd.read_sql(query, self.conn, params=(customer_ref,))
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().any():
                logger.warning(f"Dropped {df['timestamp'].isnull().sum()} rows with invalid timestamps for customer {customer_ref}")
                df = df.dropna(subset=['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                logger.info(f"Converted timezone-aware timestamps to naive for customer {customer_ref}")
            df['timestamp'] = df['timestamp'].dt.round('15min')
            valid_time_range = (df['timestamp'] >= '2000-01-01') & (df['timestamp'] <= '2030-12-31')
            if not valid_time_range.all():
                logger.warning(f"Dropped {len(df[~valid_time_range])} rows with out-of-range timestamps for customer {customer_ref}")
                df = df[valid_time_range]
            logger.info(f"Fetched {len(df)} valid records for customer {customer_ref} at 15-minute intervals")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for customer {customer_ref}: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, np.ndarray]:
        try:
            features = ['import_kwh', 'avg_import_kw', 'power_factor',
                        'phase_a_current', 'phase_a_voltage',
                        'phase_b_current', 'phase_b_voltage',
                        'phase_c_current', 'phase_c_voltage']

            # Make a copy of original import_kwh for plotting
            original_import_kwh = df['import_kwh'].copy()

            # Compute differences for import_kwh only
            df['import_kwh_diff'] = df['import_kwh'].diff().fillna(0)

            # Replace original import_kwh with diff for training
            df['import_kwh'] = df['import_kwh_diff']
            df = df.drop(columns=['import_kwh_diff'])

            # Fill missing values in other features
            df = df[features].ffill().infer_objects(copy=False).fillna(0)

            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            logger.info("Data preprocessed successfully using differences for import_kwh")
            
            return scaled_data, scaler, original_import_kwh.values  # Return original values for plotting
        except Exception as e:
            logger.error(f"Failed to preprocess data: {e}")
            raise

    def load_existing_model(self, customer_ref: int) -> tuple[nn.Module, float, float]:
        try:
            self.cur.execute("""
                SELECT model_data, mse, r2_score
                FROM customer_model
                WHERE customer_ref = %s
            """, (customer_ref,))
            result = self.cur.fetchone()
            if result:
                model_data, mse, r2_score = result
                model = BiLSTM(input_size=9)
                buffer = io.BytesIO(model_data)
                model = torch.jit.load(buffer)
                logger.info(f"Loaded existing model for customer {customer_ref}")
                return model, mse, r2_score
            logger.info(f"No existing model for customer {customer_ref}")
            return None, None, None
        except Exception as e:
            logger.error(f"Error loading model for customer {customer_ref}: {e}")
            raise

    def train_model(self, model: nn.Module, train_loader: DataLoader,
                    val_loader: DataLoader, num_epochs: int = 10, patience: int = 3) -> tuple[nn.Module, float, float]:
        try:
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            best_val_loss = float('inf')
            best_model_state = None
            best_r2_score = float('-inf')
            epochs_no_improve = 0
            early_stop = False

            for epoch in range(num_epochs):
                if early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
                model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch_x)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item() * batch_x.size(0)

                model.eval()
                val_loss = 0
                predictions, actuals = [], []
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        val_loss += criterion(outputs.squeeze(), batch_y.squeeze()).item() * batch_x.size(0)
                        predictions.extend(outputs.cpu().numpy())
                        actuals.extend(batch_y.cpu().numpy())

                val_loss /= len(val_loader.dataset)
                train_loss /= len(train_loader.dataset)
                predictions = np.concatenate(predictions)
                actuals = np.concatenate(actuals)
                r2 = r2_score(actuals.flatten(), predictions.flatten())

                logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, R² Score: {r2:.4f}')

                if r2 == 0:
                    logger.warning("Validation R² is 0. Keeping last best model.")
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                    break

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    best_r2_score = r2
                    epochs_no_improve = 0
                    logger.info(f"New best validation loss: {best_val_loss:.4f}, R²: {best_r2_score:.4f}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        early_stop = True
                        logger.info(f"No improvement in validation loss for {patience} epochs")

            if best_model_state:
                model.load_state_dict(best_model_state)
                logger.info(f"Final model restored with val loss: {best_val_loss:.4f}, R²: {best_r2_score:.4f}")

            logger.info(f"Model training completed. MSE: {best_val_loss:.4f}, R2 Score: {best_r2_score:.4f}")
            return model, best_val_loss, best_r2_score
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def predict_next_timestep(self, model: nn.Module, last_sequence: np.ndarray,
                              scaler: StandardScaler, last_kwh: float) -> tuple[np.ndarray, np.ndarray]:
        try:
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
            with torch.no_grad():
                prediction = model(input_tensor)  # shape: (1, 96, 1)
            prediction = prediction.cpu().numpy().squeeze()

            dummy_array = np.zeros((96, len(scaler.mean_)))
            dummy_array[:, 0] = prediction
            prediction_transformed = scaler.inverse_transform(dummy_array)[:, 0]
            prediction_deltas = np.maximum(prediction_transformed, 0)

            # Reconstruct absolute kWh from deltas
            prediction_abs = [last_kwh + prediction_deltas[0]]
            for i in range(1, 96):
                prediction_abs.append(prediction_abs[-1] + prediction_deltas[i])

            logger.info(f"Predicted kWh deltas: {prediction_deltas}")
            logger.info(f"Reconstructed kWh values: {prediction_abs}")
            return np.array(prediction_abs), np.array(prediction_deltas)
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            raise

    def create_prediction_plot(self, df: pd.DataFrame, predictions: np.ndarray, customer_ref: int, sequence_length: int):
        try:
            output_dir = os.path.join(self.output_base_dir, f"customer_{customer_ref}")
            os.makedirs(output_dir, exist_ok=True)

            plt.figure(figsize=(16, 6))
            last_n = df.tail(sequence_length)
            timestamps = last_n['timestamp']
            actual_kw = last_n['import_kwh']
            last_timestamp = timestamps.iloc[-1]
            prediction_times = [last_timestamp + timedelta(minutes=15 * i) for i in range(1, 97)]

            plt.plot(timestamps, actual_kw.values, 'b-', label='Historical Consumption')
            plt.plot(prediction_times, predictions, 'g--o', label='Predicted Consumption (Next 24 Hours)')
            plt.axvline(x=last_timestamp, color='r', linestyle='--', label='Prediction Point')
            plt.title(f'Customer {customer_ref} - Energy Consumption Prediction (Next 24 Hours)')
            plt.xlabel('Timestamp')
            plt.ylabel('Energy (kWh)')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=2))
            plt.tight_layout()
            output_path = os.path.join(output_dir, f'prediction_{customer_ref}_{datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved 24-hour plot at: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to create plot for customer {customer_ref}: {e}")
            raise

    def save_prediction_to_json(self, customer_ref: int, predictions_abs: np.ndarray,
                                predictions_deltas: np.ndarray, timestamp: datetime, last_kwh: float):
        try:
            output_dir = os.path.join(self.output_base_dir, f"customer_{customer_ref}")
            os.makedirs(output_dir, exist_ok=True)

            prediction_data = {
                "customer_ref": customer_ref,
                "predicted_import_kwh_96_steps": predictions_abs.tolist(),
                "predicted_import_kwh_diffs": predictions_deltas.tolist(),
                "last_import_kwh": float(last_kwh),
                "prediction_start_time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "generated_at": datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
            }

            json_path = os.path.join(output_dir, f'prediction_{customer_ref}_{datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")}.json')
            with open(json_path, 'w') as f:
                json.dump(prediction_data, f, indent=4)
            logger.info(f"Saved prediction JSON for customer {customer_ref} at: {json_path}")
            return json_path, prediction_data
        except Exception as e:
            logger.error(f"Failed to save prediction JSON for customer {customer_ref}: {e}")
            raise

    def save_prediction_to_db(self, customer_ref: int, prediction_data: dict):
        try:
            prediction_timestamp = datetime.strptime(
                prediction_data["prediction_start_time"], "%Y-%m-%d %H:%M:%S"
            )
            self.cur.execute("""
                INSERT INTO customer_prediction (
                    customer_ref, prediction_json, prediction_timestamp
                ) VALUES (%s, %s, %s)
                ON CONFLICT (customer_ref)
                DO UPDATE SET
                    prediction_json = EXCLUDED.prediction_json,
                    prediction_timestamp = EXCLUDED.prediction_timestamp
            """, (
                customer_ref,
                json.dumps(prediction_data),
                prediction_timestamp
            ))
            self.conn.commit()
            logger.info(f"Saved prediction for customer {customer_ref} to database")
        except Exception as e:
            logger.error(f"Failed to save prediction to database for customer {customer_ref}: {e}")
            self.conn.rollback()
            raise

    def save_model(self, model: nn.Module, customer_ref: int, mse: float, r2_score: float):
        try:
            buffer = io.BytesIO()
            torch.jit.save(torch.jit.script(model), buffer)
            model_data = buffer.getvalue()
            self.cur.execute("""
                INSERT INTO customer_model (customer_ref, model_data, mse, r2_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (customer_ref) DO UPDATE 
                SET model_data = EXCLUDED.model_data,
                    mse = EXCLUDED.mse,
                    r2_score = EXCLUDED.r2_score,
                    trained_at = CURRENT_TIMESTAMP
            """, (customer_ref, psycopg2.Binary(model_data), float(mse), float(r2_score)))
            self.conn.commit()
            logger.info(f"Saved model for customer {customer_ref}")
        except Exception as e:
            logger.error(f"Error saving model for customer {customer_ref}: {e}")
            raise

    def process_customer(self, customer_ref: int, sequence_length: int = 1440, batch_size: int = 32) -> dict:
        try:
            logger.info(f"Processing customer {customer_ref} for 24-hour 15-min interval prediction")
            df = self.fetch_data(customer_ref)
            if len(df) < sequence_length + 96:
                logger.warning(f"Insufficient data for customer {customer_ref}")
                return None

            last_known_kwh = df['import_kwh'].iloc[-1]
            # Get original_import_kwh for plotting
            scaled_data, scaler, original_import_kwh = self.preprocess_data(df)

            dataset = ElectricityDataset(scaled_data, sequence_length)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            model, prev_mse, prev_r2 = self.load_existing_model(customer_ref)
            if model is None:
                model = BiLSTM(input_size=9)
                logger.info(f"Created new Bi-LSTM model for customer {customer_ref}")

            model, mse, r2_score = self.train_model(model, train_loader, val_loader)

            last_sequence = scaled_data[-sequence_length:]
            predictions_abs, predictions_deltas = self.predict_next_timestep(model, last_sequence, scaler, last_known_kwh)

            last_timestamp = df['timestamp'].iloc[-1]
            next_timestamp = last_timestamp + timedelta(minutes=15)

            # Use original_import_kwh for plotting
            df_plot = df.copy()
            df_plot['import_kwh'] = original_import_kwh  # Restore original kWh for plotting

            plot_path = self.create_prediction_plot(df_plot, predictions_abs, customer_ref, sequence_length)
            json_path, prediction_data = self.save_prediction_to_json(customer_ref, predictions_abs, predictions_deltas, next_timestamp, last_known_kwh)

            try:
                self.save_prediction_to_db(customer_ref, prediction_data)
            except Exception as e:
                logger.warning(f"Could not save prediction to DB for customer {customer_ref}: {e}")

            self.save_model(model, customer_ref, mse, r2_score)

            logger.info(f"Customer {customer_ref}: Predictions saved for 96 intervals")
            return {
                'customer_ref': customer_ref,
                'predictions': predictions_abs,
                'last_known_kwh': last_known_kwh,
                'plot_path': plot_path,
                'json_path': json_path
            }
        except Exception as e:
            logger.error(f"Failed to process customer {customer_ref}: {e}")
            return None

    def run(self, sequence_length: int = 1440, batch_size: int = 32):
        try:
            self.connect_db()
            customer_refs = self.fetch_customer_refs()
            results = []
            total_increase = 0.0
            for customer_ref in customer_refs:
                result = self.process_customer(customer_ref, sequence_length, batch_size)
                if result:
                    results.append(result)
                    increase = result['predictions'][0] - result['last_known_kwh']
                    total_increase += increase
                    logger.info(f"Customer {customer_ref}: Increase in usage: {increase:.4f} kWh")
            logger.info(f"Total predicted increase in energy usage across all customers: {total_increase:.4f} kWh")
            summary_data = {
                "total_predicted_usage_increase_kWh": round(float(total_increase), 4),
                "generated_at": datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
            }
            summary_path = os.path.join(self.output_base_dir, f"total_usage_increase_{datetime.now(pytz.UTC).strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_data, f, indent=4)
            logger.info(f"Saved total predicted usage increase to: {summary_path}")
            return results
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        finally:
            self.close_db()


if __name__ == "__main__":
    pipeline = CustomerBehaviorPipeline()
    results = pipeline.run()
    for result in results:
        logger.info(f"Customer {result['customer_ref']}: Predicted kWh: {result['predictions'][:3]}..., "
                   f"Plot: {result['plot_path']}, JSON: {result['json_path']}")