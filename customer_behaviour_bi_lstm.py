import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
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
        return len(self.data) - self.sequence_length
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, 0]  # Predicting avg_import_kw
        return torch.FloatTensor(x), torch.FloatTensor([y])

class BiLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)  # *2 for bidirectional
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # Take the last time step
        return out

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
                SELECT m.timestamp, m.avg_import_kw, m.power_factor, 
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
            # Convert and validate timestamps
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            # Drop rows with invalid timestamps
            if df['timestamp'].isnull().any():
                logger.warning(f"Dropped {df['timestamp'].isnull().sum()} rows with invalid timestamps for customer {customer_ref}")
                df = df.dropna(subset=['timestamp'])
            # Strip timezone to make timestamps naive
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                logger.info(f"Converted timezone-aware timestamps to naive for customer {customer_ref}")
            # Ensure timestamps are in 15-minute intervals
            df['timestamp'] = df['timestamp'].dt.round('15min')
            # Filter out unreasonable timestamps (e.g., before 2000 or after 2030)
            valid_time_range = (df['timestamp'] >= '2000-01-01') & (df['timestamp'] <= '2030-12-31')
            if not valid_time_range.all():
                logger.warning(f"Dropped {len(df[~valid_time_range])} rows with out-of-range timestamps for customer {customer_ref}")
                df = df[valid_time_range]
            logger.info(f"Fetched {len(df)} valid records for customer {customer_ref} at 15-minute intervals")
            return df
        except Exception as e:
            logger.error(f"Error fetching data for customer {customer_ref}: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
        try:
            features = ['avg_import_kw', 'power_factor', 
                        'phase_a_current', 'phase_a_voltage',
                        'phase_b_current', 'phase_b_voltage',
                        'phase_c_current', 'phase_c_voltage']
            df = df[features].ffill().infer_objects(copy=False).fillna(0)
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(df)
            logger.info("Data preprocessed successfully")
            return scaled_data, scaler
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
                model = BiLSTM(input_size=8)
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
                    loss = criterion(outputs, batch_y)
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
                        val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                        predictions.extend(outputs.cpu().numpy().flatten())
                        actuals.extend(batch_y.cpu().numpy().flatten())
                
                val_loss = val_loss / len(val_loader.dataset)
                train_loss = train_loss / len(train_loader.dataset)
                
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
                # Early stopping logic
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    epochs_no_improve = 0
                    logger.info(f"New best validation loss: {best_val_loss:.4f}")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        early_stop = True
                        logger.info(f"No improvement in validation loss for {patience} epochs")
                
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                mse = val_loss
                r2_score = 1 - np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals, axis=0))**2)

            # Load the best model state
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                logger.info(f"Restored best model with validation loss: {best_val_loss:.4f}")
            
            logger.info(f"Model training completed. MSE: {mse:.4f}, R2 Score: {r2_score:.4f}")
            return model, mse, r2_score
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def predict_next_timestep(self, model: nn.Module, last_sequence: np.ndarray, 
                            scaler: StandardScaler) -> float:
        try:
            model.eval()
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)
            
            with torch.no_grad():
                prediction = model(input_tensor)
            
            prediction = prediction.cpu().numpy()
            dummy_array = np.zeros((1, len(scaler.mean_)))
            dummy_array[0, 0] = prediction[0, 0]
            prediction_transformed = scaler.inverse_transform(dummy_array)[0, 0]
            prediction_value = max(0, prediction_transformed)
            logger.info(f"Predicted avg_import_kw for next 15-minute interval: {prediction_value:.4f} kW")
            return prediction_value
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            raise

    def create_prediction_plot(self, df: pd.DataFrame, prediction: float, customer_ref: int, sequence_length: int):
        try:
            output_dir = os.path.join(self.output_base_dir, f"customer_{customer_ref}")
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            last_n = df.tail(sequence_length)
            timestamps = last_n['timestamp']  # Already datetime and timezone-naive from fetch_data
            actual_kw = last_n['avg_import_kw']
            
            if timestamps.empty:
                raise ValueError(f"No valid timestamps available for customer {customer_ref}")
            
            last_timestamp = timestamps.iloc[-1]
            next_timestamp = last_timestamp + timedelta(minutes=15)
            extended_timestamps = pd.Series([*timestamps, next_timestamp])
            extended_kw = np.append(actual_kw.values, prediction)
            
            logger.debug(f"Timestamp range: {extended_timestamps.min()} to {extended_timestamps.max()}")
            
            plt.plot(extended_timestamps, extended_kw, 'b-', label='Actual + Predicted (15-min intervals)')
            plt.plot(extended_timestamps.iloc[-1], extended_kw[-1], 'ro', label='Predicted Value')
            plt.axvline(x=last_timestamp, color='r', linestyle='--', label='Prediction Point')
            plt.title(f'Customer {customer_ref} - Power Consumption Prediction (15-min intervals)')
            plt.xlabel('Timestamp')
            plt.ylabel('Power (kW)')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            
            # Set date formatter
            plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
            # Use AutoDateLocator to automatically choose appropriate tick intervals
            plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.AutoDateLocator(interval_multiples=True))
            # Limit x-axis to a reasonable range (e.g., last 24 hours + 15 minutes)
            plt.gca().set_xlim([last_timestamp - timedelta(hours=24), next_timestamp])
            
            output_path = os.path.join(output_dir, f'prediction_{customer_ref}_{datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved 15-minute interval plot at: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to create plot for customer {customer_ref}: {e}")
            raise

    def save_prediction_to_json(self, customer_ref: int, prediction: float, timestamp: datetime):
        try:
            output_dir = os.path.join(self.output_base_dir, f"customer_{customer_ref}")
            os.makedirs(output_dir, exist_ok=True)
            
            prediction_data = {
                "customer_ref": customer_ref,
                "predicted_avg_import_kw": float(prediction),
                "prediction_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "generated_at": datetime.now(pytz.UTC).strftime("%Y-%m-%d %H:%M:%S %Z")
            }
            
            json_path = os.path.join(output_dir, f'prediction_{customer_ref}_{datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")}.json')
            with open(json_path, 'w') as f:
                json.dump(prediction_data, f, indent=4)
            logger.info(f"Saved prediction JSON for customer {customer_ref} at: {json_path}")
            return json_path
        except Exception as e:
            logger.error(f"Failed to save prediction JSON for customer {customer_ref}: {e}")
            raise

    def save_model(self, model: nn.Module, customer_ref: int, mse: float, r2_score: float):
        try:
            buffer = io.BytesIO()
            torch.jit.save(torch.jit.script(model), buffer)
            model_data = buffer.getvalue()
            mse = float(mse)
            r2_score = float(r2_score)
            
            self.cur.execute("""
                INSERT INTO customer_model (customer_ref, model_data, mse, r2_score)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (customer_ref) DO UPDATE 
                SET model_data = EXCLUDED.model_data,
                    mse = EXCLUDED.mse,
                    r2_score = EXCLUDED.r2_score,
                    trained_at = CURRENT_TIMESTAMP
            """, (customer_ref, psycopg2.Binary(model_data), mse, r2_score))
            self.conn.commit()
            logger.info(f"Saved model for customer {customer_ref}")
        except Exception as e:
            logger.error(f"Error saving model for customer {customer_ref}: {e}")
            raise

    def process_customer(self, customer_ref: int, sequence_length: int = 96, batch_size: int = 32) -> tuple[float, str, str]:
        try:
            logger.info(f"Processing customer {customer_ref} for 15-minute interval prediction")
            df = self.fetch_data(customer_ref)
            if len(df) < sequence_length:
                logger.warning(f"Insufficient data for customer {customer_ref}")
                return None, None, None
            
            scaled_data, scaler = self.preprocess_data(df)
            dataset = ElectricityDataset(scaled_data, sequence_length)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            model, prev_mse, prev_r2 = self.load_existing_model(customer_ref)
            if model is None:
                model = BiLSTM(input_size=8)
                logger.info(f"Created new Bi-LSTM model for customer {customer_ref}")
            
            model, mse, r2_score = self.train_model(model, train_loader, val_loader)
            last_sequence = scaled_data[-sequence_length:]
            prediction = self.predict_next_timestep(model, last_sequence, scaler)
            last_timestamp = df['timestamp'].iloc[-1]
            next_timestamp = last_timestamp + timedelta(minutes=15)
            plot_path = self.create_prediction_plot(df, prediction, customer_ref, sequence_length)
            json_path = self.save_prediction_to_json(customer_ref, prediction, next_timestamp)
            self.save_model(model, customer_ref, mse, r2_score)
            
            return prediction, plot_path, json_path
        except Exception as e:
            logger.error(f"Failed to process customer {customer_ref}: {e}")
            return None, None, None

    def run(self, sequence_length: int = 96, batch_size: int = 32):
        try:
            self.connect_db()
            customer_refs = self.fetch_customer_refs()
            results = []
            for customer_ref in customer_refs:
                prediction, plot_path, json_path = self.process_customer(customer_ref, sequence_length, batch_size)
                if prediction is not None and plot_path is not None and json_path is not None:
                    results.append({
                        'customer_ref': customer_ref,
                        'prediction': prediction,
                        'plot_path': plot_path,
                        'json_path': json_path
                    })
                    logger.info(f"Customer {customer_ref}: Prediction for next 15-minute interval: {prediction:.4f} kW, "
                               f"Plot: {plot_path}, JSON: {json_path}")
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
        logger.info(f"Customer {result['customer_ref']}: Predicted avg_import_kw for next 15-minute interval: "
                   f"{result['prediction']:.4f} kW, Plot: {result['plot_path']}, JSON: {result['json_path']}")