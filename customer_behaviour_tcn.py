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
from typing import Tuple, List
import logging
from dotenv import load_dotenv
import pytz

# Load environment variables from .env file
load_dotenv()

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'customer_behavior_tcn_day_{now}.log'),
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
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length, 0]  # Predicting avg_import_kw
        return torch.FloatTensor(x), torch.FloatTensor([y])

class TCNN(nn.Module):
    def __init__(self, input_size: int, num_channels: List[int], kernel_size: int = 3):
        super(TCNN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, 
                         padding=(kernel_size-1) * dilation, dilation=dilation),
                nn.ReLU(),
                nn.BatchNorm1d(out_channels),
                nn.Dropout(0.2)
            ]
        
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)  # (batch, sequence_length, features) -> (batch, features, sequence_length)
        x = self.network(x)
        x = x[:, :, -1]  # Take the last time step
        x = self.fc(x)
        return x

class CustomerBehaviorPipeline:
    def __init__(self, output_base_dir: str = f"customer_outputs_tcn_day_{now}"):
        """Initialize the pipeline with database configuration and TCNN parameters."""
        self.db_config = {
            'dbname': os.getenv('DB_NAME_TCN_DAY'),
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
            logger.info(f"Created base output directory: {self.output_base_dir}")

    def connect_db(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor()
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise

    def close_db(self):
        """Close database connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

    def fetch_customer_refs(self) -> List[int]:
        """Fetch all customer references from the customer table."""
        try:
            self.cur.execute("SELECT customer_ref FROM customer")
            customer_refs = [row[0] for row in self.cur.fetchall()]
            logger.info(f"Fetched {len(customer_refs)} customer references")
            if not customer_refs:
                logger.warning("No customers found in the customer table")
            return customer_refs
        except psycopg2.Error as e:
            logger.error(f"SQL error while fetching customer references: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching customer references: {e}")
            raise

    def fetch_data(self, customer_ref: int, sequence_length: int) -> pd.DataFrame:
        """Fetch historical measurement data for a customer from measurement and phase_measurement tables."""
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
            logger.info(f"Fetched {len(df)} records for customer {customer_ref}")
            return df
        except psycopg2.Error as e:
            logger.error(f"SQL error while fetching data for customer {customer_ref}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while fetching data for customer {customer_ref}: {e}")
            raise

    def preprocess_data(self, df: pd.DataFrame, sequence_length: int) -> Tuple[np.ndarray, StandardScaler]:
        """Preprocess data and create sequences."""
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

    def load_existing_model(self, customer_ref: int) -> Tuple[nn.Module, float, float]:
        """Load existing model from database if it exists."""
        try:
            self.cur.execute("""
                SELECT model_data, mse, r2_score
                FROM customer_model
                WHERE customer_ref = %s
            """, (customer_ref,))
            result = self.cur.fetchone()
            
            if result:
                model_data, mse, r2_score = result
                model = TCNN(input_size=8, num_channels=[64, 32, 16])
                buffer = io.BytesIO(model_data)
                model = torch.jit.load(buffer)
                logger.info(f"Loaded existing model for customer {customer_ref}")
                return model, mse, r2_score
            logger.info(f"No existing model found for customer {customer_ref}")
            return None, None, None
        except psycopg2.Error as e:
            logger.error(f"SQL error while loading model for customer {customer_ref}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while loading model for customer {customer_ref}: {e}")
            raise

    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                    val_loader: DataLoader, num_epochs: int = 10) -> Tuple[nn.Module, float, float]:
        """Train or incrementally train the TCNN model."""
        try:
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            total_train_loss = 0
            total_val_loss = 0
            total_samples = 0
            
            for epoch in range(num_epochs):
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
                val_samples = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                        outputs = model(batch_x)
                        val_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                        val_samples += batch_x.size(0)
                
                total_train_loss += train_loss
                total_val_loss += val_loss
                total_samples += val_samples
                
                logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader.dataset):.4f}, '
                           f'Val Loss: {val_loss/val_samples:.4f}')
            
            mse = total_val_loss / total_samples
            predictions = []
            actuals = []
            model.eval()
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    outputs = model(batch_x)
                    predictions.extend(outputs.cpu().numpy().flatten())
                    actuals.extend(batch_y.cpu().numpy().flatten())
            
            predictions = np.array(predictions)
            actuals = np.array(actuals)
            ss_tot = np.sum((actuals - np.mean(actuals))**2)
            ss_res = np.sum((actuals - predictions)**2)
            r2_score = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            logger.info(f"Model training completed. MSE: {mse:.4f}, R2 Score: {r2_score:.4f}")
            
            return model, mse, r2_score
        except Exception as e:
            logger.error(f"Failed to train model: {e}")
            raise

    def predict_next_timestep(self, model: nn.Module, last_sequence: np.ndarray, 
                            scaler: StandardScaler) -> float:
        """Predict the next timestep's avg_import_kw."""
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
            logger.info(f"Predicted avg_import_kw: {prediction_value:.4f} kW")
            return prediction_value
        except Exception as e:
            logger.error(f"Failed to make prediction: {e}")
            raise

    def create_prediction_plot(self, df: pd.DataFrame, prediction: float, customer_ref: int, sequence_length: int):
        """Create and save a plot of historical data and prediction."""
        try:
            output_dir = os.path.join(self.output_base_dir, f"customer_{customer_ref}")
            os.makedirs(output_dir, exist_ok=True)
            
            plt.figure(figsize=(12, 6))
            last_n = df.tail(sequence_length)
            timestamps = last_n['timestamp']
            actual_kw = last_n['avg_import_kw']
            
            last_timestamp = timestamps.iloc[-1]
            next_timestamp = last_timestamp + timedelta(minutes=15)
            extended_timestamps = pd.concat([pd.Series(timestamps), pd.Series([next_timestamp])], ignore_index=True)
            extended_kw = np.append(actual_kw.values, prediction)
            
            plt.plot(extended_timestamps, extended_kw, 'b-', label='Actual + Predicted avg_import_kw')
            plt.axvline(x=last_timestamp, color='r', linestyle='--', label='Prediction Point')
            plt.title(f'Customer {customer_ref} - Last {sequence_length} Timesteps and Next Prediction')
            plt.xlabel('Timestamp')
            plt.ylabel('Average Import Power (kW)')
            plt.grid(True)
            plt.legend()
            plt.xticks(rotation=45)
            
            output_path = os.path.join(output_dir, f'prediction_plot_{customer_ref}_{datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(output_path, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved prediction plot at: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to create prediction plot: {e}")
            raise

    def save_model(self, model: nn.Module, customer_ref: int, mse: float, r2_score: float):
        """Save the trained model to the database."""
        try:
            buffer = io.BytesIO()
            torch.jit.save(torch.jit.script(model), buffer)
            model_data = buffer.getvalue()
            
            # Convert numpy.float32 to Python float
            mse = float(mse) if isinstance(mse, np.floating) else mse
            r2_score = float(r2_score) if isinstance(r2_score, np.floating) else r2_score
            
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
            logger.info(f"Saved model for customer {customer_ref} to database")
        except psycopg2.Error as e:
            logger.error(f"SQL error while saving model for customer {customer_ref}: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error while saving model for customer {customer_ref}: {e}")
            raise

    def process_customer(self, customer_ref: int, sequence_length: int = 96, batch_size: int = 32) -> Tuple[float, str]:
        """Process a single customer: fetch data, train model, predict, and save plot."""
        try:
            logger.info(f"Processing customer {customer_ref}")
            
            # Fetch and preprocess data
            df = self.fetch_data(customer_ref, sequence_length)
            if len(df) < sequence_length:
                logger.warning(f"Insufficient data for customer {customer_ref}. Need at least {sequence_length} records.")
                return None, None
            
            scaled_data, scaler = self.preprocess_data(df, sequence_length)
            
            # Create datasets
            dataset = ElectricityDataset(scaled_data, sequence_length)
            train_size = int(0.8 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)
            
            # Load existing model or create new
            model, prev_mse, prev_r2 = self.load_existing_model(customer_ref)
            if model is None:
                model = TCNN(input_size=8, num_channels=[64, 32, 16])
                logger.info(f"Created new TCNN model for customer {customer_ref}")
            
            # Train or incrementally train model
            model, mse, r2_score = self.train_model(model, train_loader, val_loader)
            
            # Predict next timestep
            last_sequence = scaled_data[-sequence_length:]
            prediction = self.predict_next_timestep(model, last_sequence, scaler)
            
            # Create and save prediction plot
            plot_path = self.create_prediction_plot(df, prediction, customer_ref, sequence_length)
            
            # Save model to database
            self.save_model(model, customer_ref, mse, r2_score)
            
            return prediction, plot_path
        except Exception as e:
            logger.error(f"Failed to process customer {customer_ref}: {e}")
            return None, None

    def run(self, sequence_length: int = 96, batch_size: int = 32):
        """Run the pipeline for all customers in the database."""
        try:
            self.connect_db()
            customer_refs = self.fetch_customer_refs()
            
            results = []
            for customer_ref in customer_refs:
                prediction, plot_path = self.process_customer(customer_ref, sequence_length, batch_size)
                if prediction is not None and plot_path is not None:
                    results.append({
                        'customer_ref': customer_ref,
                        'prediction': prediction,
                        'plot_path': plot_path
                    })
                    logger.info(f"Completed processing for customer {customer_ref}. "
                               f"Prediction: {prediction:.4f} kW, Plot: {plot_path}")
                else:
                    logger.warning(f"Skipped customer {customer_ref} due to errors or insufficient data")
            
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
        logger.info(f"Customer {result['customer_ref']}: Predicted avg_import_kw for next 15-minute timestep: {result['prediction']:.4f} kW")
        logger.info(f"Customer {result['customer_ref']}: Prediction plot saved at: {result['plot_path']}")