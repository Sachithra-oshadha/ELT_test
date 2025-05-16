import pandas as pd
import psycopg2
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import logging
from datetime import datetime
import os
from dotenv import load_dotenv
import tempfile

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('customer_behavior.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CustomerBehaviorPipeline:
    def __init__(self):
        """Initialize the pipeline with database configuration and Bi-LSTM parameters."""
        self.db_config = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        self.conn = None
        self.cur = None
        self.last_processed_timestamp = {}
        self.output_base_dir = "customer_plots"
        self.time_steps = 2880  # 15-minute intervals for 1 month (4 * 24 * 30 = 2880, assuming 30-day month)
        self.scaler = StandardScaler()
        plt.style.use('seaborn-v0_8')

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

    def fetch_customer_data(self, customer_ref, min_timestamp=None):
        """Fetch measurement data for a specific customer."""
        query = """
        SELECT m.serial, m.timestamp, m.avg_import_kw, m.import_kwh, m.power_factor,
               TO_TIMESTAMP(m.timestamp / 1000) AS datetime
        FROM measurement m
        JOIN meter mt ON m.serial = mt.serial
        WHERE mt.customer_ref = %s
        """
        params = [customer_ref]
        if min_timestamp:
            query += " AND m.timestamp > %s"
            params.append(min_timestamp)
        
        try:
            df = pd.read_sql(query, self.conn, params=params)
            if len(df) < self.time_steps:
                logger.warning(f"Insufficient data for customer {customer_ref}: {len(df)} records, need {self.time_steps}")
                return pd.DataFrame()
            logger.info(f"Fetched {len(df)} records for customer {customer_ref}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for customer {customer_ref}: {e}")
            raise

    def prepare_features(self, df):
        """Prepare features for Bi-LSTM model training."""
        if df.empty:
            return None, None, None
        
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        features = ['hour', 'day_of_week', 'month', 'import_kwh', 'power_factor']
        X = df[features].fillna(0).values
        y = df['avg_import_kw'].fillna(0).values
        
        X = self.scaler.fit_transform(X) if not hasattr(self.scaler, 'mean_') else self.scaler.transform(X)
        
        X_seq, y_seq = self.create_sequences(X, y, self.time_steps)
        
        return X_seq, y_seq, df

    def create_sequences(self, X, y, time_steps):
        """Create sequences for time-series data."""
        X_seq, y_seq = [], []
        for i in range(len(X) - time_steps):
            X_seq.append(X[i:i + time_steps])
            y_seq.append(y[i + time_steps])
        if not X_seq:
            return None, None
        return np.array(X_seq), np.array(y_seq)

    def train_model(self, X, y, existing_model=None):
        """Train or incrementally update a Bi-LSTM model."""
        if X is None or y is None or len(X) < self.time_steps + 10:
            logger.warning("Insufficient data for training")
            return None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if existing_model:
            model = existing_model
            logger.info("Using existing Bi-LSTM model for incremental training")
        else:
            model = Sequential([
                Bidirectional(LSTM(32, return_sequences=False), 
                              input_shape=(self.time_steps, X.shape[2])),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            logger.info("Created new Bi-LSTM model")
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        y_pred = model.predict(X_test, verbose=0)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model training completed (MSE: {mse:.4f}, R2: {r2:.4f})")
        return model, mse, r2

    def serialize_model(self, model):
        """Serialize the Bi-LSTM model to a binary format using .keras format."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                model.save(tmp_file.name)
                tmp_file_path = tmp_file.name
            
            with open(tmp_file_path, 'rb') as f:
                buffer = BytesIO(f.read())
            
            os.unlink(tmp_file_path)
            
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            logger.error(f"Failed to serialize model: {e}")
            raise

    def deserialize_model(self, model_data):
        """Deserialize the Bi-LSTM model from binary format."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.keras', delete=False) as tmp_file:
                tmp_file.write(model_data)
                tmp_file_path = tmp_file.name
            
            model = load_model(tmp_file_path)
            
            os.unlink(tmp_file_path)
            
            return model
        except Exception as e:
            logger.error(f"Failed to deserialize model: {e}")
            raise

    def store_model(self, customer_ref, model, mse, r2):
        """Store the trained model in the customer_model table."""
        if model is None:
            logger.warning(f"No model to store for customer {customer_ref}")
            return
        
        model_data = self.serialize_model(model)
        
        query = """
        INSERT INTO customer_model (customer_ref, model_data, trained_at, mse, r2_score)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (customer_ref)
        DO UPDATE SET
            model_data = EXCLUDED.model_data,
            trained_at = EXCLUDED.trained_at,
            mse = EXCLUDED.mse,
            r2_score = EXCLUDED.r2_score,
            updated_at = CURRENT_TIMESTAMP;
        """
        params = (customer_ref, psycopg2.Binary(model_data), datetime.now(), mse, r2)
        
        try:
            self.cur.execute(query, params)
            self.conn.commit()
            logger.info(f"Stored model for customer {customer_ref} (MSE: {mse:.4f}, R2: {r2:.4f})")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to store model for customer {customer_ref}: {e}")
            raise

    def get_customers(self):
        """Fetch all customer references."""
        query = "SELECT customer_ref FROM customer;"
        try:
            self.cur.execute(query)
            customers = [row[0] for row in self.cur.fetchall()]
            logger.info(f"Found {len(customers)} customers")
            return customers
        except Exception as e:
            logger.error(f"Failed to fetch customers: {e}")
            raise

    def check_new_data(self, customer_ref):
        """Check for new data since the last model training."""
        query = """
        SELECT MAX(timestamp)
        FROM measurement m
        JOIN meter mt ON m.serial = mt.serial
        WHERE mt.customer_ref = %s;
        """
        self.cur.execute(query, (customer_ref,))
        max_timestamp = self.cur.fetchone()[0]
        
        last_trained = self.last_processed_timestamp.get(customer_ref, 0)
        if max_timestamp and max_timestamp > last_trained:
            logger.info(f"New data detected for customer {customer_ref} (max timestamp: {max_timestamp})")
            return max_timestamp
        return None

    def retrieve_model(self, customer_ref):
        """Retrieve the stored model for a customer."""
        query = """
        SELECT model_data, mse, r2_score, trained_at
        FROM customer_model
        WHERE customer_ref = %s;
        """
        try:
            self.cur.execute(query, (customer_ref,))
            result = self.cur.fetchone()
            if result:
                logger.warning(f"Ignoring existing model for {customer_ref} due to potential time_steps change")
                return None, None, None, None
            logger.warning(f"No model found for customer {customer_ref}")
            return None, None, None, None
        except Exception as e:
            logger.error(f"Failed to retrieve model for customer {customer_ref}: {e}")
            raise

    def analyze_customer_behavior(self, customer_ref):
        """Analyze customer behavior, return key metrics, and save to CSV."""
        df = self.fetch_customer_data(customer_ref)
        if df.empty:
            logger.warning(f"No data available for customer {customer_ref}")
            return None

        metrics = {
            'max_usage_kw': df['avg_import_kw'].max(),
            'min_usage_kw': df['avg_import_kw'].min(),
            'avg_usage_kw': df['avg_import_kw'].mean(),
            'total_kwh': df['import_kwh'].sum(),
            'peak_hour': df.groupby(df['datetime'].dt.hour)['avg_import_kw'].mean().idxmax(),
            'avg_power_factor': df['power_factor'].mean(),
            'usage_std': df['avg_import_kw'].std()
        }
        
        metrics_summary = "\n".join([f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}" 
                                    for key, value in metrics.items()])
        logger.info(f"Behavior metrics for customer {customer_ref}:\n{metrics_summary}")

        customer_dir = os.path.join(self.output_base_dir, str(customer_ref))
        if not os.path.exists(customer_dir):
            os.makedirs(customer_dir)
            logger.info(f"Created directory for customer {customer_ref}: {customer_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'metrics_{timestamp}.csv'
        metrics_path = os.path.join(customer_dir, filename)

        try:
            metrics_df = pd.DataFrame([metrics])
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Saved metrics to {metrics_path}")
        except Exception as e:
            logger.error(f"Failed to save metrics for customer {customer_ref}: {e}")

        return metrics

    def plot_usage_patterns(self, customer_ref, save_path=None):
        """Plot customer usage patterns and save with timestamp."""
        df = self.fetch_customer_data(customer_ref)
        if df.empty:
            logger.warning(f"No data to plot for customer {customer_ref}")
            return None

        plt.figure(figsize=(12, 6))
        hourly_usage = df.groupby(df['datetime'].dt.hour)['avg_import_kw'].mean()
        plt.plot(hourly_usage.index, hourly_usage.values, 'b-', label='Average kW')
        
        plt.title(f'Customer {customer_ref} - Hourly Usage Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Import (kW)')
        plt.grid(True)
        plt.legend()
        
        customer_dir = os.path.join(self.output_base_dir, str(customer_ref))
        if not os.path.exists(customer_dir):
            os.makedirs(customer_dir)
            logger.info(f"Created directory for customer {customer_ref}: {customer_dir}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'usage_pattern_{timestamp}.png'
        save_path = os.path.join(customer_dir, filename)
        
        plt.savefig(save_path)
        logger.info(f"Saved usage pattern plot to {save_path}")
        plt.close()
        return save_path

    def plot_feature_importance(self, customer_ref, save_path=None):
        """Placeholder: Bi-LSTM does not provide feature importance directly."""
        logger.warning(f"Feature importance not supported for Bi-LSTM for customer {customer_ref}")
        return None

    def run(self):
        """Run the customer behavior analysis and model training pipeline."""
        start_time = datetime.now()
        logger.info("Starting customer behavior pipeline")
        
        try:
            self.connect_db()
            customers = self.get_customers()
            
            for customer_ref in customers:
                max_timestamp = self.check_new_data(customer_ref)
                existing_model, _, _, _ = self.retrieve_model(customer_ref)
                
                if max_timestamp:
                    df = self.fetch_customer_data(customer_ref, self.last_processed_timestamp.get(customer_ref))
                    X, y, _ = self.prepare_features(df)
                    model, mse, r2 = self.train_model(X, y, existing_model=existing_model)
                    self.store_model(customer_ref, model, mse, r2)
                    self.last_processed_timestamp[customer_ref] = max_timestamp
                
                elif not existing_model:
                    df = self.fetch_customer_data(customer_ref)
                    X, y, _ = self.prepare_features(df)
                    model, mse, r2 = self.train_model(X, y)
                    self.store_model(customer_ref, model, mse, r2)
                    max_timestamp = df['timestamp'].max() if not df.empty else 0
                    self.last_processed_timestamp[customer_ref] = max_timestamp
                
                else:
                    logger.info(f"No new data for customer {customer_ref}, existing model found")
                
                metrics = self.analyze_customer_behavior(customer_ref)
                if metrics:
                    self.plot_usage_patterns(customer_ref)
                    self.plot_feature_importance(customer_ref)
            
            logger.info(f"Pipeline completed in {datetime.now() - start_time}")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            self.close_db()

if __name__ == "__main__":
    pipeline = CustomerBehaviorPipeline()
    pipeline.run()