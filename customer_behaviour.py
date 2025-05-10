import pandas as pd
import psycopg2
from psycopg2.extras import execute_batch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io
import logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from dotenv import load_dotenv
import os
import time

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
        """Initialize the pipeline with database configuration from environment variables."""
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
        self.output_base_dir = "customer_plots"  # Base directory for customer plots
        plt.style.use('seaborn-v0_8')  # Set matplotlib style

        # Create base output directory if it doesn't exist
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
            logger.info(f"Fetched {len(df)} records for customer {customer_ref}")
            return df
        except Exception as e:
            logger.error(f"Failed to fetch data for customer {customer_ref}: {e}")
            raise

    def prepare_features(self, df):
        """Prepare features for model training."""
        if df.empty:
            return None, None, None
        
        # Extract temporal features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['month'] = df['datetime'].dt.month
        
        # Features: hour, day_of_week, month, import_kwh, power_factor
        features = ['hour', 'day_of_week', 'month', 'import_kwh', 'power_factor']
        X = df[features].fillna(0)  # Handle NULLs
        y = df['avg_import_kw'].fillna(0)  # Target variable
        
        return X, y, df

    def train_model(self, X, y, existing_model=None):
        """Train or incrementally update a RandomForestRegressor model."""
        if X is None or y is None or len(X) < 10:  # Minimum data requirement
            return None, None, None
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        if existing_model:
            # Incremental training with warm_start
            model = existing_model
            model.n_estimators += 50  # Add more trees for incremental learning
            model.fit(X_train, y_train)
            logger.info("Incrementally updated existing model")
        else:
            # Train new model
            model = RandomForestRegressor(n_estimators=100, random_state=42, warm_start=True)
            model.fit(X_train, y_train)
            logger.info("Trained new model")
        
        # Evaluate model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        return model, mse, r2

    def serialize_model(self, model):
        """Serialize the model to a binary format."""
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        return buffer.getvalue()

    def deserialize_model(self, model_data):
        """Deserialize the model from binary format."""
        try:
            buffer = BytesIO(model_data)
            model = joblib.load(buffer)
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
                model_data, mse, r2, trained_at = result
                model = self.deserialize_model(model_data)
                logger.info(f"Retrieved model for customer {customer_ref} (trained at: {trained_at})")
                return model, mse, r2, trained_at
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
        
        # Log metrics in a formatted way
        metrics_summary = "\n".join([f"  {key}: {value:.4f}" if isinstance(value, (int, float)) else f"  {key}: {value}" 
                                    for key, value in metrics.items()])
        logger.info(f"Behavior metrics for customer {customer_ref}:\n{metrics_summary}")

        # Save metrics to CSV in customer-specific folder
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
        
        # Hourly usage pattern
        hourly_usage = df.groupby(df['datetime'].dt.hour)['avg_import_kw'].mean()
        plt.plot(hourly_usage.index, hourly_usage.values, 'b-', label='Average kW')
        
        plt.title(f'Customer {customer_ref} - Hourly Usage Pattern')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Import (kW)')
        plt.grid(True)
        plt.legend()
        
        # Create customer-specific folder
        customer_dir = os.path.join(self.output_base_dir, str(customer_ref))
        if not os.path.exists(customer_dir):
            os.makedirs(customer_dir)
            logger.info(f"Created directory for customer {customer_ref}: {customer_dir}")

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'usage_pattern_{timestamp}.png'
        save_path = os.path.join(customer_dir, filename)
        
        plt.savefig(save_path)
        logger.info(f"Saved usage pattern plot to {save_path}")
        plt.close()
        return save_path

    def plot_feature_importance(self, customer_ref, save_path=None):
        """Plot feature importance from the trained model and save with timestamp."""
        model, mse, r2, trained_at = self.retrieve_model(customer_ref)
        if model is None:
            logger.warning(f"No model available for customer {customer_ref}")
            return None

        features = ['hour', 'day_of_week', 'month', 'import_kwh', 'power_factor']
        importances = model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances, y=features)
        
        plt.title(f'Customer {customer_ref} - Feature Importance (R²: {r2:.4f})')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        # Create customer-specific folder
        customer_dir = os.path.join(self.output_base_dir, str(customer_ref))
        if not os.path.exists(customer_dir):
            os.makedirs(customer_dir)
            logger.info(f"Created directory for customer {customer_ref}: {customer_dir}")

        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'feature_importance_{timestamp}.png'
        save_path = os.path.join(customer_dir, filename)
        
        plt.savefig(save_path)
        logger.info(f"Saved feature importance plot to {save_path}")
        plt.close()
        return save_path

    def run(self):
        """Run the customer behavior analysis and model training pipeline."""
        start_time = datetime.now()
        logger.info("Starting customer behavior pipeline")
        
        try:
            self.connect_db()
            customers = self.get_customers()
            
            for customer_ref in customers:
                # Check for new data
                max_timestamp = self.check_new_data(customer_ref)
                
                # Fetch existing model
                existing_model, _, _, _ = self.retrieve_model(customer_ref)
                
                if max_timestamp:  # New data available
                    # Fetch only new data
                    df = self.fetch_customer_data(customer_ref, self.last_processed_timestamp.get(customer_ref))
                    
                    # Prepare features and train model
                    X, y, _ = self.prepare_features(df)
                    model, mse, r2 = self.train_model(X, y, existing_model=existing_model)
                    
                    # Store model
                    self.store_model(customer_ref, model, mse, r2)
                    
                    # Update last processed timestamp
                    self.last_processed_timestamp[customer_ref] = max_timestamp
                
                elif not existing_model:  # No new data, but no existing model
                    # Fetch all data for initial model training
                    df = self.fetch_customer_data(customer_ref)
                    X, y, _ = self.prepare_features(df)
                    model, mse, r2 = self.train_model(X, y)
                    self.store_model(customer_ref, model, mse, r2)
                    # Update timestamp to the max available
                    max_timestamp = df['timestamp'].max() if not df.empty else 0
                    self.last_processed_timestamp[customer_ref] = max_timestamp
                
                else:
                    logger.info(f"No new data for customer {customer_ref}, existing model found")
                
                # Generate analysis and visualizations
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
    # Run pipeline
    pipeline = CustomerBehaviorPipeline()
    pipeline.run()