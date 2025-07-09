import pandas as pd
import psycopg2
import boto3
import tempfile
import re
from psycopg2.extras import execute_batch
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data_insertion_{now}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class LoadProfilePipeline:
    def __init__(self, bucket_name, bucket_prefix=''):
        """Initialize the pipeline with S3 bucket name and optional prefix."""
        self.db_config = {
            'dbname': os.getenv('DB_NAME_S3'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }

        self.s3_config = {
            'bucket_name': bucket_name,
            'prefix': bucket_prefix,
            'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
            'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
            'region_name': os.getenv('AWS_REGION')
        }

        # Validate environment variables
        for key, value in self.db_config.items():
            if value is None:
                logger.error(f"Environment variable for {key} is not set")
                raise ValueError(f"Environment variable for {key} is not set")
        for key, value in self.s3_config.items():
            if key in ['aws_access_key_id', 'aws_secret_access_key', 'region_name'] and value is None:
                logger.error(f"S3 environment variable for {key} is not set")
                raise ValueError(f"S3 environment variable for {key} is not set")
        self.conn = None
        self.cur = None
        self.s3_client = None
        self.temp_dir = tempfile.mkdtemp()

    def connect_db(self):
        """Establish connection to PostgreSQL database."""
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor()
            logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def connect_s3(self):
        """Establish connection to S3."""
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.s3_config['aws_access_key_id'],
                aws_secret_access_key=self.s3_config['aws_secret_access_key'],
                region_name=self.s3_config['region_name']
            )
            # Test connection by listing buckets
            self.s3_client.list_buckets()
            logger.info("Successfully connected to S3")
            logger.info(f"Connected to bucket: {self.s3_config['bucket_name']}")  
        except Exception as e:
            logger.error(f"Failed to connect to S3: {e}")
            raise

    def close_db(self):
        """Close database connection."""
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
    
    def cleanup_temp_files(self):
        """Remove temporary directory and its contents."""
        try:
            for root, _, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
            os.rmdir(self.temp_dir)
            logger.info(f"Cleaned up temporary directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to clean up temporary directory: {e}")

    def is_file_processed(self, s3_key):
        """Check if a file has already been processed."""
        try:
            self.cur.execute(
                "SELECT 1 FROM processed_files WHERE s3_path = %s;",
                (s3_key,)
            )
            result = self.cur.fetchone()
            return result is not None
        except Exception as e:
            logger.error(f"Failed to check processed files: {e}")
            raise

    def mark_file_processed(self, s3_key):
        """Mark a file as processed in the database."""
        try:
            file_name = os.path.basename(s3_key)
            insert_query = """
            INSERT INTO processed_files (file_name, s3_path, processed_at)
            VALUES (%s, %s, %s);
            """
            self.cur.execute(insert_query, (file_name, s3_key, datetime.now()))
            self.conn.commit()
            logger.info(f"Marked file as processed: {s3_key}")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to mark file as processed: {e}")
            raise

    def list_s3_files(self):
        """List Excel and CSV files in the S3 bucket."""
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            pages = paginator.paginate(
                Bucket=self.s3_config['bucket_name'],
                Prefix=self.s3_config['prefix']
            )
            files = []
            for page in pages:
                if 'Contents' in page:
                    for obj in page['Contents']:
                        file_key = obj['Key']
                        if re.search(r'\.(csv|xlsx|xls)$', file_key, re.IGNORECASE):
                            files.append(file_key)
            logger.info(f"Found {len(files)} Excel/CSV files in S3 bucket")
            return files
        except Exception as e:
            logger.error(f"Failed to list S3 files: {e}")
            raise

    def download_s3_file(self, s3_key):
        """Download an S3 file to a temporary local path."""
        try:
            file_name = os.path.basename(s3_key)
            local_path = os.path.join(self.temp_dir, file_name)
            self.s3_client.download_file(
                self.s3_config['bucket_name'],
                s3_key,
                local_path
            )
            logger.info(f"Downloaded file from S3: {s3_key} to {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"Failed to download S3 file {s3_key}: {e}")
            raise

    def read_data(self, file_path):
        """Read the Excel or CSV file into a pandas DataFrame."""
        try:
            file_extension = os.path.splitext(file_path)[1].lower()
            if file_extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
                logger.info(f"Successfully read Excel file: {file_path}")
            elif file_extension == '.csv':
                df = pd.read_csv(file_path)
                logger.info(f"Successfully read CSV file: {file_path}")
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            return df
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise

    def insert_customers(self, df):
        """Insert unique customers into the customer table."""
        customers = df[['CUSTOMER_REF']].drop_duplicates().dropna()
        customer_data = [(int(row['CUSTOMER_REF']),) for _, row in customers.iterrows()]
        
        insert_query = """
        INSERT INTO customer (customer_ref)
        VALUES (%s)
        ON CONFLICT (customer_ref) DO NOTHING;
        """
        
        try:
            execute_batch(self.cur, insert_query, customer_data, page_size=1000)
            self.conn.commit()
            logger.info(f"Inserted {len(customer_data)} customers")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert customers: {e}")
            raise

    def insert_meters(self, df):
        """Insert unique meters into the meter table."""
        meters = df[['SERIAL', 'CUSTOMER_REF']].drop_duplicates().dropna()
        meter_data = [(int(row['SERIAL']), int(row['CUSTOMER_REF'])) for _, row in meters.iterrows()]
        
        insert_query = """
        INSERT INTO meter (serial, customer_ref)
        VALUES (%s, %s)
        ON CONFLICT (serial) DO NOTHING;
        """
        
        try:
            execute_batch(self.cur, insert_query, meter_data, page_size=1000)
            self.conn.commit()
            logger.info(f"Inserted {len(meter_data)} meters")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert meters: {e}")
            raise

    def insert_measurements(self, df):
        """Insert measurement data into the measurement table."""
        # Select relevant columns and handle NULLs
        measurement_cols = [
            'SERIAL', 'DATE', 'TIME', 'OBIS', 'AVG._IMPORT_KW (kW)', 'IMPORT_KWH (kWh)',
            'AVG._EXPORT_KW (kW)', 'EXPORT_KWH (kWh)', 'AVG._IMPORT_KVA (kVA)',
            'AVG._EXPORT_KVA (kVA)', 'IMPORT_KVARH (kvarh)', 'EXPORT_KVARH (kvarh)',
            'POWER_FACTOR', 'AVG._CURRENT (V)', 'AVG._VOLTAGE (V)'
        ]
        df_measurements = df[measurement_cols].copy()
        
        # Rename columns to match database
        df_measurements.columns = [
            'serial', 'date', 'time', 'obis', 'avg_import_kw', 'import_kwh',
            'avg_export_kw', 'export_kwh', 'avg_import_kva', 'avg_export_kva',
            'import_kvarh', 'export_kvarh', 'power_factor', 'avg_current', 'avg_voltage'
        ]
        
        # Combine DATE and TIME into a datetime column
        df_measurements['timestamp'] = pd.to_datetime(
            df_measurements['date'] + ' ' + df_measurements['time'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        
        # Drop rows where timestamp could not be parsed
        invalid_rows = df_measurements[df_measurements['timestamp'].isna()]
        if not invalid_rows.empty:
            logger.warning(f"Dropped {len(invalid_rows)} rows due to invalid DATE/TIME format")
            df_measurements = df_measurements.dropna(subset=['timestamp'])
        
        # Convert data types and handle NULLs
        df_measurements['serial'] = df_measurements['serial'].astype(int)
        df_measurements['obis'] = df_measurements['obis'].astype(str)
        df_measurements['power_factor'] = df_measurements['power_factor'].clip(lower=-1, upper=1)
        
        measurement_data = [
            (
                row['serial'], row['timestamp'], row['obis'],
                row['avg_import_kw'] if pd.notnull(row['avg_import_kw']) else None,
                row['import_kwh'] if pd.notnull(row['import_kwh']) else None,
                row['avg_export_kw'] if pd.notnull(row['avg_export_kw']) else None,
                row['export_kwh'] if pd.notnull(row['export_kwh']) else None,
                row['avg_import_kva'] if pd.notnull(row['avg_import_kva']) else None,
                row['avg_export_kva'] if pd.notnull(row['avg_export_kva']) else None,
                row['import_kvarh'] if pd.notnull(row['import_kvarh']) else None,
                row['export_kvarh'] if pd.notnull(row['export_kvarh']) else None,
                row['power_factor'] if pd.notnull(row['power_factor']) else None,
                row['avg_current'] if pd.notnull(row['avg_current']) else None,
                row['avg_voltage'] if pd.notnull(row['avg_voltage']) else None
            )
            for _, row in df_measurements.iterrows()
        ]
        
        insert_query = """
        INSERT INTO measurement (
            serial, timestamp, obis, avg_import_kw, import_kwh, avg_export_kw, export_kwh,
            avg_import_kva, avg_export_kva, import_kvarh, export_kvarh, power_factor,
            avg_current, avg_voltage
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (serial, timestamp) DO NOTHING;
        """
        
        try:
            execute_batch(self.cur, insert_query, measurement_data, page_size=1000)
            self.conn.commit()
            logger.info(f"Inserted {len(measurement_data)} measurements")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert measurements: {e}")
            raise

    def insert_phase_measurements(self, df):
        """Insert phase-specific measurements into the phase_measurement table."""
        phase_data = []
        
        # Combine DATE and TIME into a datetime for timestamp
        df['timestamp'] = pd.to_datetime(
            df['DATE'] + ' ' + df['TIME'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        
        # Drop rows where timestamp could not be parsed
        invalid_rows = df[df['timestamp'].isna()]
        if not invalid_rows.empty:
            logger.warning(f"Dropped {len(invalid_rows)} rows due to invalid DATE/TIME format in phase measurements")
            df = df.dropna(subset=['timestamp'])
        
        # Process Phase A, B, C measurements
        for _, row in df.iterrows():
            serial = int(row['SERIAL'])
            timestamp = row['timestamp']
            
            # Phase A
            if pd.notnull(row['PHASE_A_INST._CURRENT (A)']) or pd.notnull(row['PHASE_A_INST._VOLTAGE (V)']) or pd.notnull(row['INST._POWER_FACTOR']):
                inst_power_factor = row['INST._POWER_FACTOR'] if pd.notnull(row['INST._POWER_FACTOR']) else None
                
                phase_data.append((
                    serial, timestamp, 'A',
                    row['PHASE_A_INST._CURRENT (A)'] if pd.notnull(row['PHASE_A_INST._CURRENT (A)']) else None,
                    row['PHASE_A_INST._VOLTAGE (V)'] if pd.notnull(row['PHASE_A_INST._VOLTAGE (V)']) else None,
                    inst_power_factor
                ))
            
            # Phase B
            if pd.notnull(row['PHASE_B_INST._CURRENT (A)']) or pd.notnull(row['PHASE_B_INST._VOLTAGE (V)']):
                phase_data.append((
                    serial, timestamp, 'B',
                    row['PHASE_B_INST._CURRENT (A)'] if pd.notnull(row['PHASE_B_INST._CURRENT (A)']) else None,
                    row['PHASE_B_INST._VOLTAGE (V)'] if pd.notnull(row['PHASE_B_INST._VOLTAGE (V)']) else None,
                    None  # No INST._POWER_FACTOR for Phase B
                ))
            
            # Phase C
            if pd.notnull(row['PHASE_C_INST._CURRENT (A)']) or pd.notnull(row['PHASE_C_INST._VOLTAGE (V)']):   
                phase_data.append((
                    serial, timestamp, 'C',
                    row['PHASE_C_INST._CURRENT (A)'] if pd.notnull(row['PHASE_C_INST._CURRENT (A)']) else None,
                    row['PHASE_C_INST._VOLTAGE (V)'] if pd.notnull(row['PHASE_C_INST._VOLTAGE (V)']) else None,
                    None  # No INST._POWER_FACTOR for Phase C
                ))
        
        insert_query = """
        INSERT INTO phase_measurement (
            serial, timestamp, phase, inst_current, inst_voltage, inst_power_factor
        )
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (serial, timestamp, phase) DO NOTHING;
        """
        
        try:
            execute_batch(self.cur, insert_query, phase_data, page_size=1000)
            self.conn.commit()
            logger.info(f"Inserted {len(phase_data)} phase measurements")
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Failed to insert phase measurements: {e}")
            raise
    
    def process_file(self, s3_key):
        """Process a single S3 file."""
        local_path = None
        try:
            if self.is_file_processed(s3_key):
                logger.info(f"Skipping already processed file: {s3_key}")
                return

            local_path = self.download_s3_file(s3_key)
            df = self.read_data(local_path)

            self.insert_customers(df)
            self.insert_meters(df)
            self.insert_measurements(df)
            self.insert_phase_measurements(df)

            self.mark_file_processed(s3_key)
            logger.info(f"Successfully processed file: {s3_key}")

        except Exception as e:
            logger.error(f"Failed to process file {s3_key}: {e}")
            raise
        finally:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
                logger.info(f"Removed temporary file: {local_path}")

    def run(self):
        """Execute the full data insertion pipeline for all S3 files."""
        start_time = datetime.now()
        logger.info("Starting S3 data insertion pipeline")
        
        try:
            self.connect_db()
            self.connect_s3()
            
            s3_files = self.list_s3_files()
            if not s3_files:
                logger.warning("No Excel or CSV files found in S3 bucket")
                return
            
            for s3_key in s3_files:
                logger.info(f"Processing S3 file: {s3_key}")
                self.process_file(s3_key)
            
            logger.info(f"Pipeline completed successfully in {datetime.now() - start_time}")
        
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        finally:
            self.close_db()
            self.cleanup_temp_files()

if __name__ == "__main__":
    bucket_name = os.getenv('S3_BUCKET_NAME')
    bucket_prefix = os.getenv('S3_BUCKET_PREFIX', '')
    
    if not bucket_name:
        logger.error("S3_BUCKET_NAME environment variable not set")
        exit(1)
    
    pipeline = LoadProfilePipeline(bucket_name, bucket_prefix)
    pipeline.run()