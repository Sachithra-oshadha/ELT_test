import pandas as pd
import os
from datetime import datetime

class FileProcessor:
    def __init__(self, db, s3_client, temp_dir, logger):
        self.db = db
        self.s3_client = s3_client
        self.temp_dir = temp_dir
        self.logger = logger

    def read_data(self, file_path):
        try:
            ext = os.path.splitext(file_path)[1].lower()
            if ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, engine='openpyxl')
            elif ext == '.csv':
                df = pd.read_csv(file_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
            self.logger.info(f"Successfully read file: {file_path}")
            return df
        except Exception as e:
            self.logger.error(f"Failed to read file: {e}")
            raise

    def is_file_processed(self, s3_key):
        try:
            self.db.cur.execute(
                "SELECT 1 FROM processed_files WHERE s3_path = %s;",
                (s3_key,)
            )
            result = self.db.cur.fetchone()
            return result is not None
        except Exception as e:
            self.logger.error(f"Failed to check processed files: {e}")
            raise

    def mark_file_processed(self, s3_key):
        try:
            file_name = os.path.basename(s3_key)
            insert_query = """
            INSERT INTO processed_files (file_name, s3_path, processed_at)
            VALUES (%s, %s, %s);
            """
            self.db.execute_query(insert_query, (file_name, s3_key, datetime.now()))
            self.logger.info(f"Marked file as processed: {s3_key}")
        except Exception as e:
            self.db.conn.rollback()
            self.logger.error(f"Failed to mark file as processed: {e}")
            raise

    def insert_customers(self, df):
        customers = df[['CUSTOMER_REF']].drop_duplicates().dropna()
        customer_data = [(int(row['CUSTOMER_REF']),) for _, row in customers.iterrows()]
        insert_query = """
        INSERT INTO customer (customer_ref)
        VALUES (%s)
        ON CONFLICT (customer_ref) DO NOTHING;
        """
        try:
            self.db.execute_batch(insert_query, customer_data)
            self.logger.info(f"Inserted {len(customer_data)} customers")
        except Exception as e:
            self.logger.error(f"Failed to insert customers: {e}")
            raise

    def insert_meters(self, df):
        meters = df[['SERIAL', 'CUSTOMER_REF']].drop_duplicates().dropna()
        meter_data = [(int(row['SERIAL']), int(row['CUSTOMER_REF'])) for _, row in meters.iterrows()]
        insert_query = """
        INSERT INTO meter (serial, customer_ref)
        VALUES (%s, %s)
        ON CONFLICT (serial) DO NOTHING;
        """
        try:
            self.db.execute_batch(insert_query, meter_data)
            self.logger.info(f"Inserted {len(meter_data)} meters")
        except Exception as e:
            self.logger.error(f"Failed to insert meters: {e}")
            raise

    def insert_measurements(self, df):
        measurement_cols = [
            'SERIAL', 'DATE', 'TIME', 'OBIS', 'AVG._IMPORT_KW (kW)', 'IMPORT_KWH (kWh)',
            'AVG._EXPORT_KW (kW)', 'EXPORT_KWH (kWh)', 'AVG._IMPORT_KVA (kVA)',
            'AVG._EXPORT_KVA (kVA)', 'IMPORT_KVARH (kvarh)', 'EXPORT_KVARH (kvarh)',
            'POWER_FACTOR', 'AVG._CURRENT (V)', 'AVG._VOLTAGE (V)'
        ]
        df_measurements = df[measurement_cols].copy()
        
        df_measurements.columns = [
            'serial', 'date', 'time', 'obis', 'avg_import_kw', 'import_kwh',
            'avg_export_kw', 'export_kwh', 'avg_import_kva', 'avg_export_kva',
            'import_kvarh', 'export_kvarh', 'power_factor', 'avg_current', 'avg_voltage'
        ]
        
        df_measurements['timestamp'] = pd.to_datetime(
            df_measurements['date'] + ' ' + df_measurements['time'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        
        invalid_rows = df_measurements[df_measurements['timestamp'].isna()]
        if not invalid_rows.empty:
            self.logger.warning(f"Dropped {len(invalid_rows)} rows due to invalid DATE/TIME format")
            df_measurements = df_measurements.dropna(subset=['timestamp'])
        
        df_measurements['serial'] = df_measurements['serial'].astype(int)
        df_measurements['obis'] = df_measurements['obis'].astype(str)
        df_measurements['power_factor'] = df_measurements['power_factor'].clip(lower=-1, upper=1)
        
        measurement_data = [
            (
                row['serial'], row['timestamp'], row['obis'],
                row['avg_import_kw'] if pd.notnull(row['avg_import_kw']) else None,
                row['imshow_kwh'] if pd.notnull(row['import_kwh']) else None,
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
            avg_scurrent, avg_voltage
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (serial, timestamp) DO NOTHING;
        """

        try:
            self.db.execute_batch(insert_query, measurement_data)
            self.logger.info(f"Inserted {len(measurement_data)} measurements")
        except Exception as e:
            self.logger.error(f"Failed to insert measurements: {e}")
            raise

    def insert_phase_measurements(self, df):
        phase_data = []
        
        df['timestamp'] = pd.to_datetime(
            df['DATE'] + ' ' + df['TIME'],
            format='%Y-%m-%d %H:%M:%S',
            errors='coerce'
        )
        
        invalid_rows = df[df['timestamp'].isna()]
        if not invalid_rows.empty:
            self.logger.warning(f"Dropped {len(invalid_rows)} rows due to invalid DATE/TIME format in phase measurements")
            df = df.dropna(subset=['timestamp'])
        
        for _, row in df.iterrows():
            serial = int(row['SERIAL'])
            timestamp = row['timestamp']
            
            if pd.notnull(row['PHASE_A_INST._CURRENT (A)']) or pd.notnull(row['PHASE_A_INST._VOLTAGE (V)']) or pd.notnull(row['INST._POWER_FACTOR']):
                inst_power_factor = row['INST._POWER_FACTOR'] if pd.notnull(row['INST._POWER_FACTOR']) else None
                
                phase_data.append((
                    serial, timestamp, 'A',
                    row['PHASE_A_INST._CURRENT (A)'] if pd.notnull(row['PHASE_A_INST._CURRENT (A)']) else None,
                    row['PHASE_A_INST._VOLTAGE (V)'] if pd.notnull(row['PHASE_A_INST._VOLTAGE (V)']) else None,
                    inst_power_factor
                ))
            
            if pd.notnull(row['PHASE_B_INST._CURRENT (A)']) or pd.notnull(row['PHASE_B_INST._VOLTAGE (V)']):
                phase_data.append((
                    serial, timestamp, 'B',
                    row['PHASE_B_INST._CURRENT (A)'] if pd.notnull(row['PHASE_B_INST._CURRENT (A)']) else None,
                    row['PHASE_B_INST._VOLTAGE (V)'] if pd.notnull(row['PHASE_B_INST._VOLTAGE (V)']) else None,
                    None  # No INST._POWER_FACTOR for Phase B
                ))
            
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
            self.db.execute_batch(insert_query, phase_data)
            self.logger.info(f"Inserted {len(phase_data)} phase measurements")
        except Exception as e:
            self.logger.error(f"Failed to insert phase measurements: {e}")
            raise

    def download_file(self, s3_key):
        try:
            return self.s3_client.download_file(s3_key, self.temp_dir)
        except Exception as e:
            self.logger.error(f"Failed to download file {s3_key}: {e}")
            raise

    def process_file(self, s3_key):
        local_path = None
        try:
            if self.is_file_processed(s3_key):
                self.logger.info(f"Skipping already processed file: {s3_key}")
                return
            local_path = self.download_file(s3_key)
            df = self.read_data(local_path)
            self.insert_customers(df)
            self.insert_meters(df)
            self.insert_measurements(df)
            self.insert_phase_measurements(df)
            self.mark_file_processed(s3_key)
            self.logger.info(f"Successfully processed file: {s3_key}")
        except Exception as e:
            self.logger.error(f"Failed to process file {s3_key}: {e}")
            raise
        finally:
            if local_path and os.path.exists(local_path):
                os.remove(local_path)
                self.logger.info(f"Removed temporary file: {local_path}")