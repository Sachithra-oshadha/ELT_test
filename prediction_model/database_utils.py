from imports import *

class DatabaseManager:
    def __init__(self, db_config, logger: logging.Logger):
        self.db_config = db_config
        self.conn = None
        self.cur = None
        self.logger = logger

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor()
            self.logger.info("Connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def fetch_customer_refs(self):
        try:
            self.cur.execute("SELECT customer_ref FROM customer")
            return [row[0] for row in self.cur.fetchall()]
        except Exception as e:
            self.logger.error(f"Error fetching customer references: {e}")
            raise

    def fetch_data(self, customer_ref, end_time=None):
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
            """
            params = [customer_ref]
            if end_time:
                query += " AND m.timestamp <= %s"
                params.append(end_time)
            query += " ORDER BY m.timestamp"
            
            df = pd.read_sql(query, self.conn, params=tuple(params))
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isnull().any():
                self.logger.warning(f"Dropped {df['timestamp'].isnull().sum()} rows with invalid timestamps for customer {customer_ref}")
                df = df.dropna(subset=['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_convert('UTC').dt.tz_localize(None)
                self.logger.info(f"Converted timezone-aware timestamps to UTC naive for customer {customer_ref}")
            df['timestamp'] = df['timestamp'].dt.round('15min')
            valid_time_range = (df['timestamp'] >= '2000-01-01') & (df['timestamp'] <= '2030-12-31')
            if not valid_time_range.all():
                self.logger.warning(f"Dropped {len(df[~valid_time_range])} rows with out-of-range timestamps for customer {customer_ref}")
                df = df[valid_time_range]
            self.logger.info(f"Fetched {len(df)} valid records for customer {customer_ref} at 15-minute intervals")
            return df
        except Exception as e:
            self.logger.error(f"Error fetching data for customer {customer_ref}: {e}")
            raise

    def load_model(self, customer_ref: int):
        try:
            self.cur.execute("""
                SELECT model_data, mse, r2_score, last_trained_data_timestamp
                FROM customer_model
                WHERE customer_ref = %s
            """, (customer_ref,))
            return self.cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error loading model from DB for customer {customer_ref}: {e}")
            raise

    def save_model(self, customer_ref: int, model_data: bytes, mse: float, r2_score: float, trained_data_timestamp):
        try:
            self.cur.execute("""
                INSERT INTO customer_model (customer_ref, model_data, mse, r2_score, last_trained_data_timestamp)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (customer_ref) DO UPDATE 
                SET model_data = EXCLUDED.model_data,
                    mse = EXCLUDED.mse,
                    r2_score = EXCLUDED.r2_score,
                    last_trained_data_timestamp = EXCLUDED.last_trained_data_timestamp,
                    trained_at = CURRENT_TIMESTAMP
            """, (customer_ref, psycopg2.Binary(model_data), float(mse), float(r2_score), trained_data_timestamp))
            self.conn.commit()
            self.logger.info(f"Saved model for customer {customer_ref} with last trained data timestamp: {trained_data_timestamp}")
        except Exception as e:
            self.logger.error(f"Error saving model for customer {customer_ref}: {e}")
            self.conn.rollback()
            raise

    def save_prediction(self, data_to_insert: list, customer_ref: int):
        try:
            insert_query = """
                INSERT INTO customer_prediction (
                    customer_ref,
                    predicted_usage,
                    predicted_import_kwh,
                    prediction_timestamp
                ) VALUES (%s, %s, %s, %s)
                ON CONFLICT (customer_ref, prediction_timestamp)
                DO UPDATE SET
                    predicted_usage = EXCLUDED.predicted_usage,
                    predicted_import_kwh = EXCLUDED.predicted_import_kwh
            """
            self.cur.executemany(insert_query, data_to_insert)
            self.conn.commit()
            self.logger.info(f"Saved {len(data_to_insert)} prediction rows for customer {customer_ref}")
        except Exception as e:
            self.logger.error(f"Failed to save prediction to database for customer {customer_ref}: {e}")
            self.conn.rollback()
            raise

    def check_existing_prediction(self, customer_ref: int, current_max_timestamp):
        try:
            self.cur.execute("SELECT 1 FROM customer_prediction WHERE customer_ref=%s AND prediction_timestamp > %s", 
                        (customer_ref, current_max_timestamp))
            return self.cur.fetchone()
        except Exception as e:
            self.logger.error(f"Error checking existing prediction for customer {customer_ref}: {e}")
            raise

    def get_customers_info(self):
        try:
            self.cur.execute("""
                SELECT mt.customer_ref, 
                       MIN(m.timestamp) AT TIME ZONE 'UTC' as min_ts,
                       MAX(m.timestamp) AT TIME ZONE 'UTC' as max_ts,
                       COUNT(m.timestamp) as row_count
                FROM measurement m
                JOIN meter mt ON m.serial = mt.serial
                GROUP BY mt.customer_ref
                ORDER BY mt.customer_ref
            """)
            
            customers = self.cur.fetchall()
            results = []
            
            for row in customers:
                customer_ref, min_ts, max_ts, row_count = row
                safe_start = None
                if row_count >= 193:
                    self.cur.execute("""
                        SELECT m.timestamp AT TIME ZONE 'UTC' as ts
                        FROM measurement m
                        JOIN meter mt ON m.serial = mt.serial
                        WHERE mt.customer_ref = %s
                        ORDER BY m.timestamp
                        LIMIT 1 OFFSET 192
                    """, (customer_ref,))
                    safe_row = self.cur.fetchone()
                    if safe_row:
                        safe_start = safe_row[0].isoformat() + "Z"
                
                results.append({
                    "customer_ref": customer_ref,
                    "min_timestamp": min_ts.isoformat() + "Z" if min_ts else None,
                    "max_timestamp": max_ts.isoformat() + "Z" if max_ts else None,
                    "safe_start_time": safe_start,
                    "row_count": row_count
                })
            return results
        except Exception as e:
            self.logger.error(f"Error fetching customers info: {e}")
            raise

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        self.logger.info("Database connection closed")