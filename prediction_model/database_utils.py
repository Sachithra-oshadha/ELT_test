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

    def fetch_data(self, customer_ref):
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
                self.logger.warning(f"Dropped {df['timestamp'].isnull().sum()} rows with invalid timestamps for customer {customer_ref}")
                df = df.dropna(subset=['timestamp'])
            if df['timestamp'].dt.tz is not None:
                df['timestamp'] = df['timestamp'].dt.tz_localize(None)
                self.logger.info(f"Converted timezone-aware timestamps to naive for customer {customer_ref}")
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

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
        self.logger.info("Database connection closed")