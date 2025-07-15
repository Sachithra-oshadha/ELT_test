import psycopg2
from psycopg2.extras import execute_batch

class Database:
    def __init__(self, db_config, logger):
        self.db_config = db_config
        self.logger = logger
        self.conn = None
        self.cur = None

    def connect(self):
        try:
            self.conn = psycopg2.connect(**self.db_config)
            self.cur = self.conn.cursor()
            self.logger.info("Successfully connected to PostgreSQL database")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise

    def close(self):
        if self.cur:
            self.cur.close()
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")

    def execute_query(self, query, params=None):
        try:
            self.cur.execute(query, params or ())
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Query execution failed: {e}")
            raise

    def execute_batch(self, query, data):
        try:
            execute_batch(self.cur, query, data, page_size=1000)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            self.logger.error(f"Batch execution failed: {e}")
            raise

    def fetch_one(self, query, params):
        try:
            self.cur.execute(query, params)
            return self.cur.fetchone()
        except Exception as e:
            self.logger.error(f"Fetch failed: {e}")
            raise