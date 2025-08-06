import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logger import setup_logger
from config import DB_CONFIG, S3_CONFIG, S3_BUCKET_NAME, S3_BUCKET_PREFIX, REQUIRED_ENV_VARS
from database import Database
from s3_client import S3Client
from file_processor import FileProcessor

logger = setup_logger()

def validate_env_vars():
    for group, vars in REQUIRED_ENV_VARS.items():
        for var in vars:
            if not os.getenv(var):
                logger.error(f"Missing required environment variable: {var}")
                raise EnvironmentError(f"Missing required environment variable: {var}")

def main():
    validate_env_vars()
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")

    db = Database(DB_CONFIG, logger)
    db.connect()

    s3 = S3Client(S3_CONFIG, S3_BUCKET_NAME, S3_BUCKET_PREFIX, logger)
    s3.connect()

    processor = FileProcessor(db, s3, temp_dir, logger)

    try:
        files = s3.list_files()
        if not files:
            logger.warning("No valid files found in S3 bucket.")
            return

        for s3_key in files:
            logger.info(f"Processing file: {s3_key}")
            processor.process_file(s3_key)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
    finally:
        db.close()
        for root, _, files in os.walk(temp_dir):
            for file in files:
                os.remove(os.path.join(root, file))
        os.rmdir(temp_dir)
        logger.info("Temporary files cleaned up.")

if __name__ == "__main__":
    main()