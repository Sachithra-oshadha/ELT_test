import tempfile
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logger import setup_logger
from config import (
    DB_CONFIG,
    S3_CONFIG,
    S3_BUCKET_NAME,
    S3_BUCKET_PREFIX,
    LOCAL_INPUT_DIR,
    REQUIRED_ENV_VARS
)
from database import Database
from file_client import FileClient
from file_processor import FileProcessor

logger = setup_logger()


def validate_env_vars():
    """
    Validate required environment variables.
    S3 vars are optional because we support local fallback.
    """
    for group, vars in REQUIRED_ENV_VARS.items():
        for var in vars:
            if not os.getenv(var):
                logger.error(f"Missing required environment variable: {var}")
                raise EnvironmentError(f"Missing required environment variable: {var}")


def main():
    validate_env_vars()

    # --------------------------------------------------
    # Temporary working directory
    # --------------------------------------------------
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")

    # --------------------------------------------------
    # Database setup
    # --------------------------------------------------
    db = Database(DB_CONFIG, logger)
    db.connect()

    # --------------------------------------------------
    # Unified file client (S3 → Local fallback)
    # --------------------------------------------------
    file_client = FileClient(
        logger=logger,
        s3_config=S3_CONFIG,
        bucket_name=S3_BUCKET_NAME,
        prefix=S3_BUCKET_PREFIX,
        local_input_dir=LOCAL_INPUT_DIR
    )

    file_client.connect()

    # --------------------------------------------------
    # File processor
    # --------------------------------------------------
    processor = FileProcessor(
        db=db,
        s3_client=file_client,  # name kept for backward compatibility
        temp_dir=temp_dir,
        logger=logger
    )

    try:
        files = file_client.list_files()

        if not files:
            logger.warning("No valid CSV / Excel files found.")
            return

        for file_key in files:
            logger.info(f"Processing file: {file_key}")
            processor.process_file(file_key)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

    finally:
        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------
        db.close()

        for root, _, files in os.walk(temp_dir):
            for file in files:
                os.remove(os.path.join(root, file))

        os.rmdir(temp_dir)
        logger.info("Temporary files cleaned up.")


if __name__ == "__main__":
    main()
