import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'host': os.getenv('DB_HOST'),
    'port': os.getenv('DB_PORT')
}

def get_database_url():
    return (
        f"postgresql+asyncpg://{DB_CONFIG['user']}:"
        f"{DB_CONFIG['password']}@{DB_CONFIG['host']}:"
        f"{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    )

S3_CONFIG = {
    'aws_access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
    'aws_secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
    'region_name': os.getenv('AWS_REGION')
}

S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
S3_BUCKET_PREFIX = os.getenv('S3_BUCKET_PREFIX', '')

REQUIRED_ENV_VARS = {
    'DB': ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'DB_PORT'],
    'S3': ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_REGION', 'S3_BUCKET_NAME']
}

OUTPUT_BASE_DIR = "customer_outputs_bilstm_day"

LOCAL_INPUT_DIR = os.getenv("LOCAL_INPUT_DIR")