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