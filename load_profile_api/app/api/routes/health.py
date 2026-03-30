from fastapi import APIRouter
from app.config import DB_CONFIG

router = APIRouter()

@router.get("/health")
def health():
    return {
        "status": "ok",
        "database": {
            "name": DB_CONFIG.get("dbname"),
            "host": DB_CONFIG.get("host"),
            "port": DB_CONFIG.get("port"),
            "user": DB_CONFIG.get("user")
        }
    }