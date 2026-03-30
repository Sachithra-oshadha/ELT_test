from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from app.database import get_db
from app.repositories.prediction_repo import PredictionRepo

router = APIRouter()

# 1. Single customer exact
@router.get("/customer/{customer_ref}")
async def get_exact(customer_ref: int, timestamp: datetime, db: AsyncSession = Depends(get_db)):
    return await PredictionRepo.get_exact(customer_ref, timestamp, db)

# 2. All customers exact
@router.get("/all")
async def get_all(timestamp: datetime, db: AsyncSession = Depends(get_db)):
    return await PredictionRepo.get_all(timestamp, db)

# 3. Single customer range
@router.get("/customer/{customer_ref}/range")
async def get_range_customer(customer_ref: int, start: datetime, end: datetime, db: AsyncSession = Depends(get_db)):
    return await PredictionRepo.get_range_customer(customer_ref, start, end, db)

# 4. All customers range
@router.get("/range")
async def get_range_all(start: datetime, end: datetime, db: AsyncSession = Depends(get_db)):
    return await PredictionRepo.get_range_all(start, end, db)