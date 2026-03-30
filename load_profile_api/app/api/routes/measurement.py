from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
from app.database import get_db
from app.repositories.measurement_repo import MeasurementRepo

router = APIRouter()

# 5. Single customer exact
@router.get("/customer/{customer_ref}")
async def get_exact(customer_ref: int, timestamp: datetime, db: AsyncSession = Depends(get_db)):
    return await MeasurementRepo.get_exact_customer(customer_ref, timestamp, db)

# 6. All customers exact
@router.get("/all")
async def get_all(timestamp: datetime, db: AsyncSession = Depends(get_db)):
    return await MeasurementRepo.get_exact_all(timestamp, db)

# 7. Single customer range
@router.get("/customer/{customer_ref}/range")
async def get_range_customer(customer_ref: int, start: datetime, end: datetime, db: AsyncSession = Depends(get_db)):
    return await MeasurementRepo.get_range_customer(customer_ref, start, end, db)

# 8. All customers range
@router.get("/range")
async def get_range_all(start: datetime, end: datetime, db: AsyncSession = Depends(get_db)):
    return await MeasurementRepo.get_range_all(start, end, db)

# Extra: full customer history
@router.get("/customer/{customer_ref}/all")
async def get_full(customer_ref: int, db: AsyncSession = Depends(get_db)):
    return await MeasurementRepo.get_full_customer(customer_ref, db)