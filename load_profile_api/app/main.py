from fastapi import FastAPI
from app.api.routes import prediction, measurement, health

app = FastAPI(title="Load Profile API")

app.include_router(prediction.router, prefix="/predictions", tags=["Predictions"])
app.include_router(measurement.router, prefix="/measurements", tags=["Measurements"])
app.include_router(health.router, tags=["Health"])