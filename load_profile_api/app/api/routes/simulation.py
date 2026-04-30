import sys
import os
import pytz
import asyncio
from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime, timedelta
from typing import Dict, Any
from pydantic import BaseModel

# Add project root to sys.path to import prediction_model
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, 'prediction_model'))

try:
    from prediction_model.main import CustomerBehaviorPipeline
    from prediction_model.logger import setup_logger
    logger = setup_logger()
    pipeline = CustomerBehaviorPipeline(logger=logger)
except ImportError as e:
    print(f"Failed to import prediction model: {e}")
    pipeline = None

router = APIRouter()

class SimulationRequest(BaseModel):
    customer_ref: int
    current_time: datetime


def _get_customers_info_sync():
    """Synchronous helper for DB work — runs in a thread."""
    pipeline.connect_db()
    try:
        return pipeline.db_manager.get_customers_info()
    finally:
        pipeline.close_db()


def _simulate_step_sync(customer_ref: int, end_time: datetime):
    """Synchronous helper for simulation — runs in a thread."""
    from prediction_model.data_processing import preprocess_data
    from prediction_model.prediction_utils import predict_next_timestep
    import numpy as np

    pipeline.connect_db()
    try:
        df = pipeline.fetch_data(customer_ref, end_time=end_time)
        if df.empty:
            return {"status": "skipped", "reason": "No data available in database up to the requested time"}

        last_known_kwh = df['import_kwh'].iloc[-1]
        actual_timestamp = df['timestamp'].iloc[-1]

        sequence_length = 192
        if len(df) < sequence_length + 1:
            return {
                "status": "skipped",
                "reason": f"Insufficient historical data (fetched {len(df)} rows, need {sequence_length + 1} for sequence)"
            }

        model, _, _, last_trained_time = pipeline.load_existing_model(customer_ref)
        if model is None:
            return {"status": "skipped", "reason": "No trained model found for this customer. Please train the model first."}

        # Inference
        scaled_data, scaler, _ = preprocess_data(df, logger)
        last_sequence = scaled_data[-sequence_length:]

        lower_abs, median_abs, upper_abs = predict_next_timestep(model, last_sequence, scaler, last_known_kwh, logger)

        prediction_time = actual_timestamp + timedelta(minutes=15)

        prediction_times = [prediction_time]
        data_to_insert = []
        predictions_deltas = np.diff(np.insert(median_abs, 0, last_known_kwh))
        for i in range(1):
            data_to_insert.append((
                customer_ref,
                float(predictions_deltas[i]),
                float(median_abs[i]),
                prediction_times[i]
            ))
        pipeline.db_manager.save_prediction(data_to_insert, customer_ref)

        return {
            "status": "success",
            "customer_ref": customer_ref,
            "actual_time": actual_timestamp.isoformat() + "Z",
            "actual_import_kwh": last_known_kwh,
            "prediction_time": prediction_time.isoformat() + "Z",
            "predicted_import_kwh": median_abs[0],
            "lower_bound": lower_abs[0],
            "upper_bound": upper_abs[0]
        }
    finally:
        pipeline.close_db()


@router.get("/customers_info")
async def get_customers_info():
    if not pipeline:
        raise HTTPException(status_code=500, detail="Prediction pipeline not initialized")
    try:
        return await asyncio.to_thread(_get_customers_info_sync)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/simulate_step")
async def simulate_step(req: SimulationRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Prediction pipeline not initialized")

    try:
        result = await asyncio.to_thread(_simulate_step_sync, req.customer_ref, req.current_time)
        return result
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
