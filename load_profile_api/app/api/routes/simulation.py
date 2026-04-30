import sys
import os
import pytz
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

@router.get("/customers_info")
async def get_customers_info():
    if not pipeline:
        raise HTTPException(status_code=500, detail="Prediction pipeline not initialized")
    try:
        pipeline.connect_db()
        results = pipeline.db_manager.get_customers_info()
        pipeline.close_db()
        return results
    except Exception as e:
        if pipeline and pipeline.db_manager.conn:
            pipeline.close_db()
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/simulate_step")
async def simulate_step(req: SimulationRequest):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Prediction pipeline not initialized")

    try:
        # Use current_time as the data boundary (the "now" in the simulation)
        end_time = req.current_time

        # Run pipeline up to next_time
        # In a real system, the new data point would arrive here.
        # Since we use fetch_data with end_time, it will act as if data only exists up to next_time.
        # But wait, run() processes ALL customers. We can optimize it by temporarily filtering.
        # Let's modify pipeline.fetch_customer_refs temporarily or just ignore it.
        # Actually, let's just use the underlying methods directly for efficiency.
        
        pipeline.connect_db()
        try:
            cur = pipeline.db_manager.cur
            conn = pipeline.db_manager.conn
            
            df = pipeline.fetch_data(req.customer_ref, end_time=end_time)
            if df.empty:
                return {"status": "skipped", "reason": "No data available in database up to the requested time"}
                
            last_known_kwh = df['import_kwh'].iloc[-1]
            actual_timestamp = df['timestamp'].iloc[-1]
            
            # Import logic
            from prediction_model.data_processing import preprocess_data
            from prediction_model.prediction_utils import predict_next_timestep
            import numpy as np
            
            sequence_length = 192
            if len(df) < sequence_length + 1:
                return {
                    "status": "skipped", 
                    "reason": f"Insufficient historical data (fetched {len(df)} rows, need {sequence_length + 1} for sequence)"
                }
                
            model, _, _, last_trained_time = pipeline.load_existing_model(req.customer_ref)
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
                    req.customer_ref,
                    float(predictions_deltas[i]),
                    float(median_abs[i]),
                    prediction_times[i]
                ))
            pipeline.db_manager.save_prediction(data_to_insert, req.customer_ref)
            
            return {
                "status": "success",
                "customer_ref": req.customer_ref,
                "actual_time": actual_timestamp.isoformat() + "Z",
                "actual_import_kwh": last_known_kwh,
                "prediction_time": prediction_time.isoformat() + "Z",
                "predicted_import_kwh": median_abs[0],
                "lower_bound": lower_abs[0],
                "upper_bound": upper_abs[0]
            }
        finally:
            pipeline.close_db()
            
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
