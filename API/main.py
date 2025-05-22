from fastapi import FastAPI, HTTPException
from database import get_db_connection
from typing import List, Dict, Any
from datetime import datetime

app = FastAPI()

# --- Customer Endpoints ---
@app.get("/customer", response_model=List[Dict[str, Any]])
async def get_customers():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT customer_ref, created_at, updated_at FROM customer")
            customers = cur.fetchall()
            return customers
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.get("/customer/{customer_ref}", response_model=Dict[str, Any])
async def get_customer(customer_ref: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT customer_ref, created_at, updated_at FROM customer WHERE customer_ref = %s",
                (customer_ref,)
            )
            customer = cur.fetchone()
            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")
            return customer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

# --- Meter Endpoints ---
@app.get("/meter", response_model=List[Dict[str, Any]])
async def get_meters():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT serial, customer_ref, created_at, updated_at FROM meter")
            meters = cur.fetchall()
            return meters
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.get("/meter/{serial}", response_model=Dict[str, Any])
async def get_meter(serial: int):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT serial, customer_ref, created_at, updated_at FROM meter WHERE serial = %s",
                (serial,)
            )
            meter = cur.fetchone()
            if not meter:
                raise HTTPException(status_code=404, detail="Meter not found")
            return meter
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

# --- Measurement Endpoints ---
@app.get("/measurement", response_model=List[Dict[str, Any]])
async def get_measurements():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT serial, timestamp, obis, avg_import_kw, import_kwh, avg_export_kw, 
                       export_kwh, avg_import_kva, avg_export_kva, import_kvarh, 
                       export_kvarh, power_factor, avg_current, avg_voltage, created_at 
                FROM measurement
            """)
            measurements = cur.fetchall()
            return measurements
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.get("/measurement/{serial}/{timestamp}", response_model=Dict[str, Any])
async def get_measurement(serial: int, timestamp: str):
    conn = get_db_connection()
    try:
        # Parse timestamp string to datetime (assuming ISO format, e.g., '2025-05-22T11:26:00+05:30')
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO format (e.g., '2025-05-22T11:26:00+05:30')")
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT serial, timestamp, obis, avg_import_kw, import_kwh, avg_export_kw, 
                       export_kwh, avg_import_kva, avg_export_kva, import_kvarh, 
                       export_kvarh, power_factor, avg_current, avg_voltage, created_at 
                FROM measurement 
                WHERE serial = %s AND timestamp = %s
            """, (serial, ts))
            measurement = cur.fetchone()
            if not measurement:
                raise HTTPException(status_code=404, detail="Measurement not found")
            return measurement
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

# --- Phase Measurement Endpoints ---
@app.get("/phase_measurement", response_model=List[Dict[str, Any]])
async def get_phase_measurements():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT serial, timestamp, phase, inst_current, inst_voltage, 
                       inst_power_factor, created_at 
                FROM phase_measurement
            """)
            phase_measurements = cur.fetchall()
            return phase_measurements
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()

@app.get("/phase_measurement/{serial}/{timestamp}/{phase}", response_model=Dict[str, Any])
async def get_phase_measurement(serial: int, timestamp: str, phase: str):
    conn = get_db_connection()
    try:
        # Validate phase
        if phase not in ('A', 'B', 'C'):
            raise HTTPException(status_code=400, detail="Phase must be 'A', 'B', or 'C'")
        
        # Parse timestamp string to datetime
        try:
            ts = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use ISO format (e.g., '2025-05-22T11:26:00+05:30')")
        
        with conn.cursor() as cur:
            cur.execute("""
                SELECT serial, timestamp, phase, inst_current, inst_voltage, 
                       inst_power_factor, created_at 
                FROM phase_measurement 
                WHERE serial = %s AND timestamp = %s AND phase = %s
            """, (serial, ts, phase))
            phase_measurement = cur.fetchone()
            if not phase_measurement:
                raise HTTPException(status_code=404, detail="Phase measurement not found")
            return phase_measurement
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    finally:
        conn.close()