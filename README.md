# Microgrid Load Profile Prediction System

A full-stack Python pipeline that ingests energy meter data (from AWS S3 or local filesystem), stores it in PostgreSQL, trains per-customer Bidirectional LSTM models, and serves real-time 15-minute-ahead load predictions via a FastAPI backend and browser-based simulation UI.

## Features

- **Data Ingestion (ELT):** Reads `.csv`, `.xlsx`, and `.xls` files from AWS S3 or a local directory, with automatic fallback.
- **Database Integration:** Inserts data into normalized PostgreSQL tables (customers, meters, measurements, phase measurements).
- **Machine Learning:** Trains a quantile Bi-LSTM model per customer for next-step (15-minute) energy consumption prediction with 95% confidence intervals.
- **REST API:** FastAPI-based API for querying measurements, predictions, and running step-by-step simulation.
- **Simulation UI:** Browser-based dashboard to interactively step through time, view actual vs. predicted values, and visualize confidence bands on a real-time chart.
- **Logging:** Comprehensive logging to file and console for all pipeline stages.

## Project Structure

```
├── config.py                   # Database, S3, and output configuration
├── create.sql                  # PostgreSQL schema (tables, views, indexes, triggers)
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (DB, S3, local paths)
│
├── data_load/                  # Data ingestion pipeline
│   ├── main.py                 # Entry point — orchestrates file discovery and processing
│   ├── database.py             # Database connection and query execution (psycopg2)
│   ├── file_client.py          # Unified file client (S3 with local fallback)
│   ├── file_processor.py       # File parsing, validation, and DB insertion
│   └── logger.py               # Logger setup for data ingestion
│
├── prediction_model/           # ML training and prediction pipeline
│   ├── main.py                 # Entry point — CustomerBehaviorPipeline orchestrator
│   ├── database_utils.py       # DB operations for fetching data, saving models/predictions
│   ├── data_processing.py      # Dataset class and preprocessing (differencing, scaling)
│   ├── model_definition.py     # BiLSTMQuantile model (PyTorch)
│   ├── model_training.py       # Training loop with quantile loss and early stopping
│   ├── prediction_utils.py     # Inference and plot generation
│   ├── imports.py              # Shared imports for all prediction modules
│   └── logger.py               # Logger setup for predictions
│
├── load_profile_api/           # FastAPI REST API
│   ├── run.py                  # Uvicorn entry point
│   └── app/
│       ├── main.py             # App setup, CORS, routing, static file serving
│       ├── database.py         # Async SQLAlchemy engine (asyncpg)
│       ├── api/routes/
│       │   ├── health.py       # Health check endpoint
│       │   ├── measurement.py  # Measurement query endpoints
│       │   ├── prediction.py   # Prediction query endpoints
│       │   └── simulation.py   # Step-by-step simulation endpoint
│       └── repositories/
│           ├── measurement_repo.py  # Measurement SQL queries
│           └── prediction_repo.py   # Prediction SQL queries
│
└── ui/                         # Browser-based simulation dashboard
    ├── index.html              # Page structure
    ├── index.css               # Styling (dark theme, design tokens)
    └── index.js                # Chart.js visualization and API interaction
```

## Prerequisites

- Python 3.10+
- PostgreSQL database
- AWS S3 bucket (optional — local filesystem fallback supported)

## Setup

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/Sachithra-oshadha/ELT_test.git
    cd ELT_test
    ```

2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

3. **Configure Environment Variables:**

    Create a `.env` file in the project root:

    ```env
    DB_NAME=load_profiles_db
    DB_USER=postgres
    DB_PASSWORD=postgres
    DB_HOST=localhost
    DB_PORT=5432

    # Optional — S3 config (omit for local-only mode)
    AWS_ACCESS_KEY_ID=
    AWS_SECRET_ACCESS_KEY=
    AWS_REGION=
    S3_BUCKET_NAME=
    S3_BUCKET_PREFIX=

    # Local filesystem path for data files (used as S3 fallback)
    LOCAL_INPUT_DIR="path/to/your/data"
    ```

4. **Set Up the Database:**

    ```bash
    psql -U postgres -d load_profiles_db -f create.sql
    ```

## Usage

### 1. Ingest Data

```bash
python data_load/main.py
```

Scans the configured source (S3 or local directory) for `.csv`/`.xlsx` files and loads them into the database. Already-processed files are tracked and skipped.

### 2. Train Models & Generate Predictions

```bash
python prediction_model/main.py
```

For each customer: fetches measurement data, trains (or reuses) a Bi-LSTM model, generates a next-15-minute prediction with confidence bounds, and saves everything to the database.

### 3. Start the API Server

```bash
cd load_profile_api
python run.py
```

Starts the FastAPI server at `http://localhost:8000`. The simulation UI is served at `http://localhost:8000/ui/`.

### 4. Use the Simulation UI

- Open `http://localhost:8000/ui/` in a browser
- Select a customer from the dropdown
- Click **Simulate Next 15 Mins** to step through time
- Use **Auto Run** for continuous simulation at 3-second intervals
- The chart shows actual vs. predicted values with 95% confidence bands

## Database Schema

- **`customer`** — Customer identifiers and details
- **`meter`** — Metering devices linked to customers
- **`measurement`** — 15-minute load profile readings (import/export kW, kWh, power factor, current, voltage)
- **`phase_measurement`** — Per-phase (A/B/C) instantaneous readings
- **`customer_model`** — Serialized trained Bi-LSTM models with metrics
- **`customer_prediction`** — Predicted usage and import kWh per customer per timestamp
- **`processed_files`** — Tracks which data files have been ingested
- **`measurement_summary`** — View joining measurements with phase data for convenient querying

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/simulation/customers_info` | List customers with data availability |
| `POST` | `/simulation/simulate_step` | Run one prediction step for a customer |
| `GET` | `/measurements/customer/{id}` | Get measurement at exact timestamp |
| `GET` | `/measurements/customer/{id}/range` | Get measurements in time range |
| `GET` | `/predictions/customer/{id}` | Get prediction at exact timestamp |
| `GET` | `/predictions/customer/{id}/range` | Get predictions in time range |

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.