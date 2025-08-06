# Load Profile and Customer Behavior Analysis Pipeline

This project is a comprehensive Python-based data pipeline that processes energy meter data from Excel or CSV files stored in an AWS S3 bucket, loads it into a PostgreSQL database, and uses a Bidirectional LSTM (Bi-LSTM) model to predict customer energy consumption over the next 24 hours at 15-minute intervals. The pipeline supports data ingestion, preprocessing, model training, prediction, visualization, and storage, with robust error handling and logging.

# Features

- **File Processing**: Reads and processes .csv, .xlsx, and .xls files from an AWS S3 bucket.
- **Data Validation:** Ensures data integrity with checks for valid formats, unique entries, and conflict handling.
- **Database Integration:** Inserts data into PostgreSQL tables for customers, meters, measurements, phase measurements, customer models, predictions, and processed files.
- **S3 Integration:** Lists and downloads files from S3, tracks processed files, and uploads prediction plots.
- **Machine Learning:** Trains a Bi-LSTM model per customer for 24-hour energy consumption predictions (96 intervals), with early stopping based on validation loss.
- **Visualization:** Generates plots comparing historical and predicted energy consumption, stored in S3.
- **Logging:** Comprehensive logging to file and console for monitoring and debugging.
- **Environment Configuration:** Uses .env file for secure configuration of database and S3 credentials.

## Project Structure

```bash
├── config.py               # Database and S3 configuration
├── create.sql              # PostgreSQL database schema setup
├── data_load
    ├── main.py             # Main pipeline logic for data insertion
    ├── logger.py           # Logger setup for data insertion
    ├── database.py         # Database connection and query execution
    ├── file_processor.py   # File reading, validation, and database insertion
    ├── s3_client.py        # AWS S3 interactions for file handling
├── prediction
    ├── main.py             # Main pipeline logic for prediction
    ├── logger.py           # Logger setup for prediction
    ├── data_processing.py  # Dataset class and preprocessing logic
    ├── database_utils.py   # Database connection and data fetching utilities
    ├── imports.py          # Shared imports for all modules processing and predictions
    ├── model_definition.py # Bi-LSTM model definition
    ├── model_training.py   # Model training and evaluation logic
    ├── prediction_utils.py # Prediction, plotting, and storage utilities
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables for DB and S3
```

## Prerequisites

- Python 3.8+
- PostgreSQL database
- AWS S3 bucket with appropriate access
- Required Python packages:
    
    Includes: ``` joblib, matplotlib, numpy, openpyxl, pandas, psycopg2-binary, python-dotenv, scikit-learn, seaborn, tensorflow, torch, boto3 ```

## Setup

1.  **Clone the Repository:**

    ```bash
     git clone https://github.com/Sachithra-oshadha/ELT_test.git 
     ```

    ```bash 
    cd energy-data-pipeline 
    ```
2. **Install Dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
3. **Set Up Environment Variables:** 
    
    Create a .env file in the project root with the following:
    
    ```sql
    DB_NAME= #database name
    DB_USER= #username 
    DB_PASSWORD= #password
    DB_HOST= #host
    DB_PORT= #postgres port (by default 5432)
    
    AWS_ACCESS_KEY_ID= #aws access key
    AWS_SECRET_ACCESS_KEY= #aws secret access-key
    AWS_REGION= #aws region
    S3_BUCKET_NAME= #S3 bucket name
    S3_BUCKET_PREFIX= #S3 bucket prefix
    ```
    **Note:** Replace sensitive values (e.g., AWS credentials) with your own and never commit the .env file.

4. **Set Up the Database:** 

    Execute the create.sql script to set up the PostgreSQL schema:
    
    ```bash
    psql -U postgres -d test_load_profiles -f create.sql
    ```


5. **Run the Pipeline:**
    - For data ingestion:
        ```bash
        python data_load/main.py
        ```
        This processes .csv, .xlsx, or .xls files from the S3 bucket and inserts data into the database.
    - For predictions:
        ```bash
        python prediction/main.py
        ```
        This fetches data, trains/reuses Bi-LSTM models, generates predictions, and uploads plots to S3.

## Database Schema

The create.sql script defines the following:

- **Tables:**
    - customer: Stores customer details (customer_ref, first_name, last_name, email).
    - meter: Links meters to customers (serial, customer_ref).
    - measurement: Stores energy data (serial, timestamp, obis, import_kwh, power_factor, etc.).
    - phase_measurement: Stores phase-specific data (serial, timestamp, phase, inst_current, inst_voltage, inst_power_factor).
    - customer_model: Stores trained Bi-LSTM models (customer_ref, model_data, mse, r2_score).
    - customer_prediction: Stores predictions (customer_ref, prediction_timestamp, predicted_usage, predicted_import_kwh).
    - processed_files: Tracks processed S3 files (file_name, s3_path, processed_at).
- **View:** measurement_summary combines measurement and phase data for easier querying.
- **Indexes:** Optimize query performance on timestamp, serial, and customer_ref.
- **Triggers:** Update updated_at timestamps for customer, meter, and customer_model.

## Usage

- **Data Ingestion (data_load/main.py):**
    - Scans the S3 bucket (load-profiles-bucket) under data/ for .csv, .xlsx, or .xls files.
    - Downloads files to a temporary directory, processes them, and inserts data into customer, meter, measurement, and phase_measurement tables.
    - Tracks processed files in processed_files to prevent reprocessing.
    - Cleans up temporary files after processing.
- **Prediction Pipeline (prediction/main.py):**
    - Fetches data from measurement and phase_measurement tables.
    - Preprocesses data (differencing import_kwh, standard scaling).
    - Trains a Bi-LSTM model per customer if new data is available, using 9 input features (e.g., import_kwh, power_factor, phase measurements).
    - Generates 24-hour predictions (96 intervals) and constrains predictions to be non-negative.
    - Saves predictions to customer_prediction and models to customer_model.
    - Generates and uploads plots comparing historical and predicted consumption to S3.
- **Output:**
    - Predictions in customer_prediction (predicted_usage, predicted_import_kwh).
    - Models and metrics (mse, r2_score) in customer_model.
    - Plots in S3 under s3://load-profiles-bucket/data/customer_<id>/.
    - Logs in files like data_insertion_YYYY-MM-DD_HH-MM-SS.log (for data ingestion) or customer_behavior_bilstm_YYYY-MM-DD_HH-MM-SS.log (for predictions).

## Key Components
- **ElectricityDataset:** Custom PyTorch Dataset for sequence-based input-output pairs.
- **BiLSTM:** Bidirectional LSTM model for time-series prediction.
- **DatabaseManager:** Manages PostgreSQL connections and queries (in database_utils.py).
- **FileProcessor:** Handles file reading, validation, and database insertion (in file_processor.py).
- **CustomerBehaviorPipeline:** Orchestrates data fetching, preprocessing, training, prediction, and storage (in prediction/main.py).
- **Prediction Utilities:** Functions for predictions, plotting, and saving results (in prediction_utils.py).

## Logging

- Logs are written to files (e.g., data_insertion_YYYY-MM-DD_HH-MM-SS.log for data ingestion, customer_behavior_bilstm_YYYY-MM-DD_HH-MM-SS.log for predictions) and printed to the console.
- Log levels: 
``` INFO, WARNING, ERROR. ```
- Tracks file processing, database operations, S3 interactions, model training, and errors.

## Error Handling

- Database errors trigger transaction rollbacks and detailed logging.
- File processing errors (e.g., invalid formats, missing columns) are logged and raised.
- S3 connection or download failures are caught and logged.
- Model training skips customers with insufficient or unchanged data.

## Notes

- The prediction pipeline skips training if no new data is available since the last model training (based on last_trained_data_timestamp).
- Predictions are constrained to be non-negative.
- The Bi-LSTM model uses 9 input features and predicts 96 future intervals.
- Temporary files are stored in a temporary directory and cleaned up after processing.

## Contributing

1. Fork the repository.
2. Create a feature branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the LICENSE file for details.