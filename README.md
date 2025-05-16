Load Profile and Customer Behavior Analysis Pipeline
This repository contains Python scripts for processing load profile data and analyzing customer behavior using a PostgreSQL database and machine learning techniques. The scripts read data from Excel or CSV files, store it in a database, and perform predictive modeling and visualization of customer energy usage patterns.
Prerequisites
Before running the scripts, ensure the following are installed and configured:

Python: Version 3.8 or higher
PostgreSQL: A running PostgreSQL database server
Input File: An Excel or CSV file containing load profile data in the same directory as the Python scripts
Environment Variables: A .env file with database configuration (see Environment Variables)

Installation

Clone the Repository:
git clone <repository-url>
cd <repository-directory>


Set Up a Virtual Environment (recommended):
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install Dependencies:Install the required Python packages listed in requirements.txt:
pip install -r requirements.txt

The requirements.txt includes:

pandas>=1.5.0: Data manipulation and Excel file reading
psycopg2-binary>=2.9.0: PostgreSQL database connectivity
openpyxl>=3.0.0: Excel file reading
python-dotenv>=0.19.0: Environment variable loading
scikit-learn>=1.0.0: Machine learning (RandomForestRegressor)
joblib>=1.1.0: Model serialization
matplotlib>=3.5.0: Plotting
seaborn>=0.11.0: Enhanced visualizations
numpy>=1.21.0: Numerical operations
tensorflow>=2.16.0: Bi-LSTM modeling (for customer behavior scripts)


Configure Environment Variables:Create a .env file in the project root with the following structure:
DB_NAME=load_profile_db
DB_USER=postgres
DB_PASSWORD=your_database_password
DB_HOST=your_aws_postgres_rds_instance
DB_PORT=5432

Replace DB_PASSWORD and DB_HOST with your PostgreSQL credentials. Use the provided defaults if using the same AWS RDS instance for development.

Set Up PostgreSQL Database:Ensure the PostgreSQL database is running and includes the required tables (customer, meter, measurement, phase_measurement, customer_model). The schema must match the structure expected by the scripts (refer to INSERT queries in load_profile_pipeline.py).


Usage
1. Load Profile Pipeline (load_profile_pipeline.py)
This script reads load profile data from an Excel file and inserts it into the PostgreSQL database.

Input: An Excel file with relevant columns
Output: Data inserted into the customer, meter, measurement, and phase_measurement tables
Run Command:python load_profile_pipeline.py


Configuration: Update excel_file_path in the script to point to a valid Excel file.

2. Customer Behavior Pipeline
This pipeline includes three scripts (customer_behavior_pipeline_1day.py, customer_behavior_pipeline_1week.py, customer_behavior_pipeline_1month.py) that analyze customer energy usage, train a Bidirectional LSTM (Bi-LSTM) model for each customer, and generate visualizations. Each script uses a different time window for sequence modeling based on 15-minute interval data.

Input: Energy usage data in the PostgreSQL database (populated by load_profile_pipeline.py), stored in the measurement table with columns: serial, timestamp, avg_import_kw, import_kwh, power_factor

Output:

Trained Models: Bi-LSTM models serialized in .keras format, stored in the customer_model table (customer_ref, model_data, trained_at, mse, r2_score)
CSV Files: Behavior metrics (e.g., max usage, average usage, peak hour) saved as metrics_<timestamp>.csv in customer_plots/<customer_ref>/
PNG Plots: Hourly usage patterns saved as usage_pattern_<timestamp>.png in customer_plots/<customer_ref>/
Note: Feature importance plotting is not supported for Bi-LSTM and logs a warning


Scripts and Configurations:

customer_behavior_pipeline_1day.py: time_steps = 96 (1 day, 4 * 24 intervals). Suitable for short-term patterns.
customer_behavior_pipeline_1week.py: time_steps = 672 (1 week, 4 * 24 * 7 intervals). Balances detail and data requirements.
customer_behavior_pipeline_1month.py: time_steps = 2880 (1 month, 4 * 24 * 30 intervals, assuming 30-day month). Captures long-term trends but requires significant data.


Requirements:

PostgreSQL database with populated measurement, meter, and customer tables
.env file with database credentials
Dependencies listed in requirements.txt


Run Command:
python customer_behavior_pipeline_1week.py

Replace with customer_behavior_pipeline_1day.py or customer_behavior_pipeline_1month.py based on the desired time window.


Directory Structure
After running the scripts, the following structure is created:
├── customer_plots/                  # Customer-specific subdirectories with metrics and plots
├── data_insertion.log              # Log file for load_profile_pipeline.py
├── customer_behavior.log           # Log file for customer_behavior scripts
├── load_profile_pipeline.py        # Script for loading data
├── customer_behavior_pipeline_*.py # Scripts for behavior analysis
├── requirements.txt                # Dependencies
└── .env                            # Environment variables

Notes

Logging: Scripts log to console and files (data_insertion.log, customer_behavior.log).
Error Handling: Includes robust error handling and logging for debugging.
Incremental Training: The customer behavior scripts support incremental model training with new data.
File Paths: Ensure excel_file_path in load_profile_pipeline.py matches the Excel file location.
Database Schema: Verify tables are created with the correct schema before running scripts.
Plot Storage: Plots and metrics are saved in customer-specific folders with timestamps for versioning.

Troubleshooting

Database Connection Issues: Verify .env file and PostgreSQL server status.
Excel File Errors: Ensure the Excel file exists and has expected column names.
Missing Packages: Run pip install -r requirements.txt to install dependencies.
Plotting Issues: Confirm matplotlib and seaborn are installed correctly.

License
This project is licensed under the MIT License. See the LICENSE file for details.
