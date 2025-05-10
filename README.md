# Load Profile and Customer Behavior Analysis Pipeline

This repository contains two Python scripts for processing load profile data and analyzing customer behavior using a PostgreSQL database and machine learning techniques. The scripts are designed to read data from Excel files, store it in a database, and perform predictive modeling and visualization for customer energy usage patterns.

## Prerequisites

Before running the scripts, ensure you have the following installed:

Python: Version 3.8 or higher

PostgreSQL: A running PostgreSQL database server

Excel File or CSV File: An Excel or CSV file containing load profile data in the same directory as python files

Environment Variables: A .env file with database configuration (see below)

## Installation

1. Clone the Repository:

2. Set Up a Virtual Environment (recommended):

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install Dependencies: Install the required Python packages listed in requirements.txt:

    pip install -r requirements.txt

    The requirements.txt includes:

        pandas>=1.5.0: For data manipulation and Excel file reading
        psycopg2-binary>=2.9.0: For PostgreSQL database connectivity
        openpyxl>=3.0.0: For reading Excel files
        python-dotenv>=0.19.0: For loading environment variables
        scikit-learn>=1.0.0: For machine learning (RandomForestRegressor)
        joblib>=1.1.0: For model serialization
        matplotlib>=3.5.0: For plotting
        seaborn>=0.11.0: For enhanced visualizations
        numpy>=1.21.0: For numerical operations

4. Configure Environment Variables: Create a .env file in the project root with the following structure:

    DB_NAME=your_database_name - load_profile_db in test case

    DB_USER=your_database_user - postgres in test case
    
    DB_PASSWORD=your_database_password
    
    DB_HOST=your_database_host - localhost in test case
    
    DB_PORT=your_database_port - 5432 in test case

    Replace the values with your PostgreSQL database credentials.

    Set Up PostgreSQL Database: Ensure your PostgreSQL database is running and has the necessary tables (customer, meter, measurement, phase_measurement, customer_model). The schema for these tables should match the structure expected by the scripts (refer to the INSERT queries in load_profile_pipeline.py).

## Usage

1. Load Profile Pipeline (load_profile_pipeline.py)

    This script reads load profile data from an Excel file and inserts it into the PostgreSQL database.

    Input: An Excel file with relevant columns.

    Output: Data inserted into the customer, meter, measurement, and phase_measurement tables.

    To run:

    python load_profile_pipeline.py
    
    Ensure the Excel file path in the script (excel_file_path) points to a valid file.

2. Customer Behavior Pipeline (customer_behavior.py)

    This script analyzes customer energy usage, trains a RandomForestRegressor model for each customer, and generates visualizations.

    Input: Data in the PostgreSQL database (populated by load_profile_pipeline.py)

    Output: 
        Trained models stored in the customer_model table

        CSV files with behavior metrics (in customer_plots/<customer_ref>/)
        
        PNG plots of usage patterns and feature importance (in customer_plots/<customer_ref>/)

    To run:

    python customer_behavior.py

## Directory Structure

After running the scripts, the following directories and files will be created:

    customer_plots/: Contains subdirectories for each customer with CSV metrics and PNG plots.
    
    data_insertion.log: Log file for load_profile_pipeline.py.
    
    customer_behavior.log: Log file for customer_behavior.py.

## Notes

Logging: Both scripts log information to console and files (data_insertion.log and customer_behavior.log).

Error Handling: The scripts include robust error handling and logging for debugging.

Incremental Training: The customer_behavior.py script supports incremental model training if new data is available.

File Paths: Update the excel_file_path in load_profile_pipeline.py to match your Excel file location.

Database Schema: Ensure the database tables are created with the correct schema before running the scripts.

Plot Storage: Plots and metrics are saved in customer-specific folders with timestamps for versioning.

## Troubleshooting

Database Connection Issues: Verify the .env file and PostgreSQL server status.

Excel File Errors: Ensure the Excel file exists and has the expected column names.

Missing Packages: Run pip install -r requirements.txt to install all dependencies.

Plotting Issues: Ensure matplotlib and seaborn are installed correctly.

## License

This project is licensed under the MIT License. See the LICENSE file for details.