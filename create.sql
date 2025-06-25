-- Combined schema for storing electrical load profile data and customer behavior models
-- Database: load_profiles_db

-- Drop existing objects to ensure a clean slate
DROP VIEW IF EXISTS measurement_summary;
DROP TABLE IF EXISTS customer_model;
DROP TABLE IF EXISTS phase_measurement;
DROP TABLE IF EXISTS measurement;
DROP TABLE IF EXISTS meter;
DROP TABLE IF EXISTS customer;
DROP FUNCTION IF EXISTS update_updated_at_column;

-- Create Customer table
CREATE TABLE customer (
    customer_ref BIGINT PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT customer_ref_positive CHECK (customer_ref > 0)
);
COMMENT ON TABLE customer IS 'Stores unique customer identifiers';
COMMENT ON COLUMN customer.customer_ref IS 'Unique customer reference number';

-- Create Meter table
CREATE TABLE meter (
    serial BIGINT PRIMARY KEY,
    customer_ref BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_customer_ref FOREIGN KEY (customer_ref) REFERENCES customer (customer_ref) ON DELETE RESTRICT,
    CONSTRAINT serial_positive CHECK (serial > 0)
);
COMMENT ON TABLE meter IS 'Stores metering devices associated with customers';
COMMENT ON COLUMN meter.serial IS 'Unique serial number of the meter';
COMMENT ON COLUMN meter.customer_ref IS 'Foreign key referencing the customer';

-- Create Measurement table
CREATE TABLE measurement (
    serial BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    obis VARCHAR(10) NOT NULL,
    avg_import_kw NUMERIC(10, 4),
    import_kwh NUMERIC(12, 4),
    avg_export_kw NUMERIC(10, 4),
    export_kwh NUMERIC(12, 4),
    avg_import_kva NUMERIC(10, 4),
    avg_export_kva NUMERIC(10, 4),
    import_kvarh NUMERIC(12, 4),
    export_kvarh NUMERIC(12, 4),
    power_factor NUMERIC(5, 4),
    avg_current NUMERIC(10, 4),
    avg_voltage NUMERIC(10, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_measurement PRIMARY KEY (serial, timestamp),
    CONSTRAINT fk_meter_serial FOREIGN KEY (serial) REFERENCES meter (serial) ON DELETE CASCADE,
    CONSTRAINT obis_not_empty CHECK (obis <> ''),
    CONSTRAINT power_factor_range CHECK (power_factor IS NULL OR (power_factor >= -1 AND power_factor <= 1))
);
COMMENT ON TABLE measurement IS 'Stores load profile measurements for meters';
COMMENT ON COLUMN measurement.serial IS 'Foreign key referencing the meter';
COMMENT ON COLUMN measurement.timestamp IS 'Timestamp of the measurement (datetime)';
COMMENT ON COLUMN measurement.obis IS 'OBIS code indicating measurement type (e.g., LP for load profile)';
COMMENT ON COLUMN measurement.avg_import_kw IS 'Average import power in kilowatts';
COMMENT ON COLUMN measurement.import_kwh IS 'Cumulative import energy in kilowatt-hours';
COMMENT ON COLUMN measurement.power_factor IS 'Power factor of the measurement (-1 to 1)';

-- Create PhaseMeasurement table
CREATE TABLE phase_measurement (
    serial BIGINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    phase CHAR(1) NOT NULL CHECK (phase IN ('A', 'B', 'C')),
    inst_current NUMERIC(10, 4),
    inst_voltage NUMERIC(10, 4),
    inst_power_factor NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT pk_phase_measurement PRIMARY KEY (serial, timestamp, phase),
    CONSTRAINT fk_measurement FOREIGN KEY (serial, timestamp) REFERENCES measurement (serial, timestamp) ON DELETE CASCADE,
    CONSTRAINT inst_power_factor_range CHECK (inst_power_factor IS NULL OR (inst_power_factor >= -1 AND inst_power_factor <= 1))
);
COMMENT ON TABLE phase_measurement IS 'Stores phase-specific measurements (A, B, C)';
COMMENT ON COLUMN phase_measurement.phase IS 'Phase identifier (A, B, or C)';
COMMENT ON COLUMN phase_measurement.inst_current IS 'Instantaneous current in amperes';
COMMENT ON COLUMN phase_measurement.inst_voltage IS 'Instantaneous voltage in volts';
COMMENT ON COLUMN phase_measurement.inst_power_factor IS 'Instantaneous power factor (-1 to 1)';

-- Create CustomerModel table
CREATE TABLE customer_model (
    customer_ref BIGINT PRIMARY KEY,
    model_data BYTEA NOT NULL,
    trained_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    mse NUMERIC(15, 4),
    r2_score NUMERIC(5, 4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_customer_ref FOREIGN KEY (customer_ref) REFERENCES customer (customer_ref) ON DELETE CASCADE
);
COMMENT ON TABLE customer_model IS 'Stores trained machine learning models for customer behavior analysis';
COMMENT ON COLUMN customer_model.customer_ref IS 'Foreign key referencing the customer';
COMMENT ON COLUMN customer_model.model_data IS 'Serialized binary of the trained model';
COMMENT ON COLUMN customer_model.trained_at IS 'Timestamp when the model was last trained';
COMMENT ON COLUMN customer_model.mse IS 'Mean Squared Error of the model on validation data';
COMMENT ON COLUMN customer_model.r2_score IS 'R-squared score of the model on validation data';

-- Create indexes for performance
CREATE INDEX idx_measurement_timestamp ON measurement (timestamp);
CREATE INDEX idx_measurement_serial ON measurement (serial);
CREATE INDEX idx_phase_measurement_serial_timestamp ON phase_measurement (serial, timestamp);
CREATE INDEX idx_meter_customer_ref ON meter (customer_ref);

-- Create a function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers to update updated_at
CREATE TRIGGER update_customer_updated_at
    BEFORE UPDATE ON customer
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_meter_updated_at
    BEFORE UPDATE ON meter
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_customer_model_updated_at
    BEFORE UPDATE ON customer_model
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create a view for easier querying of combined data
CREATE VIEW measurement_summary AS
SELECT 
    m.serial,
    m.timestamp,
    m.timestamp AS datetime,
    m.obis,
    m.avg_import_kw,
    m.import_kwh,
    m.avg_export_kw,
    m.export_kwh,
    m.power_factor,
    pm_a.inst_current AS phase_a_current,
    pm_a.inst_voltage AS phase_a_voltage,
    pm_a.inst_power_factor AS phase_a_power_factor,
    pm_b.inst_current AS phase_b_current,
    pm_b.inst_voltage AS phase_b_voltage,
    pm_b.inst_power_factor AS phase_b_power_factor,
    pm_c.inst_current AS phase_c_current,
    pm_c.inst_voltage AS phase_c_voltage,
    pm_c.inst_power_factor AS phase_c_power_factor
FROM measurement m
LEFT JOIN phase_measurement pm_a ON m.serial = pm_a.serial AND m.timestamp = pm_a.timestamp AND pm_a.phase = 'A'
LEFT JOIN phase_measurement pm_b ON m.serial = pm_b.serial AND m.timestamp = pm_b.timestamp AND pm_b.phase = 'B'
LEFT JOIN phase_measurement pm_c ON m.serial = pm_c.serial AND m.timestamp = pm_c.timestamp AND pm_c.phase = 'C';
COMMENT ON VIEW measurement_summary IS 'View combining measurement and phase-specific data for easier querying';


-- 10:49 19/06/2025

ALTER TABLE customer
    ADD COLUMN first_name VARCHAR(255),
    ADD COLUMN last_name VARCHAR(255),
    ADD COLUMN email VARCHAR(255);

COMMENT ON COLUMN customer.first_name IS 'Customer''s first name';
COMMENT ON COLUMN customer.last_name IS 'Customer''s last name';
COMMENT ON COLUMN customer.email IS 'Customer''s email address';

-- 11:37 24/06/2025

CREATE TABLE customer_prediction (
    customer_ref BIGINT PRIMARY KEY,
    prediction_json JSONB NOT NULL,
    prediction_timestamp TIMESTAMP NOT NULL,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_prediction_customer FOREIGN KEY (customer_ref) REFERENCES customer(customer_ref) ON DELETE CASCADE
);

CREATE INDEX idx_prediction_customer_ref ON customer_prediction (customer_ref);  -- Newly added
CREATE INDEX idx_prediction_generated_at ON customer_prediction (generated_at);  -- Newly added

COMMENT ON TABLE customer_prediction IS 'Stores predicted electricity usage data per customer';
COMMENT ON COLUMN customer_prediction.id IS 'Unique ID for each prediction record';
COMMENT ON COLUMN customer_prediction.customer_ref IS 'Reference to the customer';
COMMENT ON COLUMN customer_prediction.prediction_json IS 'Stored prediction result in JSON format (supports querying)';
COMMENT ON COLUMN customer_prediction.file_path IS 'Optional path to the raw JSON file (local or cloud)';
COMMENT ON COLUMN customer_prediction.prediction_timestamp IS 'When the prediction applies (e.g., next 15-min interval)';
COMMENT ON COLUMN customer_prediction.generated_at IS 'When the prediction was generated';