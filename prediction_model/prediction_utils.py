from imports import *

def predict_next_timestep(model: "nn.Module", last_sequence: np.ndarray,
                          scaler: "StandardScaler", last_kwh: float, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    try:
        model.eval()
        device = next(model.parameters()).device
        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)
        
        prediction = prediction.cpu().numpy().squeeze()

        # Prepare dummy array for inverse scaling
        dummy_array = np.zeros((96, len(scaler.mean_)))
        dummy_array[:, 0] = prediction
        prediction_transformed = scaler.inverse_transform(dummy_array)[:, 0]

        # Apply non-negative constraint
        prediction_deltas = np.maximum(prediction_transformed, 0)

        # Reconstruct absolute kWh values
        prediction_abs = [last_kwh + prediction_deltas[0]]
        for i in range(1, 96):
            prediction_abs.append(prediction_abs[-1] + prediction_deltas[i])

        return np.array(prediction_abs), np.array(prediction_deltas)

    except Exception as e:
        logger.error(f"Failed to make prediction: {e}")
        raise

def create_prediction_plot(df: pd.DataFrame, predictions: np.ndarray, customer_ref: int, sequence_length: int,
                           output_base_dir: str, logger: logging.Logger) -> str:
    try:
        output_dir = os.path.join(output_base_dir, f"customer_{customer_ref}")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(16, 6))
        last_n = df.tail(sequence_length)
        timestamps = last_n['timestamp']
        actual_kw = last_n['import_kwh']
        last_timestamp = timestamps.iloc[-1]
        prediction_times = [last_timestamp + timedelta(minutes=15 * i) for i in range(1, 97)]

        plt.plot(timestamps, actual_kw.values, 'b-', label='Historical Consumption')
        plt.plot(prediction_times, predictions, 'g--o', label='Predicted Consumption (Next 24 Hours)')
        plt.axvline(x=last_timestamp, color='r', linestyle='--', label='Prediction Point')

        plt.title(f'Customer {customer_ref} - Energy Consumption Prediction (Next 24 Hours)')
        plt.xlabel('Timestamp')
        plt.ylabel('Energy (kWh)')
        plt.grid(True)
        plt.legend()
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.HourLocator(interval=2))
        plt.tight_layout()

        output_path = os.path.join(
            output_dir,
            f'prediction_{customer_ref}_{datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")}.png'
        )
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved 24-hour plot at: {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Failed to create plot for customer {customer_ref}: {e}")
        raise

def save_prediction_to_db(cur, conn, customer_ref: int, prediction_abs: np.ndarray,
                          prediction_deltas: np.ndarray, start_time: datetime, logger: logging.Logger):
    try:
        prediction_times = [start_time + timedelta(minutes=15 * i) for i in range(96)]
        data_to_insert = []

        for i in range(96):
            data_to_insert.append((
                customer_ref,
                float(prediction_deltas[i]),
                float(prediction_abs[i]),
                prediction_times[i]
            ))

        insert_query = """
            INSERT INTO customer_prediction (
                customer_ref,
                predicted_usage,
                predicted_import_kwh,
                prediction_timestamp
            ) VALUES (%s, %s, %s, %s)
            ON CONFLICT (customer_ref, prediction_timestamp)
            DO UPDATE SET
                predicted_usage = EXCLUDED.predicted_usage,
                predicted_import_kwh = EXCLUDED.predicted_import_kwh
        """

        cur.executemany(insert_query, data_to_insert)
        conn.commit()
        logger.info(f"Saved {len(data_to_insert)} prediction rows for customer {customer_ref}")

    except Exception as e:
        logger.error(f"Failed to save prediction to database for customer {customer_ref}: {e}")
        conn.rollback()
        raise

def save_model_to_db(cur, conn, model: "nn.Module", customer_ref: int,
                     mse: float, r2_score: float, trained_data_timestamp: datetime, logger: logging.Logger):
    try:
        buffer = io.BytesIO()
        torch.jit.save(torch.jit.script(model), buffer)
        model_data = buffer.getvalue()

        cur.execute("""
            INSERT INTO customer_model (customer_ref, model_data, mse, r2_score, last_trained_data_timestamp)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (customer_ref) DO UPDATE 
            SET model_data = EXCLUDED.model_data,
                mse = EXCLUDED.mse,
                r2_score = EXCLUDED.r2_score,
                last_trained_data_timestamp = EXCLUDED.last_trained_data_timestamp,
                trained_at = CURRENT_TIMESTAMP
        """, (customer_ref, psycopg2.Binary(model_data), float(mse), float(r2_score), trained_data_timestamp))

        conn.commit()
        logger.info(f"Saved model for customer {customer_ref} with last trained data timestamp: {trained_data_timestamp}")

    except Exception as e:
        logger.error(f"Error saving model for customer {customer_ref}: {e}")
        conn.rollback()
        raise