import os
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pytz
import torch

def predict_next_timestep(model, last_sequence, scaler, last_kwh, logger):
    try:
        model.eval()
        device = next(model.parameters()).device

        input_tensor = torch.FloatTensor(last_sequence).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(input_tensor)

        # Take only the first step (next 15 mins) to support both old 96-step models and new 1-step models
        prediction = prediction.cpu().numpy()[0][:1, :]  # shape becomes (1, 3)

        dummy = np.zeros((1, len(scaler.mean_)))
        transformed = np.zeros_like(prediction)

        for i in range(3):
            dummy[:, 0] = prediction[:, i]
            transformed[:, i] = scaler.inverse_transform(dummy)[:, 0]

        transformed = np.maximum(transformed, 0)

        lower, median, upper = [], [], []

        for i in range(1):
            if i == 0:
                lower.append(last_kwh + transformed[i, 0])
                median.append(last_kwh + transformed[i, 1])
                upper.append(last_kwh + transformed[i, 2])
            else:
                lower.append(lower[-1] + transformed[i, 0])
                median.append(median[-1] + transformed[i, 1])
                upper.append(upper[-1] + transformed[i, 2])

        return np.array(lower), np.array(median), np.array(upper)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def create_prediction_plot(df, median, lower, upper, customer_ref,
                          sequence_length, output_base_dir, logger):

    try:
        output_dir = os.path.join(output_base_dir, f"customer_{customer_ref}")
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(16, 6))

        last_n = df.tail(sequence_length)
        timestamps = last_n['timestamp']
        actual = last_n['import_kwh']

        last_timestamp = timestamps.iloc[-1]
        pred_times = [last_timestamp + timedelta(minutes=15 * i) for i in range(1, 2)]

        plt.plot(timestamps, actual, label="Historical")
        plt.plot(pred_times, median, linestyle='--', label="Median Prediction")

        plt.fill_between(pred_times, lower, upper, alpha=0.2, label="95% Confidence")

        plt.axvline(x=last_timestamp, linestyle='--', label="Prediction Start")

        plt.legend()
        plt.grid(True)

        output_path = os.path.join(
            output_dir,
            f'prediction_{customer_ref}_{datetime.now(pytz.UTC).strftime("%Y%m%d_%H%M%S")}.png'
        )

        plt.savefig(output_path)
        plt.close()

        return output_path

    except Exception as e:
        logger.error(f"Plot failed: {e}")
        raise
