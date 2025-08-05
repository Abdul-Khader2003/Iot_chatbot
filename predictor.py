# predictor.py
import os
import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import clickhouse_connect
import time



# Prediction interval and time window
interval_minutes = 30
lookback_days = 30
predict_days = 60
seq_len = lookback_days * 24 * 60 // interval_minutes
pred_len = predict_days * 24 * 60 // interval_minutes


def predict_next_24_hours(filter_tags: list = None, start: pd.Timestamp = None, end: pd.Timestamp = None):

    # 1. Load model (fixed compile issue with .h5)
    model = load_model("lstm_model_long_format_30mins_60.h5", compile=False)
    model.compile(optimizer='adam', loss=MeanSquaredError())

    # 2. Connect to ClickHouse
    client = clickhouse_connect.get_client(
        host=os.getenv("CLICKHOUSE_HOST"),
        port=int(os.getenv("CLICKHOUSE_PORT")),
        username=os.getenv("CLICKHOUSE_USER"),
        password=os.getenv("CLICKHOUSE_PASSWORD"),
        database=os.getenv("CLICKHOUSE_DATABASE"),
    )

    # 3. Fetch data
    query = """
    SELECT tagName, value, toDateTime(createdAt) AS createdAt
    FROM tag_data_30mins
    WHERE tagName IN ('temp', 'level', 'speed')
    ORDER BY createdAt
    """
    df_long = client.query_df(query)

    # 4. Preprocess
    df_long['createdAt'] = pd.to_datetime(df_long['createdAt'])
    df_long.set_index('createdAt', inplace=True)
    df_wide = df_long.pivot(columns='tagName', values='value')
    df_wide = df_wide.interpolate(method='time').ffill().bfill()

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(df_wide)

    # 5. Prepare input sequence for prediction
    last_seq = scaled_values[-seq_len:]
    last_seq = last_seq.reshape((1, seq_len, df_wide.shape[1]))

    # 6. Predict for fixed horizon (60 days at 30-min intervals)
    pred_scaled = model.predict(last_seq)
    pred_scaled = pred_scaled.reshape((pred_len, df_wide.shape[1]))
    predicted_values = scaler.inverse_transform(pred_scaled)

    # 7. Create timestamps for predicted data
    last_time = df_wide.index[-1]
    future_times = [last_time + timedelta(minutes=interval_minutes * i) for i in range(1, pred_len + 1)]
    forecast_df = pd.DataFrame(predicted_values, columns=df_wide.columns, index=future_times)
    forecast_df.reset_index(inplace=True)
    forecast_df.rename(columns={"index": "timestamp"}, inplace=True)

    # 8. Filter by requested tags if specified
    if filter_tags:
        filter_tags_lower = [tag.lower() for tag in filter_tags]
        available_tags = [tag for tag in filter_tags_lower if tag in forecast_df.columns]
        columns_to_return = ["timestamp"] + available_tags
        forecast_df = forecast_df[columns_to_return]

    # 9. If start and end provided, filter rows accordingly
    if start and end:
        mask = (forecast_df['timestamp'] >= start) & (forecast_df['timestamp'] <= end)
        forecast_df = forecast_df.loc[mask]
    return forecast_df.to_dict(orient="records")
