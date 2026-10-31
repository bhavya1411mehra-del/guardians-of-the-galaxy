"""Time-aware supervised direct-horizon forecasting. All outputs are PM2.5 µg/m³, NOT AQI."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sample_data import DIWALI_DATES

HORIZONS = tuple(range(1, 7))
WEATHER = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation"]
FEATURES = ["pm2_5", "pm_lag3", "pm_lag6", "pm_mean6", "hour_sin", "hour_cos", "month_sin",
            "month_cos", "diwali_window", "diwali_offset", "horizon",
            "future_temperature_2m", "future_relative_humidity_2m",
            "future_wind_speed_10m", "future_precipitation"]


def _calendar(target: pd.Series) -> pd.DataFrame:
    target = pd.to_datetime(target)
    offsets = np.column_stack([((target - d) / pd.Timedelta(days=1)).to_numpy(dtype=float) for d in DIWALI_DATES])
    nearest = offsets[np.arange(len(offsets)), np.argmin(np.abs(offsets), axis=1)]
    return pd.DataFrame({
        "hour_sin": np.sin(2 * np.pi * target.dt.hour / 24).to_numpy(),
        "hour_cos": np.cos(2 * np.pi * target.dt.hour / 24).to_numpy(),
        "month_sin": np.sin(2 * np.pi * target.dt.month / 12).to_numpy(),
        "month_cos": np.cos(2 * np.pi * target.dt.month / 12).to_numpy(),
        "diwali_window": (np.abs(nearest) <= 3).astype(float),
        "diwali_offset": np.clip(nearest, -15, 15),
    })


def make_training_rows(history: pd.DataFrame) -> pd.DataFrame:
    df = history.sort_values("time").drop_duplicates("time").reset_index(drop=True).copy()
    df["pm_lag3"] = df["pm2_5"].shift(3)
    df["pm_lag6"] = df["pm2_5"].shift(6)
    df["pm_mean6"] = df["pm2_5"].rolling(6).mean()
    # Validate time continuity, important for separated 2025/2026 historical windows.
    for lag in [3, 6]:
        df.loc[(df["time"] - df["time"].shift(lag)) != pd.Timedelta(hours=lag), f"pm_lag{lag}"] = np.nan
    df.loc[(df["time"] - df["time"].shift(5)) != pd.Timedelta(hours=5), "pm_mean6"] = np.nan
    batches = []
    for horizon in HORIZONS:
        future_time = df["time"].shift(-horizon)
        valid = (future_time - df["time"]) == pd.Timedelta(hours=horizon)
        batch = df[["time", "pm2_5", "pm_lag3", "pm_lag6", "pm_mean6"]].copy()
        batch["target"] = df["pm2_5"].shift(-horizon)
        batch["horizon"] = horizon
        for key in WEATHER:
            batch[f"future_{key}"] = df[key].shift(-horizon)
        batch = pd.concat([batch, _calendar(future_time.fillna(df["time"]))], axis=1)
        batches.append(batch[valid].dropna(subset=FEATURES + ["target"]))
    joined = pd.concat(batches, ignore_index=True).sort_values("time").reset_index(drop=True)
    return joined


def _model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=52, max_depth=11, min_samples_leaf=4,
                                 random_state=42, n_jobs=-1)


def train_model(history: pd.DataFrame) -> tuple[RandomForestRegressor, dict]:
    dataset = make_training_rows(history)
    if len(dataset) < 450:
        raise ValueError("Not enough continuous historical data to train a model (need at least 450 training examples).")
    times = np.sort(dataset["time"].unique())
    cutoff = times[int(len(times) * 0.8)]
    # Hold a six-hour gap so test targets cannot overlap any training origin.
    train = dataset[dataset["time"] < cutoff - np.timedelta64(6, "h")]
    test = dataset[dataset["time"] >= cutoff]
    if len(train) < 300 or len(test) < 80:
        raise ValueError("Not enough history for a time-based evaluation split.")
    candidate = _model()
    candidate.fit(train[FEATURES], train["target"])
    estimates = np.maximum(0, candidate.predict(test[FEATURES]))
    mae = float(mean_absolute_error(test["target"], estimates))
    persistence_mae = float(mean_absolute_error(test["target"], test["pm2_5"]))
    final = _model()
    final.fit(dataset[FEATURES], dataset["target"])
    return final, {"holdout_mae": mae, "baseline_mae": persistence_mae,
                   "training_examples": len(train), "test_examples": len(test),
                   "features": FEATURES, "rows": len(dataset)}


def forecast(history: pd.DataFrame, future_weather: pd.DataFrame,
             model: RandomForestRegressor) -> pd.DataFrame:
    df = history.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    last = df.iloc[-1]
    origin = pd.Timestamp(last["time"])
    prev = df.set_index("time")["pm2_5"]
    for lag in (3, 6):
        if origin - pd.Timedelta(hours=lag) not in prev.index:
            raise ValueError("Recent PM2.5 history has gaps, unable to create model lags.")
    last6 = prev.loc[origin - pd.Timedelta(hours=5):origin]
    if len(last6) != 6:
        raise ValueError("Recent PM2.5 history is incomplete.")
    rows = []
    for h in HORIZONS:
        target = origin + pd.Timedelta(hours=h)
        candidate = future_weather[future_weather["time"] == target]
        if candidate.empty:
            raise ValueError(f"Missing weather forecast for {target}")
        w = candidate.iloc[0]
        row = {"time": origin, "pm2_5": float(last["pm2_5"]),
               "pm_lag3": float(prev.loc[origin - pd.Timedelta(hours=3)]),
               "pm_lag6": float(prev.loc[origin - pd.Timedelta(hours=6)]),
               "pm_mean6": float(last6.mean()), "horizon": h}
        row.update({f"future_{k}": float(w[k]) for k in WEATHER})
        row.update(_calendar(pd.Series([target])).iloc[0].to_dict())
        rows.append(row)
    x = pd.DataFrame(rows)
    if x[FEATURES].isna().any().any():
        raise ValueError("Missing values in latest observations or forecast weather")
    predictions = np.maximum(0, model.predict(x[FEATURES]))
    return pd.DataFrame({"horizon": HORIZONS,
                         "time": [origin + pd.Timedelta(hours=h) for h in HORIZONS],
                         "pm2_5": predictions.round(1)})
