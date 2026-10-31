"""Fully synthetic demonstration data. NEVER present this as measured air quality."""
from __future__ import annotations

import numpy as np
import pandas as pd

CITIES = {
    "Amritsar": (31.6340, 74.8723),
    "Ludhiana": (30.9010, 75.8573),
    "Jalandhar": (31.3260, 75.5762),
    "Chandigarh": (30.7333, 76.7794),
    "Delhi": (28.6139, 77.2090),
}

DIWALI_DATES = [pd.Timestamp(v) for v in ["2023-11-12", "2024-11-01", "2025-10-20", "2026-11-08"]]


def demo_city(city: str) -> pd.DataFrame:
    """Make deterministic fake hourly data, with seasonal/festival patterns."""
    idx = pd.date_range(end=pd.Timestamp.now(tz="Asia/Kolkata").floor("h").tz_localize(None), periods=24 * 420, freq="h")
    rng = np.random.default_rng(220 + list(CITIES).index(city) * 101)
    hour = idx.hour.to_numpy()
    month = idx.month.to_numpy()
    dayofyear = idx.dayofyear.to_numpy()
    winter = np.maximum(0, np.cos(2 * np.pi * (dayofyear - 355) / 365.25))
    wind = np.clip(7 + 3 * np.sin(hour * np.pi / 12) + rng.normal(0, 1.7, len(idx)), 0.5, 30)
    humidity = np.clip(60 + 15 * np.cos(hour * np.pi / 12) + rng.normal(0, 7, len(idx)), 15, 100)
    temp = 27 + 12 * np.sin(2 * np.pi * (dayofyear - 100) / 365.25) + 4 * np.sin((hour - 7) * np.pi / 12)
    rain = np.where(rng.random(len(idx)) < 0.035, rng.uniform(0.2, 5, len(idx)), 0)
    festival_effect = np.zeros(len(idx))
    for festival in DIWALI_DATES:
        days = ((idx - festival) / pd.Timedelta(days=1)).to_numpy()
        festival_effect += np.exp(-0.5 * (days / 1.9) ** 2) * 45
    city_base = {"Amritsar": 65, "Ludhiana": 71, "Jalandhar": 66, "Chandigarh": 52, "Delhi": 87}[city]
    signal = (city_base + winter * 51 + 12 * np.cos((hour - 22) * np.pi / 12)
              - 1.2 * wind + 0.12 * humidity - rain * 3 + festival_effect)
    noise = rng.normal(0, 8, len(idx))
    pm = np.maximum(5, signal + noise).round(1)
    return pd.DataFrame({
        "time": idx, "pm2_5": pm, "temperature_2m": temp.round(1),
        "relative_humidity_2m": humidity.round(1), "wind_speed_10m": wind.round(1),
        "precipitation": rain.round(1),
    })


def fake_future_weather(history: pd.DataFrame) -> pd.DataFrame:
    """Generate synthetic future weather, not live forecasts."""
    last = history.iloc[-1]
    rows = []
    for h in range(1, 13):
        t = pd.Timestamp(last["time"]) + pd.Timedelta(hours=h)
        rows.append({
            "time": t,
            "temperature_2m": float(last["temperature_2m"]) + 1.4 * np.sin(h * np.pi / 10),
            "relative_humidity_2m": np.clip(float(last["relative_humidity_2m"]) + h * 0.3, 0, 100),
            "wind_speed_10m": max(0.2, float(last["wind_speed_10m"]) + 1.3 * np.sin(h * np.pi / 6)),
            "precipitation": 0.0,
        })
    return pd.DataFrame(rows)
