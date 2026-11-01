"""Diwali-specific measured/modelled-history comparisons and bounded provider forecasts.

Daily values are *not* the existing 1–6h Random Forest forecasts. Never silently fill
unavailable future dates with invented pollution values.
"""
from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd

from data_sources import AIR_URL, _hourly, _request

# Holiday dates confirmed against Government of India lists.
DIWALI_BY_YEAR: dict[int, date] = {
    2024: date(2024, 10, 31),
    2025: date(2025, 10, 20),
    2026: date(2026, 11, 8),
}
TARGET_YEAR = 2026
OFFSETS = tuple(range(-3, 4))
VARIABLES = ["pm2_5", "us_aqi"]


def phase_name(offset: int) -> str:
    return "Before Diwali" if offset < 0 else "On Diwali" if offset == 0 else "After Diwali"


def phase_for_date(value: date, festival_date: date = DIWALI_BY_YEAR[2026]) -> str:
    return phase_name((value - festival_date).days)


def _ensure_hourly(frame: pd.DataFrame, festival: date, year: int, source: str) -> pd.DataFrame:
    """Keep a date-relative day index, so different festival dates can be compared."""
    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).copy()
    df["date"] = df["time"].dt.date
    df["offset"] = df["date"].map(lambda d: (d - festival).days)
    df = df[df["offset"].isin(OFFSETS)].copy()
    for col in VARIABLES:
        if col not in df:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = year
    df["source"] = source
    return df[["time", "date", "offset", "year", "source", *VARIABLES]].reset_index(drop=True)


def historical_archive(lat: float, lon: float, year: int) -> pd.DataFrame:
    """Fetch historical regional CAMS forecasts/model output via Open-Meteo.

    The API may have coverage gaps, which should be shown rather than fabricated.
    """
    day = DIWALI_BY_YEAR[year]
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "start_date": (day - timedelta(days=3)).isoformat(),
        "end_date": (day + timedelta(days=3)).isoformat(),
        "hourly": "pm2_5,us_aqi", "domains": "cams_global",
    }
    result = _request(AIR_URL, params)
    return _ensure_hourly(_hourly(result, VARIABLES), day, year, "Open-Meteo CAMS historical model")


def provider_future(lat: float, lon: float) -> pd.DataFrame:
    """Current published CAMS forecast, maximum 7 days, not our Random Forest."""
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "hourly": "pm2_5,us_aqi", "domains": "cams_global",
        "forecast_days": 7,
    }
    result = _request(AIR_URL, params)
    return _ensure_hourly(_hourly(result, VARIABLES), DIWALI_BY_YEAR[TARGET_YEAR],
                          TARGET_YEAR, "Open-Meteo CAMS provider forecast")


def from_uploaded_history(history: pd.DataFrame) -> pd.DataFrame:
    """Use only genuine uploaded dates in a Diwali ±3-day window."""
    frames = []
    if "time" not in history or "pm2_5" not in history:
        return pd.DataFrame(columns=["time", "date", "offset", "year", "source", *VARIABLES])
    timestamps = pd.to_datetime(history["time"], errors="coerce")
    for year in (2024, 2025):
        pieces = history[timestamps.dt.year == year].copy()
        if not pieces.empty:
            frames.append(_ensure_hourly(pieces, DIWALI_BY_YEAR[year], year,
                                         "Uploaded historical PM2.5"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["time", "date", "offset", "year", "source", *VARIABLES])


def synthetic_archive() -> pd.DataFrame:
    """Clearly fictional demo data, never silently used in Live/CSV mode."""
    rng = np.random.default_rng(2026)
    frames = []
    for year in (2024, 2025):
        festival = DIWALI_BY_YEAR[year]
        times = pd.date_range(pd.Timestamp(festival) - pd.Timedelta(days=3), periods=7*24, freq="h")
        offsets = np.array([(t.date() - festival).days for t in times])
        hour = times.hour.to_numpy()
        baseline = 83 + (year-2024)*8 + 13 * np.cos((hour-21)*np.pi/12)
        festival_bump = np.where(offsets==0, 54, np.where(offsets==1, 35, np.where(offsets==-1, 15, 5)))
        pm = np.clip(baseline + festival_bump + rng.normal(0, 6, len(times)), 5, None)
        frame = pd.DataFrame({"time": times, "pm2_5": pm, "us_aqi": np.nan})
        frames.append(_ensure_hourly(frame, festival, year, "SYNTHETIC DEMO — NOT MEASURED"))
    return pd.concat(frames, ignore_index=True)


def daily_stats(frame: pd.DataFrame, variable: str = "pm2_5", min_samples: int = 12) -> pd.DataFrame:
    """Average within each date with coverage checks; missing dates remain missing."""
    if frame.empty or variable not in frame:
        return pd.DataFrame(columns=["year", "offset", "date", "mean", "peak", "n", "source"])
    working = frame.dropna(subset=[variable]).copy()
    if working.empty:
        return pd.DataFrame(columns=["year", "offset", "date", "mean", "peak", "n", "source"])
    aggregate = (working.groupby(["year", "offset", "date", "source"], as_index=False)
                 .agg(mean=(variable, "mean"), peak=(variable, "max"), n=(variable, "count")))
    return aggregate[aggregate["n"] >= min_samples].reset_index(drop=True)


def build_outlook(history: pd.DataFrame, future: pd.DataFrame | None = None,
                  variable: str = "pm2_5") -> pd.DataFrame:
    """Separate historical analogue from a genuine upcoming provider forecast.

    `reference` is only a prior-year median (NOT a trained forecast).
    `forecast` is populated only when a provider actually returns that date.
    """
    historical_days = daily_stats(history, variable)
    future_days = daily_stats(future, variable) if future is not None else daily_stats(pd.DataFrame(), variable)
    records = []
    for offset in OFFSETS:
        day = DIWALI_BY_YEAR[TARGET_YEAR] + timedelta(days=offset)
        old = historical_days[historical_days["offset"] == offset]
        incoming = future_days[(future_days["offset"] == offset) & (future_days["date"] == day)]
        records.append({
            "offset": offset, "date": day, "phase": phase_name(offset),
            "reference": float(old["mean"].median()) if len(old) else np.nan,
            "reference_years": int(old["year"].nunique()),
            "forecast": float(incoming.iloc[0]["mean"]) if len(incoming) else np.nan,
            "forecast_hours": int(incoming.iloc[0]["n"]) if len(incoming) else 0,
        })
    return pd.DataFrame(records)


def phase_summary(outlook: pd.DataFrame, column: str = "reference") -> dict[str, float | None]:
    output = {}
    for name in ("Before Diwali", "On Diwali", "After Diwali"):
        values = outlook.loc[outlook["phase"] == name, column].dropna()
        output[name] = float(values.mean()) if len(values) else None
    return output
