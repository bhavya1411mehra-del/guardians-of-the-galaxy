"""Open-Meteo modeled air-quality data and weather forecasts (not ground sensor readings)."""
from __future__ import annotations

from datetime import timedelta
import pandas as pd
import requests

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
TIMEZONE = "auto"


def _request(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=35, headers={"User-Agent": "Airfection student hackathon prototype"})
    response.raise_for_status()
    payload = response.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise ValueError(payload.get("reason", "API returned an error"))
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict) and item.get("error"):
                raise ValueError(item.get("reason", "Multi-location API returned an error"))
    return payload


def _hourly(payload: dict, variables: list[str]) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError("API returned no hourly data")
    result = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})
    for variable in variables:
        result[variable] = pd.to_numeric(hourly.get(variable, [None] * len(result)), errors="coerce")
    return result


def _merge(air: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    merged = air.merge(weather, on="time", how="inner").sort_values("time")
    return merged.drop_duplicates("time").reset_index(drop=True)


def get_live_history(lat: float, lon: float) -> pd.DataFrame:
    """Training data: previous Diwali 2025 plus recent 120 days, using modeled CAMS PM2.5."""
    today = pd.Timestamp.now(tz="UTC").date()
    windows = [(today - timedelta(days=122), today - timedelta(days=2))]
    if today > pd.Timestamp("2025-11-20").date():
        windows.insert(0, (pd.Timestamp("2025-10-01").date(), pd.Timestamp("2025-11-20").date()))
    frames = []
    for start, end in windows:
        if start > end:
            continue
        args = {"latitude": lat, "longitude": lon, "start_date": start.isoformat(),
                "end_date": end.isoformat(), "timezone": TIMEZONE}
        air = _request(AIR_URL, {**args, "hourly": "pm2_5", "domains": "cams_global"})
        weather = _request(ARCHIVE_URL, {**args, "hourly": WEATHER_VARS})
        frames.append(_merge(_hourly(air, ["pm2_5"]), _hourly(weather, WEATHER_VARS.split(","))))
    if not frames:
        raise RuntimeError("No historical date windows available")
    return pd.concat(frames, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)


def get_live_recent_and_future(lat: float, lon: float) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Near-current modeled PM2.5 and near-current/future forecast weather."""
    params = {"latitude": lat, "longitude": lon, "timezone": TIMEZONE,
              "past_days": 3, "forecast_days": 2}
    air = _request(AIR_URL, {**params, "hourly": "pm2_5", "domains": "cams_global"})
    weather = _request(FORECAST_URL, {**params, "hourly": WEATHER_VARS})
    air_df = _hourly(air, ["pm2_5"])
    weather_df = _hourly(weather, WEATHER_VARS.split(","))
    recent = _merge(air_df, weather_df)
    local_zone = weather.get("timezone", "UTC")
    now = pd.Timestamp.now(tz=local_zone).tz_localize(None).floor("h")
    recent = recent[recent["time"] <= now].reset_index(drop=True)
    future = weather_df[weather_df["time"] > now].reset_index(drop=True)
    return recent, future, local_zone


def get_provider_forecast(lat: float, lon: float, target_time: pd.Timestamp) -> float | None:
    """A CAMS provider forecast for another city on the map, not our own ML prediction."""
    params = {"latitude": lat, "longitude": lon, "timezone": TIMEZONE,
              "hourly": "pm2_5", "forecast_days": 2, "domains": "cams_global"}
    payload = _request(AIR_URL, params)
    hours = _hourly(payload, ["pm2_5"])
    matches = hours.loc[hours["time"] == pd.Timestamp(target_time), "pm2_5"].dropna()
    return float(matches.iloc[0]) if len(matches) else None


def get_spatial_forecast_grid(lat: float, lon: float, target_time: pd.Timestamp,
                              side: int = 5, spacing_deg: float = 0.4) -> pd.DataFrame:
    """Sample the *regional CAMS model*, NOT monitoring stations, around a location.

    Returns actual provider grid-cell centers where available, deduplicated.
    US AQI is the Open-Meteo/CAMS-derived US index, not CPCB/Indian AQI.
    Multiple requested coordinates can resolve to the same 45 km model cell.
    """
    from itertools import product
    import numpy as np

    if side < 2 or side > 7:
        raise ValueError("Map sample side must be between 2 and 7")
    offsets = np.linspace(-(side-1)/2, (side-1)/2, side) * spacing_deg
    factor = 1 / max(0.3, np.cos(np.deg2rad(lat)))
    requests_points = [(float(np.clip(lat + dy, -89.9, 89.9)),
                        float(np.clip(lon + dx * factor, -179.9, 179.9)))
                       for dy, dx in product(offsets, offsets)]
    params = {
        "latitude": ",".join(str(round(x, 4)) for x, _ in requests_points),
        "longitude": ",".join(str(round(y, 4)) for _, y in requests_points),
        "timezone": "auto", "hourly": "pm2_5,us_aqi", "forecast_days": 2,
        "domains": "cams_global", "cell_selection": "nearest",
    }
    payloads = _request(AIR_URL, params)
    if isinstance(payloads, dict):
        payloads = [payloads]
    if len(payloads) != len(requests_points):
        raise ValueError("Regional API returned an unexpected number of forecast locations")
    target = pd.Timestamp(target_time).tz_localize(None) if pd.Timestamp(target_time).tzinfo else pd.Timestamp(target_time)
    rows = []
    for (requested_lat, requested_lon), data in zip(requests_points, payloads):
        hourly = _hourly(data, ["pm2_5", "us_aqi"])
        hit = hourly[hourly["time"] == target]
        if hit.empty:
            continue
        record = hit.iloc[0]
        if pd.isna(record["pm2_5"]):
            continue
        rows.append({"lat": float(data.get("latitude", requested_lat)),
                     "lon": float(data.get("longitude", requested_lon)),
                     "pm2_5": float(record["pm2_5"]),
                     "us_aqi": float(record["us_aqi"]) if pd.notna(record["us_aqi"]) else float('nan'),
                     "source": "Open-Meteo CAMS global regional model"})
    if not rows:
        raise ValueError("CAMS supplied no regional map values for that forecast time")
    # Don't display duplicate provider cells as independent observations.
    return (pd.DataFrame(rows).drop_duplicates(subset=["lat", "lon"])
            .reset_index(drop=True))
