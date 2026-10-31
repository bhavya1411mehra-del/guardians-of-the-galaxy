"""Download real OpenAQ PM2.5 sensor histories; merge with Open-Meteo weather.

Never equate India AQI with an hourly PM2.5 concentration. One file represents
ONE monitoring location at local hourly timestamps. Network is only used when
an explicit user action requests data retrieval.
"""
from __future__ import annotations
from datetime import timedelta
from io import BytesIO
import math
import pandas as pd
import requests

from data_sources import ARCHIVE_URL, WEATHER_VARS, _hourly, _request

REQUIRED = ['time','pm2_5','temperature_2m','relative_humidity_2m','wind_speed_10m','precipitation']
WEATHER = REQUIRED[2:]
HEADERS = {'User-Agent': 'AIRFECTION research prototype (student hackathon)'}


def validate_pollution_csv(file) -> pd.DataFrame:
    raw = pd.read_csv(file)
    raw.columns = [c.strip().lower().replace(' ', '_') for c in raw.columns]
    rename = {'timestamp': 'time', 'datetime': 'time', 'date_time': 'time',
              'pm25': 'pm2_5', 'pm2.5': 'pm2_5', 'pm_2_5': 'pm2_5'}
    raw = raw.rename(columns=rename)
    if not {'time', 'pm2_5'}.issubset(raw.columns):
        raise ValueError('CSV must include time and pm2_5 columns. AQI alone is not PM2.5 data.')
    # Offsets may be provided, but train/model uses simple local clock per station.
    raw['time'] = pd.to_datetime(raw['time'], errors='coerce')
    if raw['time'].isna().any():
        raise ValueError('Some timestamps cannot be parsed. Use local YYYY-MM-DD HH:MM:SS.')
    if raw['time'].dt.tz is not None:
        raw['time'] = raw['time'].dt.tz_localize(None)
    raw['time'] = raw['time'].dt.floor('h')
    for c in ['pm2_5', *WEATHER]:
        if c in raw:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    raw = raw[(raw['pm2_5'] >= 0) & (raw['pm2_5'] <= 1500)]
    raw = raw.sort_values('time').drop_duplicates('time', keep='last').reset_index(drop=True)
    if len(raw) < 100:
        raise ValueError('Need at least 100 valid hourly PM2.5 readings to prepare a data file (more for ML training).')
    return raw


def full_training_csv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED if col not in df]
    if missing:
        raise ValueError('Missing weather columns: ' + ', '.join(missing) + '. Use Enrich with weather in Data Lab.')
    clean = df[REQUIRED].copy().sort_values('time').drop_duplicates('time').dropna()
    if len(clean) < 100:
        raise ValueError('Not enough complete hourly pollution/weather rows to process.')
    return clean.reset_index(drop=True)


def fetch_weather_for_history(air: pd.DataFrame, lat: float, lon: float,
                              max_days: int = 730) -> pd.DataFrame:
    """Fill weather variables from the Open-Meteo archive in 55-day requests.
    Timestamps in air must match the local time at coordinates (not UTC).
    """
    start = pd.Timestamp(air['time'].min()).date()
    stop = pd.Timestamp(air['time'].max()).date()
    if (stop - start).days > max_days:
        raise ValueError(f'Choose a time range of <= {max_days} days; download in separate batches for longer histories.')
    current = start
    frames = []
    while current <= stop:
        end = min(stop, current + timedelta(days=54))
        result = _request(ARCHIVE_URL, {'latitude': lat, 'longitude': lon,
                      'start_date': current.isoformat(), 'end_date': end.isoformat(),
                      'timezone': 'auto', 'hourly': WEATHER_VARS})
        frames.append(_hourly(result, WEATHER))
        current = end + timedelta(days=1)
    weather = pd.concat(frames, ignore_index=True).drop_duplicates('time')
    existing_weather = [c for c in WEATHER if c in air]
    merged = air.drop(columns=existing_weather).merge(weather, on='time', how='left')
    return full_training_csv(merged)


def openaq_get(path: str, api_key: str, params: dict | None = None) -> dict:
    if not api_key.strip():
        raise ValueError('OpenAQ API key is required. Create a free key at explore.openaq.org.')
    reply = requests.get('https://api.openaq.org/v3/' + path.lstrip('/'),
        params=params, headers={**HEADERS, 'X-API-Key': api_key.strip()}, timeout=30)
    if reply.status_code in (401,403):
        raise RuntimeError('OpenAQ rejected your key or your account cannot access that resource.')
    if reply.status_code == 429:
        raise RuntimeError('OpenAQ API rate limit reached. Try a smaller date range later.')
    reply.raise_for_status()
    payload = reply.json()
    if not isinstance(payload, dict):
        raise RuntimeError('OpenAQ API returned an unexpected response.')
    return payload


def find_pm25_sensors(api_key: str, lat: float, lon: float, radius_m: int = 25000):
    """Discover PM2.5 sensors within maximum OpenAQ 25km search radius."""
    if not 0 < radius_m <= 25000:
        raise ValueError('OpenAQ radius must be 1..25000 metres.')
    payload = openaq_get('locations', api_key,
        {'coordinates': f'{lat:.4f},{lon:.4f}', 'radius': radius_m,
         'limit':100, 'page':1})
    choices = []
    for loc in payload.get('results', []):
        coordinates = loc.get('coordinates') or {}
        sensors = loc.get('sensors') or []
        for sensor in sensors:
            p = sensor.get('parameter') or {}
            name = str(p.get('name') or '').lower().replace('.', '').replace('_','')
            unit = str(p.get('units') or '')
            if name not in ('pm25','pm2 5'):
                continue
            if unit and ('µg' not in unit and 'ug' not in unit.lower()):
                continue
            choices.append({'station': str(loc.get('name') or f"Location {loc.get('id')}"),
                            'location_id': loc.get('id'), 'sensor_id': sensor.get('id'),
                            'lat': coordinates.get('latitude'), 'lon': coordinates.get('longitude'),
                            'units': unit or 'verify original sensor unit'})
    return choices


def download_pm25_hours(api_key: str, sensor_id: int, start: str, end: str,
                         max_pages: int = 12) -> pd.DataFrame:
    """Download up to 12k hourly samples; use short windows to avoid provider limits."""
    records = []
    for page in range(1, max_pages+1):
        payload = openaq_get(f'sensors/{int(sensor_id)}/hours', api_key,
            {'date_from': start, 'date_to': end, 'limit': 1000, 'page': page})
        batch = payload.get('results', [])
        for measurement in batch:
            period = measurement.get('period') or {}
            span = (period.get('datetimeFrom') or {}).get('local')
            if span is None:
                span = (period.get('datetimeFrom') or {}).get('utc')
            if span is None:
                continue
            parameter = measurement.get('parameter') or {}
            pname = str(parameter.get('name', '')).lower().replace('.', '').replace('_','')
            if pname and pname != 'pm25':
                continue
            unit = str(parameter.get('units') or '')
            if unit and ('µg' not in unit and 'ug' not in unit.lower()):
                raise ValueError('Sensor data units are not µg/m³: ' + unit)
            dt = pd.Timestamp(span)
            records.append({'time': dt.tz_localize(None).floor('h'), 'pm2_5': measurement.get('value')})
        if len(batch) < 1000:
            break
    if not records:
        raise ValueError('No sensor hourly data returned for this date range.')
    return validate_pollution_csv(pd.DataFrame(records).to_csv(index=False))
