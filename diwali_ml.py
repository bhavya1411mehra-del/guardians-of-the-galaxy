"""Airfection Diwali 2026 *historical-pattern* forecasting.

No API key and no CAMS: trains solely on public archived Indian city-hour
CPCB-derived observations (2015-2020). Dataset's last July 2020 means the
usable festival dates are 2015-2019, NOT 2020. We do not invent sensor data.
Without future weather, this is a seasonal/Diwali analog forecast, NOT a
weather-conditioned short-term operational forecast.
"""
from __future__ import annotations
from datetime import date, timedelta
from io import BytesIO

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

TARGET = date(2026, 11, 8)
DIWALI_DATES = {
    2015: date(2015, 11, 11), 2016: date(2016, 10, 30),
    2017: date(2017, 10, 19), 2018: date(2018, 11, 7),
    2019: date(2019, 10, 27),
}
# CPCB-derived public city-hour archive; no login or API credentials.
SOURCE_URL = ("https://raw.githubusercontent.com/edusatyaki/"
              "DataAnalyticsProjrectIdeas/main/01_National_Air_Quality/"
              "data/Excel%20Dataset/city_hour.csv.gz")
DELHI_FALLBACK_URL = ("https://raw.githubusercontent.com/edusatyaki/"
                      "DataAnalyticsProjrectIdeas/main/01_National_Air_Quality/"
                      "data/Google%20Sheets%20Datasheets/delhi_hour.csv")
SOURCE_CREDIT = "CPCB-derived India city-hour dataset (2015–2020), publicly mirrored by edusatyaki"
FEATURES = ["relative_hour", "hour_sin", "hour_cos", "relative_day", "festival_evening", "after_midnight"]


def load_public_archive(get=None) -> pd.DataFrame:
    """Fetch old actual readings from public GitHub, with a Delhi fallback.

    This function is designed to run inside st.cache_data, so normal visitors
    download the file once per cache interval, not at every slider move.
    """
    request = get or requests.get
    try:
        response = request(SOURCE_URL, timeout=(8, 55))
        response.raise_for_status()
        if not response.content[:2] == b'\x1f\x8b':
            raise ValueError('Expected gzip archive, received unexpected content')
        frame = pd.read_csv(BytesIO(response.content), compression='gzip',
                            usecols=lambda col: col in {'City','Datetime','PM2.5','AQI'},
                            low_memory=False)
        if not {'City','Datetime','PM2.5'}.issubset(frame.columns):
            raise ValueError('Invalid city-hour source file columns')
    except Exception as first_error:
        try:
            response = request(DELHI_FALLBACK_URL, timeout=(8, 25))
            response.raise_for_status()
            frame = pd.read_csv(BytesIO(response.content),
                                usecols=lambda col: col in {'City','Datetime','PM2.5','AQI'},
                                low_memory=False)
            if not {'City','Datetime','PM2.5'}.issubset(frame.columns):
                raise ValueError('Invalid Delhi fallback columns')
        except Exception as second_error:
            raise RuntimeError('Public historical dataset could not be downloaded. '
                               'Check internet access and source availability. '
                               'No synthetic forecast has been substituted. '
                               f'Archive: {first_error}; fallback: {second_error}') from second_error
    return normalize_source(frame)


def normalize_source(frame: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps and retain only plausible real observations; never fill missing values."""
    required = {'City', 'Datetime', 'PM2.5'}
    if not required.issubset(frame):
        raise ValueError(f'Expected columns {sorted(required)}')
    data = pd.DataFrame({
        'city': frame['City'].astype(str).str.strip(),
        'time': pd.to_datetime(frame['Datetime'], errors='coerce'),
        'pm25': pd.to_numeric(frame['PM2.5'], errors='coerce'),
        'aqi': pd.to_numeric(frame['AQI'], errors='coerce') if 'AQI' in frame else np.nan,
    }).dropna(subset=['time'])
    data = data[(data.city != '') & (data.pm25.between(0, 1500) | data.aqi.between(0, 1200))].copy()
    data.loc[~data.pm25.between(0, 1500), 'pm25'] = np.nan
    data.loc[~data.aqi.between(0, 1200), 'aqi'] = np.nan
    # Shrink large 700k-row archive to genuine historical festival windows
    # before caching in Streamlit. 2020 ends in July, so no 2020 Diwali event.
    year = data.time.dt.year
    date_map = year.map(DIWALI_DATES)
    relative_days = (data.time.dt.normalize() - pd.to_datetime(date_map)).dt.days
    data = data[relative_days.between(-3, 3)].copy()
    # Some archive sources may have multiple readings per city/hour; average only actual records.
    data['time'] = data['time'].dt.floor('h')
    data = data.groupby(['city','time'], as_index=False)[['pm25','aqi']].mean()
    return data.sort_values(['city','time']).reset_index(drop=True)


def festival_rows(data: pd.DataFrame, city: str) -> pd.DataFrame:
    """Align previous Diwali events at hour 0, including 3 days each side."""
    obs = data[data.city.str.casefold() == city.casefold()].copy()
    if obs.empty:
        raise ValueError(f'No historical city-hour observations for {city}.')
    obs['year'] = obs.time.dt.year
    obs = obs[obs.year.isin(DIWALI_DATES)].copy()
    obs['event'] = obs.year.map(DIWALI_DATES)
    obs['relative_hour'] = [int((pd.Timestamp(t) - pd.Timestamp(d)).total_seconds() // 3600)
                            for t, d in zip(obs.time, obs.event)]
    obs = obs[obs.relative_hour.between(-72, 95)].copy()  # +/-3 days, incl after-third-day 23h
    if obs.empty:
        raise ValueError(f'No Diwali-period observations for {city}.')
    return add_features(obs)


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    h = d['relative_hour'].to_numpy(dtype=int)
    clock = np.mod(h, 24)
    d['hour_sin'] = np.sin(clock * (2*np.pi/24))
    d['hour_cos'] = np.cos(clock * (2*np.pi/24))
    d['relative_day'] = np.floor_divide(h, 24)
    d['festival_evening'] = ((np.floor_divide(h,24)==0) & (clock>=18)).astype(int)
    d['after_midnight'] = ((np.floor_divide(h,24)==1) & (clock<=5)).astype(int)
    return d


def years_with_coverage(rows: pd.DataFrame, metric: str, minimum: int = 65) -> list[int]:
    counts = rows.dropna(subset=[metric]).groupby('year').size()
    return sorted(int(year) for year, count in counts.items() if count >= minimum)


def _pred_features(hours: np.ndarray, year: int, metric: str) -> pd.DataFrame:
    df = pd.DataFrame({'relative_hour': hours.astype(int), 'year': year})
    return add_features(df)[FEATURES]


def train_diwali_outlook(history: pd.DataFrame, city: str) -> tuple[pd.DataFrame, dict]:
    """Train RandomForest on historic events, hold latest festival out for evaluation.

    Model inputs only contain date-relative features actually known for 2026.
    No leakage of future measurements or 2026 weather. Returned AQI is a
    historical-pattern model estimate, NOT an official CPCB forecast.
    """
    rows = festival_rows(history, city)
    aqi_years = years_with_coverage(rows, 'aqi')
    pm_years = years_with_coverage(rows, 'pm25')
    if len(aqi_years) >= 3:
        target = 'aqi'; years = aqi_years
        ylabel = 'Estimated Indian AQI (historical-pattern model)'; unit = 'AQI'
    elif len(pm_years) >= 3:
        target = 'pm25'; years = pm_years
        ylabel = 'Predicted PM2.5 concentration'; unit = 'µg/m³'
    else:
        raise ValueError('Too few independently observed Diwali events for ' + city +
                         '. Need at least 3 historical years with ≥65 hourly readings '
                         'in the festival window. Choose Delhi or add more data.')
    clean = rows[(rows.year.isin(years)) & rows[target].notna()].copy()
    train_years, holdout_year = years[:-1], years[-1]
    train = clean[clean.year.isin(train_years)]
    test = clean[clean.year == holdout_year]
    if len(train) < 120 or len(test) < 65:
        raise ValueError('Insufficient usable hours to evaluate this city.')

    model_args = dict(n_estimators=160, max_depth=8, min_samples_leaf=5,
                      max_features=0.9, random_state=42, n_jobs=-1)
    eval_model = RandomForestRegressor(**model_args)
    eval_model.fit(train[FEATURES], train[target])
    test_pred = eval_model.predict(test[FEATURES])
    holdout_mae = float(mean_absolute_error(test[target], test_pred))
    baseline = train.groupby('relative_hour')[target].median()
    baseline_pred = test.relative_hour.map(baseline).fillna(train[target].median())
    baseline_mae = float(mean_absolute_error(test[target], baseline_pred))
    # Refit using all available historical years for 2026 output.
    final_model = RandomForestRegressor(**model_args)
    final_model.fit(clean[FEATURES], clean[target])
    hours = np.arange(-72, 96, dtype=int)
    future_x = _pred_features(hours, 2026, target)
    prediction = np.maximum(final_model.predict(future_x), 0.)
    # Not a calibrated confidence interval. Historical cross-year variation provides an
    # approximate range around the predicted pattern, with at least holdout MAE width.
    historical_band = clean.groupby('relative_hour')[target].agg(['std','count'])
    hour_sd = historical_band['std'].reindex(hours).fillna(float(clean[target].std())).to_numpy()
    half_width = np.maximum(np.nan_to_num(hour_sd, nan=holdout_mae), holdout_mae)
    target_start = pd.Timestamp(TARGET) - pd.Timedelta(days=3)
    outcome = pd.DataFrame({
        'time': pd.date_range(target_start, periods=len(hours), freq='h'),
        'relative_hour': hours,
        'forecast': np.maximum(prediction, 0),
        'lower': np.maximum(0, prediction-half_width),
        'upper': prediction+half_width,
    })
    if target == 'aqi':
        # Display scale is CPCB's nominal 0-500 range; some historic archive
        # records exceed that, so do not claim 'official' or silently clip labels.
        outcome['forecast'] = outcome['forecast'].round(0)
        outcome['lower'] = outcome['lower'].round(0)
        outcome['upper'] = outcome['upper'].round(0)
    else:
        outcome[['forecast','lower','upper']] = outcome[['forecast','lower','upper']].round(1)
    info = {
        'city': city, 'metric': target, 'label': ylabel, 'unit': unit,
        'years': years, 'train_years': train_years, 'holdout_year': holdout_year,
        'samples': len(clean), 'holdout_mae': holdout_mae,
        'baseline_mae': baseline_mae, 'source': SOURCE_CREDIT,
        'target_date': TARGET.isoformat(),
        'limitations': 'Historical seasonal pattern only; cannot know 2026 firecracker activity, wind or rain. '
                       'Uncertainty band is indicative, not a calibrated prediction interval.',
    }
    return outcome, info


def phase_label(offset: int) -> str:
    return 'Before Diwali' if offset < 0 else 'Diwali Day' if offset == 0 else 'After Diwali'
