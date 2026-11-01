"""Measured CPCB monitoring-network PM2.5 via XKDR India Air Quality Database.

CPCB observations are not official AQI. The free public demo token accesses 2024 only.
A private token enables other years. All timestamps in Indian Standard Time.
"""
from __future__ import annotations
from datetime import date, timedelta
from io import BytesIO
import pandas as pd
import numpy as np
import requests

XKDR_API = "https://airquality.xkdr.org/v1/measurements"
PUBLIC_2024_KEY = "aqi_demo_wbf92Qx21zX-Wa_Tg8Dx1nXe"  # PUBLIC, published by XKDR


def normalize_station_csv(content: bytes | str, day: date) -> pd.DataFrame:
    """Aggregate each hour across reporting CPCB stations, with one vote per station.

    A city average is unweighted by station density and is NOT a hyperlocal estimate.
    """
    raw = pd.read_csv(BytesIO(content if isinstance(content,bytes) else content.encode('utf-8')))
    if raw.empty: return pd.DataFrame(columns=['time','pm2_5','station_count'])
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    timestamp_key = next((key for key in ('period_start','collected_at','time','timestamp') if key in raw), None)
    concentration_key = next((key for key in ('mean','value','pm2_5','pm25') if key in raw), None)
    if timestamp_key is None or concentration_key is None:
        raise ValueError('Station API CSV missing timestamp or mean/value column: '+', '.join(raw.columns))
    raw['time'] = pd.to_datetime(raw[timestamp_key],errors='coerce')
    if getattr(raw['time'].dt,'tz',None) is not None:
        raw['time']=raw['time'].dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    raw['pm2_5'] = pd.to_numeric(raw[concentration_key], errors='coerce')
    raw=raw.dropna(subset=['time','pm2_5']).copy()
    raw=raw[(raw['pm2_5']>=0)&(raw['pm2_5']<=2000)]
    raw=raw[raw.time.dt.date == day]
    raw['time']=raw['time'].dt.floor('h')
    station_col=next((k for k in ('station_id','station','station_name') if k in raw),None)
    if station_col is None:
        # Already aggregated provider response — show one feed only, do not fabricate station counts.
        raw['station_id']='provider_aggregate'
        station_col='station_id'
    # Deduplicate any multiple reports per station-hour before city aggregation.
    first=raw.groupby(['time',station_col],as_index=False)['pm2_5'].mean()
    grouped=first.groupby('time',as_index=False).agg(pm2_5=('pm2_5','mean'),station_count=(station_col,'nunique'))
    return grouped.sort_values('time').reset_index(drop=True)


def download_cpcb_festival(city: str, year: int, festival_date: date,
                            key: str = '', session=None) -> pd.DataFrame:
    """Fetch 7 real historical dates. Request each day independently to avoid demo row caps."""
    token = (key or '').strip() or PUBLIC_2024_KEY
    if year != 2024 and token == PUBLIC_2024_KEY:
        raise ValueError('The public demo key only covers 2024. Get a free personal XKDR key for '+str(year)+'.')
    request = session if session is not None else requests
    results=[]
    for offset in range(-3,4):
        day=festival_date+timedelta(days=offset)
        response=request.get(XKDR_API,params={
            'city':city,'source':'cpcb_caaqm','parameter':'PM2.5',
            'start':day.isoformat(),'end':day.isoformat(),'agg':'hourly','format':'csv',
        },headers={'Authorization':'Bearer '+token,'Accept':'text/csv'},timeout=40)
        response.raise_for_status()
        if response.headers.get('X-Truncated','').lower()=='true':
            raise ValueError('Station API truncated '+str(day)+'; please select a single station or narrow the request.')
        if not response.content or response.text.strip().startswith('{'):
            raise ValueError('Station API returned no CSV for '+str(day)+': '+response.text[:120])
        df=normalize_station_csv(response.content,day)
        if not df.empty:
            df['date']=df['time'].dt.date
            df['offset']=offset
            df['year']=year
            df['source']='CPCB monitoring stations (via XKDR Forum)'
            df['us_aqi']=np.nan
            results.append(df)
    if not results:
        raise ValueError('No CPCB measurements found for '+city+' near Diwali '+str(year)+'. Try Delhi or another city.')
    return pd.concat(results,ignore_index=True).sort_values('time').reset_index(drop=True)


def festival_evening_stats(frame:pd.DataFrame, year:int, festival: date) -> dict:
    """More relevant than broad phase averages: compare preceding evening, Diwali
    18:00-23:00 and next-day 00:00-05:00. Minimum 4 of 6 hours required.
    """
    if frame.empty: return {}
    w=frame.copy(); w['time']=pd.to_datetime(w['time'])
    w['pm2_5']=pd.to_numeric(w['pm2_5'],errors='coerce')
    categories={
        'Before • previous evening':(festival-timedelta(days=1),18,23),
        'On Diwali • evening':(festival,18,23),
        'After • next morning':(festival+timedelta(days=1),0,5),
    }
    out={}
    for name,(day,start,end) in categories.items():
        valid=w[(w.time.dt.date==day)&(w.time.dt.hour.between(start,end))].dropna(subset=['pm2_5'])
        if len(valid)>=4:
            out[name]={'pm2_5':float(valid.pm2_5.mean()),'peak':float(valid.pm2_5.max()),'hours':len(valid)}
        else:
            out[name]=None
    return out
