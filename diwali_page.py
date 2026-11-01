"""Standalone Diwali forecasting experience; no viewer API key, no CAMS."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from diwali_ml import TARGET, load_public_archive, train_diwali_outlook, phase_label


@st.cache_resource(show_spinner=False)
def _archived_data():
    return load_public_archive()


@st.cache_data(ttl=86400, show_spinner=False)
def _trained_outlook(city: str):
    return train_diwali_outlook(_archived_data(), city)


def _design(fig, height=330):
    fig.update_layout(height=height, margin=dict(l=8,r=10,t=20,b=8),
        paper_bgcolor='#ffffff', plot_bgcolor='#ffffff',
        font=dict(color='#3a5060',size=12),
        legend=dict(orientation='h', y=1.13), hovermode='x unified',
        xaxis=dict(gridcolor='#e6ecf0'),
        yaxis=dict(gridcolor='#e6ecf0'))
    return fig


def render_diwali_page(default_city='Delhi'):
    st.title("Diwali 2026 air quality forecast")
    st.write("**8 November 2026** | Hourly estimates for 5–11 November")
    st.caption("Choose a city, date and hour. Estimates are trained on historical Diwali periods, not published 2026 forecasts.")
    st.caption('Forecast type: historical-pattern machine learning estimate — not an official government AQI forecast or a guarantee about 2026 weather.')
    try:
        with st.spinner('Loading archived Indian air-quality observations (first load only)…'):
            data = _archived_data()
    except Exception as exc:
        st.error(f'Could not load real historical data: {exc}')
        st.info('The app does not substitute fictional values when the archive is unavailable. Check GitHub connectivity and try refreshing.')
        return
    available = sorted(data.city.dropna().unique().tolist())
    if not available:
        st.error('The public dataset returned no usable records.'); return
    default = default_city if default_city in available else ('Delhi' if 'Delhi' in available else available[0])
    city = st.selectbox('Choose forecast city', available, index=available.index(default), key='diwali_model_city')
    try:
        with st.spinner(f'Training and evaluating historical Diwali patterns for {city}…'):
            forecast, info = _trained_outlook(city)
    except Exception as exc:
        st.warning(str(exc))
        st.caption('This city needs more real historical festival data. Try Delhi first. No made-up forecasts are displayed.')
        return

    c1,c2 = st.columns(2)
    with c1:
        day_offset = st.slider('Day relative to Diwali', -3, 3, 0, 1,
                               format='%d days', key='festival_predict_day')
    with c2:
        hour = st.slider('Time of day (IST)', 0, 23, 20, 1, key='festival_predict_hour')
    picked_day = TARGET + pd.Timedelta(days=day_offset)
    selected_time = pd.Timestamp(picked_day) + pd.Timedelta(hours=hour)
    chosen = forecast.loc[forecast.time == selected_time]
    if chosen.empty:
        st.error('Hourly result missing: selected date/time not found.'); return
    row = chosen.iloc[0]
    day = forecast[forecast.time.dt.date == picked_day].copy()
    group = phase_label(day_offset)
    st.subheader(f'{group} | {selected_time:%d %B %Y, %I:%M %p} IST')
    k1,k2,k3 = st.columns(3)
    k1.metric(f'2026 predicted {info["unit"]}', f'{row.forecast:.0f}' if info['metric']=='aqi' else f'{row.forecast:.1f}',
              help=info['label'])
    k2.metric('Indicative low–high', f'{row.lower:.0f}–{row.upper:.0f}' if info['metric']=='aqi' else f'{row.lower:.1f}–{row.upper:.1f}')
    k3.metric('Predicted daily peak', f'{day.forecast.max():.0f}' if info['metric']=='aqi' else f'{day.forecast.max():.1f}')
    if info['metric'] == 'pm25':
        st.warning('This city lacks enough historical AQI values, so this forecast is PM2.5 µg/m³, **not the Indian AQI**.')
    else:
        st.caption('This is an ML estimate trained on historical reported AQI values. It is not a current official CPCB AQI reading.')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=day.time,y=day.upper,mode='lines',line=dict(width=0),showlegend=False,hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=day.time,y=day.lower,mode='lines',fill='tonexty',
                             fillcolor='rgba(40,94,131,.12)',line=dict(width=0),
                             name='Indicative historical variation',hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=day.time,y=day.forecast,mode='lines+markers',
                             name='AIRFECTION 2026 forecast',line=dict(color='#285e83',width=2)))
    fig.add_trace(go.Scatter(x=[selected_time], y=[row.forecast],name='Selected hour',mode='markers',
                             marker=dict(color='#b25234',size=11,line=dict(color='#fff',width=2))))
    fig.update_xaxes(tickformat='%I %p',title='Hour of day (IST)')
    fig.update_yaxes(title=info['label'])
    st.plotly_chart(_design(fig,360),use_container_width=True,config={'displayModeBar':False})

    st.subheader('Seven-day outlook | 5–11 November')
    days = forecast.assign(day=forecast.time.dt.date).groupby('day',as_index=False).agg(
        mean=('forecast','mean'),peak=('forecast','max'))
    fig2=go.Figure()
    fig2.add_trace(go.Scatter(x=forecast.time,y=forecast.forecast,mode='lines',
                              name='Hourly model forecast',line=dict(color='#285e83',width=2)))
    fig2.add_vrect(x0=pd.Timestamp(TARGET),x1=pd.Timestamp(TARGET)+pd.Timedelta(days=1),
                   fillcolor='#a9bfce',opacity=.2,line_width=0)
    fig2.update_xaxes(title='Time (India Standard Time)',tickformat='%d %b')
    fig2.update_yaxes(title=info['label'])
    st.plotly_chart(_design(fig2,270),use_container_width=True,config={'displayModeBar':False})
    st.download_button('Download hourly forecast data (CSV)',
        forecast.to_csv(index=False).encode('utf-8'),
        file_name=f'airfection_{city.lower().replace(" ","_")}_diwali_2026_predictions.csv',mime='text/csv')
    with st.expander('How was this prediction trained and tested?'):
        st.markdown('**Model:** Scikit-learn Random Forest Regressor, trained on real historical city-hour air quality records aligned by Diwali dates and hour of day.')
        st.markdown('**Training years:** '+', '.join(map(str,info['years'])))
        st.markdown(f'**Observation rows used:** {info["samples"]:,} · **Last event held out:** {info["holdout_year"]}')
        st.markdown(f'**Holdout model MAE:** {info["holdout_mae"]:.1f} {info["unit"]} · **Historical median baseline MAE:** {info["baseline_mae"]:.1f} {info["unit"]}')
        if info['holdout_mae'] >= info['baseline_mae']:
            st.warning('The model did not beat the simple historical-profile baseline on the last Diwali event. Treat the forecast as exploratory.')
        st.markdown('**What we cannot know in advance:** actual 2026 firecracker use, wind, temperature, rainfall, restrictions or unusual events. These forecasts are seasonal/festival patterns, not weather-conditioned predictions.')
        st.caption('Variation band is descriptive/indicative, not a calibrated 90% confidence interval.')
        st.markdown('[Historical dataset source (public GitHub)](https://github.com/edusatyaki/DataAnalyticsProjrectIdeas/tree/main/01_National_Air_Quality) · Original data credited to CPCB-derived India air-quality archives. Timestamps treated as IST.')
    st.caption('No API key is required to view this page. The app caches the public archive and fitted model for 24 hours to avoid retraining on slider movements.')
