"""AIRFECTION: premium Streamlit frontend for the transparent PM2.5 hackathon prototype."""
from __future__ import annotations

from html import escape
from datetime import timedelta
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st
from PIL import Image

from data_sources import get_live_history, get_live_recent_and_future, get_spatial_forecast_grid
from data_lab import (validate_pollution_csv, full_training_csv, fetch_weather_for_history,
                      find_pm25_sensors, download_pm25_hours, REQUIRED)
from health_evidence import guidance, reference_for, SOURCES, pm25_24h_mean, WHO_GUIDELINE
from forecasting import HORIZONS, forecast, train_model
from heatmap import color_for_value, demo_grid, interpolate_surface
from sample_data import CITIES, demo_city, fake_future_weather

st.set_page_config(page_title="AIRFECTION | Predict the smog before the sparkle", page_icon="🌬️", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"<style>{(Path(__file__).parent / 'ui.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

ROUTES = ["◈  Dashboard", "🪔  Diwali Forecast", "◎  Forecast Heatmap", "↗  Spike Radar", "♡  Health Shield", "▧  Sky Photo AI", "▤  Data Lab", "⚙  Model & Evidence"]
PALETTE = {"bg": "#111724", "grid": "rgba(199,214,230,.08)", "pink": "#fa90ad", "mint": "#92edc5", "amber": "#fac17f", "text": "#e9f0f9"}


@st.cache_data(ttl=1800, show_spinner=False)
def load_demo(city):
    return demo_city(city)


@st.cache_data(ttl=3600, show_spinner=False)
def load_live_history(lat, lon):
    return get_live_history(lat, lon)


@st.cache_data(ttl=1200, show_spinner=False)
def load_live_near(lat, lon):
    return get_live_recent_and_future(lat, lon)


@st.cache_data(ttl=1800, show_spinner=False)
def load_spatial_grid(lat, lon, time_string):
    return get_spatial_forecast_grid(lat, lon, pd.Timestamp(time_string))


@st.cache_resource(show_spinner=False)
def fitted(dataframe):
    return train_model(dataframe)


def label_section(kicker, title, subtitle="", right=""):
    st.markdown(
        f'<div class="section-top"><div><div class="kicker">{escape(kicker)}</div>'
        f'<div class="section-title">{escape(title)}</div></div>'
        f'{f"<span class=\"tag\">{escape(right)}</span>" if right else ""}</div>'
        f'{f"<p class=\"section-sub\">{escape(subtitle)}</p>" if subtitle else ""}',
        unsafe_allow_html=True,
    )


def panel(title, description, kind=""):
    st.markdown(f'<div class="panel {kind}"><h3>{escape(title)}</h3><p>{escape(description)}</p></div>', unsafe_allow_html=True)


def plot_style(fig, height=335):
    fig.update_layout(
        height=height, margin=dict(l=6, r=12, t=16, b=10),
        paper_bgcolor="#151c2b", plot_bgcolor="#151c2b", font=dict(color="#aab6c9", size=11),
        hovermode="x unified", showlegend=True,
        legend=dict(orientation="h", y=1.13, x=0, font=dict(size=11)),
        xaxis=dict(gridcolor=PALETTE["grid"], showline=False, zeroline=False),
        yaxis=dict(gridcolor=PALETTE["grid"], showline=False, zeroline=False),
    )
    return fig


def line_chart(latest, predictions, selected_hour):
    now = pd.Timestamp(latest["time"])
    line = pd.concat([
        pd.DataFrame({"time": [now], "pm2_5": [float(latest["pm2_5"])]}),
        predictions[["time", "pm2_5"]],
    ], ignore_index=True)
    picked = predictions.loc[predictions["horizon"] == selected_hour].iloc[0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=line["time"], y=line["pm2_5"], name="ML forecast",
                             mode="lines+markers", line=dict(color="#fa92ad", width=3, shape="spline"),
                             marker=dict(size=7, color="#ffa6bf"),
                             fill="tozeroy", fillcolor="rgba(248,113,151,.065)"))
    fig.add_trace(go.Scatter(x=[pd.Timestamp(picked["time"])], y=[picked["pm2_5"]], name=f"Selected +{selected_hour}h",
                             mode="markers", marker=dict(size=15, color="#a0f3d1", line=dict(color="#132129", width=3)),
                             hovertemplate="Selected forecast: %{y:.1f} µg/m³<extra></extra>"))
    fig.add_trace(go.Scatter(x=line["time"], y=[float(latest["pm2_5"])] * len(line), name="Current reference",
                             mode="lines", line=dict(color="#79889c", dash="dot", width=1.5), opacity=.8))
    fig.update_yaxes(title_text="PM2.5 (µg/m³)", rangemode="tozero")
    fig.update_xaxes(title_text=None, tickformat="%I %p")
    return plot_style(fig)


def is_uploaded_dataset():
    return bool(st.session_state.get("active_data_mode", "").startswith("CSV"))

def map_rows(city, coords, is_demo, target_time, selected_hour, predicted):
    """Return spatial data suitable for a surface, never unrelated city markers."""
    if is_uploaded_dataset():
        return pd.DataFrame([{"city": city, "lat": coords[0], "lon": coords[1],
                              "pm2_5": predicted, "us_aqi": np.nan,
                              "source": "Single uploaded station / ML forecast"}])
    if is_demo:
        return demo_grid(coords[0], coords[1], predicted, selected_hour)
    return load_spatial_grid(coords[0], coords[1], str(target_time))


def render_map(rows, chosen, center, metric="pm2_5", steps=34):
    """Interpolated regional forecast surface; no fake glow, no ungrounded points."""
    if isinstance(rows, list):
        frame = pd.DataFrame(rows)
    else:
        frame = rows.copy()
    if frame.empty:
        st.warning("No geographical forecast points found for this time.")
        return
    frame = frame.dropna(subset=["lat", "lon", metric]).copy()
    if frame.empty:
        st.warning("No valid data for the selected map variable.")
        return
    frame["fill"] = frame[metric].map(lambda val: color_for_value(val, metric, 245))
    layer_list = []
    surface = interpolate_surface(frame, metric, steps=steps)
    if not surface.empty:
        layer_list.append(pdk.Layer("PolygonLayer", data=surface, id="forecast_surface",
                                    get_polygon="polygon", get_fill_color="fill",
                                    filled=True, stroked=False, pickable=True,
                                    auto_highlight=False, extruded=False))
    # Actual forecast sampling points displayed as small dots, not huge blobs.
    layer_list.append(pdk.Layer("ScatterplotLayer", data=frame,
                                id="forecast_nodes", get_position="[lon, lat]",
                                get_fill_color=[247, 248, 252, 230],
                                get_radius=1750, radius_min_pixels=2, radius_max_pixels=5,
                                stroked=True, get_line_color=[22, 30, 40, 245],
                                line_width_min_pixels=1, pickable=True))
    view = pdk.ViewState(latitude=float(center[0]),longitude=float(center[1]),
                         zoom=7.2 if len(frame)>2 else 9.4, pitch=0)
    unit = "US AQI" if metric == "us_aqi" else "PM2.5 µg/m³"
    deck = pdk.Deck(map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                    initial_view_state=view,layers=layer_list,
                    tooltip={"html": f"<b>{unit}</b><br>Interpolated estimate: {{value}}<br>Colored area is not a sensor measurement",
                             "style": {"backgroundColor": "#141e2e", "color":"#f5f4f8", "borderRadius":"12px"}})
    st.pydeck_chart(deck, use_container_width=True)
    if surface.empty:
        st.warning("Too few geographically distinct forecast grid cells were available. A continuous heatmap cannot be estimated reliably, so only supplied locations are shown.")
    else:
        st.caption(f"Colored surface interpolated from {len(frame)} distinct regional forecast cells. A smoother display does NOT imply higher data resolution or neighborhood-level accuracy.")
    return len(surface)


def render_legend(metric):
    if metric == "us_aqi":
        labels = [(0,"0"),(50,"50"),(100,"100"),(150,"150"),(200,"200"),(300,"300+")]
        unit = "Open-Meteo US AQI forecast · NOT Indian CPCB AQI"
    else:
        labels = [(0,"0"),(15,"15"),(35,"35"),(75,"75"),(125,"125"),(200,"200+")]
        unit = "PM2.5 µg/m³ · NOT AQI"
    gradient = ','.join(f'rgb({",".join(map(str,color_for_value(v,metric)[:3]))})' for v,_ in labels)
    tick_html = ''.join(f'<span>{escape(label)}</span>' for _,label in labels)
    st.markdown(f'<div class="heatmap-key"><div class="gradient-bar" style="background:linear-gradient(90deg,{gradient})"></div><div class="gradient-ticks">{tick_html}</div><div class="gradient-caption">{escape(unit)} · Darker red = higher modeled pollution</div></div>', unsafe_allow_html=True)



with st.sidebar:
    st.markdown('<div class="brandline"><span class="brandmark">✺</span><span class="brand">AIRFECTION</span></div>', unsafe_allow_html=True)
    st.caption("THE FUTURE OF AIR, IN FOCUS")
    st.markdown("---")
    st.caption("WORKSPACE")
    nav = st.radio("Explore AIRFECTION", ROUTES, label_visibility="collapsed", key="nav_route")
    st.markdown('<div class="diwali-sidebar"><span>🪔</span><strong>DIWALI 2026</strong><small>8 November · before / during / after</small></div>', unsafe_allow_html=True)
    st.markdown("---")
    if nav == ROUTES[1]:
        st.caption("DIWALI FORECAST")
        st.info("Real historical CPCB-derived data · Machine-learning forecast for 2026 · No API key")
        mode = "Diwali ML historical archive"
        is_demo = False
        is_uploaded = False
        chosen_location = st.session_state.get("diwali_model_city", "Delhi")
        city = chosen_location
        latitude, longitude = CITIES.get(city, CITIES["Delhi"])
    else:
        st.caption("DATA MODE")
        mode = st.selectbox("Data source", ["Demo · Synthetic data", "Live · Open-Meteo models", "CSV · Uploaded station data"],
                            help="Demo is fictional; live is based on regional modeled environmental data, not monitoring stations.")
        is_demo = mode.startswith("Demo")
        is_uploaded = mode.startswith("CSV")
        chosen_location = st.selectbox("Location", [*CITIES.keys(), "Custom coordinates"])
        if chosen_location == "Custom coordinates":
            city = st.text_input("Your location label", value="Custom location").strip() or "Custom location"
            latitude = st.number_input("Latitude", min_value=-90., max_value=90., value=31.6340, step=.01, format="%.4f")
            longitude = st.number_input("Longitude", min_value=-180., max_value=180., value=74.8723, step=.01, format="%.4f")
        else:
            city = chosen_location
            latitude, longitude = CITIES[city]
    st.session_state["active_data_mode"] = mode
    st.markdown('<div class="sidebar-note">◉ <b>Guardians of the Galaxy</b><br>Student-built forecasting prototype<br>Not an official AQI or medical service.</div>', unsafe_allow_html=True)

if is_demo and chosen_location == "Custom coordinates" and nav not in (ROUTES[5], ROUTES[6]):
    st.error("Custom coordinates require Live mode. Choose a preset city to view synthetic demo data.")
    st.stop()

if nav == ROUTES[5]:
    st.markdown('<div class="compact-hero"><div class="eyebrow">AIRFECTION V3.1 · EXPERIMENTAL VISION</div><h1>Sky Photo AI <span style="color:#fca5b9">✦</span></h1><p>Upload a sky photo for visual smoke, haze, fog and cloud matching.</p></div>', unsafe_allow_html=True)
    st.info("This is a visual-scene classification tool, not an AQI detector. Fog, dust, smog and clouds can look similar. Results never alter pollution forecasts.")
    left, right = st.columns([1.05,1], gap="large")
    with left:
        sky_file = st.file_uploader("Upload a photo of the sky", type=["jpg","jpeg","png"], key="sky_upload", help="JPG or PNG, under 8 MB. A clear outdoor photo works best.")
        sky_image = None
        if sky_file is not None:
            if sky_file.size > 8*1024*1024:
                st.error("Image is too large. Use a photo smaller than 8 MB.")
            else:
                try:
                    from PIL import ImageOps
                    sky_image = ImageOps.exif_transpose(Image.open(BytesIO(sky_file.getvalue()))).convert("RGB")
                    st.image(sky_image, caption="Your uploaded photo", use_container_width=True)
                except Exception as exc:
                    st.error(f"Could not open this image: {exc}")
    with right:
        custom_model_path = Path(__file__).parent / "sky_prototypes.npz"
        model_options = ["Pretrained CLIP · ready without training"]
        if custom_model_path.is_file():
            model_options.append("AIRFECTION trained prototypes · your labeled data")
        selected_vision = st.selectbox("Image model", model_options)
        from importlib.util import find_spec
        if find_spec("transformers") is None:
            st.warning("Sky AI needs optional model packages. Install them in the server environment (GitHub deployment requirements or your Python environment):")
            st.code("py -m pip install -r requirements-vision.txt", language="powershell")
        st.markdown("**What happens after upload?** ① Image is analyzed by CLIP ② Scene descriptions are ranked ③ You review the result.")
        analyze = st.button("✦ Analyze my sky", type="primary", disabled=sky_image is None, use_container_width=True)
        if analyze:
            try:
                from vision_optional import classify_sky, classify_trained
                with st.spinner("Loading pretrained CLIP and analyzing photo… (first run downloads model weights)"):
                    if selected_vision.startswith("AIRFECTION trained"):
                        scored = classify_trained(sky_image, str(custom_model_path))
                    else:
                        scored = classify_sky(sky_image)
                import hashlib
                image_hash = hashlib.sha256(sky_file.getvalue()).hexdigest()
                st.session_state["sky_analysis"] = (image_hash, selected_vision, scored)
            except Exception as exc:
                st.error(f"Sky AI could not run: {exc}")
                st.code("py -m pip install -r requirements-vision.txt", language="powershell")
                st.caption("Run that command in the AIRFECTION project directory, then restart the website. CLIP's weights also require internet on first launch.")
        previous = st.session_state.get("sky_analysis")
        if previous and sky_file and previous[0] == __import__("hashlib").sha256(sky_file.getvalue()).hexdigest() and previous[1] == selected_vision:
            top = previous[2][0]
            st.markdown(f'<div class="sky-result"><div class="tiny">TOP VISUAL MATCH</div><h2>{escape(top[0])}</h2><p>Relative comparison score {top[1]*100:.0f}% · not a probability of pollution</p></div>', unsafe_allow_html=True)
            chart_df = pd.DataFrame(previous[2],columns=["Description","Relative matching score"]).set_index("Description")
            st.bar_chart(chart_df, horizontal=True)
            from vision_optional import interpretation
            if "smoke" in top[0].lower():
                st.warning(interpretation(previous[2]))
            else:
                st.info(interpretation(previous[2]))
            st.caption("These are relative matches against a fixed list, not calibrated probabilities. Never interpret a non-smoke label as 'NOT POLLUTION'.")
        elif sky_image is None:
            st.markdown('<div class="sky-empty">☁️  Your sky analysis will appear here once you upload a photo.</div>', unsafe_allow_html=True)
    with st.expander("How do we train our own AIRFECTION sky model?",expanded=False):
        st.markdown("""
**You can use pretrained CLIP immediately — no training required.** To adapt it to photos from your region:

1. Collect photos with permission; target 100–200 images per category across many days, cameras, locations and weather conditions.
2. Label each photo into `clear`, `cloudy`, `fog`, or `haze`. If you cannot confidently tell fog from haze, exclude that image or mark it for review.
3. Save files under `sky_photos/industrial_smoke/`, `sky_photos/cloudy/`, `sky_photos/clear/`, `sky_photos/fog/`, and `sky_photos/haze/`. Include genuine smoke and tricky negative examples (dark clouds, chimneys without smoke).
4. Install vision requirements, then run `py train_sky.py --data sky_photos --output sky_prototypes.npz`.
5. Restart AIRFECTION and select **AIRFECTION trained prototypes**. This trains a lightweight classifier on CLIP embeddings, *not the whole CLIP network*.
6. Evaluate on pictures from **different dates/locations/cameras**, not duplicates or near-identical photos. Keep this example photo OUT of training if using it as a test. Report per-class precision/recall and a confusion matrix.

This version adds explicit industrial-smoke candidate descriptions. Changing prompts is **not training**. To validate actual pollution detection, you would additionally need photos synchronized to trustworthy PM2.5 sensors and weather visibility; visual labels alone are not enough.
""")
    st.caption("Model: [OpenAI CLIP ViT-B/32](https://huggingface.co/openai/clip-vit-base-patch32) via Hugging Face Transformers. Sky image data is processed by the running Streamlit server; no public community upload database is included.")
    st.stop()

if nav == ROUTES[6]:
    st.markdown('<div class="compact-hero"><div class="eyebrow">REAL DATA STUDIO</div><h1>Data Lab</h1><p>Add your own PM2.5 measurements or download station readings.</p></div>', unsafe_allow_html=True)
    label_section("Data engineering studio", "Bring your own real PM2.5 data", "Upload station measurements, add weather automatically, download a training-ready CSV, and train AIRFECTION on it.", right="YOUR DATA")
    st.info("Use ONE station per file. PM2.5 must be in µg/m³, not AQI. Timestamps must represent the local clock time at the selected coordinates. No data is uploaded to a shared database; it is held in your running Streamlit session.")
    first, second = st.tabs(["1 · Upload CSV & add weather", "2 · Download OpenAQ readings"])
    with first:
        st.markdown("**Required minimum CSV columns:** `time,pm2_5` (or `timestamp,pm25`). Weather can be filled using Open-Meteo.")
        st.code("time,pm2_5\n2025-10-18 14:00:00,76.5\n2025-10-18 15:00:00,83.1", language="csv")
        uploaded = st.file_uploader("Upload your hourly PM2.5 CSV", type="csv", key="pm_csv")
        if uploaded is not None:
            file_fingerprint = (uploaded.name, len(uploaded.getvalue()), hash(uploaded.getvalue()))
            if st.session_state.get("last_csv_fingerprint") != file_fingerprint:
                st.session_state.pop("combined_dataset", None)
                st.session_state["last_csv_fingerprint"] = file_fingerprint
            try:
                prepared = validate_pollution_csv(BytesIO(uploaded.getvalue()))
                st.success(f"Validated {len(prepared):,} rows, {pd.Timestamp(prepared.time.min()):%Y-%m-%d} to {pd.Timestamp(prepared.time.max()):%Y-%m-%d}.")
                st.dataframe(prepared.head(8), use_container_width=True, hide_index=True)
                has_weather = all(col in prepared.columns for col in REQUIRED)
                if not has_weather:
                    if st.button("Automatically add matching historical weather from Open-Meteo"):
                        with st.spinner("Downloading and matching hourly weather to your station timestamps…"):
                            st.session_state["combined_dataset"] = fetch_weather_for_history(prepared, latitude, longitude)
                else:
                    st.session_state["combined_dataset"] = full_training_csv(prepared)
            except Exception as exc:
                st.error(f"File validation failed: {exc}")
        if "combined_dataset" in st.session_state:
            combined = st.session_state["combined_dataset"]
            st.metric("Complete station + weather hours", f"{len(combined):,}")
            st.dataframe(combined.head(5), use_container_width=True, hide_index=True)
            st.download_button("↓ Download merged training-ready CSV", combined.to_csv(index=False),
                               file_name="airfection_station_with_weather.csv", mime="text/csv")
            if st.button("Use this dataset to train AIRFECTION"):
                st.session_state["user_dataset"] = combined.copy()
                st.success("Dataset loaded into this session. Choose 'CSV · Uploaded station data' in the sidebar. Future weather in that mode is an assumed-weather REPLAY, not a live forecast.")
    with second:
        st.caption("Optional developer source: OpenAQ station data. The maintainer can configure OPENAQ_API_KEY in Streamlit App settings → Secrets; website visitors never enter API keys.")
        try:
            openaq_key = st.secrets.get("OPENAQ_API_KEY", "")
        except Exception:
            openaq_key = ""
        if not openaq_key:
            st.info("OpenAQ integration is not configured. The Diwali Forecast works without any key using a separate public historical dataset.")
        radius = st.slider("Search station radius (km)", 1, 25, 20)
        if st.button("Find nearby PM2.5 stations", disabled=not bool(openaq_key)):
            try:
                st.session_state["openaq_sensors"] = find_pm25_sensors(openaq_key, latitude, longitude, radius * 1000)
            except Exception as exc:
                st.error(f"OpenAQ lookup failed: {exc}")
        sensors = st.session_state.get("openaq_sensors", [])
        if sensors:
            labels = [f"{r['station']} · sensor {r['sensor_id']}" for r in sensors]
            chosen_sensor_idx = st.selectbox("Select a PM2.5 sensor", list(range(len(labels))), format_func=lambda i: labels[i])
            station = sensors[chosen_sensor_idx]
            st.caption(f"Location: {station['lat']}, {station['lon']} · Units: {station['units']}")
            d1, d2 = st.columns(2)
            with d1: since = st.date_input("From", value=pd.Timestamp('2025-10-01').date())
            with d2: until = st.date_input("To", value=pd.Timestamp('2025-11-15').date())
            if st.button("Download hourly readings from this sensor"):
                try:
                    if since >= until: raise ValueError("'To' date must be after 'From' date.")
                    st.session_state["openaq_dataframe"] = download_pm25_hours(openaq_key, station['sensor_id'], since.isoformat(), until.isoformat())
                    st.session_state["openaq_station"] = station
                except Exception as exc:
                    st.error(f"OpenAQ download failed: {exc}")
        else:
            st.caption("No station list loaded. Some locations have no OpenAQ coverage; that does not prove no physical sensors exist nearby.")
        if "openaq_dataframe" in st.session_state:
            air_data = st.session_state["openaq_dataframe"]
            st.dataframe(air_data.head(8), use_container_width=True, hide_index=True)
            st.download_button("↓ Download station PM2.5 CSV", air_data.to_csv(index=False), file_name="airfection_openaq_pm25.csv", mime="text/csv")
            st.caption("To train: return to the first tab and upload this downloaded CSV. Use the coordinates of THIS station (not necessarily the city center) when adding weather.")
    with st.expander("Where to find more historical data"):
        st.markdown("**OpenAQ:** Find a PM2.5 station and download its hourly records by date window. Requires an API key. **CPCB:** Indian official air-quality monitoring portal; verify available export methods, station metadata and usage terms. **Open-Meteo:** Weather reanalysis for matching historical hours. **Your own station CSV:** Upload a file with timestamps and PM2.5 concentrations in µg/m³.")
    st.markdown("**Data ethics:** Never mix modeled CAMS values and station measurements without identifying the different sources. Do not interpolate across cities to claim street-level accuracy.")
    st.stop()

if nav == ROUTES[1]:
    from diwali_page import render_diwali_page
    render_diwali_page(city)
    st.stop()

with st.spinner("Preparing forecast and evaluating the model…"):
    try:
        if is_uploaded:
            if "user_dataset" not in st.session_state:
                st.error("You haven't uploaded a dataset yet. Open Data Lab, prepare a CSV, and click 'Use this dataset to train AIRFECTION'.")
                st.stop()
            else:
                history = st.session_state["user_dataset"].copy()
                # Historical replay with generated future weather, NOT current live weather.
                weather = fake_future_weather(history)
                local_zone = "local time from uploaded CSV"
        elif is_demo:
            local_zone = "Asia/Kolkata"
            history = load_demo(city)
            weather = fake_future_weather(history)
        else:
            older = load_live_history(latitude, longitude)
            recent, weather, local_zone = load_live_near(latitude, longitude)
            history = (pd.concat([older, recent], ignore_index=True)
                       .drop_duplicates("time", keep="last").sort_values("time").reset_index(drop=True))
        model, stats = fitted(history)
        predictions = forecast(history, weather, model)
        latest = history.iloc[-1]
    except Exception as exc:
        st.error(f"Could not load or train the forecast: {exc}")
        st.info("Try Demo mode. Demo numbers are artificial and are never displayed as live observations.")
        st.stop()

st.markdown('''
<div class="hero"><div style="position:relative;z-index:2"><div class="eyebrow">GUARDIANS OF THE GALAXY  ·  ENVIRONMENTAL INTELLIGENCE</div>
<h1>AIRFECTION<span style="color:#ffd3e2">.</span></h1>
<p class="sub">Predict the smog before the sparkle.<br>Explore pollution before, on and after Diwali — plus the next six hours.</p>
<div class="hero-pills"><span class="pill">✧ AI forecast</span><span class="pill">◎ Future pollution map</span><span class="pill">♡ Health awareness</span></div>
</div><div class="orb"></div></div>
''', unsafe_allow_html=True)

st.write("")
label_section("Interactive time travel", "Move forward into the forecast", "Select any hour ahead. All charts, highlights and future-map predictions follow this slider.", right="HOURLY · 1—6 HOURS")
slider_col, info_col = st.columns([4, 1.15], vertical_alignment="center")
with slider_col:
    selected_hour = st.slider("Forecast time", min_value=min(HORIZONS), max_value=max(HORIZONS), value=3,
                              step=1, format="+%d hours", label_visibility="collapsed")
    marks = st.columns(len(HORIZONS))
    for idx, h in enumerate(HORIZONS):
        marks[idx].caption(f"+{h}h")
with info_col:
    st.markdown(f'<div class="panel accent" style="text-align:center;padding:12px;margin:0"><div class="tiny">SELECTED HORIZON</div><div class="timeline-hour">+{selected_hour}h</div></div>', unsafe_allow_html=True)

picked = predictions.loc[predictions["horizon"] == selected_hour].iloc[0]
current_pm = float(latest["pm2_5"])
future_pm = float(picked["pm2_5"])
diff = future_pm - current_pm
peak = predictions.loc[predictions["pm2_5"].idxmax()]
minpoint = predictions.loc[predictions["pm2_5"].idxmin()]
mode_badge = 'UPLOADED SENSOR DATA · ASSUMED FUTURE WEATHER' if is_uploaded else ('DEMO · FICTIONAL DATA' if is_demo else 'LIVE · CAMS MODELED DATA')
source_kind = "uploaded sensor" if is_uploaded else ("demo-generated" if is_demo else "Open-Meteo CAMS modeled")
forecast_label = "Illustrative/replay forecast" if (is_demo or is_uploaded) else "Model-derived forecast"

label_section("Forecast command center", f"{city} at a glance", f"{mode_badge} · Model reference time: {pd.Timestamp(latest['time']):%d %b %Y, %I:%M %p} ({local_zone})")
metric1, metric2, metric3, metric4 = st.columns(4)
metric1.metric("Current PM2.5 · µg/m³", f"{current_pm:.1f}", help=f"Latest {source_kind} concentration, not the official AQI.")
metric2.metric(f"Forecast +{selected_hour}h · µg/m³", f"{future_pm:.1f}", f"{diff:+.1f} vs current", delta_color="inverse")
metric3.metric("Forecast peak · 6h", f"{float(peak['pm2_5']):.1f}", f"+{int(peak['horizon'])}h")
metric4.metric("Holdout prediction error", f"{stats['holdout_mae']:.1f}", "MAE · µg/m³", delta_color="off")

warning_source = ("✦ <b>Your uploaded observations:</b> Forecasts use assumed future weather derived from past readings. They are demonstration/replay estimates, NOT a live forecast." if is_uploaded else
    ("✦ <b>Demo preview:</b> Every pollution value, map point and model score is derived from synthetic data." if is_demo else
     "✦ <b>Modeled environmental data:</b> Live values are regional CAMS model outputs, not verified local sensors. Do not infer street-level pollution."))
st.markdown(f'<div class="warning-note">{warning_source} All displayed numbers are <b>PM2.5 concentrations (µg/m³), NOT official Indian AQI.</b></div>', unsafe_allow_html=True)

if nav == ROUTES[0]:
    label_section("The big picture", "Your next six hours", "See the forecast curve and immediately spot periods with rising or falling concentrations.", right="MODEL FORECAST")
    left, right = st.columns([1.85, 1], gap="medium")
    with left:
        st.plotly_chart(line_chart(latest, predictions, selected_hour), use_container_width=True, config={"displayModeBar": False})
    with right:
        trend = "rising" if diff > 3 else "falling" if diff < -3 else "relatively steady"
        panel("◉ Forecast insight", f"At +{selected_hour} hours, PM2.5 is projected to be {future_pm:.1f} µg/m³, {abs(diff):.1f} µg/m³ {'above' if diff>0 else 'below' if diff<0 else 'from'} the latest reference level. The projected trend is {trend}.", "accent")
        panel("✧ Lower forecast period", f"Of the six modeled future hours, +{int(minpoint['horizon'])}h has the lowest predicted PM2.5 ({float(minpoint['pm2_5']):.1f} µg/m³). This does not establish that outdoor exposure is safe.", "green")
    label_section("Explore more", "How AIRFECTION helps", "Navigate from the sidebar to explore each feature.")
    a, b, c = st.columns(3)
    with a: panel("🪔 Diwali Outlook", "Explore AIRFECTION’s own hour-by-hour 2026 Diwali forecast trained on historical festival data.")
    with b: panel("◎ Future Heatmap", "A regional surface displaying modeled future pollution. Darker red means higher concentration.")
    with c: panel("♡ Health Shield", "Source-linked advice for vulnerable groups based on published health guidance.")
    def _open_diwali_page():
        st.session_state["nav_route"] = ROUTES[1]
    st.button("🪔 Explore the Diwali 2026 AI forecast", use_container_width=True,
              on_click=_open_diwali_page)

elif nav == ROUTES[2]:
    label_section("Live forecast geography", "Future pollution heatmap",
                  "Move the forecast slider to watch model values change. Darker red means a higher value.",
                  right=f"+{selected_hour}H FORECAST")
    with st.spinner("Preparing the regional forecast cells…"):
        try:
            grid = map_rows(city, (latitude,longitude), is_demo,
                            pd.Timestamp(picked["time"]),selected_hour,future_pm)
            map_error = None
        except Exception as exc:
            grid = pd.DataFrame([{"lat":latitude,"lon":longitude,"pm2_5":future_pm,"us_aqi":np.nan,"source":"AIRFECTION ML: selected point only"}])
            map_error = str(exc)
    if map_error:
        st.warning(f"Regional forecast unavailable: {map_error}. Showing the selected location without a fabricated heatmap.")
    has_aqi = "us_aqi" in grid.columns and grid["us_aqi"].notna().sum() >= 3
    map_choice = st.radio("Choose map measurement", ["US AQI (CAMS model)", "PM2.5 concentration" ] if has_aqi else ["PM2.5 concentration"],
                          horizontal=True, help="US AQI is provider-derived, NOT the official Indian AQI. PM2.5 uses µg/m³.")
    metric = "us_aqi" if map_choice.startswith("US AQI") else "pm2_5"
    mcol,dcol = st.columns([3,1], gap="medium")
    with mcol:
        render_map(grid, city, (latitude,longitude),metric=metric)
        render_legend(metric)
    with dcol:
        panel("◉ AIRFECTION forecast", f"Selected location: {city}. ML PM2.5 prediction at +{selected_hour}h: {future_pm:.1f} µg/m³.", "accent")
        valid = grid.dropna(subset=[metric])
        if len(valid):
            st.metric("Region's highest map value", f"{float(valid[metric].max()):.0f}" + (" US AQI" if metric=="us_aqi" else " µg/m³"))
            st.metric("Region's lowest map value", f"{float(valid[metric].min()):.0f}" + (" US AQI" if metric=="us_aqi" else " µg/m³"))
            st.metric("Regional forecast cells", len(valid))
    if is_demo:
        st.warning("DEMO MAP: Every color and measurement on this surface is synthetic. Use Live mode for regional CAMS values.")
    elif is_uploaded_dataset():
        st.warning("Uploaded data contains just one station. A spatial heatmap needs several geographically separate measurements; we will not invent them.")
    else:
        st.info("The heatmap is interpolated from Open-Meteo CAMS GLOBAL modeled forecast cells (~45 km resolution), not verified neighborhood sensors. This map's data source differs from the AIRFECTION ML point prediction.")
    with st.expander("See the data behind each colored region"):
        st.dataframe(grid.rename(columns={"lat":"Latitude","lon":"Longitude","pm2_5":"PM2.5 µg/m³","us_aqi":"US AQI","source":"Source"}),
                     use_container_width=True,hide_index=True)
        st.caption("The interpolated surface changes when the selected forecast hour changes. Data values, not the number of map points, determine the colors.")

elif nav == ROUTES[3]:
    label_section("Early warning system", "Spike Radar & Recovery Clock", "Adjust the thresholds and see whether any forecast hour crosses them.")
    left, right = st.columns(2, gap="medium")
    with left:
        panel("↗ Spike Radar", "Choose how much PM2.5 must rise above the latest reference reading before the app flags a potential spike.", "accent")
        rise = st.slider("Alert on an increase of at least (µg/m³)", 5, 100, 25, 5)
        spikes = predictions[predictions["pm2_5"] - current_pm >= rise]
        if spikes.empty:
            st.success(f"No rise ≥ {rise} µg/m³ across the next six predicted hours.")
        else:
            first = spikes.iloc[0]
            st.warning(f"Potential spike at +{int(first['horizon'])}h: +{float(first['pm2_5'])-current_pm:.1f} µg/m³ vs latest reference. This is a forecast, not a live alert.")
    with right:
        panel("◷ Recovery Clock", "Set a concentration level and identify the earliest forecast time predicted to fall below it.", "green")
        recover_under = st.slider("Find the first forecast below (µg/m³)", 10, 300, 75, 5)
        recoveries = predictions[predictions["pm2_5"] < recover_under]
        if recoveries.empty:
            st.info(f"No predicted level below {recover_under} µg/m³ within six hours.")
        else:
            first = recoveries.iloc[0]
            st.success(f"Earliest predicted level below the comparison threshold: +{int(first['horizon'])}h ({pd.Timestamp(first['time']):%I:%M %p}).")
    st.plotly_chart(line_chart(latest, predictions, selected_hour), use_container_width=True, config={"displayModeBar": False})
    st.caption("The selected concentration thresholds are personal display preferences, not medically established safe levels. Alerts only appear inside this website. Forecast recovery is not guaranteed.")

elif nav == ROUTES[4]:
    label_section("WHO + peer-reviewed research", "Health Shield", "Source-linked, general exposure advice — not diagnosis or WHO certification of AIRFECTION.", right="EVIDENCE-BASED")
    group = st.selectbox("Who are you planning for?", list(__import__('health_evidence').GROUPS))
    headline, tips, references = guidance(group)
    panel("Selected forecast • PM2.5 only", f"At +{selected_hour} hours, AIRFECTION estimates {future_pm:.1f} µg/m³. This hourly value cannot establish an individual's medical risk or official AQI category.", "accent")
    panel("♡ " + headline, "Practical precautions supported by public-health literature", "green")
    for item in tips:
        st.markdown("- " + item)
    st.markdown("**Sources supporting this guidance**")
    for url in references:
        title, summary = reference_for(url)
        st.markdown(f"- [{title}]({url}) — {summary}")
    st.divider()
    label_section("Important distinction", "WHO's 24-hour guideline is not an hourly threshold")
    previous_24h = pm25_24h_mean(history)
    if previous_24h is not None:
        c1, c2 = st.columns(2)
        c1.metric("Mean of last 24 consecutive hours", f"{previous_24h:.1f} µg/m³")
        c2.metric("WHO PM2.5 24-hour guideline", "15 µg/m³")
        st.caption(f"These data are {source_kind}; this is a 24-hour mean, not the selected hourly forecast. It is not an official AQI or safety guarantee.")
    else:
        st.info("There are not 24 consecutive hourly records available to display a 24-hour mean.")
    st.markdown(f"Source: [WHO 2021 global air quality guidelines]({WHO_GUIDELINE}). WHO also lists an annual mean guideline of 5 µg/m³. These are population-level air-quality guidelines, not individual medical thresholds.")
    with st.expander("Research library: open the original WHO documents and studies"):
        for title, url, summary in SOURCES:
            st.markdown(f"**[{title}]({url})** — {summary}")
    st.warning("This is an educational exposure-reduction tool. AIRFECTION is not WHO-certified, and the WHO does not endorse its predictions. Seek prompt medical care for severe breathing difficulty, chest pain, or other emergency symptoms. Continue prescribed medical treatment.")

elif nav == ROUTES[7]:
    label_section("Transparency first", "Inside the prediction engine", "What AIRFECTION really uses, how it was tested, and where limitations remain.", right="REAL MODEL")
    c1, c2, c3 = st.columns(3)
    c1.metric("Chronological train rows", f"{stats['training_examples']:,}")
    c2.metric("Held-out test rows", f"{stats['test_examples']:,}")
    c3.metric("Persistence baseline MAE", f"{stats['baseline_mae']:.1f} µg/m³")
    if stats["holdout_mae"] >= stats["baseline_mae"]:
        st.warning("This model did not outperform the persistence baseline on the held-out period. Accuracy improvements are needed before making strong forecast claims.")
    else:
        st.success("The model beat the persistence baseline on this holdout period. This does not guarantee future or Diwali-specific accuracy.")
    imp = (pd.DataFrame({"Feature": stats["features"], "Importance": model.feature_importances_})
           .sort_values("Importance", ascending=False).head(9).iloc[::-1])
    fig = go.Figure(go.Bar(x=imp["Importance"], y=imp["Feature"], orientation="h", marker_color="#f98bab"))
    fig.update_xaxes(title_text="Feature importance (not cause and effect)")
    st.plotly_chart(plot_style(fig, 340), use_container_width=True, config={"displayModeBar": False})
    st.markdown("**Your actual forecast table**")
    table = predictions.rename(columns={"horizon": "Hours ahead", "time": "Local forecast time", "pm2_5": "Predicted PM2.5 (µg/m³)"})
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button("↓ Download forecast CSV", data=table.to_csv(index=False).encode("utf-8"), file_name="airfection_pm25_forecast.csv", mime="text/csv")
    with st.expander("Data sources and limitations", expanded=True):
        st.markdown("""
- **Training:** Time-aware Random Forest with 1–6 hour direct forecast horizons; PM2.5 lags, rolling average, weather and Diwali calendar features.
- **Evaluation:** Later observations held out chronologically; Mean Absolute Error (MAE) compared with persistence. A historical holdout does not prove performance on an unseen Diwali event; the Diwali Forecast page also distinguishes historical analogues from published CAMS forecasts.
- **Demo mode:** Fictional generated environmental data. Scores, map values and forecasts are not real measurements.
- **Live mode:** Uses Open-Meteo CAMS modeled PM2.5 and weather. It does **not** currently use CPCB or OpenAQ ground-sensor readings.
- **Geography:** A regional, value-based interpolated surface from multiple CAMS model cells where available. CAMS global is coarse (~45 km). Uploaded single-station data cannot support a surface.
- **Health:** General information only; no personalized diagnosis, medication advice or prevention guarantees.
- **Sky photo:** Pretrained CLIP performs *visual scene matching* and may be adapted with a labeled-photo prototype classifier. It cannot measure AQI.
""")
        st.markdown("Sources: [Open-Meteo Air Quality](https://open-meteo.com/en/docs/air-quality-api) · [Open-Meteo Historical Weather](https://open-meteo.com/en/docs/historical-weather-api) · [OpenAQ](https://docs.openaq.org/) · [CPCB](https://cpcb.nic.in/) · [WHO](https://www.who.int/health-topics/air-pollution)")



st.write("")
st.markdown('<div style="border-top:1px solid #293142;padding-top:16px;color:#8798ac;font-size:.78rem;display:flex;justify-content:space-between;gap:10px;flex-wrap:wrap"><span>✺ AIRFECTION · PREDICT THE SMOG BEFORE THE SPARKLE</span><span>Built by Guardians of the Galaxy · Student hackathon prototype</span></div>', unsafe_allow_html=True)
