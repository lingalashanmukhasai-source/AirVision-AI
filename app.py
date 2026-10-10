
# app.py — AirVision Professional Dashboard

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

MODEL_PATH = OUT_DIR / "airvision_model.joblib"
METRICS_PATH = OUT_DIR / "metrics.json"
HISTORY_PATH = DATA_DIR / "pm25_weather_history.csv"

st.set_page_config(
    page_title="AirVision | Air Intelligence",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Visual theme
# --------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --av-bg: #0a101b;
        --av-panel: #111b2b;
        --av-border: rgba(156, 178, 205, .16);
        --av-text: #edf4ff;
        --av-muted: #91a4bf;
        --av-cyan: #5eead4;
        --av-blue: #60a5fa;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 0%, rgba(38, 99, 150, .19), transparent 34%),
            radial-gradient(ellipse at 95% 5%, rgba(25, 164, 144, .10), transparent 28%),
            var(--av-bg);
        color: var(--av-text);
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -.035em;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0d1624;
        border-right: 1px solid var(--av-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--av-text);
    }

    .block-container {
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    .av-topline {
        color: var(--av-cyan);
        text-transform: uppercase;
        font-size: .75rem;
        font-weight: 700;
        letter-spacing: .16em;
        margin-bottom: .6rem;
    }

    .av-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.15rem, 4vw, 3.6rem);
        font-weight: 700;
        line-height: 1.05;
        letter-spacing: -.06em;
        color: #f5f9ff;
        margin: 0;
    }

    .av-gradient {
        background: linear-gradient(90deg, #5eead4, #60a5fa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .av-subtitle {
        color: #9fb0c8;
        font-size: 1rem;
        max-width: 760px;
        line-height: 1.7;
        margin-top: .9rem;
    }

    .av-panel {
        background: linear-gradient(145deg, rgba(22, 34, 52, .96), rgba(13, 23, 37, .96));
        border: 1px solid var(--av-border);
        border-radius: 18px;
        padding: 20px 22px;
        min-height: 135px;
        box-shadow: 0 12px 34px rgba(0,0,0,.10);
    }

    .av-panel-label {
        color: #9aacc4;
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 600;
    }

    .av-panel-value {
        color: #f2f7ff;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.3;
        margin-top: .35rem;
    }

    .av-panel-note {
        color: #91a4bf;
        font-size: .82rem;
        margin-top: .3rem;
    }

    .av-section {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #f2f7ff;
        margin-top: .8rem;
        margin-bottom: .25rem;
    }

    .av-muted {
        color: #91a4bf;
        font-size: .88rem;
        line-height: 1.6;
    }

    .av-chip {
        display: inline-block;
        border: 1px solid rgba(94,234,212,.25);
        background: rgba(94,234,212,.08);
        color: #7ce9d8;
        padding: 5px 10px;
        border-radius: 999px;
        font-size: .75rem;
        font-weight: 600;
        margin-right: 5px;
    }

    div[data-testid="stButton"] > button[kind="primary"] {
        background: linear-gradient(100deg, #0d9488, #2563eb);
        color: white;
        border: 0;
        border-radius: 10px;
        min-height: 44px;
        font-weight: 700;
        transition: all .2s ease;
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover {
        border: 1px solid #5eead4;
        filter: brightness(1.12);
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background: #111b2b;
        border-radius: 10px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 600;
    }

    hr {
        border-color: rgba(156,178,205,.15);
    }

    footer {visibility: hidden;}
    #MainMenu {visibility: hidden;}

    .av-footer {
        color: #70839d;
        font-size: .78rem;
        text-align: center;
        padding-top: 1.5rem;
        line-height: 1.7;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Data services
# --------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def geocode(place):
    response = requests.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={
            "name": place,
            "count": 8,
            "language": "en",
            "format": "json",
        },
        timeout=25,
    )
    response.raise_for_status()
    return response.json().get("results", []) or []


@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(lat, lon):
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "current": (
                "temperature_2m,relative_humidity_2m,"
                "precipitation,wind_speed_10m"
            ),
            "timezone": "UTC",
        },
        timeout=25,
    )
    response.raise_for_status()
    return response.json()


def load_metrics():
    if not METRICS_PATH.exists():
        return {}
    try:
        return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def load_history():
    if not HISTORY_PATH.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(HISTORY_PATH)
    except (ValueError, OSError):
        return pd.DataFrame()


def make_features(lat, lon, weather, feature_names):
    current = weather.get("current", {})
    now = datetime.now(timezone.utc)

    values = {
        "latitude": float(lat),
        "longitude": float(lon),
        "temperature": current.get("temperature_2m"),
        "humidity": current.get("relative_humidity_2m"),
        "precipitation": current.get("precipitation"),
        "wind_speed": current.get("wind_speed_10m"),
        "month": now.month,
        "day_of_year": now.timetuple().tm_yday,
        "day_of_week": now.weekday(),
    }

    row = pd.DataFrame([
        {name: values.get(name) for name in feature_names}
    ])
    row = row.apply(pd.to_numeric, errors="coerce")

    if row.isna().any().any():
        missing = row.columns[row.isna().any()].tolist()
        raise ValueError(
            "Weather or feature data is missing: " + ", ".join(missing)
        )

    return row


def haversine_km(lat1, lon1, lat2, lon2):
    earth_radius = 6371.0
    lat1, lon1, lat2, lon2 = map(
        np.radians, [lat1, lon1, lat2, lon2]
    )
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    return 2 * earth_radius * np.arcsin(np.sqrt(a))


def nearby_stations(history, lat, lon):
    required = {"latitude", "longitude", "location_name"}
    if history.empty or not required.issubset(history.columns):
        return pd.DataFrame()

    df = history.dropna(subset=["latitude", "longitude"]).copy()
    if df.empty:
        return df

    df["distance_km"] = haversine_km(
        lat,
        lon,
        df["latitude"].astype(float).to_numpy(),
        df["longitude"].astype(float).to_numpy(),
    )

    dedupe_col = (
        "location_id" if "location_id" in df.columns else "location_name"
    )
    df = df.sort_values("distance_km").drop_duplicates(dedupe_col)
    return df.head(10)


def metric_card(label, value, note, accent="cyan"):
    accent_color = "#5eead4" if accent == "cyan" else "#60a5fa"
    st.markdown(
        f"""
        <div class="av-panel">
            <div class="av-panel-label">{label}</div>
            <div class="av-panel-value" style="color:{accent_color}">
                {value}
            </div>
            <div class="av-panel-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# Sidebar
# --------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="font-family:Space Grotesk;font-size:1.5rem;font-weight:700;">
            🌫️ Air<span style="color:#5eead4">Vision</span>
        </div>
        <div class="av-muted" style="margin-top:5px;">
            Environmental intelligence platform
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown("### 📍 Explore a location")
    with st.form("location_search", clear_on_submit=False):
        place = st.text_input(
            "City, town or place",
            value="Vijayawada, Andhra Pradesh, India",
            placeholder="e.g. Guntur, Andhra Pradesh",
        )
        submitted = st.form_submit_button(
            "Analyze location →",
            type="primary",
            use_container_width=True,
        )

    st.divider()
    st.markdown("### About this prototype")
    st.caption(
        "AirVision estimates ground-level PM2.5 at a selected location using "
        "historical station data and weather features."
    )
    st.caption("Data services: OpenAQ + Open-Meteo")
    st.caption("Units shown for predictions: µg/m³, assuming source data uses these units.")

# --------------------------------------------------
# Hero
# --------------------------------------------------
st.markdown(
    """
    <div class="av-topline">AIR QUALITY · SPATIAL INTELLIGENCE</div>
    <div class="av-title">
        See the air.<br><span class="av-gradient">Beyond the sensors.</span>
    </div>
    <div class="av-subtitle">
        Explore estimated PM2.5 at locations where monitoring stations may not
        exist. AirVision combines historical observations and local weather to
        create a model-based estimate—not a direct sensor measurement.
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)

metrics = load_metrics()
history = load_history()

# --------------------------------------------------
# Initial state / model status
# --------------------------------------------------
if not submitted:
    trained = metrics.get("trained", False) and MODEL_PATH.exists()
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card(
            "Model status",
            "READY" if trained else "NEEDS DATA",
            "Trained model found" if trained else "Run the pipeline to train",
        )
    with col2:
        metric_card(
            "Historical records",
            f"{len(history):,}" if not history.empty else "0",
            "Rows in local history file",
            "blue",
        )
    with col3:
        metric_card(
            "Monitoring locations",
            (
                f"{history['location_id'].nunique():,}"
                if not history.empty and "location_id" in history.columns
                else "—"
            ),
            "Distinct locations in the dataset",
        )

    st.markdown('<div class="av-section">Your workspace</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="av-muted">Enter a location in the left sidebar and click '
        '<b>Analyze location</b> to view its weather, estimated PM2.5 and '
        'available station context.</div>',
        unsafe_allow_html=True,
    )

    if not metrics:
        st.info(
            "No evaluation file was found yet. Run `python run_pipeline.py` "
            "in your project terminal."
        )

# --------------------------------------------------
# Location analysis
# --------------------------------------------------
if submitted:
    if not place.strip():
        st.error("Enter a location name first.")
        st.stop()

    try:
        with st.spinner("Finding your location..."):
            candidates = geocode(place.strip())

        if not candidates:
            st.error(
                "No matching place was found. Try adding the state and country."
            )
            st.stop()

        labels = []
        for candidate in candidates:
            parts = [
                candidate.get("name"),
                candidate.get("admin1"),
                candidate.get("country"),
            ]
            labels.append(", ".join(str(p) for p in parts if p))

        selected_label = st.selectbox(
            "Choose the correct location",
            labels,
            index=0,
        )
        selected_index = labels.index(selected_label)
        location = candidates[selected_index]

        lat = float(location["latitude"])
        lon = float(location["longitude"])
        current_name = labels[selected_index]

        with st.spinner("Loading local weather..."):
            weather = get_weather(lat, lon)

        current = weather.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        precipitation = current.get("precipitation")
        wind = current.get("wind_speed_10m")

        st.markdown("---")
        left, right = st.columns([2.5, 1])

        with left:
            st.markdown(
                f'<div class="av-section">📍 {current_name}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="av-muted">Coordinates: {lat:.5f}, {lon:.5f}</div>',
                unsafe_allow_html=True,
            )
        with right:
            st.markdown(
                '<span class="av-chip">LOCATION RESOLVED</span>',
                unsafe_allow_html=True,
            )

        w1, w2, w3, w4 = st.columns(4)
        with w1:
            metric_card(
                "Temperature",
                f"{temp}°C" if temp is not None else "—",
                "Current weather",
            )
        with w2:
            metric_card(
                "Relative humidity",
                f"{humidity}%" if humidity is not None else "—",
                "Current weather",
                "blue",
            )
        with w3:
            metric_card(
                "Precipitation",
                f"{precipitation} mm" if precipitation is not None else "—",
                "Current weather",
            )
        with w4:
            metric_card(
                "Wind speed",
                f"{wind} km/h" if wind is not None else "—",
                "Current weather",
                "blue",
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # Prediction
        if not MODEL_PATH.exists():
            st.error(
                "The trained model file is missing. Run "
                "`python run_pipeline.py` first. If the pipeline reports "
                "insufficient data, more valid historical observations are needed."
            )
            st.stop()

        bundle = joblib.load(MODEL_PATH)
        model = bundle["model"]
        feature_names = bundle["features"]
        features = make_features(lat, lon, weather, feature_names)
        prediction = float(model.predict(features)[0])

        st.markdown('<div class="av-section">Estimated PM2.5</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="av-muted">A model-based estimate for the selected '
            'coordinates. This is not a direct reading from a sensor.</div>',
            unsafe_allow_html=True,
        )

        pred_col, info_col = st.columns([1, 2])

        with pred_col:
            st.markdown(
                f"""
                <div class="av-panel" style="min-height:220px;">
                    <div class="av-panel-label">MODEL ESTIMATE</div>
                    <div class="av-panel-value" style="font-size:3.2rem;color:#5eead4;">
                        {prediction:.1f}
                    </div>
                    <div class="av-panel-note">µg/m³ · estimated PM2.5</div>
                    <hr style="border-color:rgba(156,178,205,.15);">
                    <div class="av-muted">Prediction generated from the loaded model.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with info_col:
            st.markdown(
                """
                <div class="av-panel" style="min-height:220px;">
                    <div class="av-panel-label">HOW TO INTERPRET THIS</div>
                    <div style="font-size:1rem;font-weight:600;margin-top:10px;color:#f2f7ff;">
                        An estimate, not a measurement
                    </div>
                    <p class="av-muted">
                        Prediction reliability depends on the training data,
                        weather-feature availability, station coverage and how
                        similar this location is to the locations used for training.
                    </p>
                    <p class="av-muted">
                        Do not use this prototype as an official AQI or health alert.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Map
        st.markdown('<div class="av-section">Location map</div>', unsafe_allow_html=True)
        map_data = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
        st.map(
            map_data,
            latitude="latitude",
            longitude="longitude",
            size=100,
            color="#5eead4",
            zoom=9,
        )

        # Nearest known station context
        stations = nearby_stations(history, lat, lon)
        if not stations.empty:
            st.markdown('<div class="av-section">Nearest known monitoring locations</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="av-muted">These stations provide context only. '
                'Their readings are not substituted for the model prediction.</div>',
                unsafe_allow_html=True,
            )
            station_cols = [
                c for c in [
                    "location_name", "latitude", "longitude", "distance_km",
                    "date", "pm25", "units"
                ] if c in stations.columns
            ]
            st.dataframe(
                stations[station_cols].rename(columns={
                    "location_name": "Station",
                    "latitude": "Latitude",
                    "longitude": "Longitude",
                    "distance_km": "Distance (km)",
                    "date": "Date",
                    "pm25": "PM2.5",
                    "units": "Units",
                }),
                use_container_width=True,
                hide_index=True,
            )

        # History and evaluation
        tab1, tab2, tab3 = st.tabs([
            "📈 Training data",
            "🧪 Model evaluation",
            "ℹ️ Methodology",
        ])

        with tab1:
            if history.empty:
                st.info("No historical training file was found.")
            else:
                st.markdown("#### Available historical observations")
                if "date" in history.columns and "pm25" in history.columns:
                    chart_df = history.copy()
                    chart_df["date"] = pd.to_datetime(
                        chart_df["date"], errors="coerce"
                    )
                    chart_df = chart_df.dropna(subset=["date", "pm25"])
                    if not chart_df.empty:
                        fig = px.scatter(
                            chart_df,
                            x="date",
                            y="pm25",
                            color=(
                                "location_name"
                                if "location_name" in chart_df.columns
                                else None
                            ),
                            hover_data=[
                                c for c in [
                                    "location_name", "location_id", "sensor_id"
                                ] if c in chart_df.columns
                            ],
                            title="PM2.5 observations by date",
                            template="plotly_dark",
                        )
                        fig.update_layout(
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            legend_title_text="Monitoring location",
                        )
                        st.plotly_chart(fig, use_container_width=True)
                st.dataframe(history.head(500), use_container_width=True)

        with tab2:
            current_metrics = load_metrics()
            if current_metrics.get("trained"):
                a, b, c = st.columns(3)
                a.metric("Held-out location RMSE", f"{current_metrics.get('rmse', float('nan')):.2f}")
                b.metric("Held-out location MAE", f"{current_metrics.get('mae', float('nan')):.2f}")
                r2 = current_metrics.get("r2")
                c.metric("R²", f"{r2:.3f}" if r2 is not None else "—")
                st.caption(current_metrics.get("warning", ""))
            else:
                st.warning(
                    current_metrics.get(
                        "message",
                        "No valid evaluation metrics are available. Train the model first.",
                    )
                )

        with tab3:
            st.markdown("#### Model inputs")
            st.write(
                "Latitude, longitude, current weather and calendar variables "
                "that are present in the trained model."
            )
            st.markdown("#### Data sources")
            st.write(
                "- OpenAQ: historical ground-level monitoring observations.\n"
                "- Open-Meteo: weather data for the selected coordinates."
            )
            st.markdown("#### Important limitation")
            st.write(
                "This baseline combines current weather with a model trained on "
                "historical observations. That temporal mismatch can reduce "
                "accuracy. Predictions outside the training region are not "
                "validated automatically."
            )

    except requests.RequestException as error:
        st.error(f"Could not connect to a data service: {error}")
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        st.error(f"AirVision could not complete the analysis: {error}")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.markdown(
    """
    <div class="av-footer">
        AIRVISION · ENVIRONMENTAL INTELLIGENCE PROTOTYPE<br>
        Estimates are not official monitoring readings, AQI determinations,
        or medical advice.
    </div>
    """,
    unsafe_allow_html=True,
)

