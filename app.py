import streamlit as st

import requests

import pandas as pd

import plotly.graph_objects as go

from datetime import datetime

from pathlib import Path

import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX



st.set_page_config(page_title="AirGuard AI | Environmental Intelligence", page_icon="🍃", layout="wide", initial_sidebar_state="expanded")



AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

CITIES = {

    "Jaipur": (26.9124, 75.7873), "Delhi": (28.6139, 77.2090), "Mumbai": (19.0760, 72.8777),

    "Bengaluru": (12.9716, 77.5946), "Kolkata": (22.5726, 88.3639), "Chennai": (13.0827, 80.2707),

    "Hyderabad": (17.3850, 78.4867), "Pune": (18.5204, 73.8567), "Lucknow": (26.8467, 80.9462),

    "Ahmedabad": (23.0225, 72.5714),

}



st.markdown("""

<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root { --navy:#102a43; --teal:#0d9488; --muted:#718198; --line:#e3eaf2; }

html, body, [class*="css"] { font-family:Inter, sans-serif; }

.stApp { background:#f3f7fb; color:#17324d; }

header[data-testid="stHeader"] { background:rgba(243,247,251,.98); height:3.6rem; min-height:3.6rem; z-index:999999; }

header[data-testid="stHeader"] button { visibility:visible !important; display:flex !important; opacity:1 !important; color:#102a43 !important; background:#ffffff !important; border:1px solid #d8e5ef !important; border-radius:10px !important; margin:7px 0 0 8px !important; box-shadow:0 2px 8px #102a4310 !important; }

[data-testid="stSidebar"] { min-width:285px; width:285px; }

[data-testid="stSidebar"] > div:first-child { padding-top:1.1rem; }

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 { color:#f4fbff; font-size:16px; margin-top:8px; }

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:#c1d5e5; }

.sidebar-note {background:#173d58;border:1px solid #2c5a75;border-radius:12px;padding:12px;margin:9px 0;color:#d9eaf5;font-size:12px;}

.sidebar-label {font-size:10px;font-weight:800;letter-spacing:1.3px;color:#82d9d0;text-transform:uppercase;margin:4px 0 8px;}

[data-testid="stSidebar"] button[kind="header"] { color:#ffffff !important; }

[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] { display:block !important; visibility:visible !important; }

[data-testid="stSidebarCollapsedControl"] { display:block !important; visibility:visible !important; opacity:1 !important; }

[data-testid="stSidebarCollapsedControl"] button { background:#ffffff !important; color:#102a43 !important; border:1px solid #d8e5ef !important; border-radius:10px !important; box-shadow:0 3px 12px #102a4320 !important; }

.block-container { max-width:1580px; padding-top:1.35rem !important; padding-bottom:2.5rem; }

[data-testid="stSidebar"] { background:#102a43; }

[data-testid="stSidebar"] * { color:#f0f7ff; }

[data-testid="stSidebar"] [data-baseweb="select"] * { color:#17324d; }

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color:#c9d9e8; }

.brandline {display:flex;align-items:center;gap:12px;min-height:66px;padding:3px 0 9px;}

.brandname {font-size:29px;font-weight:800;letter-spacing:-1.1px;color:#102a43;line-height:1.2;}

.brandtag {font-size:11px;color:#718198;margin-top:5px;letter-spacing:.3px;}

.hero {background:radial-gradient(circle at 88% 5%,#21b6a8 0%,transparent 27%),linear-gradient(115deg,#0b2338 0%,#124a66 55%,#087f78 100%);padding:30px 32px;border-radius:20px;color:white;margin:15px 0 20px;box-shadow:0 12px 34px #102a4322;border:1px solid #ffffff30;}

.hero .eyebrow {font-size:11px;font-weight:800;letter-spacing:1.7px;color:#a7e7dc;}

.hero h2 {color:white;font-size:27px;line-height:1.25;margin:10px 0 8px;}

.hero p {color:#d8edf0;font-size:13px;margin:0;}

.section-title {font-size:20px;font-weight:800;color:#17324d;margin:22px 0 4px;}

.section-subtitle {font-size:12px;color:#718198;margin-bottom:14px;}

.metric-card,.panel {background:linear-gradient(145deg,#ffffff,#fbfdff);border:1px solid #dce8f4;border-radius:16px;padding:18px;box-shadow:0 6px 22px #102a430a;}

.metric-card {min-height:125px;border-top:4px solid var(--teal);}

.metric-label {color:#718198;font-size:10px;font-weight:800;letter-spacing:.8px;text-transform:uppercase;}

.metric-value {font-size:27px;line-height:1.25;font-weight:800;color:#17324d;margin-top:12px;overflow-wrap:anywhere;}

.metric-note {font-size:11px;color:#718198;margin-top:7px;}

.notice {background:#fff8e6;border:1px solid #f3dfa4;border-radius:12px;padding:12px 15px;color:#805b0b;font-size:12px;margin:12px 0;}

.pill {display:inline-block;padding:7px 12px;border-radius:30px;font-size:11px;font-weight:800;}

.mode-caption{font-size:10px;font-weight:800;letter-spacing:1.2px;color:#718198;text-transform:uppercase;margin:4px 0 6px;}

[data-testid="stSegmentedControl"] {background:#e8f0f8;padding:5px;border-radius:16px;border:1px solid #d8e5f1;}

[data-testid="stSegmentedControl"] button {font-weight:800 !important;border-radius:12px !important;min-height:42px !important;}

[data-testid="stSegmentedControl"] button[aria-pressed="true"] {background:linear-gradient(120deg,#0d9488,#087f70) !important;color:#ffffff !important;box-shadow:0 4px 12px #0d948833 !important;}

.stButton button,.stDownloadButton button {border-radius:11px;font-weight:700;border:1px solid #cbdce9;}

.stButton button:hover,.stDownloadButton button:hover {border-color:#0d9488;color:#087f70;}

.stTabs [data-baseweb="tab"] {font-weight:700;}

[data-testid="stPlotlyChart"] {border:1px solid #dbe8f4;border-radius:16px;padding:7px;background:#fff;box-shadow:0 7px 22px #102a430a;}

.hour-card {background:linear-gradient(160deg,#ffffff,#f1f8ff);border:1px solid #dce9f5;border-radius:15px;padding:13px 8px;text-align:center;min-height:145px;box-shadow:0 5px 16px #102a4308;}

.hour-time {font-size:11px;color:#617890;font-weight:800;}

.hour-icon {font-size:26px;margin:9px 0;}

.hour-temp {font-size:20px;font-weight:800;color:#102a43;}

.hour-sub {font-size:10px;color:#718198;margin-top:5px;line-height:1.4;}

.weather-highlight {background:linear-gradient(120deg,#102a43,#145b77 60%,#0d9488);border-radius:18px;padding:22px 25px;color:white;box-shadow:0 12px 28px #102a4320;}

.weather-highlight h2 {color:white!important;font-size:28px;margin:4px 0 8px;}

.weather-highlight p {color:#d8edf0;margin:0;font-size:13px;}

.chart-heading {font-size:17px;font-weight:800;color:#17324d;margin:14px 0 4px;}

.stRadio [role="radiogroup"] {gap:8px;}

.stRadio [role="radio"] {background:#eef5fb;border:1px solid #d8e5f1;border-radius:10px;padding:8px 13px;}

.stRadio [role="radio"]:has(input:checked) {background:#d9f7f1;border-color:#0d9488;}

.stTabs [data-baseweb="tab-list"] {gap:7px;}

.stTabs [data-baseweb="tab"] {background:#e6edf5;color:#29465f;border-radius:9px;padding:9px 14px;}

.stTabs [aria-selected="true"] {background:white !important;color:#087f70 !important;}

div[data-testid="stMetric"] {background:white;border:1px solid var(--line);padding:14px;border-radius:13px;}

footer,#MainMenu {visibility:hidden;}

@media(max-width:768px){.brandname{font-size:24px}.hero{padding:20px}.hero h2{font-size:22px}.block-container{padding-top:.5rem !important}}

</style>

""", unsafe_allow_html=True)



LOGO = '<svg width="48" height="48" viewBox="0 0 48 48" fill="none" xmlns="http://www.w3.org/2000/svg"><rect width="48" height="48" rx="15" fill="#0D9488"/><path d="M34 12C22 12 13 18 13 28c0 5 4 8 9 8 10 0 13-12 12-24Z" fill="white"/><path d="M10 37c5-9 11-14 20-19" stroke="#B7F3E8" stroke-width="2.6" stroke-linecap="round"/><path d="M9 17c3-2 5-2 8-1M8 23c2-1 4-1 6 0" stroke="#B7F3E8" stroke-width="2.2" stroke-linecap="round"/></svg>'



def aqi_details(value):

    if value is None or pd.isna(value): return "Unavailable", "#718198", "AQI data is currently unavailable."

    value = float(value)

    if value <= 50: return "Good", "#16a34a", "Air quality is satisfactory. Enjoy outdoor activities as usual."

    if value <= 100: return "Moderate", "#ca8a04", "Air quality is acceptable. Unusually sensitive people may need to take care."

    if value <= 150: return "Unhealthy for Sensitive Groups", "#f97316", "Sensitive groups should consider reducing prolonged outdoor exertion."

    if value <= 200: return "Unhealthy", "#dc2626", "Consider reducing prolonged outdoor exertion."

    if value <= 300: return "Very Unhealthy", "#9333ea", "Reduce outdoor exposure where possible."

    return "Hazardous", "#7f1d1d", "Avoid prolonged outdoor exposure where possible and check official advisories."



def fmt(v, suffix=""):

    try:

        if v is None or pd.isna(v): return "—"

        return f"{float(v):.1f}{suffix}"

    except (TypeError, ValueError): return "—"



def metric_card(label, value, note, accent="#0d9488"):

    st.markdown(f'<div class="metric-card" style="border-top-color:{accent}"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>', unsafe_allow_html=True)

<<<<<<< HEAD
@st.cache_data(ttl=600, show_spinner=False)
def fetch_air_quality(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "us_aqi,pm2_5,pm10",
        "hourly": "us_aqi,pm2_5,pm10",
        "forecast_days": 2,
        "timezone": "auto",
    }
    try:
        response = requests.get(AQ_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or "current" not in data or "hourly" not in data:
            raise RuntimeError("The air-quality service returned incomplete data. Please refresh shortly.")
        return data
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 429:
            raise RuntimeError("The air-quality service is receiving too many requests. Wait a few minutes before refreshing.") from exc
        raise RuntimeError(f"The air-quality service returned HTTP {status}. Please try again later.") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The air-quality service timed out. Please try again shortly.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Could not connect to the air-quality service. Please try again shortly.") from exc
    except ValueError as exc:
        raise RuntimeError("The air-quality service returned invalid data. Please try again later.") from exc


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_weather(lat, lon):
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m,weather_code,apparent_temperature",
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset,uv_index_max,wind_speed_10m_max",
        "forecast_days": 3,
        "timezone": "auto",
    }
    try:
        response = requests.get(WEATHER_URL, params=params, timeout=20)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or "current" not in data:
            raise RuntimeError("The weather service returned incomplete data. Please refresh shortly.")
        return data
    except requests.exceptions.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else "unknown"
        if status == 429:
            raise RuntimeError("The weather service is temporarily rate-limiting requests. Wait a few minutes before refreshing.") from exc
        raise RuntimeError(f"The weather service returned HTTP {status}. Please try again later.") from exc
    except requests.exceptions.Timeout as exc:
        raise RuntimeError("The weather service timed out. Please try again shortly.") from exc
    except requests.exceptions.RequestException as exc:
        raise RuntimeError("Could not connect to the weather service. Please try again shortly.") from exc
    except ValueError as exc:
        raise RuntimeError("The weather service returned invalid data. Please try again later.") from exc
=======


@st.cache_data(ttl=600, show_spinner=False)

def fetch_air_quality(lat, lon):

    r = requests.get(
        AQ_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "current": "us_aqi,pm2_5,pm10",
            "hourly": "us_aqi,pm2_5,pm10",
            # Include recent hourly values so SARIMA trains on the selected city's
            # recent history, not on the API's future forecast values.
            "past_days": 7,
            "forecast_days": 2,
            "timezone": "auto",
        },
        timeout=25,
    )

    r.raise_for_status(); data = r.json()

    if "current" not in data or "hourly" not in data: raise ValueError("The air-quality service returned incomplete data.")

    return data
>>>>>>> 02d60ac (Add SARIMA forecasting to AirGuard AI)



@st.cache_data(ttl=600, show_spinner=False)

def fetch_weather(lat, lon):

    r = requests.get(WEATHER_URL, params={"latitude":lat,"longitude":lon,"current":"temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m","hourly":"temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,wind_speed_10m,weather_code,apparent_temperature","daily":"weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,sunrise,sunset,uv_index_max,wind_speed_10m_max","forecast_days":3,"timezone":"auto"}, timeout=25)

    r.raise_for_status(); data = r.json()

    if "current" not in data: raise ValueError("The weather service returned incomplete data.")

    return data



def make_hourly_frame(hourly, columns):

    frame = pd.DataFrame({"time":pd.to_datetime(hourly.get("time", []), errors="coerce")})

    for label, key in columns.items(): frame[label] = pd.to_numeric(hourly.get(key, []), errors="coerce")

    return frame.dropna(subset=["time"]).sort_values("time")



@st.cache_resource(show_spinner=False)

def load_spike_artifact():

    """Load the trained PM2.5 spike-classification model once per app session."""

    model_path = Path(__file__).resolve().parent / "outputs" / "spike_model.joblib"

    if not model_path.exists():

        raise FileNotFoundError(f"Model file not found: {model_path}")

    artifact = joblib.load(model_path)

    if not isinstance(artifact, dict) or "model" not in artifact or "feature_columns" not in artifact:

        raise ValueError("The saved spike model does not contain the expected model and feature_columns entries.")

    return artifact



def build_spike_features(hourly, current_time_text=None):

    """Build the same 15 past-only features used when training the spike model."""

    frame = pd.DataFrame({

        "time": pd.to_datetime(hourly.get("time", []), errors="coerce"),

        "pm25": pd.to_numeric(hourly.get("pm2_5", []), errors="coerce"),

    })

    frame = frame.dropna(subset=["time"]).sort_values("time").drop_duplicates("time", keep="last")

    if frame.empty:

        return None, "The API did not return hourly PM2.5 values."



    # Open-Meteo may return timezone-aware timestamps; use the timestamp's own

    # local-hour sequence and require 25 consecutive hourly values for features.

    frame = frame.set_index("time")

    latest_time = frame.index.max().floor("h")

    required_times = pd.date_range(end=latest_time, periods=25, freq="h")

    recent = frame.reindex(required_times)

    values = recent["pm25"]

    if values.isna().any():

        missing = int(values.isna().sum())

        return None, f"Not enough complete recent hourly PM2.5 data to calculate the model features ({missing} missing hour(s)). Refresh later and try again."



    # The model was trained with rolling statistics over prior values only.

    past = values.shift(1)

    features = {

        "pm25_lag_1h": values.iloc[-2],

        "pm25_lag_2h": values.iloc[-3],

        "pm25_lag_3h": values.iloc[-4],

        "pm25_lag_6h": values.iloc[-7],

        "pm25_lag_12h": values.iloc[-13],

        "pm25_lag_24h": values.iloc[-25],

        "pm25_mean_6h": past.iloc[-6:].mean(),

        "pm25_std_6h": past.iloc[-6:].std(),

        "pm25_mean_12h": past.iloc[-12:].mean(),

        "pm25_std_12h": past.iloc[-12:].std(),

        "pm25_mean_24h": past.iloc[-24:].mean(),

        "pm25_std_24h": past.iloc[-24:].std(),

        "hour": latest_time.hour,

        "day_of_week": latest_time.dayofweek,

        "month": latest_time.month,

    }

    return pd.DataFrame([features]), None



@st.cache_data(ttl=1800, show_spinner=False)
def calculate_sarima_forecast(time_values, pm25_values, current_time_text, horizon=24):
    """Fit a SARIMA model on recent past PM2.5 values and forecast the next hours.

    Input values are Open-Meteo model-estimated hourly PM2.5, not official
    ground-station observations. Only timestamps strictly before current_time_text
    are used to fit the model; API forecast values are deliberately excluded.
    """
    frame = pd.DataFrame({
        "time": pd.to_datetime(list(time_values), errors="coerce"),
        "pm25": pd.to_numeric(list(pm25_values), errors="coerce"),
    })
    frame = frame.dropna(subset=["time"]).sort_values("time")
    frame = frame.drop_duplicates(subset=["time"], keep="last")
    current_time = pd.to_datetime(current_time_text, errors="coerce")
    if pd.isna(current_time):
        raise ValueError("The API did not provide a valid current timestamp for SARIMA.")

    # Train only on the preceding seven days; do not leak future API forecast data.
    history = frame.loc[frame["time"] < current_time].tail(24 * 7)
    if len(history) < 48:
        raise ValueError("SARIMA needs at least 48 past hourly PM2.5 values. Refresh the data and try again.")

    series = history.set_index("time")["pm25"].resample("h").mean()
    # Fill only short gaps. Large gaps are left missing and reported rather than
    # silently inventing long stretches of pollution values.
    series = series.interpolate(method="time", limit=3, limit_area="inside")
    if series.isna().any():
        missing = int(series.isna().sum())
        raise ValueError(f"Recent hourly history contains {missing} missing value(s) after short-gap interpolation. SARIMA was not run on incomplete data.")
    if len(series) < 48:
        raise ValueError("Not enough continuous hourly PM2.5 history is available for SARIMA.")

    model = SARIMAX(
        series.astype(float),
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 24),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=50)
    prediction = fitted.get_forecast(steps=int(horizon))
    mean = prediction.predicted_mean.clip(lower=0)
    interval = prediction.conf_int()
    result = pd.DataFrame({
        "time": pd.date_range(start=current_time, periods=int(horizon), freq="h"),
        "PM2.5 forecast": mean.to_numpy(),
        "Lower 95%": interval.iloc[:, 0].clip(lower=0).to_numpy(),
        "Upper 95%": interval.iloc[:, 1].clip(lower=0).to_numpy(),
    })
    return result, len(series), float(series.iloc[-1]), float(fitted.aic)


def render_sarima_forecast(hourly, current_time_text, city):
    """Show a live, city-specific SARIMA PM2.5 forecast in the ML tab."""
    st.markdown('<div class="section-title">SARIMA PM2.5 time-series forecast</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Fits SARIMA(1,0,1) × (1,0,1,24) to recent hourly history and forecasts the next 24 hours.</div>', unsafe_allow_html=True)
    st.info("This SARIMA forecast is fitted from the selected city's recent Open-Meteo hourly PM2.5 estimates. It is separate from the Logistic Regression spike-probability model and is not an official CPCB forecast.")
    try:
        times = tuple(str(value) for value in hourly.get("time", []))
        raw_values = pd.to_numeric(pd.Series(hourly.get("pm2_5", [])), errors="coerce")
        values = tuple(None if pd.isna(value) else float(value) for value in raw_values)
        with st.spinner("Fitting SARIMA to recent hourly PM2.5 history…"):
            forecast, history_hours, last_value, aic = calculate_sarima_forecast(
                times, values, str(current_time_text), 24
            )

        c1, c2, c3 = st.columns(3)
        c1.metric("Next 24-hour average", f"{forecast['PM2.5 forecast'].mean():.1f} µg/m³")
        c2.metric("Forecast peak", f"{forecast['PM2.5 forecast'].max():.1f} µg/m³")
        c3.metric("Past hourly values used", str(history_hours))
        st.caption(f"Selected city: {city} · Latest historical PM2.5: {last_value:.1f} µg/m³ · Model AIC: {aic:.1f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast["time"], y=forecast["Upper 95%"], mode="lines",
            line={"width": 0}, showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["time"], y=forecast["Lower 95%"], mode="lines",
            line={"width": 0}, fill="tonexty", fillcolor="rgba(59,130,246,0.15)",
            name="95% confidence interval", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=forecast["time"], y=forecast["PM2.5 forecast"], mode="lines+markers",
            line={"color": "#2563eb", "width": 3}, marker={"size": 5},
            name="SARIMA forecast", hovertemplate="%{x|%a %d %b, %I:%M %p}<br>PM2.5: %{y:.1f} µg/m³<extra></extra>",
        ))
        fig.update_layout(
            height=365, margin=dict(l=18, r=18, t=25, b=18),
            paper_bgcolor="white", plot_bgcolor="#fcfdff",
            font={"family": "Inter, sans-serif", "color": "#35516b", "size": 12},
            xaxis={"title": "Local date and time", "gridcolor": "#edf2f7", "tickformat": "%I %p"},
            yaxis={"title": "PM2.5 (µg/m³)", "gridcolor": "#e5edf5", "rangemode": "tozero"},
            hovermode="x unified", legend={"orientation": "h", "y": 1.12},
        )
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("View SARIMA forecast values"):
            st.dataframe(forecast.round(2), use_container_width=True, hide_index=True)
        st.download_button(
            "Download SARIMA forecast CSV",
            forecast.to_csv(index=False).encode("utf-8"),
            file_name=f"{city.lower()}_sarima_pm25_forecast.csv",
            mime="text/csv",
        )
        st.caption("Model specification: SARIMA(1,0,1)(1,0,1,24). Confidence intervals describe model uncertainty and do not guarantee actual future readings.")
    except (ValueError, TypeError, KeyError, IndexError, RuntimeError, OSError) as exc:
        st.warning("SARIMA could not produce a forecast from the available recent hourly data.")
        with st.expander("SARIMA technical details"):
            st.code(str(exc))


def render_ml_prediction(hourly, city, current_time_text):

    st.markdown('<div class="section-title">ML pollution-spike risk</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-subtitle">A trained classifier estimates whether hourly PM2.5 may exceed 60 µg/m³ at least once in the next 24 hours.</div>', unsafe_allow_html=True)

    st.info("This is a machine-learning risk estimate, not a guaranteed forecast or an official CPCB alert. The saved model was trained using historical Jaipur data; predictions for other cities are not validated and may be unreliable.")

    try:

        artifact = load_spike_artifact()

        features, message = build_spike_features(hourly, current_time_text)

        if message:

            st.warning(message)

            return

        feature_columns = artifact["feature_columns"]

        missing_columns = [name for name in feature_columns if name not in features.columns]

        if missing_columns:

            st.error("The model's expected feature columns do not match the generated features: " + ", ".join(missing_columns))

            return

        feature_row = features[feature_columns]

        probability = float(artifact["model"].predict_proba(feature_row)[0, 1])

        threshold = float(artifact.get("threshold", 60.0))

        horizon = int(artifact.get("horizon_hours", 24))

        risk_label = "Higher model-estimated risk" if probability >= 0.5 else "Lower model-estimated risk"

        risk_color = "#dc2626" if probability >= 0.5 else "#0d9488"

        c1, c2, c3 = st.columns(3)

        with c1:

            metric_card("Spike probability", f"{probability * 100:.1f}%", risk_label, risk_color)

        with c2:

            metric_card("PM2.5 threshold", f"{threshold:.0f} µg/m³", "Event threshold used for training", "#f97316")

        with c3:

            metric_card("Prediction horizon", f"{horizon} hours", f"Selected city: {city}", "#3b82f6")

        gauge = go.Figure(go.Indicator(mode="gauge+number", value=probability * 100, number={"suffix":"%","font":{"size":38,"color":"#17324d"}}, title={"text":"Estimated chance of a PM2.5 spike","font":{"size":15,"color":"#35516b"}}, gauge={"axis":{"range":[0,100],"ticksuffix":"%"},"bar":{"color":risk_color,"thickness":0.25},"bgcolor":"white","borderwidth":0,"steps":[{"range":[0,25],"color":"#d1fae5"},{"range":[25,50],"color":"#fef3c7"},{"range":[50,75],"color":"#fed7aa"},{"range":[75,100],"color":"#fee2e2"}]}))

        gauge.update_layout(height=260, margin=dict(l=18,r=18,t=35,b=8), paper_bgcolor="white", font={"family":"Inter, sans-serif"})

        st.plotly_chart(gauge, use_container_width=True)

        if probability >= 0.5:

            st.warning("The model estimates an elevated chance of the defined PM2.5 spike event. Consider checking official local air-quality updates before planning prolonged outdoor activity.")

        else:

            st.success("The model estimates a lower chance of the defined PM2.5 spike event. This does not mean the air is guaranteed to be safe; check current readings and official guidance.")

        with st.expander("View model input features"):

            st.dataframe(feature_row.T.rename(columns={0:"Value"}), use_container_width=True)

        st.caption("Model file: outputs/spike_model.joblib · Input data: Open-Meteo hourly PM2.5 estimates · Feature generation uses past values only.")

    except FileNotFoundError as exc:

        st.error("The trained model file is missing. Put spike_model.joblib inside the project's outputs folder.")

        with st.expander("Technical details"):

            st.code(str(exc))

    except (KeyError, ValueError, TypeError, OSError, AttributeError) as exc:

        st.error("The ML prediction could not be calculated. Check that the saved model artifact and feature columns match the training script.")

        with st.expander("Technical details"):

            st.code(str(exc))



def plot_line(frame, x, y, label, color="#0d9488", unit=""):

    """Bright, accessible chart with explicit axis titles, readable ticks and hover labels."""

    clean = frame.dropna(subset=[x, y]).copy()

    fig = go.Figure()

    fig.add_trace(go.Scatter(

        x=clean[x], y=clean[y], mode="lines+markers", name=label,

        line={"color": color, "width": 3.5, "shape": "spline"},

        marker={"size": 6, "color": color, "line": {"color": "#ffffff", "width": 1.5}},

        fill="tozeroy", fillcolor="rgba(13,148,136,0.12)",

        hovertemplate=f"<b>%{{x|%a, %d %b · %I:%M %p}}</b><br>{label}: %{{y:.1f}} {unit}<extra></extra>"

    ))

    fig.update_layout(

        height=365, margin=dict(l=18, r=18, t=25, b=18),

        paper_bgcolor="white", plot_bgcolor="#fcfdff",

        font={"family":"Inter, sans-serif", "color":"#35516b", "size":12},

        xaxis={"title":{"text":"Local date and time", "font":{"size":12}}, "showgrid":True,

               "gridcolor":"#edf2f7", "tickformat":"%I %p", "tickfont":{"size":10},

               "linecolor":"#d8e4ef", "zeroline":False},

        yaxis={"title":{"text":unit or label, "font":{"size":12}}, "gridcolor":"#e5edf5",

               "rangemode":"tozero", "tickfont":{"size":10}, "linecolor":"#d8e4ef",

               "zeroline":False, "automargin":True},

        hovermode="x unified", showlegend=False,

        hoverlabel={"bgcolor":"#102a43", "font":{"color":"white", "family":"Inter"}}

    )

    return fig



with st.sidebar:

    st.markdown(f'<div style="display:flex;align-items:center;gap:10px;padding:8px 0 12px">{LOGO}<div><div style="font-size:23px;font-weight:800;line-height:1.1">AirGuard <span style="color:#55d6c2">AI</span></div><div style="font-size:9px;letter-spacing:1.5px;margin-top:5px">ENVIRONMENTAL INTELLIGENCE</div></div></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)

    st.markdown("### Dashboard & location")

    city = st.selectbox("Choose a city", list(CITIES.keys()), index=0, help="All readings and forecasts update for this city.")

    if st.button("↻  Refresh live data", use_container_width=True, type="primary"):

        fetch_air_quality.clear(); fetch_weather.clear(); st.rerun()

    st.markdown(f'<div class="sidebar-note"><b>📍 Selected location</b><br>{city}, India<br><span style="color:#a9c6d8;font-size:10px">{CITIES[city][0]:.4f}° N · {CITIES[city][1]:.4f}° E</span></div>', unsafe_allow_html=True)

    st.markdown('<div class="sidebar-label">Explore</div>', unsafe_allow_html=True)

    st.markdown("""- 🏠 **Dashboard** — current conditions

- 📈 **Forecast** — next 24 hours

- 📊 **Trends** — hourly changes

- 💚 **Health insights** — practical guidance""")

    st.divider()

    st.markdown('<div class="sidebar-label">Data & project</div>', unsafe_allow_html=True)

    st.markdown("**Live data sources**")

    st.caption("Air quality: Open-Meteo Air Quality API")

    st.caption("Weather: Open-Meteo Forecast API")

    st.caption("Model-based estimates; not official CPCB station readings.")

    with st.expander("About AirGuard AI"):

        st.write("AirGuard AI brings air quality, weather, short-term forecasts and health context into one dashboard.")

        st.caption("Forecasts are estimates and can differ from local monitoring stations.")

    st.markdown('<div class="sidebar-note"><b>System status</b><br>🟢 API connection checked on refresh<br>🔄 Data cache: 10 minutes</div>', unsafe_allow_html=True)



lat, lon = CITIES[city]

# Keep the mode switch in the main dashboard header so it remains easy to find

# even when the sidebar is collapsed.

head1, head2 = st.columns([2.7, 1.3], vertical_alignment="center")

with head1:

    st.markdown(f'<div class="brandline">{LOGO}<div><div class="brandname">AirGuard <span style="color:#0d9488">AI</span></div><div class="brandtag">Live environmental monitoring · Forecasts · Health insights</div></div></div>', unsafe_allow_html=True)

with head2:

    st.markdown('<div class="mode-caption">Choose dashboard</div>', unsafe_allow_html=True)

    view = st.segmented_control("Dashboard mode", options=["Air Quality", "Weather"], default="Air Quality", key="view_mode_main", label_visibility="collapsed", format_func=lambda x: "🍃  Air Quality" if x == "Air Quality" else "☁️  Weather")

    if view is None:

        view = "Air Quality"

    st.markdown('<div style="text-align:right;margin-top:7px"><span style="background:#dcfce7;color:#166534;padding:6px 10px;border-radius:30px;font-size:10px;font-weight:800">● LIVE DATA</span></div>', unsafe_allow_html=True)



try:

    if view == "Air Quality":

        with st.spinner(f"Loading air-quality data for {city}…"): data = fetch_air_quality(lat, lon)

        current, hourly = data["current"], data["hourly"]

        aqi, pm25, pm10, timestamp = current.get("us_aqi"), current.get("pm2_5"), current.get("pm10"), current.get("time", "—")

        category, color, advice = aqi_details(aqi)

        st.markdown(f'<div class="hero"><div class="eyebrow">AIR QUALITY INTELLIGENCE · {city.upper()}</div><h2>Understand today’s air. Prepare for tomorrow.</h2><p>Track current pollution estimates and explore the next 24 hours.</p><p style="margin-top:13px;font-size:12px">📍 {city}, India &nbsp; · &nbsp; Local API time: {timestamp}</p></div>', unsafe_allow_html=True)

        st.markdown('<div class="notice"><b>Data note:</b> AQI uses the US AQI scale and Open-Meteo modelled data. This is not an official Indian CPCB station reading. The forecast below is from the API, not a separately trained ML model.</div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Current air quality</div><div class="section-subtitle">Latest values returned by the live air-quality API</div>', unsafe_allow_html=True)

        cols = st.columns(4)

        with cols[0]: metric_card("Current US AQI", str(int(float(aqi))) if aqi is not None and pd.notna(aqi) else "—", category, color)

        with cols[1]: metric_card("PM2.5", fmt(pm25," µg/m³"), "Fine particulate matter", "#0d9488")

        with cols[2]: metric_card("PM10", fmt(pm10," µg/m³"), "Particulate matter", "#3b82f6")

        with cols[3]: metric_card("Health category", category, "Based on US AQI scale", color)

        t1,t2,t3,t4,t5 = st.tabs(["◉ Overview", "⌁ Next 24 Hours", "♡ Health Guidance", "ⓘ About", "🤖 ML Prediction"])

        with t1:

            left,right = st.columns([1.2,1])

            with left:

                st.markdown('<div class="section-title">Current AQI indicator</div>', unsafe_allow_html=True)

                gauge = go.Figure(go.Indicator(mode="gauge+number", value=float(aqi) if aqi is not None and pd.notna(aqi) else 0, number={"font":{"size":46,"color":"#17324d"},"valueformat":".0f"}, title={"text":f"<b>{category}</b>","font":{"size":16,"color":color}}, gauge={"axis":{"range":[0,500],"tickfont":{"size":11,"color":"#35516b"},"ticks":"outside","tickcolor":"#9aafc2"},"bar":{"color":color,"thickness":.25},"bgcolor":"white","borderwidth":0,"steps":[{"range":[0,50],"color":"#dcfce7"},{"range":[50,100],"color":"#fef9c3"},{"range":[100,150],"color":"#ffedd5"},{"range":[150,200],"color":"#fee2e2"},{"range":[200,300],"color":"#ede9fe"},{"range":[300,500],"color":"#e5e7eb"}]}))

                gauge.update_layout(height=310, margin=dict(l=15,r=15,t=50,b=10), paper_bgcolor="white", font={"family":"Inter, sans-serif", "color":"#35516b"})

                st.plotly_chart(gauge, use_container_width=True)

            with right:

                st.markdown('<div class="section-title">Air quality status</div>', unsafe_allow_html=True)

                st.markdown(f'<div class="panel"><span class="pill" style="background:{color}20;color:{color}">{category.upper()}</span><h3 style="margin-top:17px;color:#17324d">{city}, India</h3><p style="color:#718198;font-size:13px">{advice}</p><hr style="border:0;border-top:1px solid #e3eaf2"><p style="color:#718198;font-size:12px"><b>Coordinates:</b> {lat}, {lon}<br><b>API timestamp:</b> {timestamp}</p></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-title">AQI category guide</div>', unsafe_allow_html=True)

            cats=[("Good","0–50","#16a34a"),("Moderate","51–100","#ca8a04"),("Sensitive groups","101–150","#f97316"),("Unhealthy","151–200","#dc2626"),("Very unhealthy","201–300","#9333ea"),("Hazardous","301+","#7f1d1d")]

            ccols=st.columns(6)

            for col,(name,band,shade) in zip(ccols,cats):

                with col: st.markdown(f'<div class="panel" style="padding:11px 7px;text-align:center"><div style="height:5px;background:{shade};border-radius:8px;margin-bottom:10px"></div><b style="font-size:11px">{name}</b><div style="font-size:11px;color:#718198;margin-top:5px">{band}</div></div>',unsafe_allow_html=True)

        with t2:

            st.markdown('<div class="section-title">Next 24-hour pollution forecast</div><div class="section-subtitle">Hourly forecast from the air-quality API</div>', unsafe_allow_html=True)

            frame=make_hourly_frame(hourly,{"AQI":"us_aqi","PM2.5":"pm2_5","PM10":"pm10"})

            start=pd.to_datetime(timestamp,errors="coerce")

            if pd.notna(start): frame=frame[(frame.time>=start)&(frame.time<=start+pd.Timedelta(hours=24))]

            if frame.empty: st.warning("No hourly forecast data was returned. Try refreshing.")

            else:

                vals=frame.AQI.dropna(); pvals=frame["PM2.5"].dropna(); m1,m2,m3=st.columns(3)

                m1.metric("Average forecast AQI",f"{vals.mean():.0f}" if not vals.empty else "—")

                m2.metric("Peak forecast AQI",f"{vals.max():.0f}" if not vals.empty else "—")

                m3.metric("Average forecast PM2.5",f"{pvals.mean():.1f} µg/m³" if not pvals.empty else "—")

                metric=st.radio("Choose forecast metric",["AQI","PM2.5","PM10"],horizontal=True,key="aq_metric")

                st.plotly_chart(plot_line(frame,"time",metric,metric,unit="US AQI" if metric=="AQI" else f"{metric} (µg/m³)"),use_container_width=True)

                with st.expander("View hourly forecast data"): st.dataframe(frame,use_container_width=True,hide_index=True)

                st.download_button("Download 24-hour forecast CSV",frame.to_csv(index=False).encode(),file_name=f"{city.lower()}_24_hour_air_quality.csv",mime="text/csv")

        with t3:

            st.markdown('<div class="section-title">Air quality and health</div>',unsafe_allow_html=True)

            st.markdown(f'<div class="panel"><span class="pill" style="background:{color}20;color:{color}">CURRENT CATEGORY: {category.upper()}</span><h3 style="margin-top:16px">Guidance for {city}</h3><p>{advice}</p><hr style="border:0;border-top:1px solid #e3eaf2"><p>• Check official local air-quality advisories.<br>• Consider limiting strenuous outdoor exercise when pollution is elevated.<br>• People with respiratory conditions should follow their clinician’s advice.<br>• This dashboard is informational and is not a medical diagnosis.</p></div>',unsafe_allow_html=True)

        with t4:

            st.markdown('<div class="section-title">About AirGuard AI</div>',unsafe_allow_html=True)

            st.write("AirGuard AI is a student project dashboard for live air-quality monitoring and short-term pollution forecasting.")

            st.markdown("**Features**\n- Current model-estimated US AQI\n- PM2.5 and PM10 readings\n- Hourly 24-hour forecast\n- Forecast chart and CSV export\n- City selection and health guidance")

            st.markdown("**Data source:** [Open-Meteo Air Quality API](https://open-meteo.com/en/docs/air-quality-api)")

            st.info("The API forecast is not a prediction from a separately trained AirGuard AI machine-learning model. Use the ML Prediction tab to run the separately trained PM2.5 spike classifier.")

        with t5:

            render_ml_prediction(hourly, city, timestamp)
            st.markdown("---")
            render_sarima_forecast(hourly, timestamp, city)

    else:

        with st.spinner(f"Loading weather data for {city}…"):

            data = fetch_weather(lat, lon)

        current, hourly, daily = data["current"], data.get("hourly", {}), data.get("daily", {})

        wcode = int(current.get("weather_code", 0) or 0)

        weather_names = {0:"Clear sky",1:"Mainly clear",2:"Partly cloudy",3:"Overcast",45:"Fog",48:"Rime fog",51:"Light drizzle",53:"Drizzle",55:"Dense drizzle",61:"Light rain",63:"Rain",65:"Heavy rain",71:"Light snow",73:"Snow",75:"Heavy snow",80:"Rain showers",81:"Rain showers",82:"Heavy rain showers",95:"Thunderstorm",96:"Thunderstorm with hail",99:"Thunderstorm with heavy hail"}

        weather_icons = {0:"☀️",1:"🌤️",2:"⛅",3:"☁️",45:"🌫️",48:"🌫️",51:"🌦️",53:"🌦️",55:"🌧️",61:"🌦️",63:"🌧️",65:"🌧️",71:"🌨️",73:"❄️",75:"❄️",80:"🌦️",81:"🌧️",82:"⛈️",95:"⛈️",96:"⛈️",99:"⛈️"}

        condition = weather_names.get(wcode, "Variable conditions")

        icon = weather_icons.get(wcode, "🌤️")

        temp, feels = current.get("temperature_2m"), current.get("apparent_temperature")

        humidity, wind, rain = current.get("relative_humidity_2m"), current.get("wind_speed_10m"), current.get("precipitation")

        timestamp = current.get("time", "—")

        st.markdown(f'<div class="weather-highlight"><div style="font-size:10px;font-weight:800;letter-spacing:1.7px;color:#9de8de">LIVE WEATHER INTELLIGENCE · {city.upper()}</div><div style="display:flex;align-items:center;gap:22px;flex-wrap:wrap"><div style="font-size:60px;line-height:1.1">{icon}</div><div><h2>{fmt(temp,"°C")} · {condition}</h2><p>{city}, India · Feels like {fmt(feels,"°C")} · Updated {timestamp}</p><p style="margin-top:8px">A clear view of the conditions now and what to expect through the day.</p></div></div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Weather right now</div><div class="section-subtitle">Current conditions and atmospheric indicators for your selected city</div>', unsafe_allow_html=True)

        wc = st.columns(5)

        cards = [("Temperature",fmt(temp," °C"),condition,"#f59e0b"),("Feels like",fmt(feels," °C"),"Perceived temperature","#f97316"),("Humidity",fmt(humidity,"%"),"Relative humidity","#3b82f6"),("Wind speed",fmt(wind," km/h"),"10 m wind","#0d9488"),("Precipitation",fmt(rain," mm"),"Current interval","#8b5cf6")]

        for col, item in zip(wc, cards):

            with col: metric_card(*item)



        # Next-hour strip: real hourly values, not decorative/sample numbers.

        wf = make_hourly_frame(hourly, {"Temperature":"temperature_2m","Feels like":"apparent_temperature","Humidity":"relative_humidity_2m","Rain chance":"precipitation_probability","Rain":"precipitation","Wind speed":"wind_speed_10m","Code":"weather_code"})

        now = pd.to_datetime(timestamp, errors="coerce")

        if pd.notna(now):

            wf = wf[(wf.time >= now.floor("h")) & (wf.time <= now.floor("h") + pd.Timedelta(hours=24))].copy()

        st.markdown('<div class="section-title">Next 24 hours</div><div class="section-subtitle">Hourly forecast cards · scroll across to compare temperature, rain chance and wind</div>', unsafe_allow_html=True)

        if wf.empty:

            st.warning("No hourly weather forecast was returned. Use Refresh live data to try again.")

        else:

            strip = wf.head(24)

            hourly_cols = st.columns(6)

            for idx, (_, row) in enumerate(strip.iterrows()):

                code = int(row["Code"]) if pd.notna(row.get("Code")) else 0

                hour_icon = weather_icons.get(code, "🌤️")

                label = "Now" if idx == 0 else row["time"].strftime("%I %p").lstrip("0")

                with hourly_cols[idx % 6]:

                    st.markdown(f'<div class="hour-card"><div class="hour-time">{label}</div><div class="hour-icon">{hour_icon}</div><div class="hour-temp">{fmt(row["Temperature"],"°")}</div><div class="hour-sub">Feels {fmt(row["Feels like"],"°")}<br>💧 {fmt(row["Rain chance"],"%")} rain<br>💨 {fmt(row["Wind speed"]," km/h")}</div></div>', unsafe_allow_html=True)

                if idx % 6 == 5 and idx < len(strip)-1:

                    hourly_cols = st.columns(6)



        wt1, wt2, wt3 = st.tabs(["📊 Weather analytics", "🗓️ 3-day outlook", "ℹ️ Data & definitions"])

        with wt1:

            st.markdown('<div class="chart-heading">Hourly weather trend</div><div class="section-subtitle">Use the metric buttons to explore the next 24 hours. Hover over points for exact local time and value.</div>', unsafe_allow_html=True)

            if not wf.empty:

                wxmetric = st.segmented_control("Weather chart metric", ["Temperature", "Feels like", "Rain chance", "Humidity", "Wind speed"], default="Temperature", key="weather_metric_btn", label_visibility="visible", format_func=lambda x: {"Temperature":"🌡️ Temperature","Feels like":"♨️ Feels like","Rain chance":"🌧️ Rain chance","Humidity":"💧 Humidity","Wind speed":"💨 Wind speed"}[x])

                wxmetric = wxmetric or "Temperature"

                units = {"Temperature":"°C","Feels like":"°C","Rain chance":"%","Humidity":"%","Wind speed":"km/h"}

                colors = {"Temperature":"#f97316","Feels like":"#ef4444","Rain chance":"#3b82f6","Humidity":"#0ea5e9","Wind speed":"#8b5cf6"}

                chart = plot_line(wf.head(24), "time", wxmetric, wxmetric, color=colors[wxmetric], unit=units[wxmetric])

                chart.update_layout(height=410, xaxis={"title":"Forecast time (local)","tickformat":"%I %p","dtick":3*60*60*1000,"showgrid":True,"gridcolor":"#eaf0f7","linecolor":"#cbd9e7","automargin":True}, yaxis={"title":units[wxmetric],"gridcolor":"#e4edf6","rangemode":"tozero","automargin":True}, margin=dict(l=35,r=20,t=30,b=45))

                if wxmetric == "Temperature":

                    chart.add_hline(y=35, line_dash="dot", line_color="#f59e0b", annotation_text="Hot weather reference · 35°C", annotation_position="top left")

                st.plotly_chart(chart, use_container_width=True, config={"displaylogo":False,"responsive":True})

                st.markdown('<div style="margin-top:20px;margin-bottom:10px;padding:14px 16px;border-radius:14px;background:linear-gradient(110deg,#e7f8f5,#f1f7ff);border:1px solid #cce9e5"><div style="font-size:18px;font-weight:800;color:#123b4a">⚡ Quick read</div><div style="font-size:12px;color:#526b7b;margin-top:4px">A simple summary of the selected metric for the next 24 hours</div></div>', unsafe_allow_html=True)

                if wxmetric in wf.columns:

                    series = wf.head(24)[wxmetric].dropna()

                    if not series.empty:

                        quick_values = [(f"24-hour average", f"{series.mean():.1f} {units[wxmetric]}", "Average across available hourly forecasts", "#0d9488"), ("Highest forecast", f"{series.max():.1f} {units[wxmetric]}", "Highest value in the next 24 hours", "#f97316"), ("Lowest forecast", f"{series.min():.1f} {units[wxmetric]}", "Lowest value in the next 24 hours", "#3b82f6")]

                        qcols = st.columns(3)

                        for qcol, (qtitle, qvalue, qnote, qcolor) in zip(qcols, quick_values):

                            with qcol:

                                st.markdown(f'<div style="height:100%;min-height:125px;padding:17px;border-radius:14px;background:#ffffff;border:1px solid #d8e6ef;border-top:4px solid {qcolor};box-shadow:0 5px 18px rgba(16,42,67,.06)"><div style="font-size:11px;font-weight:800;letter-spacing:.5px;text-transform:uppercase;color:#61778a">{qtitle}</div><div style="font-size:27px;font-weight:850;color:#17324d;margin-top:12px;overflow-wrap:anywhere">{qvalue}</div><div style="font-size:11px;line-height:1.5;color:#718198;margin-top:7px">{qnote}</div></div>', unsafe_allow_html=True)

                    else:

                        st.info("Quick read will appear when hourly forecast values are available for this metric.")

                with st.expander("View full hourly weather table"):

                    display = wf.head(24).copy()

                    display["time"] = display["time"].dt.strftime("%a, %d %b %I:%M %p")

                    st.dataframe(display.drop(columns=["Code"], errors="ignore"), use_container_width=True, hide_index=True)

                st.download_button("⬇ Download 24-hour weather forecast (CSV)", wf.head(24).drop(columns=["Code"],errors="ignore").to_csv(index=False).encode("utf-8"), file_name=f"{city.lower()}_24_hour_weather.csv", mime="text/csv", use_container_width=True)

        with wt2:

            st.markdown('<div class="section-title">3-day weather outlook</div><div class="section-subtitle">Daily high and low temperatures, precipitation probability and wind outlook when provided by the API</div>', unsafe_allow_html=True)

            if daily.get("time"):

                dcols = st.columns(min(3, len(daily["time"])))

                for i, day in enumerate(daily["time"][:3]):

                    hi=(daily.get("temperature_2m_max") or [None]*3)[i]; lo=(daily.get("temperature_2m_min") or [None]*3)[i]

                    pop=(daily.get("precipitation_probability_max") or [None]*3)[i]; uv=(daily.get("uv_index_max") or [None]*3)[i]; wmax=(daily.get("wind_speed_10m_max") or [None]*3)[i]

                    code=int((daily.get("weather_code") or [0]*3)[i] or 0)

                    with dcols[i]:

                        st.markdown(f'<div class="panel" style="min-height:230px;text-align:center"><div style="font-weight:800;color:#52677d">{pd.to_datetime(day).strftime("%A, %d %b")}</div><div style="font-size:36px;margin:13px 0">{weather_icons.get(code,"🌤️")}</div><div style="font-size:26px;font-weight:800;color:#102a43">{fmt(hi,"°")} <span style="font-size:15px;color:#718198">/ {fmt(lo,"°")}</span></div><div style="font-size:12px;color:#718198;margin-top:8px">{weather_names.get(code,"Variable conditions")}</div><hr style="border:0;border-top:1px solid #e3eaf2;margin:12px 0"><div style="font-size:12px;text-align:left;line-height:2;color:#52677d">🌧️ Rain chance <b>{fmt(pop,"%")}</b><br>💨 Max wind <b>{fmt(wmax," km/h")}</b><br>☀️ Max UV index <b>{fmt(uv)}</b></div></div>', unsafe_allow_html=True)

            else:

                st.info("Daily outlook is not available in this response. Check again after refreshing.")

        with wt3:

            st.markdown('<div class="section-title">About this weather data</div>', unsafe_allow_html=True)

            st.write("Temperature, feels-like temperature, relative humidity, precipitation, wind and weather codes are returned by the Open-Meteo Forecast API for the selected city coordinates.")

            st.markdown("**Source:** [Open-Meteo Weather API](https://open-meteo.com/en/docs)")

            st.markdown("**How to read the dashboard**")

            st.markdown("""- **Temperature:** air temperature at approximately 2 metres above ground.

- **Feels like:** modelled apparent temperature.

- **Rain chance:** forecast probability of precipitation for that hour.

- **Wind speed:** forecast wind speed at 10 metres.

- **Humidity:** relative humidity percentage.""")

            st.info("Forecasts are model-based estimates and may differ from observations at a nearby weather station. Use official alerts for severe weather.")

    st.markdown("---")

    st.markdown('<div style="text-align:center;color:#8291a5;font-size:11px;padding:10px">AIRGUARD AI · ENVIRONMENTAL INTELLIGENCE<br>Powered by Open-Meteo · Educational project dashboard</div>',unsafe_allow_html=True)
<<<<<<< HEAD
except RuntimeError as exc:
    st.warning(str(exc))
=======

>>>>>>> 02d60ac (Add SARIMA forecasting to AirGuard AI)
except requests.exceptions.Timeout:

    st.error("The live data service took too long to respond. Please use Refresh live data and try again.")

except requests.exceptions.RequestException as exc:

    st.error("Could not connect to the live data service. Check your internet connection and try again.")

    with st.expander("Technical details"): st.code(str(exc))

except (ValueError, KeyError, TypeError, IndexError) as exc:

    st.error("The data service returned unexpected or incomplete data.")

    with st.expander("Technical details"): st.code(str(exc))
