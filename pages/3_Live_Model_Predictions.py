
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------
# AirGuard AI - Live Model Predictions
# --------------------------------------------------

st.set_page_config(
    page_title="AirGuard AI | Live Model Predictions",
    page_icon="🌿",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "outputs" / "spike_model.joblib"

AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

CITIES = {
    "Jaipur": (26.9124, 75.7873),
    "Delhi": (28.6139, 77.2090),
    "Mumbai": (19.0760, 72.8777),
    "Bengaluru": (12.9716, 77.5946),
    "Kolkata": (22.5726, 88.3639),
    "Chennai": (13.0827, 80.2707),
    "Hyderabad": (17.3850, 78.4867),
    "Pune": (18.5204, 73.8567),
    "Lucknow": (26.8467, 80.9462),
    "Ahmedabad": (23.0225, 72.5714),
}

LAGS = [1, 2, 3, 6, 12, 24]
ROLLING_WINDOWS = [6, 12, 24]


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=600, show_spinner=False)
def fetch_air_quality(lat, lon):
    response = requests.get(
        AQ_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "hourly": "pm2_5",
            "past_days": 2,
            "forecast_days": 2,
            "timezone": "auto",
        },
        timeout=25,
    )
    response.raise_for_status()
    return response.json()


def make_features(pm25):
    """Build features using the same definitions as model training."""
    features = pd.DataFrame(index=pm25.index)

    for lag in LAGS:
        features[f"pm25_lag_{lag}h"] = pm25.shift(lag)

    past_values = pm25.shift(1)

    for hours in ROLLING_WINDOWS:
        features[f"pm25_mean_{hours}h"] = (
            past_values.rolling(hours, min_periods=hours).mean()
        )
        features[f"pm25_std_{hours}h"] = (
            past_values.rolling(hours, min_periods=hours).std()
        )

    features["hour"] = pm25.index.hour
    features["day_of_week"] = pm25.index.dayofweek
    features["month"] = pm25.index.month

    return features


st.title("AirGuard AI — Live Model Predictions")
st.caption(
    "Live air-quality inputs • Machine-learning spike prediction "
    "• Historical SARIMA evaluation"
)

st.info(
    "The air-quality inputs are modelled estimates from Open-Meteo, "
    "not official CPCB station measurements. The Logistic Regression "
    "model was trained on historical Jaipur data. Predictions for other "
    "cities are experimental and have not been independently validated."
)

# Sidebar
with st.sidebar:
    st.header("Prediction Settings")
    city = st.selectbox("Select city", list(CITIES.keys()), index=0)
    refresh = st.button("Refresh live data", type="primary")

if refresh:
    fetch_air_quality.clear()

lat, lon = CITIES[city]

# Load the saved model
try:
    artifact = load_model()
    classifier = artifact["model"]
    feature_columns = artifact["feature_columns"]
    threshold = float(artifact["threshold"])
    horizon = int(artifact["horizon_hours"])
except Exception as exc:
    st.error(f"Could not load the saved spike model: {exc}")
    st.stop()

# Fetch live/history data
try:
    with st.spinner(f"Fetching recent air-quality data for {city}..."):
        data = fetch_air_quality(lat, lon)
except Exception as exc:
    st.error(f"Could not fetch air-quality data: {exc}")
    st.stop()

hourly = data.get("hourly", {})
times = hourly.get("time", [])
values = hourly.get("pm2_5", [])

if not times or len(times) != len(values):
    st.error("The API did not return a usable hourly PM2.5 series.")
    st.stop()

frame = pd.DataFrame({
    "time": pd.to_datetime(times, errors="coerce"),
    "pm25": pd.to_numeric(values, errors="coerce"),
}).dropna(subset=["time"])

frame = (
    frame.drop_duplicates(subset="time")
    .sort_values("time")
    .set_index("time")
)

# Exclude forecast hours: the model must use past/current inputs only.
current_time = pd.to_datetime(
    data.get("current", {}).get("time", times[0]),
    errors="coerce",
)

if pd.isna(current_time):
    st.error("Could not determine the API's current local time.")
    st.stop()

current_hour = current_time.floor("h")
history = frame.loc[frame.index <= current_hour, "pm25"]

if history.empty:
    st.error("No historical PM2.5 values are available.")
    st.stop()

# Display latest available observed/modelled input value.
latest_time = history.index[-1]
latest_value = history.iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Selected City", city)
col2.metric("Latest PM2.5", f"{latest_value:.1f} µg/m³")
col3.metric("Latest Data Timestamp", latest_time.strftime("%d %b, %H:%M"))

st.divider()
st.subheader(f"Next {horizon}-hour pollution-spike prediction")

# Validate continuous hourly history without filling missing values.
expected_index = pd.date_range(
    end=current_hour,
    periods=25,
    freq="h",
)

recent = history.reindex(expected_index)

if len(recent) != 25 or recent.isna().any():
    missing_count = int(recent.isna().sum())
    st.warning(
        "A prediction was not generated because the latest 25 hourly "
        f"PM2.5 values are not continuous and complete. Missing hours: "
        f"{missing_count}. No missing values were filled or guessed."
    )
else:
    features = make_features(history)
    prediction_row = features.loc[[current_hour]]

    prediction_row = prediction_row.reindex(columns=feature_columns)

    if prediction_row.isna().any().any():
        st.warning(
            "The feature row contains missing values, so no prediction "
            "was generated."
        )
    else:
        probability = float(
            classifier.predict_proba(prediction_row)[0, 1]
        )
        predicted_class = int(
            classifier.predict(prediction_row)[0]
        )

        left, right = st.columns(2)

        left.metric(
            "Estimated spike probability",
            f"{probability * 100:.1f}%",
        )

        if predicted_class == 1:
            right.warning(
                f"Spike predicted: at least one hourly PM2.5 value may "
                f"exceed {threshold:.0f} µg/m³ during the next {horizon} hours."
            )
        else:
            right.success(
                f"No spike predicted by the model for the next {horizon} hours."
            )

        st.caption(
            "This is an experimental model output, not a guarantee or "
            "a calibrated health-risk probability. It predicts the "
            "historical training target; it does not generate hourly "
            "PM2.5 forecast values."
        )

# Recent input chart
st.divider()
st.subheader("Recent hourly PM2.5 input history")

chart_data = history.tail(48).rename("PM2.5")
st.line_chart(chart_data)

with st.expander("View recent input data"):
    display = history.tail(48).rename("PM2.5").reset_index()
    display.columns = ["Local timestamp", "PM2.5 (µg/m³)"]
    st.dataframe(display, use_container_width=True, hide_index=True)

# SARIMA status
st.divider()
st.subheader("SARIMA forecasting model")

st.write(
    "The saved SARIMA model was fitted to historical Jaipur data ending "
    "in February 2020. Its historical test-set evaluation reported "
    "MAE of approximately 5.93 µg/m³ and RMSE of approximately "
    "7.43 µg/m³."
)

st.warning(
    "These are historical evaluation metrics, not current live accuracy. "
    "The saved model has not been updated with recent data, so this page "
    "does not present it as a valid current live forecast."
)

st.caption(
    "AirGuard AI | Educational project | Open-Meteo modelled data"
)
