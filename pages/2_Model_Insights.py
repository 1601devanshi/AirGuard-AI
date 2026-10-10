
from pathlib import Path
import json

import pandas as pd
import streamlit as st

# --------------------------------------------------
# AirGuard AI - Model Insights
# This page reads saved evaluation outputs.
# It does not change or replace app.py.
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"

st.set_page_config(
    page_title="AirGuard AI | Model Insights",
    page_icon="🌫️",
    layout="wide",
)

st.title("🧠 AirGuard AI — Model Insights")
st.caption(
    "Evaluation results for the trained PM2.5 forecasting "
    "and pollution-spike prediction models."
)

st.info(
    "These results are based on historical evaluation data. "
    "They are not live predictions for every city in the dashboard."
)


def load_csv(filename):
    path = OUTPUTS / filename
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Could not read {filename}: {exc}")
        return None


def show_csv_metrics(df):
    if df is None or df.empty:
        st.warning("No evaluation metrics found.")
        return

    # Display the saved metrics without assuming specific column names.
    st.dataframe(df, use_container_width=True, hide_index=True)


# --------------------------------------------------
# 1. SARIMA - PM2.5 Forecasting
# --------------------------------------------------

st.header("1. SARIMA — PM2.5 Forecasting")
st.write(
    "SARIMA forecasts PM2.5 concentration using historical "
    "time-series patterns. The saved test results are shown below."
)

sarima_metrics = load_csv("sarima_metrics.csv")
sarima_forecast = load_csv("sarima_test_forecast.csv")

if sarima_metrics is not None:
    st.subheader("SARIMA evaluation metrics")
    show_csv_metrics(sarima_metrics)
else:
    st.warning("Missing outputs/sarima_metrics.csv")

plot_path = OUTPUTS / "sarima_forecast_plot.png"
if plot_path.exists():
    st.subheader("Actual vs predicted PM2.5")
    st.image(
        str(plot_path),
        caption="Saved SARIMA historical test forecast",
        use_container_width=True,
    )
else:
    st.warning("Missing outputs/sarima_forecast_plot.png")

if sarima_forecast is not None:
    st.subheader("SARIMA test predictions")
    st.dataframe(
        sarima_forecast.tail(100),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download SARIMA test predictions",
        data=sarima_forecast.to_csv(index=False).encode("utf-8"),
        file_name="sarima_test_forecast.csv",
        mime="text/csv",
    )
else:
    st.warning("Missing outputs/sarima_test_forecast.csv")


st.divider()


# --------------------------------------------------
# 2. Logistic Regression - Pollution Spike Prediction
# --------------------------------------------------

st.header("2. Logistic Regression — Pollution Spike Detection")
st.write(
    "The classifier estimates whether at least one hourly PM2.5 "
    "reading will exceed 60 µg/m³ during the next 24 hours, "
    "using the features defined by the training pipeline."
)

metrics_path = OUTPUTS / "spike_model_metrics.json"

if metrics_path.exists():
    try:
        with open(metrics_path, "r", encoding="utf-8") as file:
            spike_metrics = json.load(file)

        st.subheader("Classifier evaluation metrics")

        if isinstance(spike_metrics, dict):
            metric_columns = st.columns(min(4, max(1, len(spike_metrics))))
            for index, (name, value) in enumerate(spike_metrics.items()):
                with metric_columns[index % len(metric_columns)]:
                    if isinstance(value, (int, float)):
                        st.metric(name.replace("_", " ").title(), f"{value:.3f}")
                    else:
                        st.metric(name.replace("_", " ").title(), str(value))

            with st.expander("View raw metrics"):
                st.json(spike_metrics)
        else:
            st.json(spike_metrics)

    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read spike model metrics: {exc}")
else:
    st.warning("Missing outputs/spike_model_metrics.json")

spike_predictions = load_csv("spike_test_predictions.csv")

if spike_predictions is not None:
    st.subheader("Spike model test predictions")
    st.dataframe(
        spike_predictions.tail(100),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download spike test predictions",
        data=spike_predictions.to_csv(index=False).encode("utf-8"),
        file_name="spike_test_predictions.csv",
        mime="text/csv",
    )
else:
    st.warning("Missing outputs/spike_test_predictions.csv")

model_path = OUTPUTS / "spike_model.joblib"
if model_path.exists():
    st.success("Logistic Regression model artifact found: spike_model.joblib")
else:
    st.warning("Logistic Regression model artifact not found.")


st.divider()


# --------------------------------------------------
# 3. Weather-aware health-risk layer
# --------------------------------------------------

st.header("3. Weather-aware Health-Risk Layer")
st.write(
    "This is the explainable risk-assessment layer. It uses "
    "weather and air-quality information to support health guidance; "
    "it is not a third trained machine-learning model."
)

st.markdown(
    """
    - **SARIMA:** estimates future PM2.5 concentrations from historical data.
    - **Logistic Regression:** classifies the probability of a PM2.5 spike
      using the trained model's input features.
    - **Risk layer:** interprets available pollution and weather conditions
      to present contextual health guidance.
    """
)

st.caption(
    "Model evaluation, live API forecasts, and health-risk guidance are "
    "different outputs. Do not interpret historical test predictions as "
    "live forecasts."
)
