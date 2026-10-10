
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 1. File paths
DATA_FILE = Path("data/processed/jaipur_air_quality_hourly.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load cleaned data
df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

pm25 = df.loc[
    "2020-01-01 00:00":"2020-02-29 23:00",
    "pm25"
].asfreq("h")

split_date = pd.Timestamp("2020-02-23 00:00")

train = pm25.loc[pm25.index < split_date]
test = pm25.loc[pm25.index >= split_date]

print("=" * 50)
print("AIRGUARD AI - FORECAST EVALUATION")
print("=" * 50)

print("Training observations:", train.notna().sum())
print("Testing observations:", test.notna().sum())


# 3. Train SARIMA using the same settings as before
print("\nTraining SARIMA...")

model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

fitted_model = model.fit(disp=False, maxiter=200)

sarima_forecast = pd.Series(
    np.asarray(
        fitted_model.get_forecast(
            steps=len(test)
        ).predicted_mean
    ),
    index=test.index
)


# 4. Previous-day baseline
# Predict each hour using the reading from 24 hours earlier.
baseline_forecast = pm25.reindex(
    test.index - pd.Timedelta(hours=24)
)

baseline_forecast.index = test.index


# 5. Evaluate both models fairly
def calculate_metrics(actual, predicted):
    valid = actual.notna() & predicted.notna()

    actual_valid = actual.loc[valid]
    predicted_valid = predicted.loc[valid]

    return {
        "Evaluated hours": int(valid.sum()),
        "MAE": mean_absolute_error(
            actual_valid, predicted_valid
        ),
        "RMSE": np.sqrt(
            mean_squared_error(
                actual_valid, predicted_valid
            )
        )
    }


sarima_metrics = calculate_metrics(
    test, sarima_forecast
)

baseline_metrics = calculate_metrics(
    test, baseline_forecast
)

comparison = pd.DataFrame([
    {
        "Model": "SARIMA",
        **sarima_metrics
    },
    {
        "Model": "Previous-day baseline",
        **baseline_metrics
    }
])

print("\nMODEL COMPARISON")
print(comparison.round(3).to_string(index=False))


# 6. Save metrics
comparison.to_csv(
    OUTPUT_DIR / "forecast_model_comparison.csv",
    index=False
)


# 7. Save predictions
predictions = pd.DataFrame({
    "timestamp": test.index,
    "actual_pm25": test.to_numpy(),
    "sarima_forecast": sarima_forecast.to_numpy(),
    "previous_day_baseline": baseline_forecast.to_numpy()
})

predictions.to_csv(
    OUTPUT_DIR / "forecast_comparison_predictions.csv",
    index=False
)


# 8. Generate comparison graph
plt.figure(figsize=(14, 6))

plt.plot(
    test.index,
    test,
    label="Actual PM2.5",
    linewidth=1.5
)

plt.plot(
    sarima_forecast.index,
    sarima_forecast,
    label="SARIMA forecast",
    linewidth=1.3
)

plt.plot(
    baseline_forecast.index,
    baseline_forecast,
    label="Previous-day baseline",
    linewidth=1.3
)

plt.title("AirGuard AI: Forecast Model Comparison")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "forecast_model_comparison.png",
    dpi=150
)

plt.show()

print("\nFiles saved successfully in outputs/:")
print("1. forecast_model_comparison.csv")
print("2. forecast_comparison_predictions.csv")
print("3. forecast_model_comparison.png")

