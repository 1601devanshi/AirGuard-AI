
from pathlib import Path

import numpy as np
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

# --------------------------------------------------
# 1. Load the cleaned dataset
# --------------------------------------------------

DATA_FILE = Path("data/processed/jaipur_air_quality_hourly.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

# Use the selected 60-day window
data = df.loc["2020-01-01 00:00":"2020-02-29 23:00", "pm25"]
data = data.asfreq("h")

print("=" * 55)
print("AIRGUARD AI - SARIMA FORECASTING")
print("=" * 55)
print("Selected period:", data.index.min(), "to", data.index.max())
print("Total hours:", len(data))
print("Valid PM2.5 readings:", data.notna().sum())
print("Missing PM2.5 readings:", data.isna().sum())

# --------------------------------------------------
# 2. Chronological train-test split
# --------------------------------------------------

split_date = pd.Timestamp("2020-02-23 00:00")

train = data.loc[data.index < split_date]
test = data.loc[data.index >= split_date]

print("\nTraining hours:", len(train))
print("Testing hours:", len(test))
print("Training observations:", train.notna().sum())
print("Testing observations:", test.notna().sum())

if train.notna().sum() < 500 or test.notna().sum() == 0:
    raise ValueError("Not enough valid data for model training/testing.")

# --------------------------------------------------
# 3. Build the SARIMA model
# Seasonal period 24 represents hourly daily seasonality
# --------------------------------------------------

print("\nTraining SARIMA model...")

model = SARIMAX(
    train,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

results = model.fit(disp=False, maxiter=200)

print("\nModel training completed.")
print(results.summary())

# --------------------------------------------------
# 4. Forecast the seven-day test period
# --------------------------------------------------

forecast_result = results.get_forecast(steps=len(test))

predictions = pd.Series(
    np.asarray(forecast_result.predicted_mean),
    index=test.index,
    name="predicted_pm25"
)

confidence = forecast_result.conf_int()

lower = np.asarray(confidence.iloc[:, 0])
upper = np.asarray(confidence.iloc[:, 1])

# Evaluate only hours where actual PM2.5 is available
valid = test.notna()

actual_values = test.loc[valid]
predicted_values = predictions.loc[valid]

mae = mean_absolute_error(actual_values, predicted_values)
rmse = np.sqrt(mean_squared_error(actual_values, predicted_values))

print("\n" + "=" * 55)
print("MODEL EVALUATION")
print("=" * 55)
print(f"Evaluated hours: {valid.sum()}")
print(f"MAE:  {mae:.2f} µg/m³")
print(f"RMSE: {rmse:.2f} µg/m³")

# --------------------------------------------------
# 5. Save predictions and evaluation results
# --------------------------------------------------

forecast_df = pd.DataFrame({
    "timestamp": test.index,
    "actual_pm25": test.to_numpy(),
    "predicted_pm25": predictions.to_numpy(),
    "lower_confidence": lower,
    "upper_confidence": upper
})

forecast_df.to_csv(
    OUTPUT_DIR / "sarima_test_forecast.csv",
    index=False
)

metrics = pd.DataFrame([{
    "model": "SARIMA(1,0,1)(1,0,1,24)",
    "training_observations": int(train.notna().sum()),
    "testing_observations": int(valid.sum()),
    "MAE": mae,
    "RMSE": rmse
}])

metrics.to_csv(OUTPUT_DIR / "sarima_metrics.csv", index=False)

# --------------------------------------------------
# 6. Plot actual vs predicted PM2.5
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    test.index,
    test,
    label="Actual PM2.5",
    linewidth=1.5
)

plt.plot(
    predictions.index,
    predictions,
    label="SARIMA forecast",
    linewidth=1.5
)

plt.fill_between(
    test.index,
    lower,
    upper,
    alpha=0.15,
    label="95% confidence interval"
)

plt.title("AirGuard AI: PM2.5 Forecast vs Actual Values")
plt.xlabel("Date")
plt.ylabel("PM2.5 (µg/m³)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_DIR / "sarima_forecast_plot.png", dpi=150)
plt.show()

print("\nSaved files:")
print("- outputs/sarima_test_forecast.csv")
print("- outputs/sarima_metrics.csv")
print("- outputs/sarima_forecast_plot.png")
print("\nSARIMA experiment completed.")



# --------------------------------------------------
# 7. Fit final SARIMA model on the full selected window
# --------------------------------------------------

print("\nFitting final SARIMA model on the full selected window...")

final_model = SARIMAX(
    data,
    order=(1, 0, 1),
    seasonal_order=(1, 0, 1, 24),
    enforce_stationarity=False,
    enforce_invertibility=False
)

final_results = final_model.fit(disp=False, maxiter=200)

model_path = OUTPUT_DIR / "sarima_model.pkl"
final_results.save(model_path)

print(f"Saved fitted SARIMA model: {model_path}")
print("The saved model uses the selected Jaipur historical window.")

