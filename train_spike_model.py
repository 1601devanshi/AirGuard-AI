
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from pandas.api.indexers import FixedForwardWindowIndexer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_FILE = Path("data/processed/jaipur_air_quality_hourly.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 60.0
HORIZON_HOURS = 24

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp")
df = df.drop_duplicates("timestamp")
df = df.set_index("timestamp")

pm25 = df["pm25"].asfreq("h")

# --------------------------------------------------
# 3. Create past-only features
# --------------------------------------------------

features = pd.DataFrame(index=pm25.index)

# Previous PM2.5 readings
for lag in [1, 2, 3, 6, 12, 24]:
    features[f"pm25_lag_{lag}h"] = pm25.shift(lag)

# Statistics based only on readings before the prediction time
for hours in [6, 12, 24]:
    past_values = pm25.shift(1)

    features[f"pm25_mean_{hours}h"] = (
        past_values.rolling(
            hours, min_periods=hours
        ).mean()
    )

    features[f"pm25_std_{hours}h"] = (
        past_values.rolling(
            hours, min_periods=hours
        ).std()
    )

# Time features
features["hour"] = features.index.hour
features["day_of_week"] = features.index.dayofweek
features["month"] = features.index.month

# --------------------------------------------------
# 4. Create future target separately
# --------------------------------------------------

# Target is 1 if ANY of the next 24 hourly readings
# exceeds 60 µg/m³. Future values are used only to
# construct labels, never as model input features.

forward_window = FixedForwardWindowIndexer(
    window_size=HORIZON_HOURS
)

future_max = (
    pm25.shift(-1)
    .rolling(
        window=forward_window,
        min_periods=HORIZON_HOURS
    )
    .max()
)

target = (future_max > THRESHOLD).astype(float)
target[future_max.isna()] = np.nan

# Combine features and target; remove incomplete rows
dataset = features.copy()
dataset["target"] = target

dataset = dataset.replace(
    [np.inf, -np.inf], np.nan
).dropna()

if len(dataset) < 100:
    raise ValueError(
        f"Only {len(dataset)} complete rows found. "
        "More usable data is needed."
    )

X = dataset[features.columns]
y = dataset["target"].astype(int)


# Chronological split with a 24-hour gap.
# The gap prevents future-window target overlap
# between training and testing samples.

split_index = int(len(dataset) * 0.80)
gap = HORIZON_HOURS

train_end = split_index - gap

if train_end <= 0 or split_index >= len(dataset):
    raise ValueError("Not enough data for a chronological split with a gap.")

X_train = X.iloc[:train_end]
y_train = y.iloc[:train_end]

X_test = X.iloc[split_index:]
y_test = y.iloc[split_index:]

# --------------------------------------------------
# 6. Train Logistic Regression
# --------------------------------------------------

classifier = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining Logistic Regression...")
classifier.fit(X_train, y_train)

# --------------------------------------------------
# 7. Evaluate on unseen later observations
# --------------------------------------------------

probabilities = classifier.predict_proba(X_test)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)

auc = (
    roc_auc_score(y_test, probabilities)
    if y_test.nunique() == 2
    else None
)

print("\nMODEL EVALUATION")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 score:  {f1:.3f}")

if auc is not None:
    print(f"ROC-AUC:   {auc:.3f}")

print("\nClassification report:")
print(classification_report(
    y_test, predictions, zero_division=0
))

print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
print(confusion_matrix(
    y_test, predictions, labels=[0, 1]
))

# --------------------------------------------------
# 8. Save test predictions and metrics
# --------------------------------------------------

prediction_df = pd.DataFrame({
    "timestamp": X_test.index,
    "actual_spike_next_24h": y_test.to_numpy(),
    "predicted_spike_next_24h": predictions,
    "spike_probability": probabilities
})

prediction_df.to_csv(
    OUTPUT_DIR / "spike_test_predictions.csv",
    index=False
)

metrics = {
    "target_definition": (
        "Any hourly PM2.5 reading above "
        "60 µg/m³ during the next 24 hours"
    ),
    "threshold": THRESHOLD,
    "horizon_hours": HORIZON_HOURS,
    "samples": int(len(dataset)),
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "accuracy": float(accuracy),
    "precision": float(precision),
    "recall": float(recall),
    "f1_score": float(f1),
    "roc_auc": float(auc) if auc is not None else None,
    "confusion_matrix": confusion_matrix(
        y_test, predictions, labels=[0, 1]
    ).tolist()
}

with open(
    OUTPUT_DIR / "spike_model_metrics.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(metrics, file, indent=2)

# Save trained model and required feature names
joblib.dump(
    {
        "model": classifier,
        "feature_columns": list(X.columns),
        "threshold": THRESHOLD,
        "horizon_hours": HORIZON_HOURS
    },
    OUTPUT_DIR / "spike_model.joblib"
)

print("\nSaved files:")
print("- outputs/spike_test_predictions.csv")
print("- outputs/spike_model_metrics.json")
print("- outputs/spike_model.joblib")
print("\nSpike detection experiment completed.")

