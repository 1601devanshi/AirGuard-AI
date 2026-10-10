
from pathlib import Path

import pandas as pd

file_path = Path("data/processed/jaipur_air_quality_hourly.csv")

df = pd.read_csv(file_path, parse_dates=["timestamp"])
df = df.sort_values("timestamp")

print("=" * 55)
print("AIRGUARD AI - DATA QUALITY REPORT")
print("=" * 55)

print("\nDataset shape:", df.shape)
print("First timestamp:", df["timestamp"].min())
print("Last timestamp:", df["timestamp"].max())

print("\nMissing values:")
print(df[["pm25", "pm10"]].isna().sum())

print("\nValid observations:")
print("PM2.5:", df["pm25"].notna().sum())
print("PM10:", df["pm10"].notna().sum())
print("Both pollutants:", df[["pm25", "pm10"]].notna().all(axis=1).sum())

print("\nPollutant statistics:")
print(df[["pm25", "pm10"]].describe().round(2))

print("\nMissing-data percentages:")
print((df[["pm25", "pm10"]].isna().mean() * 100).round(2))

# Examine consecutive hours with no measurements for either pollutant.
both_missing = df[["pm25", "pm10"]].isna().all(axis=1)
group_id = both_missing.ne(both_missing.shift()).cumsum()

gaps = (
    df.loc[both_missing]
    .assign(group=group_id[both_missing].values)
    .groupby("group")
    .agg(
        start=("timestamp", "min"),
        end=("timestamp", "max"),
        missing_hours=("timestamp", "size"),
    )
    .sort_values("missing_hours", ascending=False)
)

print("\n10 longest gaps where BOTH pollutants are missing:")
print(gaps.head(10).to_string(index=False) if not gaps.empty else "No gaps found.")

print("\nReport complete.")

