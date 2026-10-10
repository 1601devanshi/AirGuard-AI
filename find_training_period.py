
from pathlib import Path

import pandas as pd

FILE = Path("data/processed/jaipur_air_quality_hourly.csv")

df = pd.read_csv(FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

for column in ["pm25", "pm10"]:
    print("\n" + "=" * 60)
    print(f"CONTINUOUS DATA ANALYSIS: {column.upper()}")
    print("=" * 60)

    valid = df[column].dropna()

    if valid.empty:
        print("No valid observations.")
        continue

    # Identify continuous hourly runs without missing measurements.
    groups = valid.index.to_series().diff().ne(
        pd.Timedelta(hours=1)
    ).cumsum()

    runs = (
        valid.groupby(groups)
        .agg(
            start=lambda x: x.index.min(),
            end=lambda x: x.index.max(),
            observations="size",
        )
        .reset_index(drop=True)
    )

    runs["duration_days"] = (
        (runs["end"] - runs["start"]).dt.total_seconds() / 86400
    )

    runs = runs.sort_values("observations", ascending=False)

    print("\nTop 10 continuous runs:")
    print(
        runs.head(10).to_string(
            index=False,
            formatters={"duration_days": "{:.1f}".format},
        )
    )

    best = runs.iloc[0]

    print("\nLongest run summary:")
    print("Start:", best["start"])
    print("End:", best["end"])
    print("Valid hourly observations:", int(best["observations"]))
    print("Duration in days:", round(best["duration_days"], 1))

print("\nAnalysis complete.")
