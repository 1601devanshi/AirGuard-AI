
from pathlib import Path

import pandas as pd

FILE = Path("data/processed/jaipur_air_quality_hourly.csv")
df = pd.read_csv(FILE, parse_dates=["timestamp"])
df = df.sort_values("timestamp").set_index("timestamp")

# Look for windows that contain at least 80% valid PM2.5 readings.
WINDOWS = [7, 14, 30, 60]

for days in WINDOWS:
    hours = days * 24
    rolling_valid = df["pm25"].notna().rolling(hours).sum()

    best_end = rolling_valid.idxmax()
    best_count = int(rolling_valid.max())

    if pd.isna(best_end):
        continue

    best_start = best_end - pd.Timedelta(hours=hours - 1)
    completeness = best_count / hours * 100

    window = df.loc[best_start:best_end]

    print("\n" + "=" * 50)
    print(f"BEST {days}-DAY WINDOW")
    print("=" * 50)
    print("Start:", best_start)
    print("End:", best_end)
    print(f"Valid PM2.5 hours: {best_count}/{hours}")
    print(f"PM2.5 completeness: {completeness:.2f}%")
    print(f"Valid PM10 hours: {window['pm10'].notna().sum()}/{len(window)}")
    print(f"Both pollutants available: {window[['pm25', 'pm10']].notna().all(axis=1).sum()}/{len(window)}")

print("\nWindow analysis complete.")
