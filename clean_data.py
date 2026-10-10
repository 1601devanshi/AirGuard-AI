
import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "jaipur_air_quality_hourly.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pollutant(filename, column):
    path = RAW_DIR / filename

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    rows = []

    for record in payload.get("results", []):
        timestamp = (
            record.get("period", {})
            .get("datetimeFrom", {})
            .get("utc")
        )
        value = record.get("value")

        if timestamp is None or value is None:
            continue

        try:
            timestamp = pd.to_datetime(timestamp, utc=True)
            value = float(value)
        except (ValueError, TypeError):
            continue

        if value < 0:
            continue

        rows.append({"timestamp": timestamp, column: value})

    df = pd.DataFrame(rows)

    if df.empty:
        raise ValueError(f"No usable records in {filename}")

    duplicates = df.duplicated(subset="timestamp").sum()

    # Average duplicate observations at the same timestamp.
    df = df.groupby("timestamp", as_index=False)[column].mean()
    df = df.set_index("timestamp").sort_index()

    print(f"\n{column.upper()}")
    print(f"Records in raw file: {len(payload.get('results', []))}")
    print(f"Usable unique timestamps: {len(df)}")
    print(f"Duplicate rows found: {duplicates}")
    print(f"First UTC timestamp: {df.index.min()}")
    print(f"Last UTC timestamp: {df.index.max()}")

    return df


def main():
    pm25 = load_pollutant("pm25_hourly_raw.json", "pm25")
    pm10 = load_pollutant("pm10_hourly_raw.json", "pm10")

    # Merge the original observation timestamps without dropping readings.
    combined = pd.concat([pm25, pm10], axis=1).sort_index()

    # Convert the observations to Jaipur local time.
    combined.index = (
        combined.index
        .tz_convert("Asia/Kolkata")
        .tz_localize(None)
    )

    # Convert observations to hourly bins.
    # Readings within the same local hour are averaged.
    hourly = combined.resample("h").mean()

    hourly.index.name = "timestamp"
    hourly = hourly.reset_index()

    hourly.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 45)
    print("AIRGUARD AI - CLEANING COMPLETE")
    print("=" * 45)
    print(f"Saved to: {OUTPUT_FILE}")
    print(f"Hourly rows: {len(hourly):,}")
    print(f"PM2.5 valid hourly bins: {hourly['pm25'].notna().sum():,}")
    print(f"PM10 valid hourly bins: {hourly['pm10'].notna().sum():,}")
    print(f"Both pollutants missing: {hourly[['pm25', 'pm10']].isna().all(axis=1).sum():,}")
    print(f"PM2.5 missing: {hourly['pm25'].isna().sum():,}")
    print(f"PM10 missing: {hourly['pm10'].isna().sum():,}")
    print(f"First local timestamp: {hourly['timestamp'].min()}")
    print(f"Last local timestamp: {hourly['timestamp'].max()}")
    print("\nFirst five rows:")
    print(hourly.head().to_string(index=False))


if __name__ == "__main__":
    main()
