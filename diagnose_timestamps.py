
import json
from collections import Counter
from pathlib import Path

import pandas as pd

for filename in [
    "pm25_hourly_raw.json",
    "pm10_hourly_raw.json",
]:
    path = Path("data/raw") / filename

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    timestamps = []

    for record in data.get("results", []):
        timestamp = (
            record.get("period", {})
            .get("datetimeFrom", {})
            .get("utc")
        )

        if timestamp:
            timestamps.append(pd.to_datetime(timestamp, utc=True))

    timestamps = pd.DatetimeIndex(timestamps)
    local_times = timestamps.tz_convert("Asia/Kolkata")

    print("\n" + "=" * 55)
    print(filename)
    print("=" * 55)
    print("Raw timestamps:", len(timestamps))
    print("Unique timestamps:", timestamps.nunique())

    print("\nUTC minute distribution:")
    print(Counter(timestamps.minute).most_common())

    print("\nLocal-time minute distribution:")
    print(Counter(local_times.minute).most_common())

    # Check how many readings match the current hourly grid.
    grid = pd.date_range(
        start=timestamps.min(),
        end=timestamps.max(),
        freq="h",
    )

    matching = timestamps.isin(grid).sum()

    print("\nReadings matching current UTC hourly grid:", matching)
    print("Readings not matching that grid:", len(timestamps) - matching)

print("\nTimestamp diagnosis complete.")
