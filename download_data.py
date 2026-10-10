
import os
import json
import time
import calendar
from pathlib import Path
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENAQ_API_KEY")
if not API_KEY:
    raise ValueError("API key not found. Check your .env file.")

BASE_URL = "https://api.openaq.org/v3/sensors"
HEADERS = {"X-API-Key": API_KEY}

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SENSORS = {
    "pm25": 15195,
    "pm10": 15162,
}

START = datetime(2018, 3, 9, tzinfo=timezone.utc)
END = datetime(2022, 11, 1, tzinfo=timezone.utc)

LIMIT = 300
MAX_RETRIES = 3


def month_ranges():
    """Yield monthly date ranges within START and END."""
    year, month = START.year, START.month

    while True:
        first = datetime(year, month, 1, tzinfo=timezone.utc)

        if first >= END:
            break

        next_month = (
            datetime(year + 1, 1, 1, tzinfo=timezone.utc)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=timezone.utc)
        )

        start = max(first, START)
        end = min(next_month, END)

        yield start, end

        year, month = next_month.year, next_month.month


def save_progress(path, sensor_id, results, completed_months):
    data = {
        "sensor_id": sensor_id,
        "datetime_from": START.isoformat(),
        "datetime_to": END.isoformat(),
        "records_downloaded": len(results),
        "downloaded_months": sorted(completed_months),
        "results": results,
    }

    temp_path = path.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(data, file)

    temp_path.replace(path)


def download_month(sensor_id, start, end):
    all_results = []
    page = 1

    params_base = {
        "datetime_from": start.isoformat().replace("+00:00", "Z"),
        "datetime_to": end.isoformat().replace("+00:00", "Z"),
        "limit": LIMIT,
    }

    while True:
        params = {**params_base, "page": page}
        response = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = requests.get(
                    f"{BASE_URL}/{sensor_id}/hours",
                    headers=HEADERS,
                    params=params,
                    timeout=60,
                )

                if response.status_code == 200:
                    break

                print(
                    f"HTTP {response.status_code} for page {page}; "
                    f"attempt {attempt}/{MAX_RETRIES}"
                )
                print(response.text[:300])

                if response.status_code not in (408, 429, 500, 502, 503, 504):
                    response.raise_for_status()

            except requests.RequestException as exc:
                print(f"Request failed: {exc}")

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)

        if response is None or response.status_code != 200:
            raise RuntimeError(
                f"Failed to download page {page} for "
                f"{start.date()} to {end.date()}"
            )

        data = response.json()
        rows = data.get("results", [])

        if not rows:
            break

        all_results.extend(rows)
        print(f"    Page {page}: {len(rows)} records")

        if len(rows) < LIMIT:
            break

        page += 1
        time.sleep(0.2)

    return all_results


def download_sensor(name, sensor_id):
    output_path = OUTPUT_DIR / f"{name}_hourly_raw.json"

    # Resume only files created by this version of the script.
    results = []
    completed_months = set()

    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))

            if "downloaded_months" in existing:
                results = existing.get("results", [])
                completed_months = set(existing["downloaded_months"])
                print(f"Resuming {name.upper()}: {len(results)} saved records")
            else:
                print("Existing file has no progress information; starting fresh.")
        except (OSError, json.JSONDecodeError):
            print("Existing file could not be read; starting fresh.")

    print(f"\nDownloading {name.upper()} (sensor {sensor_id})")

    for start, end in month_ranges():
        month_key = start.strftime("%Y-%m")

        if month_key in completed_months:
            continue

        print(f"  Month: {month_key}")

        try:
            month_data = download_month(sensor_id, start, end)

            results.extend(month_data)
            completed_months.add(month_key)

            save_progress(
                output_path, sensor_id, results, completed_months
            )

            print(
                f"  Saved progress: {len(results)} total records "
                f"to {output_path}"
            )

        except (RuntimeError, requests.RequestException, ValueError) as exc:
            print(f"  Month {month_key} failed: {exc}")
            print("  Progress so far is saved. Run the script again to retry.")
            return False

    print(f"Finished {name.upper()}: {len(results)} records saved.")
    return True


def main():
    print("AirGuard AI - Monthly OpenAQ Downloader")
    print("=" * 45)

    for name, sensor_id in SENSORS.items():
        download_sensor(name, sensor_id)

    print("\nCheck data/raw for your JSON files.")


if __name__ == "__main__":
    main()