
import json
import urllib.request
import urllib.error
import os
from pathlib import Path

LOCATION_ID = 5635
API_URL = f"https://api.openaq.org/v3/locations/{LOCATION_ID}/sensors"

# Read the key from an environment variable
api_key = os.environ.get("OPENAQ_API_KEY", "").strip()

if not api_key:
    # Ask through a normal input prompt, not hidden input
    api_key = input("Paste your OpenAQ API key here: ").strip()

if not api_key:
    print("No API key entered. Please try again.")
    raise SystemExit(1)

request = urllib.request.Request(
    API_URL,
    headers={"X-API-Key": api_key, "Accept": "application/json"}
)

try:
    print("Connecting to OpenAQ...")

    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read().decode("utf-8"))

    sensors = data.get("results", [])

    if not sensors:
        print("No sensors returned. Check the location ID.")
    else:
        output_file = Path("data") / "raw" / "jaipur_sensors.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            json.dumps(sensors, indent=2),
            encoding="utf-8"
        )

        print(f"\nFound {len(sensors)} sensors:")
        for sensor in sensors:
            parameter = sensor.get("parameter") or {}
            coverage = sensor.get("coverage") or {}
            print(
                f"ID: {sensor.get('id')} | "
                f"Parameter: {parameter.get('name')} | "
                f"Units: {parameter.get('units')} | "
                f"Coverage: {coverage}"
            )

        print(f"\nSensor details saved to: {output_file}")

except urllib.error.HTTPError as error:
    print(f"HTTP error: {error.code}")
    print("Check your API key and OpenAQ access.")
except urllib.error.URLError as error:
    print(f"Connection error: {error.reason}")
except Exception as error:
    print(f"Error: {type(error).__name__}: {error}")