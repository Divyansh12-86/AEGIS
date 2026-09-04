import random
import time
import requests
from datetime import datetime, timezone

API_URL = "http://localhost:8000/telemetry/ingest"
ASSETS = ["transformer_01", "feeder_breaker_02"]

def generate_reading(asset_id: str) -> dict:
    return {
        "asset_id": asset_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "temperature_c": round(random.uniform(50, 95), 1),
        "current_a": round(random.uniform(100, 400), 1),
        "voltage_v": round(random.uniform(370, 445), 1),
        "load_pct": round(random.uniform(40, 100), 1),
    }

if __name__ == "__main__":
    while True:
        for asset_id in ASSETS:
            reading = generate_reading(asset_id)
            requests.post(API_URL, json=reading)
        time.sleep(3)
