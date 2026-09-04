import json
from datetime import datetime, timezone
from pathlib import Path

LOG_PATH = Path("data/audit_log.jsonl")

def log_decision(payload: dict) -> None:
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    LOG_PATH.parent.mkdir(exist_ok=True)
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(payload) + "\n")
