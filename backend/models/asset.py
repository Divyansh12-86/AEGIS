from pydantic import BaseModel
from typing import Optional
from backend.models.telemetry import TelemetryReading

class AssetState(BaseModel):
    asset_id: str
    name: str
    last_updated: str
    temperature_c: float
    voltage_v: float
    current_a: float
    load_pct: float
    feeder_id: Optional[str] = None
