from pydantic import BaseModel
from typing import Optional

class TelemetryReading(BaseModel):
    asset_id: str
    timestamp: str
    temperature_c: float
    current_a: float
    voltage_v: float
    partial_discharge_pc: Optional[float] = None
    load_pct: float

class TelemetryBatch(BaseModel):
    readings: list[TelemetryReading]
