from fastapi import APIRouter
from backend.models.asset import TelemetryReading
from backend.services import digital_twin

router = APIRouter(prefix="/telemetry", tags=["telemetry"])

@router.post("/ingest")
def ingest(reading: TelemetryReading):
    return digital_twin.ingest_telemetry(reading)

@router.get("/assets")
def list_assets():
    return digital_twin.list_asset_states()
