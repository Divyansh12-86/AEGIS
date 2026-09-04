from backend.models.asset import TelemetryReading, AssetState

# in-memory store for MVP; swap for a DB later
_asset_states: dict[str, AssetState] = {}

def ingest_telemetry(reading: TelemetryReading) -> AssetState:
    state = AssetState(
        asset_id=reading.asset_id,
        name=_asset_states.get(reading.asset_id, AssetState).name if reading.asset_id in _asset_states else reading.asset_id,
        last_updated=reading.timestamp,
        temperature_c=reading.temperature_c,
        voltage_v=reading.voltage_v,
        current_a=reading.current_a,
        load_pct=reading.load_pct,
    )
    _asset_states[reading.asset_id] = state
    return state

def get_asset_state(asset_id: str) -> AssetState | None:
    return _asset_states.get(asset_id)

def list_asset_states() -> list[AssetState]:
    return list(_asset_states.values())
