from fastapi import APIRouter, HTTPException
from backend.services import digital_twin, health_intelligence

router = APIRouter(prefix="/health-intel", tags=["health"])

@router.get("/{asset_id}")
def get_health(asset_id: str):
    state = digital_twin.get_asset_state(asset_id)
    if not state:
        raise HTTPException(404, "Asset not found")

    index = health_intelligence.compute_health_index(state)
    return {
        "asset_id": asset_id,
        "health_index": index,
        "failure_probability": health_intelligence.estimate_failure_probability(index),
        "rul_days": health_intelligence.estimate_rul_days(index),
    }
