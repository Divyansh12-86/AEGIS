from fastapi import APIRouter, HTTPException
from backend.services import digital_twin, decision_planner, health_intelligence, digital_engineer

router = APIRouter(prefix="/validation", tags=["validation"])

@router.get("/{asset_id}")
def validate_all_strategies(asset_id: str):
    state = digital_twin.get_asset_state(asset_id)
    if not state:
        raise HTTPException(404, "Asset not found")

    health = health_intelligence.compute_health_index(state)
    candidates = decision_planner.generate_candidate_strategies(state, health)

    results = [digital_engineer.validate_strategy(s, state) for s in candidates]
    return {"asset_id": asset_id, "validations": [r.model_dump() for r in results]}
