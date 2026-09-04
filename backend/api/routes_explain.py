from fastapi import APIRouter, HTTPException
from backend.services import digital_twin, decision_planner, health_intelligence, nl_explainer

router = APIRouter(prefix="/explain", tags=["explain"])

@router.get("/{asset_id}/{strategy_name}")
def explain(asset_id: str, strategy_name: str):
    state = digital_twin.get_asset_state(asset_id)
    if not state:
        raise HTTPException(404, "Asset not found")

    health = health_intelligence.compute_health_index(state)
    candidates = decision_planner.generate_candidate_strategies(state, health)

    match = next((s for s in candidates if s.name.lower() == strategy_name.lower()), None)
    if not match:
        raise HTTPException(404, "Strategy not found for this asset")

    return {"strategy": match.model_dump(), "explanation": nl_explainer.explain_strategy(match)}
