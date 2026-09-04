from fastapi import APIRouter, HTTPException
from backend.services import digital_twin, health_intelligence, decision_planner, strategy_engine, safety_verification, nl_explainer
from backend.core.audit_log import log_decision

router = APIRouter(prefix="/plans", tags=["plans"])

@router.get("/{asset_id}")
def get_plans(asset_id: str):
    state = digital_twin.get_asset_state(asset_id)
    if not state:
        raise HTTPException(404, "Asset not found")

    health = health_intelligence.compute_health_index(state)
    candidates = decision_planner.generate_candidate_strategies(state, health)
    safe = safety_verification.verify_and_filter(candidates, state)
    ranked = strategy_engine.rank_strategies(safe)

    result = [
        {**s.model_dump(), "explanation": nl_explainer.explain_strategy(s)}
        for s in ranked
    ]

    log_decision({"asset_id": asset_id, "health_index": health, "strategies": result})
    return {"asset_id": asset_id, "health_index": health, "strategies": result}
