from backend.models.asset import AssetState
from backend.services import decision_planner, safety_verification

def make_state(temp=75, voltage=410):
    return AssetState(
        asset_id="test_asset",
        name="Test Asset",
        last_updated="2026-09-04T00:00:00Z",
        temperature_c=temp,
        voltage_v=voltage,
        current_a=250,
        load_pct=85,
    )

def test_safe_conditions_pass_validation():
    state = make_state(temp=70, voltage=410)
    candidates = decision_planner.generate_candidate_strategies(state, health_index=60)
    approved = safety_verification.verify_and_filter(candidates, state)
    assert len(approved) > 0

def test_overheating_rejects_strategies():
    state = make_state(temp=150, voltage=410)  # exceeds THERMAL_LIMIT_C
    candidates = decision_planner.generate_candidate_strategies(state, health_index=20)
    approved = safety_verification.verify_and_filter(candidates, state)
    rejected = [s for s in candidates if s.status == "Rejected"]
    assert len(rejected) > 0
    assert all(s.status != "Rejected" for s in approved)

def test_voltage_out_of_range_rejects_strategies():
    state = make_state(temp=70, voltage=500)  # exceeds VOLTAGE_LIMITS
    candidates = decision_planner.generate_candidate_strategies(state, health_index=60)
    approved = safety_verification.verify_and_filter(candidates, state)
    assert len(approved) == 0
