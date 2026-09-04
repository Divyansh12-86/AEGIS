from backend.models.asset import AssetState
from backend.services import decision_planner

def make_state():
    return AssetState(
        asset_id="test_asset",
        name="Test Asset",
        last_updated="2026-09-04T00:00:00Z",
        temperature_c=75,
        voltage_v=410,
        current_a=250,
        load_pct=85,
    )

def test_generates_multiple_strategies():
    state = make_state()
    strategies = decision_planner.generate_candidate_strategies(state, health_index=55)
    assert len(strategies) >= 3

def test_all_strategies_reference_correct_asset():
    state = make_state()
    strategies = decision_planner.generate_candidate_strategies(state, health_index=55)
    assert all(s.asset_id == state.asset_id for s in strategies)

def test_strategy_names_are_unique():
    state = make_state()
    strategies = decision_planner.generate_candidate_strategies(state, health_index=55)
    names = [s.name for s in strategies]
    assert len(names) == len(set(names))
