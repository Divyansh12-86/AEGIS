from backend.models.asset import AssetState
from backend.services import health_intelligence

def make_state(temp=50, load=50):
    return AssetState(
        asset_id="test_asset",
        name="Test Asset",
        last_updated="2026-09-04T00:00:00Z",
        temperature_c=temp,
        voltage_v=400,
        current_a=200,
        load_pct=load,
    )

def test_health_index_full_score_at_low_temp_load():
    state = make_state(temp=50, load=50)
    assert health_intelligence.compute_health_index(state) == 100.0

def test_health_index_penalizes_high_temperature():
    state = make_state(temp=90, load=50)
    score = health_intelligence.compute_health_index(state)
    assert score < 100.0

def test_health_index_bounded_between_0_and_100():
    state = make_state(temp=200, load=200)
    score = health_intelligence.compute_health_index(state)
    assert 0.0 <= score <= 100.0

def test_failure_probability_inverse_of_health():
    assert health_intelligence.estimate_failure_probability(100) == 0.0
    assert health_intelligence.estimate_failure_probability(0) == 1.0

def test_rul_days_scales_with_health():
    low = health_intelligence.estimate_rul_days(10)
    high = health_intelligence.estimate_rul_days(90)
    assert high > low
