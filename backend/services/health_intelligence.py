from backend.models.asset import AssetState

def compute_health_index(state: AssetState) -> float:
    """Toy scoring — replace with trained model / physics-informed model."""
    temp_penalty = max(0, state.temperature_c - 60) * 1.5
    load_penalty = max(0, state.load_pct - 80) * 0.8
    score = 100 - temp_penalty - load_penalty
    return max(0.0, min(100.0, score))

def estimate_failure_probability(health_index: float) -> float:
    return round(max(0.0, (100 - health_index) / 100), 2)

def estimate_rul_days(health_index: float) -> int:
    return int(max(0, (health_index / 100) * 90))
