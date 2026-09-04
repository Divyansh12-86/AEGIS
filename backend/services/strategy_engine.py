from backend.models.strategy import Strategy

WEIGHTS = {"cost": 0.3, "risk": 0.35, "downtime": 0.2, "rul_gain": 0.15}

RISK_SCORE = {"None": 0, "Low": 1, "Medium": 2, "High": 3, "Very High": 4}

def score_strategy(s: Strategy) -> float:
    cost_term = -s.estimated_cost / 1000
    risk_term = -RISK_SCORE[s.risk] * 10
    downtime_term = -s.downtime_hours * 2
    rul_term = s.rul_gain_days * 0.5
    return (
        WEIGHTS["cost"] * cost_term
        + WEIGHTS["risk"] * risk_term
        + WEIGHTS["downtime"] * downtime_term
        + WEIGHTS["rul_gain"] * rul_term
    )

def rank_strategies(strategies: list[Strategy]) -> list[Strategy]:
    ranked = sorted(strategies, key=score_strategy, reverse=True)
    if ranked:
        ranked[0].status = "Recommended"
    return ranked
