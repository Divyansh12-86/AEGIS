from backend.models.strategy import Strategy

def explain_strategy(strategy: Strategy) -> str:
    return (
        f"{strategy.name} is recommended because it addresses the asset's degradation "
        f"with an estimated cost of ${strategy.estimated_cost:,.0f}, {strategy.risk.lower()} risk, "
        f"and an expected remaining-useful-life gain of {strategy.rul_gain_days} days, "
        f"while satisfying all engineering and safety constraints."
    )
