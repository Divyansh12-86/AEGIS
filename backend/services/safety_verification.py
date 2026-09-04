from backend.models.strategy import Strategy
from backend.models.asset import AssetState
from backend.services.digital_engineer import validate_strategy

def verify_and_filter(strategies: list[Strategy], state: AssetState) -> list[Strategy]:
    approved = []
    for s in strategies:
        result = validate_strategy(s, state)
        if result.approved:
            approved.append(s)
        else:
            s.status = "Rejected"
    return approved
