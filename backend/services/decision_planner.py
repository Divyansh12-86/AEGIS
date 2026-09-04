from backend.models.asset import AssetState
from backend.models.strategy import Strategy
import uuid

def generate_candidate_strategies(state: AssetState, health_index: float) -> list[Strategy]:
    strategies = [
        Strategy(
            id=str(uuid.uuid4()), asset_id=state.asset_id, name="Replace",
            description="Replace the asset immediately.",
            estimated_cost=50000, risk="Very Low", downtime_hours=8, rul_gain_days=365,
        ),
        Strategy(
            id=str(uuid.uuid4()), asset_id=state.asset_id, name="Load Reduction",
            description="Reduce feeder load by 15% to lower thermal stress.",
            estimated_cost=2000, risk="Low", downtime_hours=0, rul_gain_days=30,
        ),
        Strategy(
            id=str(uuid.uuid4()), asset_id=state.asset_id, name="Load Transfer",
            description="Transfer load to an adjacent feeder.",
            estimated_cost=8000, risk="Medium", downtime_hours=1, rul_gain_days=45,
        ),
        Strategy(
            id=str(uuid.uuid4()), asset_id=state.asset_id, name="Wait",
            description="Delay maintenance; monitor closely.",
            estimated_cost=0, risk="High", downtime_hours=0, rul_gain_days=0,
        ),
    ]
    return strategies
