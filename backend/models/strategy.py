from pydantic import BaseModel
from typing import Literal

RiskLevel = Literal["None", "Low", "Medium", "High", "Very High"]

class Strategy(BaseModel):
    id: str
    asset_id: str
    name: str                     # e.g. "Replace", "Load Reduction"
    description: str
    estimated_cost: float
    risk: RiskLevel
    downtime_hours: float
    rul_gain_days: float
    status: Literal["Available", "Recommended", "Critical", "Rejected"] = "Available"

class ValidatedStrategy(Strategy):
    approved: bool
    validation_reasons: list[str] = []
