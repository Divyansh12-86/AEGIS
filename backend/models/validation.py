from pydantic import BaseModel

class ValidationResult(BaseModel):
    strategy_id: str
    approved: bool
    voltage_ok: bool
    thermal_ok: bool
    topology_ok: bool
    schedule_ok: bool
    regulatory_ok: bool
    reasons: list[str]
