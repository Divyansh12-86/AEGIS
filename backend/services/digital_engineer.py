from backend.models.asset import AssetState
from backend.models.strategy import Strategy
from backend.models.validation import ValidationResult
from backend.core.constraints import VOLTAGE_LIMITS, THERMAL_LIMIT_C, CRITICAL_FEEDERS

def validate_strategy(strategy: Strategy, state: AssetState) -> ValidationResult:
    reasons = []

    voltage_ok = VOLTAGE_LIMITS["min"] <= state.voltage_v <= VOLTAGE_LIMITS["max"]
    reasons.append("Voltage within limits" if voltage_ok else "Voltage out of range")

    thermal_ok = state.temperature_c <= THERMAL_LIMIT_C
    reasons.append("Temperature below threshold" if thermal_ok else "Temperature exceeds threshold")

    topology_ok = state.feeder_id not in CRITICAL_FEEDERS or strategy.name != "Wait"
    reasons.append("No critical feeder conflict" if topology_ok else "Critical feeder affected by delay")

    schedule_ok = True   # stub: integrate with production schedule
    regulatory_ok = True # stub: integrate IEEE rule checks

    approved = all([voltage_ok, thermal_ok, topology_ok, schedule_ok, regulatory_ok])

    return ValidationResult(
        strategy_id=strategy.id,
        approved=approved,
        voltage_ok=voltage_ok,
        thermal_ok=thermal_ok,
        topology_ok=topology_ok,
        schedule_ok=schedule_ok,
        regulatory_ok=regulatory_ok,
        reasons=reasons,
    )
