"""Acceptance tests for demo_predictx.py (PredictX Priorities 1, 2, and 3).

Verifies that the interactive CLI demonstration runs cleanly, produces
valid structured outputs, correctly maps 3-tier advisory maintenance decisions,
and gracefully triggers fallback alerts when the structured model is unreliable.
"""
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

# Ensure scripts directory is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.demo_predictx import run_demo, run_trajectory, run_unreliable_fallback


def test_single_snapshot_demo_execution():
    """Verify single snapshot demo runs without error and outputs core sections."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_demo(asset_type="switchgear", cycle=42)
    output = buf.getvalue()

    assert "AEGIS (EGPM)" in output
    assert "[1] Causal Telemetry" in output
    assert "[2] Discovered Event" in output
    assert "[3] Temporal Grammar" in output
    assert "[4] Conformal RUL" in output
    assert "[5] Causal Diagnosis" in output
    assert "[6] Advisory Decision" in output
    assert "Counterfactual Simulation CF1" in output


def test_trajectory_progression_execution():
    """Priority 1 & 2: Verify multi-stage progression across States 1 -> 4."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_trajectory()
    output = buf.getvalue()

    # Verify physical progression through 4 sequential states
    assert "Regime: State 1 (Healthy / Nominal Operation)" in output
    assert "Regime: State 2 (Bedding-In / Thermal Drift)" in output
    assert "Regime: State 3 (Incipient Micro-Fretting & Wear)" in output
    assert "Regime: State 4 (Accelerated Thermal Degradation)" in output

    # Priority 2: Verify 3-tier advisory decision progression
    assert "[NOMINAL: STANDARD MONITORING]" in output
    assert "[ELEVATED: INCREASE TELEMETRY FREQUENCY]" in output
    assert "[CRITICAL: SCHEDULE MAINTENANCE INSPECTION]" in output

    # Verify explicit duration dwell effect on RUL
    assert "due to dwell" in output
    assert "Conformal RUL" in output


def test_unreliable_structured_fallback():
    """Priority 1: Verify system response when structured representation is unreliable."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        run_unreliable_fallback()
    output = buf.getvalue()

    assert "[STRUCTURED_REPRESENTATION_UNRELIABLE]" in output
    assert "SUPPRESSES UNRELIABLE STRUCTURED RUL ESTIMATE" in output
    assert "ACTIVATES CONSERVATIVE RAW-SENSOR HEURISTIC ALARM" in output
    assert "HIGH-PRIORITY MANUAL INSPECTION REQUIRED" in output
    assert "Fallback threshold heuristic is provisional and advisory" in output
