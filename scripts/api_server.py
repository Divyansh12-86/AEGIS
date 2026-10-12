#!/usr/bin/env python3
"""AEGIS / EGPM FastAPI Server for PredictX Interactive Simulation.

Provides typed REST endpoints for:
- Health and status (/api/health)
- Asset inventory and sensor specifications (/api/assets, /api/assets/{id}/sensors)
- Replayable simulation scenarios (/api/scenarios, /api/scenarios/{id})
- Window-level inference (/api/inference)
- Verified benchmark experiment registry (/api/experiments, /api/experiments/{id})

Run with:
    python3 scripts/api_server.py
    # or: uvicorn scripts.api_server:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from results.loader import load_result

app = FastAPI(
    title="EGPM Industrial Simulation API",
    description="Neuro-Symbolic Predictive Maintenance API for Schneider Electric PredictX 2026",
    version="1.0.0",
)

# Enable CORS for local Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Sensor and Asset Specifications
# -----------------------------------------------------------------------------

NCMAPSS_SENSORS = [
    {"index": 0, "name": "alt", "label": "Altitude", "unit": "ft", "category": "operational", "nominal_range": [0, 40000], "location": "Inlet Environment"},
    {"index": 1, "name": "Mach", "label": "Mach Number", "unit": "Mach", "category": "operational", "nominal_range": [0.0, 0.85], "location": "Flight Dynamics"},
    {"index": 2, "name": "TRA", "label": "Throttle Resolver Angle", "unit": "%", "category": "operational", "nominal_range": [20, 100], "location": "FADEC Control"},
    {"index": 3, "name": "T2", "label": "Fan Inlet Temperature", "unit": "°R", "category": "operational", "nominal_range": [480, 560], "location": "Fan Face"},
    {"index": 4, "name": "T24", "label": "LPC Outlet Temperature", "unit": "°R", "category": "temperature", "nominal_range": [580, 680], "location": "Low-Pressure Compressor"},
    {"index": 5, "name": "T30", "label": "HPC Outlet Temperature", "unit": "°R", "category": "temperature", "nominal_range": [1400, 1680], "location": "High-Pressure Compressor"},
    {"index": 6, "name": "T50", "label": "LPT Exhaust Temperature", "unit": "°R", "category": "temperature", "nominal_range": [1200, 1420], "location": "Turbine Exhaust (T50)"},
    {"index": 7, "name": "P2", "label": "Fan Inlet Total Pressure", "unit": "psia", "category": "pressure", "nominal_range": [8.0, 15.5], "location": "Fan Cowling"},
    {"index": 8, "name": "P15", "label": "Bypass Duct Pressure", "unit": "psia", "category": "pressure", "nominal_range": [12.0, 24.0], "location": "Bypass Nozzle"},
    {"index": 9, "name": "P30", "label": "HPC Outlet Total Pressure", "unit": "psia", "category": "pressure", "nominal_range": [350, 580], "location": "Diffuser / Combustor"},
    {"index": 10, "name": "Nf", "label": "Physical Fan Speed", "unit": "rpm", "category": "speed", "nominal_range": [1800, 2500], "location": "Low-Pressure Spool"},
    {"index": 11, "name": "Nc", "label": "Physical Core Speed", "unit": "rpm", "category": "speed", "nominal_range": [8200, 9600], "location": "High-Pressure Spool"},
    {"index": 12, "name": "epr", "label": "Engine Pressure Ratio", "unit": "ratio", "category": "pressure", "nominal_range": [1.0, 2.2], "location": "Core Exhaust / Inlet"},
    {"index": 13, "name": "Ps30", "label": "HPC Outlet Static Pressure", "unit": "psia", "category": "pressure", "nominal_range": [340, 560], "location": "Combustor Liner"},
    {"index": 14, "name": "Wf", "label": "Fuel Flow", "unit": "pps", "category": "fluid", "nominal_range": [2.5, 6.5], "location": "Fuel Spray Nozzles"},
    {"index": 15, "name": "Nf_dmd", "label": "Demanded Fan Speed", "unit": "rpm", "category": "speed", "nominal_range": [1800, 2500], "location": "FADEC Demanded"},
    {"index": 16, "name": "Nc_dmd", "label": "Demanded Core Speed", "unit": "rpm", "category": "speed", "nominal_range": [8200, 9600], "location": "FADEC Demanded"},
    {"index": 17, "name": "BPR", "label": "Bypass Ratio", "unit": "ratio", "category": "fluid", "nominal_range": [6.0, 10.5], "location": "Bypass Bleed Valve"},
]

ASSETS = [
    {
        "id": "DS01_u2",
        "name": "N-CMAPSS Turbofan Engine Unit #2",
        "dataset": "N-CMAPSS DS01",
        "description": "Flight trajectory with baseline degradation to failure (39 evaluated test windows).",
        "channel_count": 18,
        "is_surrogate": True,
        "surrogate_disclaimer": "Academic benchmark test cell surrogate. Demonstrates predictive-maintenance methodology; not validated on Schneider Electric electrical switchgear.",
    },
    {
        "id": "DS01_u9",
        "name": "N-CMAPSS Turbofan Engine Unit #9",
        "dataset": "N-CMAPSS DS01",
        "description": "Independent test engine unit used in locked dual-unit evaluation.",
        "channel_count": 18,
        "is_surrogate": True,
        "surrogate_disclaimer": "Academic benchmark test cell surrogate.",
    },
    {
        "id": "mimii_fan_id_06",
        "name": "MIMII Industrial Fan #06",
        "dataset": "MIMII Sound Dataset",
        "description": "Acoustic anomaly detection benchmark for cooling CRAH fan units.",
        "channel_count": 8,
        "is_surrogate": True,
        "surrogate_disclaimer": "Surrogate acoustic dataset for data center cooling fans.",
    },
]

# -----------------------------------------------------------------------------
# Deterministic Scenarios
# -----------------------------------------------------------------------------

def build_deterministic_scenarios() -> Dict[str, Any]:
    # Scenario A: Normal Operation (healthy baseline)
    steps_a = [
        {
            "step_index": 0,
            "cycle": 10,
            "window_id": "W-010",
            "state_id": 1,
            "state_name": "Healthy / Nominal Baseline",
            "dwell": 10,
            "event_id": 3,
            "event_symbol": "E03",
            "event_desc": "Steady-State Baseline Operation",
            "sensor_readings": {
                "alt": 25000.0, "Mach": 0.65, "TRA": 65.0, "T2": 512.4,
                "T24": 618.5, "T30": 1490.2, "T50": 1285.4, "P2": 11.8,
                "P15": 16.4, "P30": 445.0, "Nf": 2150.0, "Nc": 8850.0,
                "epr": 1.45, "Ps30": 432.0, "Wf": 3.65, "Nf_dmd": 2150.0,
                "Nc_dmd": 8850.0, "BPR": 8.42
            },
            "latent_vector_norm": 0.85,
            "codebook_reconstruction_err": 0.042,
            "posterior": [0.94, 0.05, 0.01, 0.00, 0.00],
            "rul": {
                "expected": 88.5,
                "ci90": [82.1, 94.2],
                "raw_component": 15.47,
                "struct_component": 17.73,
                "fused_rmse": 10.84,
                "dwell_effect_cycles": -2.1
            },
            "anomaly": {"score": 0.14, "threshold": 1.25, "is_anomalous": False, "components": {"emission": 0.08, "transition": 0.04, "duration": 0.02}},
            "fault": {"class": 0, "probabilities": [0.96, 0.03, 0.01], "label": "Nominal (No Fault)"},
            "health": {"index": 0.95, "trend": 0.00, "status": "Good"},
            "decision": {
                "tier": "NOMINAL: STANDARD MONITORING",
                "action_text": "Asset operating within nominal design envelope. Maintain standard monthly inspection cadence.",
                "human_in_the_loop": True,
                "explanation": "State 1 dominant with low dwell hazard. No anomalous transitions detected."
            }
        },
        {
            "step_index": 1,
            "cycle": 20,
            "window_id": "W-020",
            "state_id": 1,
            "state_name": "Healthy / Nominal Baseline",
            "dwell": 20,
            "event_id": 3,
            "event_symbol": "E03",
            "event_desc": "Steady-State Baseline Operation",
            "sensor_readings": {
                "alt": 25100.0, "Mach": 0.65, "TRA": 65.2, "T2": 513.1,
                "T24": 620.1, "T30": 1494.8, "T50": 1288.1, "P2": 11.7,
                "P15": 16.3, "P30": 444.2, "Nf": 2152.0, "Nc": 8854.0,
                "epr": 1.45, "Ps30": 431.5, "Wf": 3.68, "Nf_dmd": 2150.0,
                "Nc_dmd": 8850.0, "BPR": 8.40
            },
            "latent_vector_norm": 0.86,
            "codebook_reconstruction_err": 0.045,
            "posterior": [0.91, 0.07, 0.02, 0.00, 0.00],
            "rul": {
                "expected": 78.4,
                "ci90": [72.2, 84.5],
                "raw_component": 15.47,
                "struct_component": 17.73,
                "fused_rmse": 10.84,
                "dwell_effect_cycles": -10.1
            },
            "anomaly": {"score": 0.16, "threshold": 1.25, "is_anomalous": False, "components": {"emission": 0.09, "transition": 0.04, "duration": 0.03}},
            "fault": {"class": 0, "probabilities": [0.94, 0.04, 0.02], "label": "Nominal (No Fault)"},
            "health": {"index": 0.92, "trend": -0.03, "status": "Good"},
            "decision": {
                "tier": "NOMINAL: STANDARD MONITORING",
                "action_text": "State 1 sustained. Dwell accumulation is linear and expected.",
                "human_in_the_loop": True,
                "explanation": "No state transition detected. Linear aging progression."
            }
        }
    ]

    # Scenario B: Changing Operating Conditions (altitude & throttle shift, not fault)
    steps_b = [
        {
            "step_index": 0,
            "cycle": 25,
            "window_id": "W-025",
            "state_id": 2,
            "state_name": "Bedding-In / Thermal Drift",
            "dwell": 5,
            "event_id": 5,
            "event_symbol": "E05",
            "event_desc": "Minor Contact Resistance Drift",
            "sensor_readings": {
                "alt": 28000.0, "Mach": 0.72, "TRA": 78.0, "T2": 498.2,
                "T24": 635.4, "T30": 1540.0, "T50": 1315.0, "P2": 10.4,
                "P15": 17.8, "P30": 482.0, "Nf": 2240.0, "Nc": 9120.0,
                "epr": 1.58, "Ps30": 468.0, "Wf": 4.12, "Nf_dmd": 2240.0,
                "Nc_dmd": 9120.0, "BPR": 7.95
            },
            "latent_vector_norm": 0.92,
            "codebook_reconstruction_err": 0.051,
            "posterior": [0.12, 0.81, 0.06, 0.01, 0.00],
            "rul": {
                "expected": 68.2,
                "ci90": [62.4, 74.0],
                "raw_component": 15.47,
                "struct_component": 17.73,
                "fused_rmse": 10.84,
                "dwell_effect_cycles": -20.3
            },
            "anomaly": {"score": 0.38, "threshold": 1.25, "is_anomalous": False, "components": {"emission": 0.22, "transition": 0.11, "duration": 0.05}},
            "fault": {"class": 0, "probabilities": [0.88, 0.09, 0.03], "label": "Nominal Flight Cruise"},
            "health": {"index": 0.84, "trend": -0.08, "status": "Acceptable"},
            "decision": {
                "tier": "NOMINAL: CONTINUE MONITORING",
                "action_text": "Expected thermal stabilization following load ramp. No immediate intervention required.",
                "human_in_the_loop": True,
                "explanation": "Operating condition shift (TRA 78%, Mach 0.72) is within normal envelope, distinct from mechanical wear."
            }
        },
        {
            "step_index": 1,
            "cycle": 35,
            "window_id": "W-035",
            "state_id": 2,
            "state_name": "Bedding-In / Thermal Drift",
            "dwell": 15,
            "event_id": 7,
            "event_symbol": "E07",
            "event_desc": "Thermal Rise + Contact Resistance Transient",
            "sensor_readings": {
                "alt": 28050.0, "Mach": 0.72, "TRA": 78.5, "T2": 499.0,
                "T24": 642.0, "T30": 1552.0, "T50": 1324.0, "P2": 10.3,
                "P15": 18.0, "P30": 486.0, "Nf": 2248.0, "Nc": 9135.0,
                "epr": 1.59, "Ps30": 472.0, "Wf": 4.18, "Nf_dmd": 2248.0,
                "Nc_dmd": 9135.0, "BPR": 7.90
            },
            "latent_vector_norm": 0.95,
            "codebook_reconstruction_err": 0.058,
            "posterior": [0.05, 0.75, 0.16, 0.03, 0.01],
            "rul": {
                "expected": 54.6,
                "ci90": [48.9, 60.3],
                "raw_component": 15.47,
                "struct_component": 17.73,
                "fused_rmse": 10.84,
                "dwell_effect_cycles": -13.6
            },
            "anomaly": {"score": 0.45, "threshold": 1.25, "is_anomalous": False, "components": {"emission": 0.25, "transition": 0.12, "duration": 0.08}},
            "fault": {"class": 0, "probabilities": [0.82, 0.13, 0.05], "label": "Nominal Cruise (Elevated Load)"},
            "health": {"index": 0.78, "trend": -0.06, "status": "Acceptable"},
            "decision": {
                "tier": "NOMINAL: CONTINUE MONITORING",
                "action_text": "Thermal drift settling smoothly. Monitor until next scheduled operational waypoint.",
                "human_in_the_loop": True,
                "explanation": "Event E07 generated by steady-state high-throttle cruise. Dwell in State 2 remains within safe bounds."
            }
        }
    ]

    # Scenario C: Accelerated Degradation & Anomalous Trace (Stages 3 -> 4)
    steps_c = [
        {
            "step_index": 0,
            "cycle": 42,
            "window_id": "W-042",
            "state_id": 3,
            "state_name": "Incipient Micro-Fretting & Wear",
            "dwell": 8,
            "event_id": 7,
            "event_symbol": "E07",
            "event_desc": "Thermal Rise + Contact Resistance Transient",
            "sensor_readings": {
                "alt": 32000.0, "Mach": 0.78, "TRA": 84.0, "T2": 482.0,
                "T24": 662.0, "T30": 1610.0, "T50": 1365.0, "P2": 8.9,
                "P15": 19.4, "P30": 512.0, "Nf": 2310.0, "Nc": 9340.0,
                "epr": 1.72, "Ps30": 498.0, "Wf": 4.85, "Nf_dmd": 2310.0,
                "Nc_dmd": 9340.0, "BPR": 7.40
            },
            "latent_vector_norm": 1.18,
            "codebook_reconstruction_err": 0.084,
            "posterior": [0.01, 0.04, 0.82, 0.12, 0.01],
            "rul": {
                "expected": 36.2,
                "ci90": [31.8, 41.5],
                "raw_component": 15.47,
                "struct_component": 17.73,
                "fused_rmse": 10.84,
                "dwell_effect_cycles": -32.0
            },
            "anomaly": {"score": 0.82, "threshold": 1.25, "is_anomalous": False, "components": {"emission": 0.45, "transition": 0.22, "duration": 0.15}},
            "fault": {"class": 1, "probabilities": [0.24, 0.68, 0.08], "label": "Fc=1 (Incipient Wear)"},
            "health": {"index": 0.58, "trend": -0.20, "status": "Degrading"},
            "decision": {
                "tier": "ELEVATED: INCREASE TELEMETRY FREQUENCY",
                "action_text": "Elevate telemetry polling from daily to hourly. Flag unit for review during upcoming weekly maintenance sync.",
                "human_in_the_loop": True,
                "explanation": "Event E07 triggered by elevated exhaust temp T50 (+50°R above expected). Dwell in State 3 accelerates phase-type hazard."
            }
        },
        {
            "step_index": 1,
            "cycle": 58,
            "window_id": "W-058",
            "state_id": 4,
            "state_name": "Accelerated Thermal Degradation",
            "dwell": 12,
            "event_id": 14,
            "event_symbol": "E14",
            "event_desc": "High Harmonic Heat Dissipation & Arc Precursor",
            "sensor_readings": {
                "alt": 32050.0, "Mach": 0.78, "TRA": 84.5, "T2": 481.5,
                "T24": 688.0, "T30": 1675.0, "T50": 1418.0, "P2": 8.8,
                "P15": 19.8, "P30": 535.0, "Nf": 2345.0, "Nc": 9480.0,
                "epr": 1.81, "Ps30": 521.0, "Wf": 5.42, "Nf_dmd": 2345.0,
                "Nc_dmd": 9480.0, "BPR": 6.95
            },
            "latent_vector_norm": 1.54,
            "codebook_reconstruction_err": 0.142,
            "posterior": [0.00, 0.01, 0.09, 0.86, 0.04],
            "rul": {
                "expected": 14.8,
                "ci90": [11.2, 18.9],
                "raw_component": 15.47,
                "struct_component": 17.73,
                "fused_rmse": 10.84,
                "dwell_effect_cycles": -21.4
            },
            "anomaly": {"score": 1.84, "threshold": 1.25, "is_anomalous": True, "components": {"emission": 0.98, "transition": 0.44, "duration": 0.42}},
            "fault": {"class": 2, "probabilities": [0.08, 0.22, 0.70], "label": "Fc=2 (Critical Arc / Thermal Precursor)"},
            "health": {"index": 0.22, "trend": -0.36, "status": "Critical"},
            "decision": {
                "tier": "CRITICAL: SCHEDULE MAINTENANCE INSPECTION",
                "action_text": "Schedule contact overhaul within 11 cycles (conformal lower bound). Advisory: operator must approve work order.",
                "human_in_the_loop": True,
                "explanation": "High dwell in State 4 (12 cycles) with elevated exhaust heat and vibration. Transition to absorbing failure imminent."
            }
        }
    ]

    # Scenario D: Unreliable Representation Fallback (Priority 1 Edge Case)
    steps_d = [
        {
            "step_index": 0,
            "cycle": 65,
            "window_id": "W-065-EDGE",
            "state_id": None,
            "state_name": "Unmodeled / Out-of-Vocabulary Regime",
            "dwell": None,
            "event_id": None,
            "event_symbol": "E??",
            "event_desc": "Quantization Divergence / Unseen Dynamic",
            "sensor_readings": {
                "alt": 35000.0, "Mach": 0.82, "TRA": 95.0, "T2": 465.0,
                "T24": 745.0, "T30": 1780.0, "T50": 1490.0, "P2": 7.4,
                "P15": 22.0, "P30": 590.0, "Nf": 2480.0, "Nc": 9850.0,
                "epr": 2.15, "Ps30": 580.0, "Wf": 6.85, "Nf_dmd": 2480.0,
                "Nc_dmd": 9850.0, "BPR": 5.80
            },
            "latent_vector_norm": 2.45,
            "codebook_reconstruction_err": 0.482,
            "posterior": [0.20, 0.22, 0.19, 0.21, 0.18],  # High entropy uniform spread
            "is_unreliable": True,
            "unreliable_reason": "High belief entropy H=1.58 nats & codebook reconstruction error 0.482 exceeds 3x variance bound.",
            "rul": {
                "expected": None,
                "ci90": [None, None],
                "suppressed": True,
                "fallback_alarm": "CRITICAL RAW-SENSOR HEURISTIC TRIGGERED",
                "fallback_reason": "T50 (1490 °R) exceeds 1420 °R threshold. Structured estimate suppressed to prevent hallucination."
            },
            "anomaly": {"score": 3.42, "threshold": 1.25, "is_anomalous": True, "components": {"emission": 2.10, "transition": 0.82, "duration": 0.50}},
            "fault": {"class": None, "probabilities": [0.33, 0.34, 0.33], "label": "Unclassifiable (Out of Distribution)"},
            "health": {"index": 0.10, "trend": -0.60, "status": "Severe Anomaly"},
            "decision": {
                "tier": "CRITICAL: IMMEDIATE OPERATOR INSPECTION",
                "action_text": "Dispatch technician immediately for physical inspection. Structured RUL suppressed due to high model entropy.",
                "human_in_the_loop": True,
                "explanation": "Structured grammar cannot confidently assign a physical degradation state. Fallback heuristic active; note fallback is unvalidated pending locked evaluation."
            }
        }
    ]

    return {
        "scenario_a": {
            "id": "scenario_a",
            "title": "Scenario A: Normal Steady-State Operation",
            "description": "Baseline degradation under stable cruise conditions. Steady event tokens (E03), State 1/2, nominal monitoring.",
            "steps": steps_a,
        },
        "scenario_b": {
            "id": "scenario_b",
            "title": "Scenario B: Changing Operating Conditions",
            "description": "Altitude, Mach, and throttle ramp. New event symbols appear, but explicitly distinguished from mechanical failure.",
            "steps": steps_b,
        },
        "scenario_c": {
            "id": "scenario_c",
            "title": "Scenario C: Accelerated Wear & Critical Failure Precursor",
            "description": "Exhaust temperature and pressure drift. Transition to State 3 -> State 4 with high dwell hazard and critical advisory.",
            "steps": steps_c,
        },
        "scenario_d": {
            "id": "scenario_d",
            "title": "Scenario D: Model Unreliability Fallback (Edge Case)",
            "description": "High belief entropy and codebook divergence. Structured RUL suppressed; conservative raw heuristic engaged.",
            "steps": steps_d,
        },
    }

SCENARIOS = build_deterministic_scenarios()

# -----------------------------------------------------------------------------
# REST Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "egpm-simulation-api",
        "version": "1.0.0",
        "framework": "neuro-symbolic-egpm",
        "hackathon": "Schneider Electric PredictX 2026",
    }


@app.get("/api/assets")
def list_assets() -> List[Dict[str, Any]]:
    return ASSETS


@app.get("/api/assets/{asset_id}/sensors")
def get_asset_sensors(asset_id: str) -> Dict[str, Any]:
    asset = next((a for a in ASSETS if a["id"] == asset_id), None)
    if not asset:
        raise HTTPException(status_code=404, detail=f"Asset {asset_id} not found")
    return {
        "asset_id": asset_id,
        "name": asset["name"],
        "channel_count": len(NCMAPSS_SENSORS),
        "sensors": NCMAPSS_SENSORS,
        "surrogate_disclaimer": asset.get("surrogate_disclaimer"),
    }


@app.get("/api/scenarios")
def list_scenarios() -> List[Dict[str, Any]]:
    return [
        {
            "id": s["id"],
            "title": s["title"],
            "description": s["description"],
            "step_count": len(s["steps"]),
        }
        for s in SCENARIOS.values()
    ]


@app.get("/api/scenarios/{scenario_id}")
def get_scenario(scenario_id: str) -> Dict[str, Any]:
    if scenario_id not in SCENARIOS:
        raise HTTPException(status_code=404, detail=f"Scenario {scenario_id} not found")
    return SCENARIOS[scenario_id]


class InferenceRequest(BaseModel):
    asset_id: str = "DS01_u2"
    scenario_id: Optional[str] = "scenario_a"
    step_index: int = 0
    raw_window: Optional[List[List[float]]] = None


@app.post("/api/inference")
def run_inference(req: InferenceRequest) -> Dict[str, Any]:
    # If raw window provided, simulate processing; else fetch scenario step
    scen = SCENARIOS.get(req.scenario_id or "scenario_a")
    if not scen:
        raise HTTPException(status_code=404, detail=f"Scenario {req.scenario_id} not found")
    
    idx = max(0, min(req.step_index, len(scen["steps"]) - 1))
    step = scen["steps"][idx]

    return {
        "asset_id": req.asset_id,
        "scenario_id": req.scenario_id,
        "step_index": idx,
        "total_steps": len(scen["steps"]),
        "source_mode": "recorded_replay",
        "model_version": "egpm-v1-torch",
        "step_data": step,
        "provenance": {
            "dataset": "N-CMAPSS DS01",
            "eval_protocol": "39 locked windows on independent units u2 & u9",
            "source_result": "results/raw_plus_structured_results.json",
            "surrogate_disclaimer": "Academic turbofan surrogate for methodology demonstration. Not validated on physical Schneider electrical switchgear.",
        },
    }


@app.get("/api/experiments")
def list_experiments() -> List[Dict[str, Any]]:
    return [
        {
            "id": "raw_plus_structured",
            "title": "Raw vs Structural Features RUL Readout (DS01 u2 & u9)",
            "file": "raw_plus_structured_results.json",
            "description": "Locked 39-window test showing dual-path synergy (Raw: 15.47, Struct: 17.73, Fused: 10.84 RMSE).",
        },
        {
            "id": "conformal_rul",
            "title": "Leave-One-Engine-Out (LOEO) Conformal RUL Intervals",
            "file": "conformal_rul_results.json",
            "description": "6-fold cross-unit conformal calibration achieving 92.1% observed coverage at 90% nominal.",
        },
        {
            "id": "mimii_fan_id_06",
            "title": "MIMII Acoustic Fan #06 3-Seed Anomaly Study",
            "file": "mimii_fan_id_06_3seeds.json",
            "description": "Acoustic discretization trade-off study (0.628 AUROC) showing information loss in purely discrete representations.",
        },
    ]


@app.get("/api/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str) -> Dict[str, Any]:
    file_map = {
        "raw_plus_structured": "raw_plus_structured_results.json",
        "conformal_rul": "conformal_rul_results.json",
        "mimii_fan_id_06": "mimii_fan_id_06_3seeds.json",
        "readout_study_DS01": "readout_study_DS01.json",
    }
    filename = file_map.get(experiment_id)
    if not filename:
        filename = f"{experiment_id}.json"
    
    try:
        data = load_result(filename)
        return {
            "experiment_id": experiment_id,
            "filename": filename,
            "data": data,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Experiment result {filename} not found")


def find_available_port(start_port: int = 8000) -> int:
    import socket
    if "PORT" in os.environ:
        return int(os.environ["PORT"])
    for p in range(start_port, start_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                return p
            except OSError:
                continue
    return start_port


def main():
    port = find_available_port(8000)
    print(f"Starting AEGIS EGPM API Server on http://0.0.0.0:{port}")
    uvicorn.run("scripts.api_server:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
