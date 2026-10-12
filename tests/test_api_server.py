"""Test suite for EGPM FastAPI Simulation Server."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from scripts.api_server import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_assets_endpoint(client):
    res = client.get("/api/assets")
    assert res.status_code == 200
    data = res.json()
    assert len(data) >= 2
    assert any(a["id"] == "DS01_u2" for a in data)


def test_asset_sensors_endpoint(client):
    res = client.get("/api/assets/DS01_u2/sensors")
    assert res.status_code == 200
    data = res.json()
    assert data["channel_count"] == 18
    assert len(data["sensors"]) == 18
    assert data["sensors"][0]["name"] == "alt"
    assert data["sensors"][6]["name"] == "T50"


def test_scenarios_and_inference(client):
    res = client.get("/api/scenarios")
    assert res.status_code == 200
    scenarios = res.json()
    assert len(scenarios) >= 4

    # Run inference on Scenario A step 0
    inf_res = client.post(
        "/api/inference",
        json={"asset_id": "DS01_u2", "scenario_id": "scenario_a", "step_index": 0},
    )
    assert inf_res.status_code == 200
    step = inf_res.json()["step_data"]
    assert step["state_id"] == 1
    assert step["rul"]["expected"] == pytest.approx(88.5, abs=0.1)
    assert step["decision"]["tier"] == "NOMINAL: STANDARD MONITORING"


def test_unreliable_scenario_fallback(client):
    # Run inference on Scenario D (unreliable edge case)
    inf_res = client.post(
        "/api/inference",
        json={"asset_id": "DS01_u2", "scenario_id": "scenario_d", "step_index": 0},
    )
    assert inf_res.status_code == 200
    step = inf_res.json()["step_data"]
    assert step["is_unreliable"] is True
    assert step["rul"]["suppressed"] is True
    assert "CRITICAL" in step["decision"]["tier"]


def test_experiments_endpoint(client):
    res = client.get("/api/experiments")
    assert res.status_code == 200
    exps = res.json()
    assert len(exps) >= 3

    detail_res = client.get("/api/experiments/raw_plus_structured")
    assert detail_res.status_code == 200
    data = detail_res.json()
    assert "raw_18" in data["data"]
