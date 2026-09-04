# AEGIS — Autonomous Engineering Guidance & Intervention System

**The Decision Intelligence Platform for Industrial Electrical Infrastructure**

Built for the Schneider Electric PredictX 2026 Innovation Challenge.

AEGIS sits above predictive maintenance systems (SCADA, EcoStruxure, anomaly detectors) and converts raw telemetry into safe, explainable, economically-optimized operational decisions — instead of just alerts.

```
Telemetry → Digital Twin → Health Intelligence → Decision Planner
   → Safety Verification (Digital Engineer) → Operations Strategy Engine
   → NL Explainer → Operator Approval → Execution
```

## Table of Contents

- [Features](#features)
- [Repository Structure](#repository-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [License](#license)

## Features

- **Digital Twin** — live synchronized state per electrical asset from ingested telemetry
- **Health Intelligence** — Health Index, failure probability, Remaining Useful Life (RUL)
- **Decision Planner** — generates multiple candidate maintenance strategies (never just one)
- **Digital Engineer / Safety Verification** — deterministic validation against voltage, thermal, topology, schedule, and IEEE regulatory constraints; rejects unsafe plans
- **Operations Strategy Engine** — ranks approved plans by cost, risk, downtime, and asset-life impact
- **NL Explainer** — human-readable justification for the recommended plan
- **Audit Log** — every generated decision set persisted for traceability
- **Streamlit Strategy Explorer** — interactive dashboard for operator review and approval

## Repository Structure

```
aegis/
├── README.md
├── .gitignore
├── .env.example
├── docker-compose.yml
├── requirements.txt
│
├── backend/
│   ├── main.py                     # FastAPI app entrypoint
│   ├── config.py                   # settings/env config
│   │
│   ├── api/
│   │   ├── routes_telemetry.py
│   │   ├── routes_health.py
│   │   ├── routes_plans.py
│   │   ├── routes_validation.py
│   │   └── routes_explain.py
│   │
│   ├── services/
│   │   ├── digital_twin.py         # Service 1: ingestion + asset state
│   │   ├── health_intelligence.py  # Service 2: health index, RUL, failure prob
│   │   ├── decision_planner.py     # Service 3: candidate strategy generation
│   │   ├── strategy_engine.py      # cost/risk/downtime/RUL optimization + ranking
│   │   ├── digital_engineer.py     # engineering constraint validator
│   │   ├── safety_verification.py  # deterministic safety gate
│   │   └── nl_explainer.py         # natural language explanation generator
│   │
│   ├── models/
│   │   ├── asset.py                # Asset, AssetState, TelemetryReading
│   │   ├── strategy.py             # Strategy / ValidatedStrategy
│   │   ├── validation.py           # ValidationResult
│   │   └── telemetry.py
│   │
│   ├── core/
│   │   ├── topology.py             # electrical topology graph
│   │   ├── constraints.py          # voltage/thermal/IEEE limits
│   │   └── audit_log.py            # decision audit logging (JSONL)
│   │
│   └── simulator/
│       └── modbus_simulator.py     # synthetic sensor data generator
│
├── dashboard/
│   ├── app.py                      # Streamlit entrypoint
│   ├── pages/
│   │   ├── 1_Overview.py
│   │   ├── 2_Strategy_Explorer.py
│   │   └── 3_Audit_Log.py
│   └── components/
│       ├── strategy_table.py
│       └── health_gauge.py
│
├── tests/
│   ├── test_health_intelligence.py
│   ├── test_decision_planner.py
│   └── test_safety_verification.py
│
└── data/
    └── sample_assets.json
```

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Pydantic |
| Dashboard | Streamlit + Plotly |
| Simulation | Custom Modbus-style synthetic generator |
| Data | In-memory (MVP) → JSONL audit log |
| Packaging | Docker / docker-compose |
| Language | Python 3.11+ |

## Getting Started

### Prerequisites

- Python 3.11+
- pip
- (optional) Docker & docker-compose

### Installation

```bash
git clone https://github.com/<your-org>/aegis.git
cd aegis
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

### Run with Docker

```bash
docker-compose up --build
```

- Backend API → `http://localhost:8000`
- Dashboard → `http://localhost:8501`

### Run Locally (no Docker)

```bash
# Terminal 1 — API
uvicorn backend.main:app --reload --port 8000

# Terminal 2 — Simulator (feeds fake telemetry)
python backend/simulator/modbus_simulator.py

# Terminal 3 — Dashboard
streamlit run dashboard/app.py
```

Visit `http://localhost:8501`.

## API Reference

Base URL: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service liveness check |
| `POST` | `/telemetry/ingest` | Ingest a telemetry reading for an asset |
| `GET` | `/telemetry/assets` | List all known asset states |
| `GET` | `/plans/{asset_id}` | Generate, validate, rank & explain strategies for an asset |

Example:

```bash
curl -X POST http://localhost:8000/telemetry/ingest \
  -H "Content-Type: application/json" \
  -d '{
        "asset_id": "transformer_01",
        "timestamp": "2026-09-04T10:00:00Z",
        "temperature_c": 88.5,
        "current_a": 320.4,
        "voltage_v": 410.2,
        "load_pct": 92.1
      }'

curl http://localhost:8000/plans/transformer_01
```

Interactive Swagger docs: `http://localhost:8000/docs`

## Testing

```bash
pytest tests/ -v
```

## Roadmap

| Version | Focus |
|---|---|
| v1 (MVP) | Digital Twin, Health Intelligence, Decision Planner, Strategy Explorer, Digital Engineer, Safety Verification, Streamlit Dashboard, Modbus Simulator, NL Interface |
| v2 | Fleet-wide optimization, multi-asset coordination, weather-aware planning |
| v3 | Multi-agent Digital Engineers, cross-site optimization, RL-based planning |
| v4 | Autonomous maintenance orchestration, spare-parts optimization, supply-chain integration, native EcoStruxure deployment |

## License

MIT

---

**Owner:** Atharva Mendhulkar
**Challenge:** Schneider Electric PredictX 2026 Innovation Challenge
