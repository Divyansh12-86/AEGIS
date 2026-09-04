from fastapi import FastAPI
from backend.api import routes_telemetry, routes_plans, routes_health, routes_validation, routes_explain

app.include_router(routes_health.router)
app.include_router(routes_validation.router)
app.include_router(routes_explain.router)


app = FastAPI(title="AEGIS", version="1.0")

app.include_router(routes_telemetry.router)
app.include_router(routes_plans.router)

@app.get("/health")
def health_check():
    return {"status": "ok"}
