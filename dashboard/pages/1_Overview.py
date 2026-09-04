import streamlit as st
import requests
from dashboard.components.health_gauge import render_health_gauge
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AEGIS — Overview", layout="wide")
st.title("Asset Overview")

assets = requests.get(f"{API_BASE}/telemetry/assets").json()

if not assets:
    st.info("No telemetry yet. Start the Modbus simulator to populate asset data.")
else:
    cols = st.columns(min(3, len(assets)))
    for i, asset in enumerate(assets):
        with cols[i % len(cols)]:
            st.subheader(asset["asset_id"])
            health = requests.get(f"{API_BASE}/health-intel/{asset['asset_id']}").json()
            render_health_gauge(health["health_index"], title=asset["asset_id"])
            st.caption(f"Failure risk: {health['failure_probability']*100:.0f}% | RUL: {health['rul_days']} days")
