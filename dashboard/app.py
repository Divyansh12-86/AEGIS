import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


st.set_page_config(page_title="AEGIS", layout="wide")
st.title("AEGIS — Decision Intelligence for Industrial Infrastructure")

assets = requests.get(f"{API_BASE}/telemetry/assets").json()
asset_ids = [a["asset_id"] for a in assets]

if asset_ids:
    selected = st.selectbox("Select Asset", asset_ids)
    if st.button("Generate Strategies"):
        plans = requests.get(f"{API_BASE}/plans/{selected}").json()
        st.metric("Health Index", f"{plans['health_index']:.1f}%")
        for s in plans["strategies"]:
            with st.expander(f"{s['name']} — {s['status']}"):
                st.write(f"Cost: ${s['estimated_cost']:,.0f} | Risk: {s['risk']} | Downtime: {s['downtime_hours']}h")
                st.write(s["explanation"])
else:
    st.info("No telemetry yet — start the Modbus simulator.")
