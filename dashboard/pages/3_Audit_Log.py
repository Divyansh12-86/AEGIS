import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="AEGIS — Audit Log", layout="wide")
st.title("Decision Audit Log")

LOG_PATH = Path("data/audit_log.jsonl")

if not LOG_PATH.exists():
    st.info("No decisions logged yet. Generate strategies from the Strategy Explorer first.")
else:
    entries = []
    with open(LOG_PATH) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))

    entries.reverse()  # most recent first

    st.caption(f"{len(entries)} logged decision(s)")

    for entry in entries:
        with st.expander(f"{entry.get('timestamp', 'unknown time')} — {entry.get('asset_id', 'unknown asset')}"):
            st.json(entry)
