import plotly.graph_objects as go
import streamlit as st

def render_health_gauge(health_index: float, title: str = "Health Index"):
    color = "#2ecc71" if health_index >= 70 else "#f39c12" if health_index >= 40 else "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_index,
        title={"text": title},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "#3a1010"},
                {"range": [40, 70], "color": "#3a2f10"},
                {"range": [70, 100], "color": "#103a1a"},
            ],
        },
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)
