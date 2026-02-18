import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import pandas as pd

from src.hopsworks_client import get_hopsworks_project
from src.batch_inference import run_batch_inference

# -----------------------------
# Page config + styling
# -----------------------------
st.set_page_config(page_title="AQI Predictor", layout="wide")

st.markdown(
    """
    <style>
      .kpi-card {
        padding: 18px 18px 14px 18px;
        border-radius: 16px;
        border: 1px solid rgba(255,255,255,0.08);
        background: rgba(255,255,255,0.03);
        box-shadow: 0 8px 24px rgba(0,0,0,0.22);
      }
      .kpi-title { font-size: 14px; opacity: 0.85; margin-bottom: 6px; }
      .kpi-value { font-size: 42px; font-weight: 800; line-height: 1.0; margin: 0; }
      .kpi-sub   { font-size: 13px; opacity: 0.85; margin-top: 6px; }
      .pill {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.10);
        background: rgba(255,255,255,0.04);
      }
      .muted { opacity: 0.75; }
      .section-title { margin-top: 10px; margin-bottom: 6px; font-size: 18px; font-weight: 800; }
      .divider { height: 1px; background: rgba(255,255,255,0.08); margin: 14px 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🌫️ AQI Predictor")
st.markdown("<div class='muted'>📅 1 / 2 / 3 Day Forecast</div>", unsafe_allow_html=True)

# -----------------------------
# Hopsworks FS init
# -----------------------------
@st.cache_resource
def get_fs():
    project = get_hopsworks_project()
    return project.get_feature_store()

fs = get_fs()

# -----------------------------
# Controls
# -----------------------------
left, right = st.columns([1, 2], gap="large")

with left:
    st.markdown("<div class='section-title'>Controls</div>", unsafe_allow_html=True)
    if st.button("🚀 Run Prediction Now", use_container_width=True):
        with st.spinner("Running batch inference..."):
            run_batch_inference()
        st.success("Done! Refreshing...")
        st.rerun()

with right:
    st.markdown("<div class='section-title'>Latest Results</div>", unsafe_allow_html=True)
    st.markdown("<div class='muted'>Shows the most recent inference run stored in Hopsworks.</div>", unsafe_allow_html=True)

# -----------------------------
# Read predictions from Hopsworks
# -----------------------------
pred_fg = fs.get_feature_group("aqi_predictions_v2", version=1)
pred_df = pred_fg.read()

# Parse times
pred_df["event_time"] = pd.to_datetime(pred_df["event_time"], errors="coerce", utc=True)
pred_df["source_feature_time"] = pd.to_datetime(pred_df["source_feature_time"], errors="coerce", utc=True)
pred_df = pred_df.dropna(subset=["event_time", "source_feature_time"])

# Pick the latest run (by source_feature_time)
latest_run_time = pred_df["source_feature_time"].max()
latest_df = pred_df[pred_df["source_feature_time"] == latest_run_time].copy()

# Keep only next 3 horizons
latest_df = latest_df.sort_values(["horizon"], ascending=[True]).head(3)

# If empty, show help
if latest_df.empty:
    st.warning("No predictions found yet. Click **Run Prediction Now**.")
    st.stop()

# Model metadata (show on dashboard)
# If different horizons use different models, show them all
# -----------------------------
# -----------------------------
st.caption(f"Latest inference run (UTC): {latest_run_time}")

st.markdown("### 🤖 Forecasting Model Used")
st.markdown(
    "**AQI Forecasting Model (XGBoost Regression)**  \n"
    "<span style='opacity:0.7;'>Machine learning model trained on historical AQI and pollutant data.</span>",
    unsafe_allow_html=True
)


# -----------------------------
# Premium AQI labeling
# -----------------------------
def aqi_band(aqi: float):
    # Returns (label, emoji)
    if aqi <= 50:
        return ("Good", "🟢")
    elif aqi <= 100:
        return ("Moderate", "🟡")
    elif aqi <= 150:
        return ("Unhealthy (SG)", "🟠")
    elif aqi <= 200:
        return ("Unhealthy", "🔴")
    elif aqi <= 300:
        return ("Very Unhealthy", "🟣")
    else:
        return ("Hazardous", "⚫")

# Convert to local date display if you want (Pakistan is UTC+5)
# We'll show both: local date label and keep UTC internally
local_tz = "Asia/Karachi"

cards = st.columns(3, gap="large")

for i, row in enumerate(latest_df.itertuples(index=False)):
    event_time_utc = row.event_time
    event_time_local = event_time_utc.tz_convert(local_tz)

    pred_aqi = float(row.predicted_aqi)
    # If your model ever outputs negatives, clamp for display only:
    pred_aqi_display = max(0.0, min(500.0, pred_aqi))

    band, emoji = aqi_band(pred_aqi_display)

    with cards[i]:
        st.markdown("<div class='kpi-card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='kpi-title'>📅 {event_time_local.strftime('%a, %d %b %Y')}</div>", unsafe_allow_html=True)
        st.markdown(f"<p class='kpi-value'>{pred_aqi_display:.1f}</p>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='kpi-sub'><span class='pill'>{emoji} {band}</span> "
            f"<span class='muted'> • Horizon: {int(row.horizon)} day(s)</span></div>",
            unsafe_allow_html=True,
        )
  
        
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# Optional: also show a clean table below
show_table = st.toggle("Show raw table", value=False)
if show_table:
    show_df = latest_df.copy()
    show_df["event_time_local"] = show_df["event_time"].dt.tz_convert(local_tz)
    show_df = show_df[["event_time_local", "event_time", "horizon", "predicted_aqi", "model_name", "model_version", "source_feature_time"]]
    st.dataframe(show_df, width="stretch")
# -----------------------------
# Footer
# -----------------------------
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)


# 🌈 AQI Level Guide
# -----------------------------
st.markdown("### 🌈 AQI Health Levels Guide")

aqi_legend = """
🟢 **0 – 50 (Good)**  
Air quality is satisfactory, and air pollution poses little or no risk.

🟡 **51 – 100 (Moderate)**  
Air quality is acceptable; some pollutants may be a concern for sensitive individuals.

🟠 **101 – 150 (Unhealthy for Sensitive Groups)**  
Sensitive groups may experience health effects.

🔴 **151 – 200 (Unhealthy)**  
Everyone may begin to experience health effects.

🟣 **201 – 300 (Very Unhealthy)**  
Health alert: everyone may experience more serious effects.

⚫ **301 – 500 (Hazardous)**  
Health warning of emergency conditions.
"""

st.markdown(aqi_legend)

st.markdown(
    """
    <div style="text-align:center; opacity:0.6; font-size:13px;">
        AQI Forecasting System • Built with Streamlit & Hopsworks <br>
        Machine Learning Powered Environmental Intelligence <br>
        © 2026 AQI Predictor Project
    </div>
    """,
    unsafe_allow_html=True
)


