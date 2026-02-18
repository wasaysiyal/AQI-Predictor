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

def safe_read_fg(fg, *, name: str):
    """
    Try Query Service first (fast). If it fails on Streamlit Cloud, fall back to Spark.
    """
    try:
        return fg.read()
    except Exception as e:
        st.warning(f"⚠️ Query Service failed while reading **{name}**. Falling back to Spark (slower).")
        try:
            return fg.read(read_options={"use_spark": True})
        except Exception as e2:
            st.error(f"❌ Failed reading **{name}** using both Query Service and Spark.")
            st.exception(e2)
            st.stop()

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
pred_df = safe_read_fg(pred_fg, name="aqi_predictions_v2")

pred_df["event_time"] = pd.to_datetime(pred_df.get("event_time"), errors="coerce", utc=True)
pred_df["source_feature_time"] = pd.to_datetime(pred_df.get("source_feature_time"), errors="coerce", utc=True)
pred_df = pred_df.dropna(subset=["event_time", "source_feature_time"])

if pred_df.empty:
    st.warning("No predictions found yet. Click **Run Prediction Now**.")
    st.stop()

latest_run_time = pred_df["source_feature_time"].max()
latest_df = pred_df[pred_df["source_feature_time"] == latest_run_time].copy()
latest_df = latest_df.sort_values(["horizon"], ascending=[True]).head(3)

st.caption(f"Latest inference run (UTC): {latest_run_time}")

st.markdown("### 🤖 Forecasting Model Used")
st.markdown(
    "**AQI Forecasting Model (XGBoost Regression)**  \n"
    "<span style='opacity:0.7;'>Machine learning model trained on historical AQI and pollutant data.</span>",
    unsafe_allow_html=True
)

def aqi_band(aqi: float):
    if aqi <= 50:  return ("Good", "🟢")
    if aqi <= 100: return ("Moderate", "🟡")
    if aqi <= 150: return ("Unhealthy for Sensitive Groups", "🟠")
    if aqi <= 200: return ("Unhealthy", "🔴")
    if aqi <= 300: return ("Very Unhealthy", "🟣")
    return ("Hazardous", "⚫")

local_tz = "Asia/Karachi"
cards = st.columns(3, gap="large")

for i, row in enumerate(latest_df.itertuples(index=False)):
    event_time_local = row.event_time.tz_convert(local_tz)
    pred_aqi_display = max(0.0, min(500.0, float(row.predicted_aqi)))
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

# -----------------------------
# Chart (history + forecast)
# -----------------------------
st.markdown("### 📈 AQI Trend & Forecast")

feat_fg = fs.get_feature_group("daily_aqi_features_v2", version=1)
hist_df = safe_read_fg(feat_fg, name="daily_aqi_features_v2")

hist_df["event_time"] = pd.to_datetime(hist_df.get("event_time"), errors="coerce", utc=True)
hist_df = hist_df.dropna(subset=["event_time"]).sort_values("event_time").tail(14)

forecast_df = latest_df[["event_time", "predicted_aqi"]].rename(columns={"predicted_aqi": "aqi_daily"})
chart_df = pd.concat([hist_df[["event_time", "aqi_daily"]], forecast_df], ignore_index=True).sort_values("event_time")

chart_df_display = chart_df.copy()
chart_df_display["event_time"] = chart_df_display["event_time"].dt.tz_convert(local_tz)

st.line_chart(chart_df_display.set_index("event_time")[["aqi_daily"]], height=350, width="stretch")

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# -----------------------------
# AQI Guide + Footer
# -----------------------------
st.markdown("### 🌈 AQI Health Levels Guide")
st.markdown(
    """
🟢 **0–50 (Good)** – Satisfactory  
🟡 **51–100 (Moderate)** – Acceptable, some risk for sensitive groups  
🟠 **101–150 (Unhealthy for Sensitive Groups)**  
🔴 **151–200 (Unhealthy)**  
🟣 **201–300 (Very Unhealthy)**  
⚫ **301–500 (Hazardous)**
"""
)

st.markdown(
    """
    <div style="text-align:center; opacity:0.6; font-size:13px;">
        AQI Forecasting System • Built with Streamlit & Hopsworks <br>
        Machine Learning Powered Environmental Intelligence <br>
        © 2026 AQI Predictor Project
    </div>
    """,
    unsafe_allow_html=True,
)
