
# app/app.py
import streamlit as st
import pandas as pd

from src.hopsworks_client import get_hopsworks_project
from src.batch_inference import run_batch_inference


st.set_page_config(page_title="AQI Predictor", layout="wide")

st.title("🌫️ AQI Predictor")
st.subheader("📅 1 / 2 / 3 Day Forecast")


@st.cache_resource
def get_fs():
    project = get_hopsworks_project()
    return project.get_feature_store()


fs = get_fs()

# -----------------------------
# Button: manual trigger
# -----------------------------
if st.button("🚀 Run Prediction Now"):
    with st.spinner("Running batch inference..."):
        run_batch_inference()
    st.success("Done! Refreshing table...")
    st.rerun()


# -----------------------------
# Read predictions from Hopsworks
# -----------------------------
pred_fg = fs.get_feature_group("aqi_predictions_v2", version=1)
pred_df = pred_fg.read()

pred_df["event_time"] = pd.to_datetime(pred_df["event_time"], errors="coerce")
pred_df = pred_df.dropna(subset=["event_time"])

pred_df = (
    pred_df.sort_values(["event_time", "horizon"], ascending=[False, True])
          .head(30)
)

st.dataframe(pred_df, width="stretch")
