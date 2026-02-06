# src/batch_inference.py
import time
import requests
import pandas as pd
import joblib

from src.hopsworks_client import get_hopsworks_project

PRED_FG_NAME = "aqi_predictions_v2"
PRED_FG_VERSION = 1

WEEKDAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

BASE_FEATURES = ["aqi_daily", "pm10_mean", "pm2_5_mean", "ozone_mean", "no2_mean", "so2_mean", "co_mean", "weekday"]

MODELS = [
    ("aqi_xgb_day1", 1),
    ("aqi_xgb_day2", 2),
    ("aqi_xgb_day3", 3),
]

def _wait_for_materialization(fg, timeout_s: int = 15 * 60, poll_s: int = 15):
    start = time.time()
    while True:
        try:
            state = fg.materialization_job.get_state()
        except Exception:
            state = None

        if state and str(state).upper() not in ("RUNNING", "STARTING"):
            return

        if time.time() - start > timeout_s:
            raise TimeoutError("Materialization job did not finish within timeout.")
        time.sleep(poll_s)

def _safe_insert_with_wait(fg, df, max_attempts: int = 5):
    for attempt in range(1, max_attempts + 1):
        try:
            fg.insert(df, write_options={"upsert": True})
            _wait_for_materialization(fg)
            return
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError) as e:
            print(f"⚠️ Insert failed (attempt {attempt}/{max_attempts}): {e}")
            try:
                _wait_for_materialization(fg)
                print("✅ Insert likely succeeded (job finished after connection drop).")
                return
            except Exception:
                if attempt == max_attempts:
                    raise
            time.sleep(min(2 ** attempt, 30))

def _load_latest_model(project, model_name: str):
    mr = project.get_model_registry()
    models = mr.get_models(model_name)
    latest = max(models, key=lambda m: m.version)

    model_dir = latest.download()
    model_path = f"{model_dir}/{model_name}.joblib"
    clf = joblib.load(model_path)
    return clf, latest.version

def run_batch_inference():
    project = get_hopsworks_project()
    fs = project.get_feature_store()


    # --- read latest features ---
    feat_fg = fs.get_feature_group("daily_aqi_features", version=1)
    feat_df = feat_fg.read()

    feat_df["event_time_dt"] = pd.to_datetime(feat_df["event_time"], errors="coerce")
    feat_df = feat_df.dropna(subset=["event_time_dt"]).sort_values("event_time_dt")

    latest = feat_df.iloc[-1:].copy()
    source_feature_time = latest["event_time_dt"].iloc[0]

    # weekday -> int
    latest["weekday"] = latest["weekday"].astype(str).str.strip().map(WEEKDAY_MAP)

    # ✅ build X with ONLY base features (no labels, no event_time)
    X = latest[BASE_FEATURES].copy()

    # numeric enforcement
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna()

    if len(X) != 1:
        raise RuntimeError("Latest feature row has NaNs after numeric conversion.")

    # create prediction FG
    pred_fg = fs.get_or_create_feature_group(
    name=PRED_FG_NAME,
    version=PRED_FG_VERSION,
    primary_key=["event_time"],
    description="Daily AQI 1/2/3-day predictions",
    online_enabled=False,   # ✅ FIX: online store doesn’t support timestamp PK
)

    rows = []
    for model_name, horizon in MODELS:
        clf, model_version = _load_latest_model(project, model_name)

        pred_time = source_feature_time + pd.Timedelta(days=horizon)
        pred = float(clf.predict(X)[0])

        rows.append(
            {
                "event_time": pd.to_datetime(pred_time, utc=True),
                "horizon": int(horizon),
                "predicted_aqi": float(pred),
                "source_feature_time": pd.to_datetime(source_feature_time, utc=True),
                "model_name": model_name,
                "model_version": int(model_version),
            }
        )

    pred_df = pd.DataFrame(rows)

    _safe_insert_with_wait(pred_fg, pred_df)

    print("✅ Stored predictions:")
    print(pred_df)

if __name__ == "__main__":
    run_batch_inference()
