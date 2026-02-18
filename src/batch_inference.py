# src/batch_inference.py
import time
import requests
import pandas as pd
import joblib

from src.hopsworks_client import get_hopsworks_project

# ✅ MUST match what feature_store_upload writes to
FEATURE_FG_NAME = "daily_aqi_features_v2"
FEATURE_FG_VERSION = 1

PRED_FG_NAME = "aqi_predictions_v2"
PRED_FG_VERSION = 1

WEEKDAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

BASE_FEATURES = [
    "aqi_daily", "pm10_mean", "pm2_5_mean", "ozone_mean",
    "no2_mean", "so2_mean", "co_mean", "weekday"
]

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

    # ✅ anchor date = today's UTC midnight
    today_local = pd.Timestamp.now().normalize()
    today_utc = today_local.tz_localize("UTC")

    print(f"\n✅ Inference anchor today_utc: {today_utc}")

    # -----------------------------
    # 1) Read latest features (FROM v2)
    # -----------------------------
    feat_fg = fs.get_feature_group(FEATURE_FG_NAME, version=FEATURE_FG_VERSION)
    feat_df = feat_fg.read(read_options={"use_hive": False})


    feat_df["event_time_dt"] = pd.to_datetime(feat_df["event_time"], errors="coerce", utc=True)
    feat_df = feat_df.dropna(subset=["event_time_dt"]).sort_values("event_time_dt")

    latest_dt = feat_df["event_time_dt"].max()
    if latest_dt < today_utc - pd.Timedelta(days=1):
        print(
            f"⚠️ Features look stale. Latest feature row is {latest_dt}, "
            f"but today_utc is {today_utc}."
        )
        print("⚠️ Run feature pipeline (feature_store_upload) to refresh rows.\n")

    # use latest available feature row (should now be >= yesterday/today after upload)
    latest = feat_df.iloc[-1:].copy()
    source_feature_time = today_utc  # show run anchor in dashboard
    print(f"✅ Latest feature row used: {latest['event_time_dt'].iloc[0]}\n")

    # weekday -> int
    latest["weekday"] = latest["weekday"].astype(str).str.strip().map(WEEKDAY_MAP)

    # build X
    X = latest[BASE_FEATURES].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.dropna()

    if len(X) != 1:
        raise RuntimeError("Latest feature row has NaNs after numeric conversion.")

    print("✅ X (features sent to models):")
    print(X.to_string(index=False))

    # -----------------------------
    # 2) Create prediction FG
    # -----------------------------
    pred_fg = fs.get_or_create_feature_group(
        name=PRED_FG_NAME,
        version=PRED_FG_VERSION,
        primary_key=["event_time"],
        description="Daily AQI 1/2/3-day predictions",
        online_enabled=False,
    )

    # -----------------------------
    # 3) Predict for today / tomorrow / day-after
    #    (event_time = today_utc + (horizon-1))
    # -----------------------------
    rows = []
    for model_name, horizon in MODELS:
        clf, model_version = _load_latest_model(project, model_name)

        # ✅ horizon mapping:
        # day1 -> today, day2 -> tomorrow, day3 -> day after
        pred_time = today_utc + pd.Timedelta(days=(horizon - 1))

        raw_pred = float(clf.predict(X)[0])
        print(f"RAW PRED [{model_name}] = {raw_pred}")

        rows.append(
            {
                "event_time": pred_time,
                "horizon": int(horizon),
                "predicted_aqi": float(raw_pred),
                "source_feature_time": today_utc,
                "model_name": model_name,
                "model_version": int(model_version),
            }
        )

    pred_df = pd.DataFrame(rows)

    _safe_insert_with_wait(pred_fg, pred_df)

    print("\n✅ Stored predictions:")
    print(pred_df)


if __name__ == "__main__":
    run_batch_inference()
