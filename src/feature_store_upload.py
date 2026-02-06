# src/feature_store_upload.py

import time
import requests
import pandas as pd

from src.hopsworks_client import get_hopsworks_project
from src.data_fetcher import fetch_daily_features


def _wait_for_materialization(fg, timeout_s: int = 15 * 60, poll_s: int = 15):
    """
    Wait until materialization job finishes (or timeout).
    Handles the case where a retry sees "job already running".
    """
    start = time.time()
    while True:
        try:
            state = fg.materialization_job.get_state()
        except Exception:
            state = None

        # if state is None, we still wait a bit and retry
        if state and str(state).upper() not in ("RUNNING", "STARTING"):
            # final state
            try:
                final = fg.materialization_job.get_final_state()
                print(f"✅ Materialization job final state: {final}")
            except Exception:
                pass
            return

        if time.time() - start > timeout_s:
            raise TimeoutError("Materialization job did not finish within timeout.")

        time.sleep(poll_s)


def upload_daily_features(lat=24.8607, lon=67.0011, days: int = 3, online_enabled=True):
    df = fetch_daily_features(lat=lat, lon=lon, days=days)

    # ✅ enforce schema: your daily_aqi_features FG expects event_time as STRING
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce").dt.date.astype(str)

    print("Data ready for upload:\n", df)

    project = get_hopsworks_project()
    fs = project.get_feature_store()

    fg = fs.get_or_create_feature_group(
        name="daily_aqi_features",
        version=1,
        primary_key=["event_time"],
        description="Daily features and target for AQI forecasting",
        online_enabled=online_enabled,
    )

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            fg.insert(df, write_options={"upsert": True})
            # ✅ always wait so next step reads the latest data
            _wait_for_materialization(fg)
            print("✅ Upload completed successfully!")
            return

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout, OSError) as e:
            print(f"⚠️ Insert failed (attempt {attempt}/{max_attempts}): {e}")

            # ✅ DO NOT re-trigger insert immediately.
            # Connection often drops AFTER job starts.
            # So: wait for the running materialization job, then proceed.
            try:
                _wait_for_materialization(fg)
                print("✅ Upload likely succeeded (job finished after connection drop).")
                return
            except Exception:
                if attempt == max_attempts:
                    raise

            wait_s = min(2 ** attempt, 30)
            print(f"Retrying insert in {wait_s}s ...")
            time.sleep(wait_s)


if __name__ == "__main__":
    upload_daily_features()
