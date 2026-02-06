# src/training_dataset.py
import os
import pandas as pd

from src.hopsworks_client import get_hopsworks_project

FG_NAME = "daily_aqi_features"
FG_VERSION = 1
OUT_PATH = "artifacts/train_data.parquet"


def create_training_data():
    project = get_hopsworks_project()
    fs = project.get_feature_store()

    fg = fs.get_feature_group(name=FG_NAME, version=FG_VERSION)
    df = fg.read()

    # --- normalize schema ---
    df["event_time_dt"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time_dt"])

    df["aqi_daily"] = pd.to_numeric(df["aqi_daily"], errors="coerce")
    df = df.dropna(subset=["aqi_daily"])

    # keep latest row per day if duplicates exist
    df = df.sort_values("event_time_dt").drop_duplicates(subset=["event_time"], keep="last")
    df = df.sort_values("event_time_dt").reset_index(drop=True)

    # ✅ labels for 1/2/3-day ahead
    df["label_aqi_day1"] = df["aqi_daily"].shift(-1)
    df["label_aqi_day2"] = df["aqi_daily"].shift(-2)
    df["label_aqi_day3"] = df["aqi_daily"].shift(-3)

    # drop rows without labels (last 3 rows)
    df = df.dropna(subset=["label_aqi_day1", "label_aqi_day2", "label_aqi_day3"]).reset_index(drop=True)

    # keep event_time as string (consistent)
    df = df.drop(columns=["event_time_dt"])

    os.makedirs("artifacts", exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)

    print("Saved training data:", OUT_PATH)
    print("Shape:", df.shape)
    print(df.tail(5))


if __name__ == "__main__":
    create_training_data()
