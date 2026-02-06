# src/training_dataset.py
import os
import pandas as pd

ARTIFACT_DIR = "artifacts"
TRAIN_DATA_PATH = os.path.join(ARTIFACT_DIR, "train_data.parquet")

FG_NAME = "daily_aqi_features"
FG_VERSION = 1

# Only REAL columns (avoid broken metadata feature name)
BASE_FEATURES = [
    "event_time",
    "aqi_daily",
    "pm10_mean",
    "pm2_5_mean",
    "ozone_mean",
    "no2_mean",
    "so2_mean",
    "co_mean",
    "weekday",
]

def create_training_data():
    from src.hopsworks_client import get_hopsworks_project

    project = get_hopsworks_project()
    fs = project.get_feature_store()

    fg = fs.get_feature_group("daily_aqi_features_v2", version=1)


    # ✅ IMPORTANT: avoid fg.read() (it selects the broken feature name)
    df = fg.select(BASE_FEATURES).read()

    # Make sure event_time is sortable
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    df = df.dropna(subset=["event_time"]).sort_values("event_time").reset_index(drop=True)

    # ✅ create 1/2/3 day labels
    df["label_aqi_day1"] = df["aqi_daily"].shift(-1)
    df["label_aqi_day2"] = df["aqi_daily"].shift(-2)
    df["label_aqi_day3"] = df["aqi_daily"].shift(-3)

    df = df.dropna().reset_index(drop=True)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df.to_parquet(TRAIN_DATA_PATH, index=False)
    print(f"Saved training data: {TRAIN_DATA_PATH}")
    print("Shape:", df.shape)
    print(df.tail())

if __name__ == "__main__":
    create_training_data()
