# src/train.py
import os
import json
import joblib
import pandas as pd

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.hopsworks_client import get_hopsworks_project

ARTIFACT_DIR = "artifacts"
TRAIN_DATA_PATH = os.path.join(ARTIFACT_DIR, "train_data.parquet")

LABELS = ["label_aqi_day1", "label_aqi_day2", "label_aqi_day3"]
MIN_ROWS_FOR_TRAINING = 30

WEEKDAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}

def rmse(y_true, y_pred) -> float:
    return mean_squared_error(y_true, y_pred) ** 0.5


def _prep_df(df: pd.DataFrame) -> pd.DataFrame:
    # keep event_time string
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce").dt.date.astype(str)

    # weekday numeric
    if "weekday" in df.columns:
        df["weekday"] = df["weekday"].astype(str).str.strip().map(WEEKDAY_MAP)

    # force numeric for all non-date columns except weekday (already numeric)
    for c in df.columns:
        if c not in ["event_time", "weekday"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def _train_one_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def _eval(model, X_val, y_val, X_test, y_test, name: str):
    out = {}
    if len(X_val) > 0:
        pred = model.predict(X_val)
        out["val_mae"] = float(mean_absolute_error(y_val, pred))
        out["val_rmse"] = float(rmse(y_val, pred))
    if len(X_test) > 0:
        pred = model.predict(X_test)
        out["test_mae"] = float(mean_absolute_error(y_test, pred))
        out["test_rmse"] = float(rmse(y_test, pred))
    print(f"{name}: {out}")
    return out


def _register_model_to_hopsworks(model_path: str, model_name: str, description: str):
    project = get_hopsworks_project()
    mr = project.get_model_registry()

    hw_model = mr.python.create_model(name=model_name, description=description)
    hw_model.save(model_path)  # auto new version
    print(f"✅ Registered: {model_name} ({model_path})")


def train_and_register():
    if not os.path.exists(TRAIN_DATA_PATH):
        raise RuntimeError("artifacts/train_data.parquet not found. Run: python -m src.training_dataset")

    df = pd.read_parquet(TRAIN_DATA_PATH)
    df = _prep_df(df)

    # must have labels
    for lab in LABELS:
        if lab not in df.columns:
            raise RuntimeError(f"{lab} missing. Rebuild training dataset.")

    if len(df) < MIN_ROWS_FOR_TRAINING:
        raise RuntimeError(f"Too few rows for training: {len(df)}. Need at least {MIN_ROWS_FOR_TRAINING}.")

    # ✅ IMPORTANT: X must NOT include any label columns
    X = df.drop(columns=["event_time"] + LABELS)
    # (event_time is not a model feature; labels must never be in X)

    # split once, reuse
    X_train, X_tmp, idx_train, idx_tmp = train_test_split(
        X, df.index, test_size=0.33, random_state=42
    )
    X_val, X_test, idx_val, idx_test = train_test_split(
        X_tmp, idx_tmp, test_size=0.5, random_state=42
    )

    metrics = {}

    # --- Day 1: 3 models ---
    y1 = df["label_aqi_day1"].astype(float)
    y1_train, y1_val, y1_test = y1.loc[idx_train], y1.loc[idx_val], y1.loc[idx_test]

    lr = _train_one_model(LinearRegression(), X_train, y1_train)
    rf = _train_one_model(RandomForestRegressor(n_estimators=300, random_state=42), X_train, y1_train)
    xgb1 = _train_one_model(
        XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.07,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ),
        X_train, y1_train
    )

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    lr_path = os.path.join(ARTIFACT_DIR, "aqi_lr_day1.joblib")
    rf_path = os.path.join(ARTIFACT_DIR, "aqi_rf_day1.joblib")
    xgb1_path = os.path.join(ARTIFACT_DIR, "aqi_xgb_day1.joblib")

    joblib.dump(lr, lr_path)
    joblib.dump(rf, rf_path)
    joblib.dump(xgb1, xgb1_path)

    metrics["aqi_lr_day1"] = _eval(lr, X_val, y1_val, X_test, y1_test, "aqi_lr_day1")
    metrics["aqi_rf_day1"] = _eval(rf, X_val, y1_val, X_test, y1_test, "aqi_rf_day1")
    metrics["aqi_xgb_day1"] = _eval(xgb1, X_val, y1_val, X_test, y1_test, "aqi_xgb_day1")

    _register_model_to_hopsworks(lr_path, "aqi_lr_day1", "Linear Regression day1 AQI")
    _register_model_to_hopsworks(rf_path, "aqi_rf_day1", "RandomForest day1 AQI")
    _register_model_to_hopsworks(xgb1_path, "aqi_xgb_day1", "XGBoost day1 AQI")

    # --- Day 2: XGB ---
    y2 = df["label_aqi_day2"].astype(float)
    y2_train, y2_val, y2_test = y2.loc[idx_train], y2.loc[idx_val], y2.loc[idx_test]

    xgb2 = _train_one_model(
        XGBRegressor(
            n_estimators=450, max_depth=4, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ),
        X_train, y2_train
    )
    xgb2_path = os.path.join(ARTIFACT_DIR, "aqi_xgb_day2.joblib")
    joblib.dump(xgb2, xgb2_path)
    metrics["aqi_xgb_day2"] = _eval(xgb2, X_val, y2_val, X_test, y2_test, "aqi_xgb_day2")
    _register_model_to_hopsworks(xgb2_path, "aqi_xgb_day2", "XGBoost day2 AQI")

    # --- Day 3: XGB ---
    y3 = df["label_aqi_day3"].astype(float)
    y3_train, y3_val, y3_test = y3.loc[idx_train], y3.loc[idx_val], y3.loc[idx_test]

    xgb3 = _train_one_model(
        XGBRegressor(
            n_estimators=450, max_depth=4, learning_rate=0.06,
            subsample=0.9, colsample_bytree=0.9, random_state=42
        ),
        X_train, y3_train
    )
    xgb3_path = os.path.join(ARTIFACT_DIR, "aqi_xgb_day3.joblib")
    joblib.dump(xgb3, xgb3_path)
    metrics["aqi_xgb_day3"] = _eval(xgb3, X_val, y3_val, X_test, y3_test, "aqi_xgb_day3")
    _register_model_to_hopsworks(xgb3_path, "aqi_xgb_day3", "XGBoost day3 AQI")

    # save metrics
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Saved metrics comparison: {metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_register()
