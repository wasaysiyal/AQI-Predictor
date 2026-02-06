# src/feature_engineering.py

import pandas as pd


def hourly_to_daily_features(hourly: dict) -> pd.DataFrame:
    """
    Converts Open-Meteo hourly payload to daily features.

    Strategy:
    - event_time = date
    - aqi_daily = daily MAX of hourly european_aqi (common AQI daily reporting)
    - pollutants = daily MEAN
    - weekday = day name
    """
    df = pd.DataFrame({"event_time": pd.to_datetime(hourly["time"])})

    # numeric hourly series
    for k in [
        "european_aqi",
        "pm10",
        "pm2_5",
        "ozone",
        "nitrogen_dioxide",
        "sulphur_dioxide",
        "carbon_monoxide",
    ]:
        df[k] = pd.to_numeric(hourly.get(k, []), errors="coerce")

    df["date"] = df["event_time"].dt.date

    agg = {
        "european_aqi": "max",          # ✅ target-like daily AQI
        "pm10": "mean",
        "pm2_5": "mean",
        "ozone": "mean",
        "nitrogen_dioxide": "mean",
        "sulphur_dioxide": "mean",
        "carbon_monoxide": "mean",
    }

    daily = df.groupby("date", as_index=False).agg(agg).rename(
        columns={
            "date": "event_time",
            "european_aqi": "aqi_daily",
            "pm10": "pm10_mean",
            "pm2_5": "pm2_5_mean",
            "ozone": "ozone_mean",
            "nitrogen_dioxide": "no2_mean",
            "sulphur_dioxide": "so2_mean",
            "carbon_monoxide": "co_mean",
        }
    )

    daily["event_time"] = pd.to_datetime(daily["event_time"])
    daily["weekday"] = daily["event_time"].dt.day_name()

    return daily
