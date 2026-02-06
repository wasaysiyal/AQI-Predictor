# src/data_fetcher.py
import os
import requests
from datetime import date, timedelta

from src.config import settings
from src.feature_engineering import hourly_to_daily_features


def _date_range_from_days(days: int, end_yesterday: bool = True) -> tuple[str, str]:
    """
    Return (start_date, end_date) inclusive as YYYY-MM-DD for the PAST `days`.
    If end_yesterday=True, end_date = yesterday (recommended for daily AQI stability).
    """
    if days < 2:
        raise ValueError("days must be >= 2")

    end = date.today() - timedelta(days=1) if end_yesterday else date.today()
    start = end - timedelta(days=days - 1)
    return start.isoformat(), end.isoformat()


def _air_quality_base_url() -> str:
    """
    Always return a valid absolute base URL.
    Priority:
      1) settings.weather.base_url (if valid)
      2) env var AIR_QUALITY_BASE_URL
      3) Open-Meteo default
    """
    # 1) from config (if present)
    base = getattr(getattr(settings, "weather", None), "base_url", None)
    if isinstance(base, str) and base.strip().startswith(("http://", "https://")):
        return base.strip().rstrip("/")

    # 2) from env
    env_base = os.getenv("AIR_QUALITY_BASE_URL", "").strip()
    if env_base.startswith(("http://", "https://")):
        return env_base.rstrip("/")

    # 3) safe default (Open-Meteo)
    return "https://air-quality-api.open-meteo.com/v1"


def fetch_air_quality_raw(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    base_url = _air_quality_base_url()
    url = f"{base_url}/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "european_aqi,pm10,pm2_5,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide",
        "timezone": "auto",
        "start_date": start_date,
        "end_date": end_date,
    }

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"Open-Meteo API error {resp.status_code}: {resp.text}")
    return resp.json()


def fetch_daily_features(lat: float, lon: float, days: int = 4):
    start_date, end_date = _date_range_from_days(days, end_yesterday=True)
    raw = fetch_air_quality_raw(lat=lat, lon=lon, start_date=start_date, end_date=end_date)
    hourly = raw["hourly"]
    return hourly_to_daily_features(hourly)
