# src/data_fetcher.py
import requests
from datetime import date, timedelta

from src.config import settings
from src.feature_engineering import hourly_to_daily_features


from datetime import date, timedelta

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



def fetch_air_quality_raw(lat: float, lon: float, start_date: str, end_date: str) -> dict:
    url = f"{settings.weather.base_url}/air-quality"
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

