import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

# Load .env file
load_dotenv()


# -------------------------
# Weather / AQI (Open-Meteo)
# -------------------------
@dataclass(frozen=True)
class WeatherConfig:
    base_url: str


# -------------------------
# Hopsworks
# -------------------------
@dataclass(frozen=True)
class HopsworksConfig:
    api_key: str
    project: str
    api_host: str


# -------------------------
# App Settings
# -------------------------
@dataclass(frozen=True)
class Settings:
    env: str
    weather: WeatherConfig = field(default_factory=WeatherConfig)
    hopsworks: HopsworksConfig = field(default_factory=HopsworksConfig)


# -------------------------
# Instantiate settings
# -------------------------
settings = Settings(
    env=os.getenv("ENV", "local"),

    weather=WeatherConfig(
        base_url=os.getenv("WEATHER_API_BASE_URL", ""),
    ),

    hopsworks=HopsworksConfig(
        api_key=os.getenv("HOPSWORKS_API_KEY", ""),
        project=os.getenv("HOPSWORKS_PROJECT", ""),
        api_host=os.getenv("HOPSWORKS_API_HOST", ""),
    ),
)
