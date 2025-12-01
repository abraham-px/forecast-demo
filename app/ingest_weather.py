"""Simple weather ingest: fetch 24h Open-Meteo data at 30 min cadence and write to Influx."""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Dict

import pandas as pd
import requests
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

LOG = logging.getLogger("ingest_weather")
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_FIELDS: Dict[str, str] = {
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
}
FORECAST_HOURS = os.getenv("FORECAST_HOURS")
if FORECAST_HOURS is None:
    FORECAST_HOURS = 47
else:
    FORECAST_HOURS = int(FORECAST_HOURS)

HALF_HOURLY_POINTS = FORECAST_HOURS * 2 + 1


def get_config() -> Dict[str, str]:
    config = {
        "latitude": os.getenv("SITE_LATITUDE"),
        "longitude": os.getenv("SITE_LONGITUDE"),
        "timezone": os.getenv("SITE_TIMEZONE", "UTC"),
        "model": os.getenv("OPEN_METEO_MODEL", "best_match"),
        "start_date": os.getenv("START_DATE"),
        "measurement": os.getenv("WEATHER_MEASUREMENT", "weather_forecast"),
        "influx_url": os.getenv("INFLUX_URL"),
        "influx_token": os.getenv("INFLUX_TOKEN"),
        "influx_org": os.getenv("INFLUX_ORG"),
        "influx_bucket": os.getenv("INFLUX_BUCKET"),
        "verify_ssl": False,
    }
    missing = [key for key in ("latitude", "longitude", "influx_url", "influx_token", "influx_org", "influx_bucket") if not config[key]]
    if missing:
        raise SystemExit(f"Missing required environment values: {', '.join(missing)}")
    return config


def fetch_forecast(config: Dict[str, str]) -> pd.DataFrame:
    params = {
        "latitude": float(config["latitude"]),
        "longitude": float(config["longitude"]),
        "hourly": list(HOURLY_FIELDS.keys()),
        "models": config["model"],
        "timezone": config["timezone"],
        "forecast_days": 2,
    }
    LOG.info("Requesting Open-Meteo data for (%s,%s)", params["latitude"], params["longitude"])
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    raw = response.json().get("hourly")
    if not raw:
        raise RuntimeError("Open-Meteo response missing hourly block")

    frame = pd.DataFrame(raw)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time")[list(HOURLY_FIELDS.keys())]
    frame = frame.rename(columns=HOURLY_FIELDS).sort_index()

    start_override = config.get("start_date")
    if start_override:
        ts = pd.Timestamp(start_override)
        if ts.tzinfo is None:
            ts = ts.tz_localize(config["timezone"])
        else:
            ts = ts.tz_convert(config["timezone"])
        window_start = ts.tz_convert(timezone.utc)
    else:
        window_start = datetime.now(tz=timezone.utc)
    window_end = window_start + timedelta(hours=FORECAST_HOURS)
    frame = frame[(frame.index >= window_start) & (frame.index <= window_end)]
    frame = frame.resample("30min").interpolate("time").ffill().bfill()
    return frame.iloc[:HALF_HOURLY_POINTS]


def write_to_influx(config: Dict[str, str], forecast: pd.DataFrame) -> int:
    if forecast.empty:
        LOG.warning("No weather rows to write")
        return 0

    tags = {"provider": "open-meteo"}
    with InfluxDBClient(
        url=config["influx_url"],
        token=config["influx_token"],
        org=config["influx_org"],
        verify_ssl=config["verify_ssl"],
    ) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        points = []
        for timestamp, row in forecast.iterrows():
            point = Point(config["measurement"]).time(timestamp.to_pydatetime(), WritePrecision.S)
            for key, value in tags.items():
                point = point.tag(key, value)
            for column, column_value in row.items():
                if pd.isna(column_value):
                    continue
                point = point.field(column, float(column_value))
            points.append(point)
        write_api.write(bucket=config["influx_bucket"], org=config["influx_org"], record=points)
    return len(points)


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
    config = get_config()
    forecast = fetch_forecast(config)
    written = write_to_influx(config, forecast)
    LOG.info("Ingest complete. Points written: %s", written)


if __name__ == "__main__":
    main()
