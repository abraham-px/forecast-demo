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
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
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


def _compute_window(config: Dict[str, str]) -> tuple[datetime, datetime, bool]:
    """Return (window_start_utc, window_end_utc, use_archive)."""
    now_utc = datetime.now(tz=timezone.utc)
    start_override = config.get("start_date")
    if start_override:
        ts = pd.Timestamp(start_override)
        if ts.tzinfo is None:
            ts = ts.tz_localize(config["timezone"])  # local date/time
        else:
            ts = ts.tz_convert(config["timezone"])  # normalize to local tz
        window_start = ts.tz_convert(timezone.utc).to_pydatetime()
    else:
        window_start = now_utc
    window_end = window_start + timedelta(hours=FORECAST_HOURS)
    use_archive = start_override is not None and window_end <= now_utc
    return window_start, window_end, use_archive


def _request_json(url: str, params: dict) -> dict:
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_forecast(config: Dict[str, str]) -> pd.DataFrame:
    window_start, window_end, use_archive = _compute_window(config)

    if use_archive:
        # Historical day via archive API
        params = {
            "latitude": float(config["latitude"]),
            "longitude": float(config["longitude"]),
            "hourly": list(HOURLY_FIELDS.keys()),
            "start_date": window_start.date().isoformat(),
            "end_date": window_end.date().isoformat(),
            "timezone": "UTC",
        }
        LOG.info(
            "Requesting Open-Meteo archive for (%s,%s) start=%s end=%s",
            params["latitude"],
            params["longitude"],
            params["start_date"],
            params["end_date"],
        )
        payload = _request_json(OPEN_METEO_ARCHIVE_URL, params)
    else:
        # Live forecast
        params = {
            "latitude": float(config["latitude"]),
            "longitude": float(config["longitude"]),
            "hourly": list(HOURLY_FIELDS.keys()),
            "models": config["model"],
            "timezone": "UTC",
            "forecast_days": 2,
        }
        LOG.info(
            "Requesting Open-Meteo forecast for (%s,%s) start=%s horizon=%sh",
            params["latitude"],
            params["longitude"],
            window_start.isoformat(),
            FORECAST_HOURS,
        )
        payload = _request_json(OPEN_METEO_URL, params)

    hourly = payload.get("hourly")
    if not hourly:
        LOG.warning("Open-Meteo response missing or empty 'hourly' block")
        return pd.DataFrame()

    frame = pd.DataFrame(hourly)
    if "time" not in frame.columns:
        LOG.warning("Open-Meteo hourly payload missing 'time' column")
        return pd.DataFrame()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time")[list(HOURLY_FIELDS.keys())]
    frame = frame.rename(columns=HOURLY_FIELDS).sort_index()

    # Trim to requested window and upsample to 30 min
    mask = (frame.index >= window_start) & (frame.index <= window_end)
    frame = frame.loc[mask]
    if frame.empty:
        LOG.warning("No rows found in the requested window")
        return frame
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
