"""Simple weather ingest: fetch multi-day Open-Meteo data at 30 min cadence and write to Influx."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
import math
import os
from typing import Dict, Iterable

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


@dataclass(frozen=True)
class WeatherConfig:
    latitude: float
    longitude: float
    timezone: str
    model: str
    start_date: str | None
    measurement: str
    influx_url: str
    influx_token: str
    influx_org: str
    influx_bucket: str
    forecast_hours: int

    @property
    def half_hour_points(self) -> int:
        return self.forecast_hours * 2

    @classmethod
    def from_env(cls) -> WeatherConfig:
        env = os.getenv
        required = {
            "SITE_LATITUDE": env("SITE_LATITUDE"),
            "SITE_LONGITUDE": env("SITE_LONGITUDE"),
            "INFLUX_URL": env("INFLUX_URL"),
            "INFLUX_TOKEN": env("INFLUX_TOKEN"),
            "INFLUX_ORG": env("INFLUX_ORG"),
            "INFLUX_BUCKET": env("INFLUX_BUCKET"),
        }
        missing = [key for key, value in required.items() if not value]
        if missing:
            raise SystemExit(f"Missing required environment values: {', '.join(missing)}")
        return cls(
            latitude=float(required["SITE_LATITUDE"]),
            longitude=float(required["SITE_LONGITUDE"]),
            timezone=env("SITE_TIMEZONE", "UTC"),
            model=env("OPEN_METEO_MODEL", "best_match"),
            start_date=env("START_DATE"),
            measurement=env("WEATHER_MEASUREMENT", "weather_forecast"),
            influx_url=required["INFLUX_URL"],
            influx_token=required["INFLUX_TOKEN"],
            influx_org=required["INFLUX_ORG"],
            influx_bucket=required["INFLUX_BUCKET"],
            forecast_hours=int(env("FORECAST_HOURS", "96")),
        )


def _request_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _window(config: WeatherConfig) -> tuple[datetime, datetime, bool]:
    now_utc = datetime.now(tz=timezone.utc)
    if config.start_date:
        ts = pd.Timestamp(config.start_date)
        ts = ts.tz_localize(config.timezone) if ts.tzinfo is None else ts.tz_convert(config.timezone)
        start = ts.tz_convert("UTC").to_pydatetime()
    else:
        start = now_utc
    end = start + timedelta(hours=config.forecast_hours)
    return start, end, bool(config.start_date and end <= now_utc)


def _build_params(config: WeatherConfig, fields: Iterable[str], *, archive: bool, start: datetime, end: datetime) -> dict:
    base = {
        "latitude": config.latitude,
        "longitude": config.longitude,
        "hourly": list(fields),
    }
    if archive:
        base.update(
            {
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
                "timezone": "UTC",
            }
        )
    else:
        base.update(
            {
                "models": config.model,
                "timezone": "UTC",
                "forecast_days": max(1, math.ceil(config.forecast_hours / 24)),
            }
        )
    return base


def fetch_forecast(config: WeatherConfig) -> pd.DataFrame:
    start, end, use_archive = _window(config)
    params = _build_params(config, HOURLY_FIELDS.keys(), archive=use_archive, start=start, end=end)
    url = OPEN_METEO_ARCHIVE_URL if use_archive else OPEN_METEO_URL
    LOG.info(
        "Requesting Open-Meteo %s for (%s,%s) window_start=%s horizon=%sh",
        "archive" if use_archive else "forecast",
        config.latitude,
        config.longitude,
        start.isoformat(),
        config.forecast_hours,
    )
    payload = _request_json(url, params)
    hourly = payload.get("hourly")
    if not hourly:
        LOG.warning("Open-Meteo response missing hourly block")
        return pd.DataFrame()
    frame = pd.DataFrame(hourly)
    if "time" not in frame:
        LOG.warning("Open-Meteo payload missing time column")
        return pd.DataFrame()
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time")[list(HOURLY_FIELDS.keys())]
    frame = frame.rename(columns=HOURLY_FIELDS).sort_index()
    window = frame[(frame.index >= start) & (frame.index < end)]
    if window.empty:
        LOG.warning("Forecast window produced no rows")
        return window
    window = window.resample("30min").interpolate("time").ffill().bfill()
    return window.iloc[: config.half_hour_points]


def write_to_influx(config: WeatherConfig, forecast: pd.DataFrame) -> int:
    if forecast.empty:
        LOG.warning("No weather rows to write")
        return 0
    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=False,
    ) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        records = []
        for timestamp, row in forecast.iterrows():
            point = Point(config.measurement).time(timestamp.to_pydatetime(), WritePrecision.S)
            for field, value in row.items():
                if not pd.isna(value):
                    point = point.field(field, float(value))
            records.append(point)
        write_api.write(bucket=config.influx_bucket, org=config.influx_org, record=records)
    return len(forecast.index)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = WeatherConfig.from_env()
    rows = fetch_forecast(config)
    written = write_to_influx(config, rows)
    LOG.info("Ingest complete. Points written: %s", written)


if __name__ == "__main__":
    main()
