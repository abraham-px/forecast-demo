"""Weather ingestion job for EPC1522 demo.

This module fetches forecast data from Open-Meteo, interpolates it to the
30-minute cadence required by downstream load/PV forecasters, and persists the
result into the shared InfluxDB bucket (weather_forecast table).
"""
from __future__ import annotations

import argparse
import logging
import math
import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Optional

import pandas as pd
import requests

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:  # pragma: no cover - handled at runtime
    InfluxDBClient = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    WritePrecision = None  # type: ignore[assignment]
    SYNCHRONOUS = None  # type: ignore[assignment]

LOGGER = logging.getLogger("ingest_weather")
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
HOURLY_VARIABLES = [
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "direct_normal_irradiance",
    "diffuse_radiation",
]
RENAME_MAP = {
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
}
FIELD_COLUMNS = list(RENAME_MAP.values())


@dataclass
class WeatherIngestConfig:
    """Typed configuration for a weather ingest run."""

    latitude: float
    longitude: float
    hours: int
    site: str
    model: str
    measurement: str
    influx_url: str
    influx_token: str
    influx_org: str
    influx_bucket: str
    verify_ssl: bool


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def fetch_open_meteo_forecast(
    latitude: float,
    longitude: float,
    *,
    hours: int,
    model: str,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """Fetch hourly forecast data from Open-Meteo and return a DataFrame."""

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": HOURLY_VARIABLES,
        "forecast_days": max(1, math.ceil(hours / 24)),
        "timezone": timezone,
        "models": model,
    }
    LOGGER.info(
        "Requesting Open-Meteo forecast for lat=%s lon=%s horizon=%sh model=%s",
        latitude,
        longitude,
        hours,
        model,
    )
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not hourly:
        raise RuntimeError("Open-Meteo response missing 'hourly' block")

    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = (
        frame.set_index("time")
        .loc[:, list(RENAME_MAP.keys())]
        .rename(columns=RENAME_MAP)
        .sort_index()
    )
    frame = frame[~frame.index.duplicated(keep="last")]

    # Limit to requested horizon starting from now (UTC)
    now = pd.Timestamp.now(tz="UTC")
    end_time = now + pd.Timedelta(hours=hours)
    trimmed = frame[(frame.index >= now) & (frame.index <= end_time)]
    if trimmed.empty:
        LOGGER.warning(
            "Forecast response did not overlap requested window; defaulting to head slice"
        )
        trimmed = frame.head(hours + 1)
    return trimmed


def resample_to_half_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate the hourly data to 30-minute resolution."""

    if df.empty:
        return df
    resampled = df.resample("30min").interpolate(method="time").ffill().bfill()
    return resampled


def write_weather_to_influx(
    df: pd.DataFrame,
    measurement: str,
    bucket: str,
    org: str,
    url: str,
    token: str,
    *,
    verify_ssl: bool,
    tags: Optional[Dict[str, str]] = None,
) -> int:
    """Persist the DataFrame into InfluxDB and return number of points written."""

    if InfluxDBClient is None or Point is None:
        raise RuntimeError(
            "influxdb-client package is not installed. Install it before running ingestion."
        )
    if df.empty:
        LOGGER.warning("No weather rows to persist; skipping write")
        return 0

    tags = tags or {}
    records = []
    for timestamp, row in df.iterrows():
        point = Point(measurement).time(timestamp.to_pydatetime(), WritePrecision.NS)
        for column in FIELD_COLUMNS:
            value = row.get(column)
            if pd.isna(value):
                continue
            point = point.field(column, float(value))
        for key, value in tags.items():
            if value is None:
                continue
            point = point.tag(key, value)
        records.append(point)

    if not records:
        LOGGER.warning("All rows were empty after filtering; nothing to write")
        return 0

    LOGGER.info(
        "Writing %s points to Influx bucket=%s measurement=%s",
        len(records),
        bucket,
        measurement,
    )
    with InfluxDBClient(url=url, token=token, org=org, verify_ssl=verify_ssl) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        write_api.write(bucket=bucket, org=org, record=records)
    return len(records)


def build_config_from_args(args: argparse.Namespace) -> WeatherIngestConfig:
    return WeatherIngestConfig(
        latitude=args.latitude,
        longitude=args.longitude,
        hours=args.hours,
        site=args.site,
        model=args.model,
        measurement=args.measurement,
        influx_url=args.influx_url,
        influx_token=args.influx_token,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        verify_ssl=args.verify_ssl,
    )


def run_ingest(config: WeatherIngestConfig) -> int:
    forecast = fetch_open_meteo_forecast(
        config.latitude,
        config.longitude,
        hours=config.hours,
        model=config.model,
    )
    interpolated = resample_to_half_hour(forecast)
    interpolated = interpolated.iloc[: int(config.hours * 2) + 1]
    tags = {"provider": "open-meteo", "site": config.site}
    written = write_weather_to_influx(
        interpolated,
        config.measurement,
        config.influx_bucket,
        config.influx_org,
        config.influx_url,
        config.influx_token,
        verify_ssl=config.verify_ssl,
        tags=tags,
    )
    LOGGER.info("Weather ingest finished. Points written: %s", written)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest weather forecasts into InfluxDB")
    parser.add_argument(
        "--latitude", type=float, required=True, help="Site latitude in decimal degrees"
    )
    parser.add_argument(
        "--longitude", type=float, required=True, help="Site longitude in decimal degrees"
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=int(os.getenv("FORECAST_HOURS", 48)),
        help="Forecast horizon in hours (default: 48)",
    )
    parser.add_argument(
        "--site",
        default=os.getenv("SITE_NAME", "default"),
        help="Site identifier stored as tag in Influx",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPEN_METEO_MODEL", "best_match"),
        help="Open-Meteo model identifier (e.g., best_match, jma_msm)",
    )
    parser.add_argument(
        "--measurement",
        default=os.getenv("WEATHER_MEASUREMENT", "weather_forecast"),
        help="Influx measurement name for persisted weather data",
    )
    parser.add_argument("--influx-url", default=os.getenv("INFLUX_URL"), help="InfluxDB URL")
    parser.add_argument(
        "--influx-token", default=os.getenv("INFLUX_TOKEN"), help="InfluxDB API token"
    )
    parser.add_argument(
        "--influx-org", default=os.getenv("INFLUX_ORG"), help="InfluxDB organization"
    )
    parser.add_argument(
        "--influx-bucket",
        default=os.getenv("INFLUX_BUCKET", "AIML"),
        help="InfluxDB bucket for writing weather data",
    )
    parser.add_argument(
        "--verify-ssl",
        default=os.getenv("INFLUX_VERIFY_SSL", "true"),
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="Toggle TLS certificate verification",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)

    required = {
        "influx_url": args.influx_url,
        "influx_token": args.influx_token,
        "influx_org": args.influx_org,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(
            f"Missing required Influx configuration values: {', '.join(missing)}"
        )

    return args


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    run_ingest(config)


if __name__ == "__main__":
    main()
