"""Data ingest: fetch Open-Meteo weather and aggregate PXC net load."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math
import os
from typing import Dict, Iterable, Sequence

import pandas as pd
import requests
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

LOG = logging.getLogger("ingest_data")
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY_FIELDS: Dict[str, str] = {
    "temperature_2m": "temp_air",
    "wind_speed_10m": "wind_speed",
    "shortwave_radiation": "ghi",
    "direct_normal_irradiance": "dni",
    "diffuse_radiation": "dhi",
}
PXC_MEASUREMENTS: tuple[str, ...] = ("PcsData", "Interconnection")
PXC_FIELDS: tuple[str, ...] = ("essPcsActvPwr", "poiActvPwr", "pvActvPwr")


@dataclass(frozen=True)
class IngestConfig:
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
    pxc_bucket: str
    historical_measurement: str
    netload_field: str
    solar_field: str
    forecast_hours: int

    @property
    def half_hour_points(self) -> int:
        return self.forecast_hours * 2

    @classmethod
    def from_env(cls) -> IngestConfig:
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
            pxc_bucket=env("PXC_BUCKET", "PXC"),
            historical_measurement=env("HISTORICAL_MEASUREMENT", "historical_data"),
            netload_field=env("NETLOAD_FIELD", "netload_kw"),
            solar_field=env("SOLAR_FIELD", "solarPV_kw"),
            forecast_hours=int(env("FORECAST_HOURS", "96")),
        )


def _request_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _now_utc(config: IngestConfig) -> datetime:
    """Return the current timestamp expressed in UTC but aligned with the site timezone."""
    local_now = pd.Timestamp.now(tz=config.timezone)
    return local_now.tz_convert("UTC").to_pydatetime()


def _flux_filter(column: str, values: Sequence[str]) -> str:
    clauses = [f'r["{column}"] == "{value}"' for value in values]
    return " or ".join(clauses)


def _window(config: IngestConfig) -> tuple[datetime, datetime, bool]:
    now_utc = _now_utc(config)
    if config.start_date:
        ts = pd.Timestamp(config.start_date)
        ts = ts.tz_localize(config.timezone) if ts.tzinfo is None else ts.tz_convert(config.timezone)
        start = ts.tz_convert("UTC").to_pydatetime()
    else:
        start = now_utc
    end = start + timedelta(hours=config.forecast_hours)
    return start, end, bool(config.start_date and end <= now_utc)


def _build_params(config: IngestConfig, fields: Iterable[str], *, archive: bool, start: datetime, end: datetime) -> dict:
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


def fetch_forecast(config: IngestConfig) -> pd.DataFrame:
    start, end, use_archive = _window(config)
    params = _build_params(config, HOURLY_FIELDS.keys(), archive=use_archive or bool(config.start_date), start=start, end=end)
    url = OPEN_METEO_ARCHIVE_URL if (use_archive or bool(config.start_date)) else OPEN_METEO_URL
    LOG.info(
        "Requesting Open-Meteo %s for (%s,%s) window_start=%s horizon=%sh",
        "archive" if url == OPEN_METEO_ARCHIVE_URL else "forecast",
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


def write_weather_to_influx(config: IngestConfig, forecast: pd.DataFrame) -> int:
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


def _as_dataframe(result: pd.DataFrame | Sequence[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(result, list):
        frames = [frame for frame in result if not frame.empty]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    return result


def fetch_historical_data(config: IngestConfig) -> pd.DataFrame:
    stop = _now_utc(config)
    start = stop - timedelta(days=7)
    measurement_filter = _flux_filter("_measurement", PXC_MEASUREMENTS)
    field_filter = _flux_filter("_field", PXC_FIELDS)
    flux = f"""
from(bucket: "{config.pxc_bucket}")
  |> range(start: {start.isoformat()}, stop: {stop.isoformat()})
  |> filter(fn: (r) => {measurement_filter})
  |> filter(fn: (r) => {field_filter})
  |> aggregateWindow(every: 30m, fn: mean, createEmpty: false)
  |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> keep(columns: ["_time","{PXC_FIELDS[0]}","{PXC_FIELDS[1]}","{PXC_FIELDS[2]}"])
  |> sort(columns:["_time"])
"""
    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=False,
        timeout=300,
    ) as client:
        query_api = client.query_api()
        raw = query_api.query_data_frame(org=config.influx_org, query=flux)
    frame = _as_dataframe(raw)
    if frame.empty or "_time" not in frame:
        LOG.warning("No PXC rows returned for the last 7 days")
        return pd.DataFrame()
    value_columns = [col for col in PXC_FIELDS if col in frame.columns]
    if not value_columns:
        LOG.warning("PXC query returned timestamps without expected fields")
        return pd.DataFrame()
    columns = ["_time"] + value_columns
    frame = frame[columns]
    frame = frame.rename(columns={"_time": "time"})
    frame["time"] = pd.to_datetime(frame["time"], utc=True)
    frame = frame.set_index("time").sort_index()
    frame = frame.resample("30min").mean()
    result = pd.DataFrame(index=frame.index)
    result[config.netload_field] = frame.reindex(columns=PXC_FIELDS).sum(axis=1, min_count=1)
    if "pvActvPwr" in frame.columns:
        result[config.solar_field] = frame["pvActvPwr"]
    else:
        LOG.warning("pvActvPwr not found in PXC data; solar field will be empty")
    result = result.dropna(how="all")
    return result


def write_historical_data(config: IngestConfig, historical: pd.DataFrame) -> int:
    if historical.empty:
        LOG.warning("No historical rows to write")
        return 0
    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=False,
    ) as client:
        write_api = client.write_api(write_options=SYNCHRONOUS)
        records = []
        for timestamp, row in historical.iterrows():
            point = Point(config.historical_measurement).time(timestamp.to_pydatetime(), WritePrecision.S)
            wrote_field = False
            netload_value = row.get(config.netload_field)
            if netload_value is not None and not pd.isna(netload_value):
                point = point.field(config.netload_field, float(netload_value))
                wrote_field = True
            solar_value = row.get(config.solar_field)
            if solar_value is not None and not pd.isna(solar_value):
                point = point.field(config.solar_field, float(solar_value))
                wrote_field = True
            if wrote_field:
                records.append(point)
        if not records:
            return 0
        write_api.write(bucket=config.influx_bucket, org=config.influx_org, record=records)
    return len(records)


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = IngestConfig.from_env()
    forecast_rows = fetch_forecast(config)
    weather_written = write_weather_to_influx(config, forecast_rows)
    historical_rows = fetch_historical_data(config)
    historical_written = write_historical_data(config, historical_rows)
    LOG.info(
        "Ingest complete. Weather points written=%s Historical points written=%s",
        weather_written,
        historical_written,
    )


if __name__ == "__main__":
    main()
