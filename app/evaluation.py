"""Daily evaluation job for load and PV forecasts."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:  # pragma: no cover
    InfluxDBClient = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    WritePrecision = None  # type: ignore[assignment]
    SYNCHRONOUS = None  # type: ignore[assignment]

LOGGER = logging.getLogger("evaluation")


@dataclass
class EvaluationConfig:
    window_hours: int
    actual_measurement: str
    forecast_measurement: str
    evaluation_measurement: str
    influx_url: str
    influx_token: str
    influx_org: str
    influx_bucket: str
    verify_ssl: bool


def build_config_from_env() -> EvaluationConfig:
    return EvaluationConfig(
        window_hours=int(os.getenv("EVALUATION_WINDOW_HOURS", "24")),
        actual_measurement=os.getenv("HISTORICAL_MEASUREMENT", "historical_actuals"),
        forecast_measurement=os.getenv("FORECAST_MEASUREMENT", "forecasts"),
        evaluation_measurement=os.getenv("EVALUATION_MEASUREMENT", "evaluations"),
        influx_url=os.getenv("INFLUX_URL", ""),
        influx_token=os.getenv("INFLUX_TOKEN", ""),
        influx_org=os.getenv("INFLUX_ORG", ""),
        influx_bucket=os.getenv("INFLUX_BUCKET", ""),
        verify_ssl=(
            str(os.getenv("INFLUX_VERIFY_SSL", "true")).lower()
            in {"1", "true", "yes", "on"}
        ),
    )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _build_filter_clause(values: Iterable[str], field: str) -> str:
    clauses = [f'r["{field}"] == "{value}"' for value in values]
    return " or ".join(clauses)


def query_influx(
    client: InfluxDBClient,
    bucket: str,
    measurement: str,
    fields: Iterable[str],
    *,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    field_clause = _build_filter_clause(fields, "_field")
    field_filter_block = (
        f'  |> filter(fn: (r) => {field_clause})\n' if field_clause else ""
    )
    flux = f"""
from(bucket: "{bucket}")
  |> range(start: time(v: "{start.isoformat()}"), stop: time(v: "{stop.isoformat()}"))
  |> filter(fn: (r) => r["_measurement"] == "{measurement}")
{field_filter_block}  |> keep(columns: ["_time", "_field", "_value"])
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
"""
    flux = "\n".join(line for line in flux.splitlines() if line.strip())
    frames = client.query_api().query_data_frame(org=client.org, query=flux)
    if isinstance(frames, list):
        frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frames.empty:
        return pd.DataFrame()
    frames = frames.rename(columns={"_time": "timestamp"})
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], utc=True)
    frames = frames.set_index("timestamp")
    columns = [c for c in fields if c in frames.columns]
    return frames[columns].sort_index()


def fetch_actuals(client: InfluxDBClient, config: EvaluationConfig, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    fields = ["netload_kw", "pv_kw"]
    return query_influx(
        client,
        config.influx_bucket,
        config.actual_measurement,
        fields=fields,
        start=start,
        stop=stop,
    )


def fetch_forecasts(client: InfluxDBClient, config: EvaluationConfig, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    fields = ["load_forecast_kw", "solarpv_forecast_kw"]
    return query_influx(
        client,
        config.influx_bucket,
        config.forecast_measurement,
        fields=fields,
        start=start,
        stop=stop,
    )


def compute_metric(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    valid = pd.concat([actual, forecast], axis=1, join="inner").dropna()
    if valid.empty:
        return {}
    y_true = valid.iloc[:, 0].to_numpy(dtype=float)
    y_pred = valid.iloc[:, 1].to_numpy(dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    denom = np.maximum(np.abs(y_true), 1e-3)
    mape = np.mean(np.abs((y_true - y_pred) / denom)) * 100.0
    return {"mae": float(mae), "rmse": float(rmse), "mape": float(mape), "count": len(valid)}


def write_evaluation(
    metrics: dict[str, dict[str, float]],
    *,
    client: InfluxDBClient,
    config: EvaluationConfig,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> None:
    if Point is None:
        raise RuntimeError("influxdb-client package is required to write evaluation metrics")
    issued_at = pd.Timestamp.now(tz="UTC")
    point = (
        Point(config.evaluation_measurement)
        .time(issued_at.to_pydatetime(), WritePrecision.NS)
        .field("window_hours", float(config.window_hours))
        .field("window_start", window_start.isoformat())
        .field("window_end", window_end.isoformat())
    )
    for prefix, values in metrics.items():
        for key, value in values.items():
            point = point.field(f"{prefix}_{key}", float(value))
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=config.influx_bucket, org=config.influx_org, record=point)


def run_evaluation(config: EvaluationConfig) -> None:
    if InfluxDBClient is None:
        raise RuntimeError("influxdb-client package is required for evaluation")
    now = pd.Timestamp.now(tz="UTC")
    start = now - pd.Timedelta(hours=config.window_hours)
    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=config.verify_ssl,
    ) as client:
        actuals = fetch_actuals(client, config, start=start, stop=now)
        forecasts = fetch_forecasts(client, config, start=start, stop=now)
        if actuals.empty or forecasts.empty:
            LOGGER.warning("Insufficient data for evaluation (actuals or forecasts empty)")
            return
        combined = actuals.join(forecasts, how="inner")
        load_metrics = compute_metric(
            combined["netload_kw"], combined["load_forecast_kw"]
        ) if {"netload_kw", "load_forecast_kw"} <= set(combined.columns) else {}
        pv_metrics = compute_metric(
            combined["pv_kw"], combined["solarpv_forecast_kw"]
        ) if {"pv_kw", "solarpv_forecast_kw"} <= set(combined.columns) else {}
        metrics = {}
        if load_metrics:
            metrics["load"] = load_metrics
        if pv_metrics:
            metrics["pv"] = pv_metrics
        if not metrics:
            LOGGER.warning("No overlapping series available for evaluation")
            return
        write_evaluation(metrics, client=client, config=config, window_start=start, window_end=now)
        LOGGER.info("Evaluation written for window %s → %s", start, now)


def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    config = build_config_from_env()
    run_evaluation(config)


if __name__ == "__main__":
    main()
