"""Load forecasting agent for the EPC1522 demo.

This module pulls historical load plus aligned weather forecasts from InfluxDB,
rebuilds the engineered feature set used in the notebook experiments, loads the
pre-trained XGBoost model from ``model/xgb_model.pkl``, runs a recursive
multi-step forecast, and writes the resulting horizon back to InfluxDB.
"""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import jpholiday
import numpy as np
import pandas as pd

import joblib


try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:  # pragma: no cover - handled at runtime
    InfluxDBClient = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    WritePrecision = None  # type: ignore[assignment]
    SYNCHRONOUS = None  # type: ignore[assignment]

LOGGER = logging.getLogger("load_forecast")
FORECAST_FEATURES = [
    "Temperature",
    "temp_business_hr",
    "temp_lag_1_day",
    "temp_rolling_mean_3hr",
    "temp_rolling_std_3hr",
    "temp_rolling_mean_24hr",
    "temp_squared",
    "hour_sin",
    "hour_cos",
    "is_business_hour",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "is_weekday",
    "time_frame",
    "is_holiday",
]


@dataclass
class LoadForecastConfig:
    site: str
    horizon_hours: int
    history_hours: int
    weather_measurement: str
    forecast_measurement: str
    model_path: Path
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


def load_xgb_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    LOGGER.info("Loading model artifact from %s", model_path)
    if joblib is not None:
        return joblib.load(model_path)
    with model_path.open("rb") as fh:
        return pickle.load(fh)


def _build_flux_filter_list(values: Iterable[str], field: str) -> str:
    clauses = [f'r["{field}"] == "{value}"' for value in values]
    if not clauses:
        return ""
    return " or ".join(clauses)


def query_influx_frame(
    client: InfluxDBClient,
    bucket: str,
    measurement: str,
    fields: Iterable[str],
    *,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    site: Optional[str] = None,
) -> pd.DataFrame:
    field_filter = _build_flux_filter_list(fields, "_field")
    field_filter_block = (
        f'  |> filter(fn: (r) => {field_filter})\n' if field_filter else ""
    )
    tag_filter_block = (
        f'  |> filter(fn: (r) => r["site"] == "{site}")\n' if site else ""
    )
    flux = f"""
from(bucket: \"{bucket}\")
  |> range(start: time(v: \"{start.isoformat()}\"), stop: time(v: \"{stop.isoformat()}\"))
  |> filter(fn: (r) => r[\"_measurement\"] == \"{measurement}\")
{field_filter_block}{tag_filter_block}  |> keep(columns: [\"_time\", \"_field\", \"_value\"])
  |> pivot(rowKey: [\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")
  |> sort(columns: [\"_time\"])
"""
    # Normalize whitespace introduced by optional blocks
    flux = "\n".join(line for line in flux.splitlines() if line.strip())
    query_api = client.query_api()
    frames = query_api.query_data_frame(org=client.org, query=flux)
    if isinstance(frames, list):
        frames = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if frames.empty:
        return pd.DataFrame()
    frames = frames.rename(columns={"_time": "timestamp"})
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], utc=True)
    frames = frames.set_index("timestamp").sort_index()
    frames = frames.loc[:, [col for col in fields if col in frames.columns]]
    return frames


def fetch_weather(client: InfluxDBClient, config: LoadForecastConfig, *, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    weather = query_influx_frame(
        client,
        config.influx_bucket,
        config.weather_measurement,
        fields=["temp_air", "ghi", "dni", "dhi", "wind_speed"],
        start=start,
        stop=stop,
        site=config.site,
    )
    if weather.empty:
        raise RuntimeError("No weather data returned from Influx. Run ingest first.")
    weather = weather.rename(columns={"temp_air": "Temperature"})
    weather = weather.sort_index().asfreq("30min", method="pad")
    return weather


def engineer_features(df: pd.DataFrame, *, dropna: bool = False) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat = df_feat.reset_index().rename(columns={"index": "timestamp"})
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], utc=True)

    df_feat["hour"] = df_feat["timestamp"].dt.hour
    df_feat["day_of_week"] = df_feat["timestamp"].dt.dayofweek
    df_feat["month"] = df_feat["timestamp"].dt.month
    df_feat["is_weekday"] = df_feat["day_of_week"].apply(lambda x: 1 if x < 6 else 0)
    df_feat["is_holiday"] = df_feat["timestamp"].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
    df_feat["is_business_hour"] = ((df_feat["hour"] >= 8) & (df_feat["hour"] <= 22)).astype(int)
    df_feat["time_frame"] = df_feat["hour"] * 2 + (df_feat["timestamp"].dt.minute // 30) + 1

    df_feat["temp_business_hr"] = df_feat["Temperature"] * df_feat["is_business_hour"]
    df_feat["temp_lag_1_day"] = (
        df_feat["Temperature"].shift(96).fillna(df_feat["Temperature"])
    )
    df_feat["temp_rolling_mean_3hr"] = (
        df_feat["Temperature"]
        .shift(1)
        .rolling(window=6, min_periods=1)
        .mean()
        .fillna(df_feat["Temperature"])
    )
    df_feat["temp_rolling_std_3hr"] = (
        df_feat["Temperature"]
        .shift(1)
        .rolling(window=6, min_periods=1)
        .std()
        .fillna(0.0)
    )
    df_feat["temp_rolling_mean_24hr"] = (
        df_feat["Temperature"]
        .shift(1)
        .rolling(window=48, min_periods=1)
        .mean()
        .fillna(df_feat["Temperature"])
    )
    df_feat["temp_squared"] = df_feat["Temperature"] ** 2

    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["day_of_week_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["day_of_week_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)

    keep_cols = set(FORECAST_FEATURES)
    drop_cols = [col for col in df_feat.columns if col not in keep_cols and col != "timestamp"]
    df_feat = df_feat.drop(columns=drop_cols)
    df_feat = df_feat.set_index("timestamp").sort_index()

    if dropna:
        df_feat = df_feat.dropna()
    else:
        df_feat = df_feat.dropna(subset=["Temperature"])
    return df_feat


def predict_forecast(
    model,
    features: pd.DataFrame,
    *,
    start_time: pd.Timestamp,
    horizon_steps: int,
) -> pd.DataFrame:
    if features.empty:
        raise RuntimeError("Feature frame is empty; cannot forecast")

    future_rows = features[features.index >= start_time]
    if future_rows.empty:
        raise RuntimeError("No feature rows available for requested forecast window")
    future_rows = future_rows.iloc[:horizon_steps]
    if len(future_rows) < horizon_steps:
        LOGGER.warning(
            "Requested %s steps but only %s feature rows available",
            horizon_steps,
            len(future_rows),
        )
    predictions = model.predict(future_rows[FORECAST_FEATURES])
    return pd.DataFrame(
        {
            "timestamp": future_rows.index,
            "load_forecast_kw": predictions,
        }
    )


def write_forecasts(
    df: pd.DataFrame,
    *,
    measurement: str,
    client: InfluxDBClient,
    bucket: str,
    org: str,
    tags: Optional[Dict[str, str]] = None,
) -> int:
    if InfluxDBClient is None or Point is None:
        raise RuntimeError("influxdb-client package is required to write forecasts")
    if df.empty:
        LOGGER.warning("No forecast rows to write")
        return 0

    records: List[Point] = []
    tags = tags or {}
    issued_at = pd.Timestamp.now(tz="UTC")
    for _, row in df.iterrows():
        point = (
            Point(measurement)
            .time(pd.Timestamp(row["timestamp"]).to_pydatetime(), WritePrecision.NS)
            .field("load_forecast_kw", float(row["load_forecast_kw"]))
            .field("issued_at", issued_at.isoformat())
        )
        for key, value in tags.items():
            point = point.tag(key, value)
        records.append(point)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org=org, record=records)
    return len(records)


def build_config_from_args(args: argparse.Namespace) -> LoadForecastConfig:
    return LoadForecastConfig(
        site=args.site,
        horizon_hours=args.horizon_hours,
        history_hours=args.history_hours,
        weather_measurement=args.weather_measurement,
        forecast_measurement=args.forecast_measurement,
        model_path=Path(args.model_path),
        influx_url=args.influx_url,
        influx_token=args.influx_token,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        verify_ssl=args.verify_ssl,
    )


def run_forecast(config: LoadForecastConfig) -> int:
    model = load_xgb_model(config.model_path)
    now = pd.Timestamp.now(tz="UTC").floor("30min")
    history_start = now - pd.Timedelta(hours=config.history_hours)
    forecast_end = now + pd.Timedelta(hours=config.horizon_hours)

    if InfluxDBClient is None:
        raise RuntimeError("influxdb-client package is required for load forecast agent")

    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=config.verify_ssl,
    ) as client:
        weather_df = fetch_weather(client, config, start=history_start, stop=forecast_end)
        features = engineer_features(weather_df)
        horizon_steps = int(config.horizon_hours * 2)
        forecast_df = predict_forecast(
            model,
            features,
            start_time=now,
            horizon_steps=horizon_steps,
        )
        tags = {"site": config.site, "model": config.model_path.stem}
        written = write_forecasts(
            forecast_df,
            measurement=config.forecast_measurement,
            client=client,
            bucket=config.influx_bucket,
            org=config.influx_org,
            tags=tags,
        )
        LOGGER.info("Load forecast completed. Rows written: %s", written)
        return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run load forecast inference and persist results")
    parser.add_argument("--site", default=os.getenv("SITE_NAME", "default"), help="Site tag")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=int(os.getenv("LOAD_FORECAST_HOURS", 24)),
        dest="horizon_hours",
        help="Forecast horizon in hours (default: 24)",
    )
    parser.add_argument(
        "--history-hours",
        type=int,
        default=int(os.getenv("LOAD_HISTORY_HOURS", 72)),
        dest="history_hours",
        help="Historic window to build features (default: 72)",
    )
    parser.add_argument(
        "--weather-measurement",
        default=os.getenv("WEATHER_MEASUREMENT", "weather_forecast"),
        help="Measurement storing weather forecasts",
    )
    parser.add_argument(
        "--forecast-measurement",
        default=os.getenv("FORECAST_MEASUREMENT", "forecasts"),
        help="Measurement to store load forecasts",
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("LOAD_MODEL_PATH", "model/xgb_model.pkl"),
        help="Path to the serialized XGBoost model",
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
        help="InfluxDB bucket for reading/writing",
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
        help="Logging verbosity",
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
        parser.error(f"Missing required Influx configuration values: {', '.join(missing)}")

    return args


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    run_forecast(config)


if __name__ == "__main__":
    main()
