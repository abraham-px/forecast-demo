"""Load forecasting agent for the EPC1522 demo.

This module pulls aligned weather forecasts from InfluxDB, rebuilds a lightweight
feature set, loads the pre-trained XGBoost model from ``model/xgb_model.pkl``,
generates a 24h (configurable) forecast, and writes the horizon back to InfluxDB.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

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
    # "Temperature",
    "temp_business_hr",
    # "temp_lag_1_day",
    # "temp_rolling_mean_3hr",
    # "temp_rolling_std_3hr",
    # "temp_rolling_mean_24hr",
    # "temp_squared",
    "hour_sin",
    "hour_cos",
    # "season",
    "is_business_hour",
    "month_sin",
    "month_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    # "is_weekday",
    "time_frame",
    "is_holiday",
    # "week_of_year",
] # must be kept in sync with model training


@dataclass
class LoadForecastConfig:
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
    timezone: str


def build_config_from_env() -> LoadForecastConfig:
    return LoadForecastConfig(
        horizon_hours=int(os.getenv("LOAD_FORECAST_HOURS", "24")),
        history_hours=int(os.getenv("LOAD_HISTORY_HOURS", "72")),
        weather_measurement=os.getenv("WEATHER_MEASUREMENT", "weather_forecast"),
        forecast_measurement=os.getenv("FORECAST_MEASUREMENT", "forecasts"),
        model_path=Path(os.getenv("LOAD_MODEL_PATH", "model/xgb_model.pkl")),
        influx_url=os.getenv("INFLUX_URL", ""),
        influx_token=os.getenv("INFLUX_TOKEN", ""),
        influx_org=os.getenv("INFLUX_ORG", ""),
        influx_bucket=os.getenv("INFLUX_BUCKET", ""),
        verify_ssl=(
            str(os.getenv("INFLUX_VERIFY_SSL", "true")).lower()
            in {"1", "true", "yes", "on"}
        ),
        timezone=os.getenv("SITE_TIMEZONE", "UTC"),
    )


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


def _resolve_start(ts_env: Optional[str], tz: str) -> pd.Timestamp:
    if ts_env:
        ts = pd.Timestamp(ts_env)
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
    else:
        ts = pd.Timestamp.now(tz=tz)
    return ts.tz_localize(None)


def _calculate_season(month: int) -> int:
    if month in (12, 1, 2):
        return 0  # winter
    if month in (3, 4, 5):
        return 1  # spring
    if month in (6, 7, 8):
        return 2  # summer
    return 3  # autumn


def query_influx_frame(
    client: InfluxDBClient,
    bucket: str,
    measurement: str,
    fields: Iterable[str],
    *,
    start: pd.Timestamp,
    stop: pd.Timestamp,
    site: Optional[str] = None,
    timezone: str = "UTC",
) -> pd.DataFrame:
    field_filter = _build_flux_filter_list(fields, "_field")
    field_filter_block = (
        f'  |> filter(fn: (r) => {field_filter})\n' if field_filter else ""
    )
    tag_filter_block = (
        f'  |> filter(fn: (r) => r["site"] == "{site}")\n' if site else ""
    )
    tz_start = start.tz_localize(timezone) if start.tzinfo is None else start.tz_convert(timezone)
    tz_stop = stop.tz_localize(timezone) if stop.tzinfo is None else stop.tz_convert(timezone)
    flux = f"""
from(bucket: \"{bucket}\")
  |> range(start: time(v: \"{tz_start.isoformat()}\"), stop: time(v: \"{tz_stop.isoformat()}\")) 
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
    frames["timestamp"] = pd.to_datetime(frames["timestamp"], utc =False)
    frames = frames.set_index("timestamp").sort_index()
    frames = frames.loc[:, [col for col in fields if col in frames.columns]]
    return frames


def fetch_weather(
    client: InfluxDBClient,
    config: LoadForecastConfig,
    *,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    weather = query_influx_frame(
        client,
        config.influx_bucket,
        config.weather_measurement,
        fields=["temp_air", "ghi", "dni", "dhi", "wind_speed"],
        start=start,
        stop=stop,
        site=None,
        timezone=config.timezone,
    )
    if weather.empty:
        raise RuntimeError("No weather data returned from Influx. Run ingest first.")
    weather = weather.rename(columns={"temp_air": "Temperature"})
    weather = weather.sort_index().asfreq("30min", method="pad")
    return weather


def engineer_features(df: pd.DataFrame, *, dropna: bool = False) -> pd.DataFrame:
    df_feat = df.copy()
    df_feat = df_feat.reset_index().rename(columns={"index": "timestamp"})
    df_feat["timestamp"] = pd.to_datetime(df_feat["timestamp"], utc=False)

    df_feat["hour"] = df_feat["timestamp"].dt.hour
    df_feat["day_of_week"] = df_feat["timestamp"].dt.dayofweek
    df_feat["month"] = df_feat["timestamp"].dt.month
    df_feat["is_weekday"] = df_feat["day_of_week"].apply(lambda x: 1 if x < 6 else 0)
    df_feat["is_holiday"] = df_feat["timestamp"].apply(lambda x: 1 if jpholiday.is_holiday(x) else 0)
    df_feat["is_business_hour"] = ((df_feat["hour"] >= 8) & (df_feat["hour"] <= 19)).astype(int) # technos business hours are 8:00 AM to 7:00 PM
    df_feat["time_frame"] = df_feat["hour"] * 2 + (df_feat["timestamp"].dt.minute // 30) + 1
    df_feat["week_of_year"] = df_feat["timestamp"].dt.isocalendar().week.astype(int)
    df_feat["season"] = df_feat["month"].apply(_calculate_season)

    df_feat["temp_business_hr"] = df_feat["Temperature"] * df_feat["is_business_hour"]
    df_feat["temp_lag_1_day"] = (
        df_feat["Temperature"].shift(96).fillna(df_feat["Temperature"])
    )
    df_feat["temp_rolling_mean_3hr"] = (
        df_feat["Temperature"]
        .shift(1)
        .rolling(window=12, min_periods=1)
        .mean()
        .fillna(df_feat["Temperature"])
    )
    df_feat["temp_rolling_std_3hr"] = (
        df_feat["Temperature"]
        .shift(1)
        .rolling(window=12, min_periods=1)
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

    base_features = set(FORECAST_FEATURES)
    keep_cols = base_features | {"timestamp"}
    drop_cols = [col for col in df_feat.columns if col not in keep_cols]
    df_feat = df_feat.drop(columns=drop_cols)
    df_feat = df_feat.set_index("timestamp").sort_index()

    # Only enforce dropna when explicitly requested; otherwise fill forward/back.
    if dropna:
        df_feat = df_feat.dropna()
    else:
        df_feat = df_feat.ffill().bfill()
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
    inputs = future_rows.reindex(columns=FORECAST_FEATURES)
    if inputs.isnull().any().any():
        LOGGER.warning("Feature matrix contains NaNs; filling forward/backward")
        inputs = inputs.ffill().bfill()
    predictions = model.predict(inputs)
    return pd.DataFrame(
        {"timestamp": inputs.index, "load_forecast_kw": predictions.astype(float)}
    )


def write_forecasts(
    df: pd.DataFrame,
    *,
    measurement: str,
    client: InfluxDBClient,
    bucket: str,
    org: str,
) -> int:
    if InfluxDBClient is None or Point is None:
        raise RuntimeError("influxdb-client package is required to write forecasts")
    if df.empty:
        LOGGER.warning("No forecast rows to write")
        return 0

    records: List[Point] = []

    for _, row in df.iterrows():
        point = (
            Point(measurement)
            .time(pd.Timestamp(row["timestamp"]).to_pydatetime(), WritePrecision.NS)
            .field("load_forecast_kw", float(row["load_forecast_kw"]))
        )
        records.append(point)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org=org, record=records)
    return len(records)


def run_forecast(config: LoadForecastConfig) -> int:
    model = load_xgb_model(config.model_path)
    start_override = os.getenv("LOAD_START_DATE") or os.getenv("START_DATE")
    window_start = _resolve_start(start_override, config.timezone).floor("30min")
    history_start = window_start - pd.Timedelta(hours=config.history_hours)
    forecast_end = window_start + pd.Timedelta(hours=config.horizon_hours)

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
        horizon_steps = int(config.horizon_hours * 2)  # 30-min steps
        forecast_df = predict_forecast(
            model,
            features,
            start_time=window_start,
            horizon_steps=horizon_steps,
        )
        local_end = window_start + pd.Timedelta(hours=config.horizon_hours)
        LOGGER.info("Load window local %s -> %s", window_start, local_end)
        forecast_df["timestamp"] = pd.to_datetime(forecast_df["timestamp"])
        forecast_df["timestamp"] = (
            forecast_df["timestamp"]
            .dt.tz_localize(config.timezone)
            .dt.tz_convert("UTC")
        )
        local_start = window_start.tz_convert(config.timezone)
        local_end = (
            window_start + pd.Timedelta(hours=config.horizon_hours)
        ).tz_convert(config.timezone)
        LOGGER.info(
            "Load window UTC %s -> %s | local %s -> %s",
            window_start,
            window_start + pd.Timedelta(hours=config.horizon_hours),
            local_start,
            local_end,
        )
        written = write_forecasts(
            forecast_df,
            measurement=config.forecast_measurement,
            client=client,
            bucket=config.influx_bucket,
            org=config.influx_org,
        )
        LOGGER.info("Load forecast completed. Rows written: %s", written)
        return written


def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    config = build_config_from_env()
    run_forecast(config)


if __name__ == "__main__":
    main()
