"""Lightweight sequential scheduler for EPC1522 demo agents.

The scheduler pulls configuration from environment variables (see `.env`) and
runs the weather ingestion, load forecast, and PV simulation jobs in order.
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence, Tuple

from evaluation import EvaluationConfig, run_evaluation
from ingest_weather import WeatherIngestConfig, run_ingest
from load_forecast import LoadForecastConfig, run_forecast
from solarpv_simu import SolarPVSimConfig, run_simulation

LOGGER = logging.getLogger("scheduler")


@dataclass
class SchedulerConfig:
    interval_minutes: int
    run_once: bool
    initial_delay_seconds: int
    evaluation_interval_minutes: int


@dataclass
class SchedulerState:
    last_evaluation_epoch: float | None = None


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None else default


def build_weather_config_from_env() -> WeatherIngestConfig:
    return WeatherIngestConfig(
        latitude=_env_float("SITE_LATITUDE", 0.0),
        longitude=_env_float("SITE_LONGITUDE", 0.0),
        hours=_env_int("FORECAST_HOURS", 48),
        site=_env_str("SITE_NAME", "default"),
        model=_env_str("OPEN_METEO_MODEL", "best_match"),
        measurement=_env_str("WEATHER_MEASUREMENT", "weather_forecast"),
        influx_url=_env_str("INFLUX_URL", ""),
        influx_token=_env_str("INFLUX_TOKEN", ""),
        influx_org=_env_str("INFLUX_ORG", ""),
        influx_bucket=_env_str("INFLUX_BUCKET", ""),
        verify_ssl=_to_bool(os.getenv("INFLUX_VERIFY_SSL"), True),
    )


def build_load_config_from_env() -> LoadForecastConfig:
    return LoadForecastConfig(
        site=_env_str("SITE_NAME", "default"),
        horizon_hours=_env_int("LOAD_FORECAST_HOURS", 24),
        history_hours=_env_int("LOAD_HISTORY_HOURS", 72),
        weather_measurement=_env_str("WEATHER_MEASUREMENT", "weather_forecast"),
        forecast_measurement=_env_str("FORECAST_MEASUREMENT", "forecasts"),
        model_path=Path(_env_str("LOAD_MODEL_PATH", "model/xgb_model.pkl")),
        influx_url=_env_str("INFLUX_URL", ""),
        influx_token=_env_str("INFLUX_TOKEN", ""),
        influx_org=_env_str("INFLUX_ORG", ""),
        influx_bucket=_env_str("INFLUX_BUCKET", ""),
        verify_ssl=_to_bool(os.getenv("INFLUX_VERIFY_SSL"), True),
    )


def build_pv_config_from_env() -> SolarPVSimConfig:
    return SolarPVSimConfig(
        site=_env_str("SITE_NAME", "default"),
        horizon_hours=_env_int("PV_FORECAST_HOURS", 24),
        weather_measurement=_env_str("WEATHER_MEASUREMENT", "weather_forecast"),
        forecast_measurement=_env_str("FORECAST_MEASUREMENT", "forecasts"),
        influx_url=_env_str("INFLUX_URL", ""),
        influx_token=_env_str("INFLUX_TOKEN", ""),
        influx_org=_env_str("INFLUX_ORG", ""),
        influx_bucket=_env_str("INFLUX_BUCKET", ""),
        verify_ssl=_to_bool(os.getenv("INFLUX_VERIFY_SSL"), True),
        latitude=_env_float("SITE_LATITUDE", 0.0),
        longitude=_env_float("SITE_LONGITUDE", 0.0),
        altitude=_env_float("SITE_ALTITUDE", 0.0),
        timezone=_env_str("SITE_TIMEZONE", "UTC"),
        surface_tilt=_env_float("PV_SURFACE_TILT", 20.0),
        surface_azimuth=_env_float("PV_SURFACE_AZIMUTH", 180.0),
        dc_capacity_kw=_env_float("PV_DC_CAPACITY_KW", 0.0),
        ac_capacity_kw=_env_float("PV_AC_CAPACITY_KW", 0.0),
        gamma_pdc=_env_float("PV_GAMMA_PDC", -0.003),
        albedo=_env_float("PV_ALBEDO", 0.2),
        model_name=_env_str("PV_MODEL_NAME", "pvlib_pvwatts"),
    )

def build_evaluation_config_from_env() -> EvaluationConfig:
    return EvaluationConfig(
        site=_env_str("SITE_NAME", "default"),
        window_hours=_env_int("EVALUATION_WINDOW_HOURS", 24),
        actual_measurement=_env_str("HISTORICAL_MEASUREMENT", "historical_actuals"),
        forecast_measurement=_env_str("FORECAST_MEASUREMENT", "forecasts"),
        evaluation_measurement=_env_str("EVALUATION_MEASUREMENT", "evaluations"),
        influx_url=_env_str("INFLUX_URL", ""),
        influx_token=_env_str("INFLUX_TOKEN", ""),
        influx_org=_env_str("INFLUX_ORG", ""),
        influx_bucket=_env_str("INFLUX_BUCKET", ""),
        verify_ssl=_to_bool(os.getenv("INFLUX_VERIFY_SSL"), True),
    )


def run_weather_job() -> None:
    config = build_weather_config_from_env()
    run_ingest(config)


def run_load_job() -> None:
    config = build_load_config_from_env()
    run_forecast(config)


def run_pv_job() -> None:
    config = build_pv_config_from_env()
    run_simulation(config)

def run_evaluation_job() -> None:
    config = build_evaluation_config_from_env()
    run_evaluation(config)

def execute_cycle(state: SchedulerState, evaluation_interval_minutes: int) -> None:
    jobs: Sequence[Tuple[str, Callable[[], None]]] = (
        ("weather_ingest", run_weather_job),
        ("load_forecast", run_load_job),
        ("pv_simulation", run_pv_job),
    )
    for name, job in jobs:
        LOGGER.info("Starting job: %s", name)
        try:
            job()
            LOGGER.info("Job succeeded: %s", name)
        except Exception:
            LOGGER.exception("Job failed: %s", name)
    if should_run_evaluation(state, evaluation_interval_minutes):
        LOGGER.info("Starting job: evaluation")
        try:
            run_evaluation_job()
            state.last_evaluation_epoch = time.time()
            LOGGER.info("Job succeeded: evaluation")
        except Exception:
            LOGGER.exception("Job failed: evaluation")


def parse_args() -> SchedulerConfig:
    parser = argparse.ArgumentParser(description="Sequential scheduler for EPC1522 demo")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=_env_int("SCHEDULE_INTERVAL_MINUTES", 30),
        help="Minutes between scheduler cycles",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        default=_to_bool(os.getenv("SCHEDULER_RUN_ONCE"), False),
        help="Run a single cycle instead of looping",
    )
    parser.add_argument(
        "--initial-delay",
        type=int,
        default=_env_int("SCHEDULER_INITIAL_DELAY_SECONDS", 0),
        help="Seconds to sleep before first cycle",
    )
    parser.add_argument(
        "--evaluation-interval-minutes",
        type=int,
        default=_env_int("EVALUATION_INTERVAL_MINUTES", 24 * 60),
        help="Minutes between evaluation job executions",
    )
    parser.add_argument(
        "--log-level",
        default=_env_str("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return SchedulerConfig(
        interval_minutes=args.interval_minutes,
        run_once=args.run_once,
        initial_delay_seconds=args.initial_delay,
        evaluation_interval_minutes=args.evaluation_interval_minutes,
    )


def should_run_evaluation(state: SchedulerState, interval_minutes: int) -> bool:
    if interval_minutes <= 0:
        return False
    now = time.time()
    if state.last_evaluation_epoch is None:
        return True
    return now - state.last_evaluation_epoch >= interval_minutes * 60


def main() -> None:
    config = parse_args()
    state = SchedulerState()
    if config.initial_delay_seconds:
        LOGGER.info("Sleeping %s seconds before first run", config.initial_delay_seconds)
        time.sleep(config.initial_delay_seconds)

    while True:
        LOGGER.info("Running scheduled cycle")
        execute_cycle(state, config.evaluation_interval_minutes)
        if config.run_once:
            break
        LOGGER.info(
            "Sleeping %s minutes before next cycle", config.interval_minutes
        )
        time.sleep(config.interval_minutes * 60)


if __name__ == "__main__":
    main()
def build_evaluation_config_from_env() -> EvaluationConfig:
    return EvaluationConfig(
        site=_env_str("SITE_NAME", "default"),
        window_hours=_env_int("EVALUATION_WINDOW_HOURS", 24),
        actual_measurement=_env_str("HISTORICAL_MEASUREMENT", "historical_actuals"),
        forecast_measurement=_env_str("FORECAST_MEASUREMENT", "forecasts"),
        evaluation_measurement=_env_str("EVALUATION_MEASUREMENT", "evaluations"),
        influx_url=_env_str("INFLUX_URL", ""),
        influx_token=_env_str("INFLUX_TOKEN", ""),
        influx_org=_env_str("INFLUX_ORG", ""),
        influx_bucket=_env_str("INFLUX_BUCKET", ""),
        verify_ssl=_to_bool(os.getenv("INFLUX_VERIFY_SSL"), True),
    )
