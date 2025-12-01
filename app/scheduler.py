"""Python scheduler for EPC1522 agents with independent intervals per job."""
from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence

LOGGER = logging.getLogger("scheduler")


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


@dataclass
class SchedulerConfig:
    poll_seconds: int
    run_once: bool
    initial_delay_seconds: int
    job_names: Sequence[str] = field(default_factory=list)


@dataclass
class ScheduledJob:
    name: str
    interval_minutes: int
    runner: Callable[[], None]
    last_run_epoch: float | None = None

    def is_due(self, now: float) -> bool:
        if self.interval_minutes <= 0:
            return False
        if self.last_run_epoch is None:
            return True
        return now - self.last_run_epoch >= self.interval_minutes * 60


def build_load_config_from_env():
    from load_forecast import LoadForecastConfig  # lazy import
    return LoadForecastConfig(
        site=_env_str("SITE_NAME", "default"),
        horizon_hours=_env_int("LOAD_FORECAST_HOURS", 24),
        history_hours=_env_int("LOAD_HISTORY_HOURS", 72),
        weather_measurement=_env_str("WEATHER_MEASUREMENT", "weather_forecast"),
        actual_measurement=_env_str("HISTORICAL_MEASUREMENT", "historical_actuals"),
        forecast_measurement=_env_str("FORECAST_MEASUREMENT", "forecasts"),
        model_path=Path(_env_str("LOAD_MODEL_PATH", "model/xgb_model.pkl")),
        influx_url=_env_str("INFLUX_URL", ""),
        influx_token=_env_str("INFLUX_TOKEN", ""),
        influx_org=_env_str("INFLUX_ORG", ""),
        influx_bucket=_env_str("INFLUX_BUCKET", ""),
        verify_ssl=_to_bool(os.getenv("INFLUX_VERIFY_SSL"), True),
    )


def build_pv_config_from_env():
    from solarpv_simu import SolarPVSimConfig  # lazy import
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


def build_evaluation_config_from_env():
    from evaluation import EvaluationConfig  # lazy import
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
    # Import inside the function so weather container doesn't need heavy deps
    import ingest_weather  # lazy import
    ingest_weather.main()


def run_load_job() -> None:
    from load_forecast import run_forecast  # lazy import
    run_forecast(build_load_config_from_env())


def run_pv_job() -> None:
    from solarpv_simu import run_simulation  # lazy import
    run_simulation(build_pv_config_from_env())


def run_evaluation_job() -> None:
    from evaluation import run_evaluation  # lazy import
    run_evaluation(build_evaluation_config_from_env())


def build_job_definitions() -> Dict[str, ScheduledJob]:
    return {
        "weather_ingest": ScheduledJob(
            name="weather_ingest",
            interval_minutes=_env_int("WEATHER_INTERVAL_MINUTES", 180),
            runner=run_weather_job,
        ),
        "load_forecast": ScheduledJob(
            name="load_forecast",
            interval_minutes=_env_int("LOAD_INTERVAL_MINUTES", 60),
            runner=run_load_job,
        ),
        "pv_simulation": ScheduledJob(
            name="pv_simulation",
            interval_minutes=_env_int("PV_INTERVAL_MINUTES", 60),
            runner=run_pv_job,
        ),
        "evaluation": ScheduledJob(
            name="evaluation",
            interval_minutes=_env_int("EVALUATION_INTERVAL_MINUTES", 24 * 60),
            runner=run_evaluation_job,
        ),
    }


def select_jobs(requested: Sequence[str]) -> List[ScheduledJob]:
    definitions = build_job_definitions()
    jobs: List[ScheduledJob] = []
    for name in requested:
        job = definitions.get(name.strip())
        if job is None:
            LOGGER.warning("Unknown job requested: %s", name)
            continue
        if job.interval_minutes <= 0:
            LOGGER.info("Job %s disabled via interval %s", job.name, job.interval_minutes)
            continue
        jobs.append(job)
    return jobs


def run_job(job: ScheduledJob) -> None:
    LOGGER.info("Starting job: %s", job.name)
    try:
        job.runner()
        LOGGER.info("Job succeeded: %s", job.name)
    except Exception:
        LOGGER.exception("Job failed: %s", job.name)
    finally:
        job.last_run_epoch = time.time()


def parse_args() -> SchedulerConfig:
    parser = argparse.ArgumentParser(description="Independent scheduler for EPC1522 agents")
    parser.add_argument(
        "--poll-seconds",
        type=int,
        default=_env_int("SCHEDULER_POLL_SECONDS", 60),
        help="Seconds between checks for due jobs",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        default=_to_bool(os.getenv("SCHEDULER_RUN_ONCE"), False),
        help="Execute each enabled job once and exit",
    )
    parser.add_argument(
        "--initial-delay",
        type=int,
        default=_env_int("SCHEDULER_INITIAL_DELAY_SECONDS", 0),
        help="Seconds to wait before starting the scheduler loop",
    )
    parser.add_argument(
        "--jobs",
        default=_env_str(
            "SCHEDULER_JOBS",
            "weather_ingest,load_forecast,pv_simulation,evaluation",
        ),
        help="Comma-separated list of jobs to run",
    )
    parser.add_argument(
        "--log-level",
        default=_env_str("LOG_LEVEL", "INFO"),
        help="Logging level (default INFO)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    job_names = [name.strip() for name in args.jobs.split(",") if name.strip()]
    return SchedulerConfig(
        poll_seconds=max(args.poll_seconds, 5),
        run_once=args.run_once,
        initial_delay_seconds=max(args.initial_delay, 0),
        job_names=job_names,
    )


def run_scheduler(config: SchedulerConfig) -> None:
    jobs = select_jobs(config.job_names)
    if not jobs:
        LOGGER.warning("No jobs enabled. Scheduler exiting.")
        return

    if config.initial_delay_seconds:
        LOGGER.info("Sleeping %s seconds before starting jobs", config.initial_delay_seconds)
        time.sleep(config.initial_delay_seconds)

    completed: set[str] = set()
    while True:
        now = time.time()
        for job in jobs:
            if job.is_due(now):
                run_job(job)
                if config.run_once:
                    completed.add(job.name)
        if config.run_once and completed == {job.name for job in jobs}:
            LOGGER.info("run-once mode complete; exiting")
            break
        time.sleep(config.poll_seconds)


def main() -> None:
    config = parse_args()
    run_scheduler(config)


if __name__ == "__main__":
    main()
