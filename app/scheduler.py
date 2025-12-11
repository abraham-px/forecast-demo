"""Python scheduler for EPC1522 agents with independent intervals per job."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

LOGGER = logging.getLogger("scheduler")


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


def run_weather_job() -> None:
    # Import inside the function so weather container doesn't need heavy deps
    import ingest_data  # lazy import

    ingest_data.main()


def run_load_job() -> None:
    from load_forecast import build_config_from_env, run_forecast  # lazy import

    run_forecast(build_config_from_env())


def run_pv_job() -> None:
    from solarpv_simu import build_config_from_env, run_simulation  # lazy import

    run_simulation(build_config_from_env())


def run_evaluation_job() -> None:
    from evaluation import build_config_from_env, run_evaluation  # lazy import

    run_evaluation(build_config_from_env())


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


def build_scheduler_config_from_env() -> SchedulerConfig:
    jobs_env = _env_str(
        "SCHEDULER_JOBS",
        "weather_ingest,load_forecast,pv_simulation,evaluation",
    )
    job_names = [name.strip() for name in jobs_env.split(",") if name.strip()]
    return SchedulerConfig(
        poll_seconds=max(_env_int("SCHEDULER_POLL_SECONDS", 60), 5),
        run_once=_to_bool(os.getenv("SCHEDULER_RUN_ONCE"), False),
        initial_delay_seconds=max(_env_int("SCHEDULER_INITIAL_DELAY_SECONDS", 0), 0),
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
    logging.basicConfig(
        level=_env_str("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    config = build_scheduler_config_from_env()
    run_scheduler(config)


if __name__ == "__main__":
    main()
