# Repository Guidelines

## Project Structure & Module Organization
The containerized stack lives under `app/`, split by agent: `ingest_weather.py`, `load_forecast.py`, `solarpv_simu.py`, `evaluation.py`, and the orchestrating `scheduler.py`. Historical seed CSVs sit in `historical_data/`, while serialized models live in `model/` (currently `xgb_model.pkl`). Runtime configuration flows through `.env`, and Docker entrypoints are defined by `docker-compose.yml` plus the `dockerfile`. Keep new assets within these folders so the mounted volumes in `docker compose` stay predictable.

## Build, Test, and Development Commands
Core workflows run via Docker Compose:
```bash
docker compose up --build -d      # build image, copy .env, start scheduler
docker compose stop && docker compose start   # pause/resume stack
```
Troubleshooting and data backfills run ad hoc:
```bash
docker compose run --rm forecast-agents \
  python app/backfill_historical.py --csv-path historical_data/technos-1118.csv

docker compose exec forecast-agents python app/ingest_weather.py
```
These commands execute inside the container, so they honor `.env` without extra flags.

## Coding Style & Naming Conventions
Python sources use 4-space indentation, snake_case functions, and module-per-agent organization (`app/<agent>.py`). Follow PEP 8, prioritize clear docstrings on public functions, and guard runnable scripts with `if __name__ == "__main__":`. Configuration keys mirror Influx measurement names (`WEATHER_MEASUREMENT`, `FORECAST_MEASUREMENT`), so extend them consistently when introducing new fields. Dependencies belong in `requirements.txt`; import order should follow stdlib, third-party, then local modules.

## Testing Guidelines
There is no pytest suite; validation relies on running agents against InfluxDB. When modifying ingestion or forecasting logic, run the relevant `docker compose exec` command and confirm new points appear in the expected measurements. For end-to-end checks, tail logs with `docker compose logs -f forecast-agents` and verify scheduler intervals in Influx. Document manual checks in the PR description until automated tests are added.

## Commit & Pull Request Guidelines
Recent history favors short, imperative summaries (e.g., `fix issues4`, `update docker-compose file naming`). Keep messages under ~72 characters, optionally referencing tracker IDs (`fix: adjust scheduler delay`). Each PR should describe the change, outline test steps or log snippets, note `.env` additions, and attach screenshots when UI dashboards or Influx visualizations change. Request review before merging, and ensure Docker builds succeed locally.

## Security & Configuration Tips
Never commit real Influx tokens or site coordinates—use `.env.example` style placeholders. Validate new environment variables in `.env` and document defaults in README. When touching `model/`, confirm artifacts exclude customer data, and prefer mounting secrets through Portainer rather than embedding them inside the repository.
