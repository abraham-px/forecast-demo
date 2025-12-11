## EPC1522 Load & PV Forecasting testing

This repository packages a lightweight end‑to‑end forecasting demo for an EPC1522
running Portainer‑managed containers. The stack runs independent agents on their
own schedules via a simple Python scheduler:

1. **Data ingest** (`app/ingest_data.py`) pulls Open‑Meteo data every 3 h,
   aligns it to 30‑minute cadence, and writes to `weather_forecast`. It also
   queries the PXC bucket, rolls one week of `PcsData`/`Interconnection` into
   30‑minute historical rows (`NETLOAD_kw`, `solarPV_kw`) stored in
   `historical_data`. Horizon is set via `FORECAST_HOURS` for half hourly intervals.
2. **PV simulation** (`app/solarpv_simu.py`) runs pvlib ModelChain on those weather
   rows and writes 24h of AC PV power (`solarpv_forecast_kw`) to `forecasts`.
3. **Load forecaster** (`app/load_forecast.py`) consumes only weather/time
   features—no netload lags are required. By default it writes a 24h horizon to
   `forecasts` using the bundled XGBoost model.
4. **Evaluation** (`app/evaluation.py`) computes daily accuracy metrics (optional).

The `app/scheduler.py` module triggers jobs independently using per‑job intervals
from `.env` (`WEATHER/LOAD/PV_INTERVAL_MINUTES`).



## Prerequisites

- Docker or Docker Compose v2
- Access to the target InfluxDB instance (URL, token, org, bucket)
- EPC1522 (or any x86_64 host) running Portainer for deployment


### Python Dependencies

`requirements.forecast.txt` specifies the exact versions installed in the forecast container:
`numpy`, `pandas`, `requests`, `influxdb-client`, `jpholiday`, `joblib`,
`pvlib`, `xgboost`, etc. These are installed automatically during the Docker
build phase.

`requirements.weather.txt` contains the minimal ingest deps (`influxdb-client`,
`pandas`, `requests`).


## Configuration (`.env`)

All runtime parameters flow through `.env`; key entries:

| Category | Variables |
| --- | --- |
| Site | `SITE_NAME`, `SITE_LATITUDE`, `SITE_LONGITUDE`, `SITE_TIMEZONE` |
| Influx | `INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET`, `INFLUX_VERIFY_SSL` |
| Measurements | `WEATHER_MEASUREMENT`, `HISTORICAL_MEASUREMENT`, `FORECAST_MEASUREMENT` |
| Weather ingest | `FORECAST_HOURS`, `OPEN_METEO_MODEL` |
| Net load ingest | `PXC_BUCKET`, `HISTORICAL_MEASUREMENT`, `NETLOAD_FIELD`, `SOLAR_FIELD` |
| Load forecast | `LOAD_FORECAST_HOURS`, `LOAD_HISTORY_HOURS`, `LOAD_MODEL_PATH` |
| PV simulation | `PV_FORECAST_HOURS`, `PV_SURFACE_TILT`, `PV_SURFACE_AZIMUTH`, `PV_DC_CAPACITY_KW`, `PV_AC_CAPACITY_KW`, `PV_GAMMA_PDC`, `PV_ALBEDO`, `PV_MODEL_NAME` |
| Evaluation | `EVALUATION_WINDOW_HOURS`, `EVALUATION_MEASUREMENT`, `EVALUATION_INTERVAL_MINUTES` |
| Scheduler | `SCHEDULE_INTERVAL_MINUTES`, `SCHEDULER_INITIAL_DELAY_SECONDS`, `SCHEDULER_RUN_ONCE` |

> Update `.env` before building the container to point to the correct InfluxDB
> endpoint and site metadata. Portainer can mount this file directly.


## Build & Run

```bash
# build split images
docker compose build --no-cache

# start 
docker compose up -d
```


### Stopping / restarting

```bash
docker compose down -v
docker compose up -d
```


## Start Dates (Backfill)

Set `START_DATE=YYYY-MM-DD` (or ISO datetime) in `.env` to backfill ingest for a
fixed window. When `START_DATE` is set, the ingest script forces Open‑Meteo’s
archive API regardless of orbit overlap with “now”. `PV_START_DATE` can be set
separately for PV; otherwise it falls back to `START_DATE`.



Most options are already defined in `.env`, so flags are only necessary when
overriding defaults.


## Verifying the Demo

1. Optionally set `START_DATE`, `PV_START_DATE`, and `LOAD_FORECAST_HOURS` in `.env`
   for deterministic windows.
2. Start services: `docker compose up -d weather-ingest forecast-agents`.
3. Monitor logs: `docker compose logs -f weather-ingest forecast-agents`.
4. Inspect Influx: confirm rows in `weather_forecast` and `forecasts`
   (`solarpv_forecast_kw`, `load_forecast_kw`).
5. Adjust per-job intervals via `.env` and restart services if needed.

The demo is ready to clone on the EPC device, configure through Portainer, and
showcase automated weather ingest + PV and load forecasting (evaluation optional).
Tune `.env` to change intervals and horizons without code changes.

### Project Structure

```
app/
├── ingest_data.py          # weather + net load agent (weather container)
├── load_forecast.py        # load agent (forecast container)
├── solarpv_simu.py         # PV agent (forecast container)
├── evaluation.py           # optional daily metrics job
└── scheduler.py            # shared interval scheduler

docker-compose.yml          # defines weather + forecast services
dockerfile.weather          # minimal weather image
dockerfile.forecast         # full forecast image (pvlib/xgboost)
requirements.forecast.txt   # forecast dependencies
requirements.weather.txt    # weather-only dependencies
model/xgb_model.pkl         # model artifact
.env                        # all configuration (env-only agents)
```
