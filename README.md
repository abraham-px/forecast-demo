## EPC1522 Load & PV Forecasting Demo

This repository packages a lightweight end-to-end forecasting demo for an EPC1522
running Portainer-managed containers. The container hosts three agents plus a
simple scheduler:

1. **Weather ingestion** (`app/ingest_weather.py`) pulls Open-Meteo data, resamples
   it to 30-minute cadence, and writes to InfluxDB (`weather_forecast` measurement).
2. **Load forecaster** (`app/load_forecast.py`) loads the pre-trained XGBoost model,
   builds the same non-autoregressive feature set used in the notebook, and writes
   48 half-hour load forecasts (24 h horizon) to Influx (`forecasts` measurement,
   `load_forecast_kw` field).
3. **PV simulation** (`app/solarpv_simu.py`) uses pvlib to convert the weather rows
   into AC PV power forecasts at the same 30 min cadence, storing the output as
   `solarpv_forecast_kw`.
4. **Evaluation** (`app/evaluation.py`) compares the last 24h (configurable) of
   forecasts vs. historical data, computes MAE/RMSE/MAPE via scikit-learn, and
   writes accuracy metrics to the `evaluations` measurement once per day.

The `app/scheduler.py` module invokes those jobs sequentially on an interval
defined in `.env`, so a single container can manage the entire pipeline.


### Contents

- `app/`: agents, scheduler, and the one-off historical backfill script
- `historical_data/`: CSVs for seeding the `historical_actuals` measurement
- `model/xgb_model.pkl`: serialized load model artifact
- `AGENTS.md`: role definitions and data contracts
- `.env`: editable config consumed by every module


## Prerequisites

- Docker or Docker Compose v2
- Access to the target InfluxDB instance (URL, token, org, bucket)
- CNC machinery friendly network path to Open-Meteo API
- EPC1522 (or any x86_64 host) running Portainer for deployment


### Python Dependencies

`requirements.txt` specifies the exact versions installed in the container:
`numpy`, `pandas`, `requests`, `influxdb-client`, `jpholiday`, `joblib`,
`pvlib`, `xgboost`, etc. These are installed automatically during the Docker
build phase.


## Configuration (`.env`)

All runtime parameters flow through `.env`; key entries:

| Category | Variables |
| --- | --- |
| Site | `SITE_NAME`, `SITE_LATITUDE`, `SITE_LONGITUDE`, `SITE_TIMEZONE` |
| Influx | `INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET`, `INFLUX_VERIFY_SSL` |
| Measurements | `WEATHER_MEASUREMENT`, `HISTORICAL_MEASUREMENT`, `FORECAST_MEASUREMENT` |
| Weather ingest | `FORECAST_HOURS`, `OPEN_METEO_MODEL` |
| Load forecast | `LOAD_FORECAST_HOURS`, `LOAD_HISTORY_HOURS`, `LOAD_MODEL_PATH` |
| PV simulation | `PV_FORECAST_HOURS`, `PV_SURFACE_TILT`, `PV_SURFACE_AZIMUTH`, `PV_DC_CAPACITY_KW`, `PV_AC_CAPACITY_KW`, `PV_GAMMA_PDC`, `PV_ALBEDO`, `PV_MODEL_NAME` |
| Evaluation | `EVALUATION_WINDOW_HOURS`, `EVALUATION_MEASUREMENT`, `EVALUATION_INTERVAL_MINUTES` |
| Scheduler | `SCHEDULE_INTERVAL_MINUTES`, `SCHEDULER_INITIAL_DELAY_SECONDS`, `SCHEDULER_RUN_ONCE` |

> Update `.env` before building the container to point to the correct InfluxDB
> endpoint and site metadata. Portainer can mount this file directly.


## Build & Run

```bash
docker compose up --build -d
```

This command builds `forecast-agents` (see `dockercompose.yml`), injects `.env`,
mounts `./model` for easy artifact updates, and starts the scheduler service
(`python app/scheduler.py`). Logs show each job being executed in turn.

### Stopping / restarting

```bash
docker compose stop
docker compose start
```


## Backfilling Historical Data

Before running forecasts, populate `historical_actuals` so evaluation dashboards
and manual checks have context. The repository includes
`historical_data/technos-1118.csv` and a helper script:

```bash
docker compose run --rm forecast-agents \
  python app/backfill_historical.py --csv-path historical_data/technos-1118.csv
```

Environment variables (`HISTORICAL_MEASUREMENT`, `SITE_NAME`, etc.) are honored by
the script, so no extra flags are needed unless you point to a different file.


## Manual Agent Runs

You can invoke any agent directly inside the container for troubleshooting:

```bash
docker compose exec forecast-agents python app/ingest_weather.py --latitude ... --longitude ...
docker compose exec forecast-agents python app/load_forecast.py --run-once  # default horizon
docker compose exec forecast-agents python app/solarpv_simu.py
```

Most options are already defined in `.env`, so flags are only necessary when
overriding defaults.


## Verifying the Demo

1. **Backfill** historical data (optional but recommended).
2. **Start** the stack via `docker compose up`.
3. **Monitor logs**: `docker compose logs -f forecast-agents`.
4. **Inspect InfluxDB** measurements: confirm new rows in
   `weather_forecast`, `forecasts`, `historical_actuals`.
5. **Adjust schedules** via `.env` and restart the container if needed.

The demo is now ready to clone on the EPC device, configure through Portainer,
and showcase fully automated load + PV forecasting. Adjust the `.env` file to
match other sites or Influx targets without touching the source code.
