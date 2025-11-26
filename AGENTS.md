# AGENTS

## Mission Context
- Target hardware: EPC1522 running Portainer-managed containers
- Objective: run load forecasting (XGBoost) and solar PV generation simulation (pvlib) side-by-side, using a light scheduler inside a single containerized service
- Data backbone: shared InfluxDB bucket (`AIML`) reachable at `https://192.168.2.10:58086` with organization `PXC`
- Weather source: Open-Meteo (see `fetch-weather.py` for request pattern)
- Device-level data: actual load and PV telemetry captured for backfilling historical and evaluation tables

## Shared Data Contracts
| Table | Key Fields | Owner | Notes |
| --- | --- | --- | --- |
| `weather_forecast` | `timestamp`, `ghi`, `dni`, `dhi`, `temperature`, `wind_speed`, metadata (`lat`, `lon`, `provider`, `model`) | Weather Ingestion Agent | Populated on every API call so downstream agents can build their feature sets. |
| `historical_actuals` | `timestamp`, `load_kw`, `pv_kw`, `netload_kw`, data quality flags | Historian & Evaluation Agent | Used to retrain models, run evaluation notebooks, and build custom dashboards. |
| `forecasts` | `timestamp`, `load_forecast_kw`, `pv_forecast_kw`, `netload_forecast_kw`, `model_version`, `issued_at` | Forecast Agents (Load + PV) | Each forecaster writes to its column set; scheduler enforces no overwrite collisions. |
| `evaluations` | `issued_at`, `metric_window`, `mae_load`, `mae_pv`, `mape_load`, `mape_pv`, `notes` | Historian & Evaluation Agent | Used to document demo accuracy and trigger retraining if thresholds are exceeded. |

## Agent Directory

### 1. Orchestration & Scheduler Agent
- **Purpose:** Coordinate the periodic execution of ingestion, forecasting, and evaluation tasks inside the shared container (e.g., lightweight cron, APScheduler, or simple loop).
- **Inputs:** Global schedule config, Influx connection env vars, task dependency graph.
- **Outputs:** Triggered jobs, structured logs, failure notifications (stdout + Portainer health).
- **Key Behaviors:**
  - Stagger job order: Weather → Load Forecast → PV Simulation → Evaluation.
  - Retry transient network/API failures with exponential backoff.
  - Surface job status to Portainer (exit codes, health check endpoint).

### 2. Weather Ingestion Agent
- **Purpose:** Pull hourly forecasts from Open-Meteo (see `fetch-weather.py`) and harmonize them for pvlib/XGBoost.
- **Inputs:** Latitude/longitude per site, API window (now→+48h), pvlib variable mapping, scheduler trigger.
- **Outputs:** Rows appended to `weather_forecast` in Influx (`ghi`, `dni`, `dhi`, `temperature`, `wind_speed`).
- **Key Behaviors:**
  - Respect API limits; chunk long windows when replaying history.
  - Enrich data with timezone, site ID, and provider metadata before writing to Influx using the bucket token.
  - Keep recent 72h forecasts cached locally (in-memory/pickle) to avoid duplicate writes when scheduler jitter occurs.

### 3. Load Forecast Agent (XGBoost)
- **Purpose:** Generate 48-step load forecasts using the trained gradient boosted model referenced in `model/`.
- **Inputs:**
  - Weather-derived features queried from `weather_forecast` (temperature, irradiance, wind, etc.).
  - Business-hour calendar flags (Uyama: 08:00–22:00) and optional holiday table.
  - Stored scaler/encoder artifacts from training notebook (`load-pv-forecat.ipynb`).
- **Outputs:** `load_forecast_kw` for each timestamp, written into `forecasts` table with model metadata.
- **Key Behaviors:**
  - Perform feature engineering consistent with notebook (lag-less or lag features based on mode flag).
  - Validate input completeness before inference; fall back to persistence model if gaps exist.
  - Log MAE per run using last 24h of historical actuals when available.

### 4. Solar PV Simulation Agent (pvlib)
- **Purpose:** Convert latest weather forecast into PV power predictions using pvlib and site-specific array parameters.
- **Inputs:** Weather rows (GHI, DNI, DHI, temperature, wind), PV system spec (DC capacity 165 kW, inverter data), location metadata from notebook.
- **Outputs:** `pv_forecast_kw` (AC) for each forecast horizon plus optional `dc_kw` diagnostic column, stored in `forecasts`.
- **Key Behaviors:**
  - Use pvlib clearsky + SAPM/CEC module models consistent with prior experiments.
  - Account for temperature derating via module temperature models driven by `temp_air` and `wind_speed`.
  - Align timestamps with load forecasts to enable combined net-load calculations.

### 5. Historian & Evaluation Agent
- **Purpose:** Maintain truth data and quantify accuracy for demo storytelling.
- **Inputs:** Actual load/PV telemetry pushed from EPC PLC, forecast outputs, evaluation schedule (e.g., hourly rolling MAE).
- **Outputs:**
  - `historical_actuals` table entries for every measurement interval.
  - `evaluations` summary rows with MAE/MAPE for load and PV plus qualitative notes.
- **Key Behaviors:**
  - Detect missing actuals and flag them (data_quality=true/false) to prevent misleading metrics.
  - Provide quick diff charts (stored PNG/HTML) to Portainer logs or S3 bucket for demo evidence.
  - Trigger retraining workflow manually (out of scope for demo) when sustained error breaches threshold.

## Interaction Pattern
1. **Scheduler kicks off ingestion** at minute 0; weather agent fetches and writes new forecast horizon.
2. **Load forecaster queries** the fresh weather rows + latest historical load to emit 48×30‑min forecasts.
3. **PV simulator consumes** the same weather rows plus PV config to create matching solar forecasts.
4. **Historian captures actuals** arriving from PLC and benchmarks forecasts, refreshing dashboard views.
5. **Evaluation agent writes metrics** back to Influx, giving an auditable trail for the Portainer-hosted demo.

## Implementation Notes
- Keep `.env` or Portainer stack secrets for Influx (`INFLUX_URL`, `INFLUX_TOKEN`, `INFLUX_ORG`, `INFLUX_BUCKET`).
- Provide a simple CLI (`python app/main.py --job weather`) so each agent can run standalone for troubleshooting.
- When cloning to the EPC device, mount the Influx CA cert if TLS is enforced.
- Notebook `load-pv-forecat.ipynb` holds feature logic; convert critical parts into reusable Python modules under `app/`.
