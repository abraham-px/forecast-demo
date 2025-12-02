"""Solar PV simulation agent for the EPC1522 demo.

Reads the latest weather forecast from InfluxDB, runs a pvlib-based PV system
simulation, and writes AC power forecasts into the shared `forecasts` measurement.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import pandas as pd

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:  # pragma: no cover
    InfluxDBClient = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    WritePrecision = None  # type: ignore[assignment]
    SYNCHRONOUS = None  # type: ignore[assignment]

try:  # pragma: no cover
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain
    from pvlib.pvsystem import PVSystem
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
except ImportError:
    Location = None  # type: ignore[assignment]
    ModelChain = None  # type: ignore[assignment]
    PVSystem = None  # type: ignore[assignment]
    TEMPERATURE_MODEL_PARAMETERS = None  # type: ignore[assignment]

LOGGER = logging.getLogger("solarpv_simu")
WEATHER_FIELDS = ["ghi", "dni", "dhi", "temp_air", "wind_speed"]
HALF_HOURLY_POINTS = 48  # 24h at 30-minute cadence


@dataclass
class SolarPVSimConfig:
    horizon_hours: int
    weather_measurement: str
    forecast_measurement: str
    influx_url: str
    influx_token: str
    influx_org: str
    influx_bucket: str
    verify_ssl: bool
    latitude: float
    longitude: float
    altitude: float
    timezone: str
    surface_tilt: float
    surface_azimuth: float
    dc_capacity_kw: float
    ac_capacity_kw: float
    gamma_pdc: float
    albedo: float
    model_name: str


def build_config_from_env() -> SolarPVSimConfig:
    return SolarPVSimConfig(
        horizon_hours=int(os.getenv("PV_FORECAST_HOURS", "24")),
        weather_measurement=os.getenv("WEATHER_MEASUREMENT", "weather_forecast"),
        forecast_measurement=os.getenv("FORECAST_MEASUREMENT", "forecasts"),
        influx_url=os.getenv("INFLUX_URL", ""),
        influx_token=os.getenv("INFLUX_TOKEN", ""),
        influx_org=os.getenv("INFLUX_ORG", ""),
        influx_bucket=os.getenv("INFLUX_BUCKET", ""),
        verify_ssl=(
            str(os.getenv("INFLUX_VERIFY_SSL", "true")).lower()
            in {"1", "true", "yes", "on"}
        ),
        latitude=float(os.getenv("SITE_LATITUDE", "0")),
        longitude=float(os.getenv("SITE_LONGITUDE", "0")),
        altitude=float(os.getenv("SITE_ALTITUDE", "0")),
        timezone=os.getenv("SITE_TIMEZONE", "UTC"),
        surface_tilt=float(os.getenv("PV_SURFACE_TILT", "20")),
        surface_azimuth=float(os.getenv("PV_SURFACE_AZIMUTH", "180")),
        dc_capacity_kw=float(os.getenv("PV_DC_CAPACITY_KW", "0")),
        ac_capacity_kw=float(os.getenv("PV_AC_CAPACITY_KW", "0")),
        gamma_pdc=float(os.getenv("PV_GAMMA_PDC", "-0.003")),
        albedo=float(os.getenv("PV_ALBEDO", "0.2")),
        model_name=os.getenv("PV_MODEL_NAME", "pvlib_pvwatts"),
    )


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


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


def _resolve_start(ts_env: Optional[str], tz: str) -> pd.Timestamp:
    if ts_env:
        ts = pd.Timestamp(ts_env)
        if ts.tzinfo is None:
            ts = ts.tz_localize(tz)
        else:
            ts = ts.tz_convert(tz)
        return ts.tz_convert("UTC")
    return pd.Timestamp.now(tz="UTC")


def fetch_weather(
    client: InfluxDBClient,
    config: SolarPVSimConfig,
    *,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    # Pad the query window slightly to tolerate ingest alignment differences
    pad = pd.Timedelta(minutes=60)
    LOGGER.info(
        "Fetching weather window UTC start=%s end=%s (pad=%s)",
        start.isoformat(),
        stop.isoformat(),
        pad,
    )
    weather = query_influx_frame(
        client,
        config.influx_bucket,
        config.weather_measurement,
        fields=WEATHER_FIELDS,
        start=start - pad,
        stop=stop + pad,
        site=None,
    )
    if weather.empty:
        raise RuntimeError("No weather data available for PV simulation")
    # Restrict to the exact window after padding
    weather = weather[(weather.index >= start) & (weather.index <= stop)]
    if weather.empty:
        raise RuntimeError(
            "Weather data missing after trimming to target window. "
            "Run ingest first or adjust START_DATE/PV_START_DATE."
        )
    # Align to 30-minute steps and keep only 48 rows
    weather = weather.sort_index().asfreq("30min")
    weather = weather.ffill().bfill()
    LOGGER.info(
        "Weather rows after trim/resample: %s (UTC %s -> %s)",
        len(weather.index),
        weather.index.min(),
        weather.index.max(),
    )
    return weather.iloc[:HALF_HOURLY_POINTS]


def build_pv_system(config: SolarPVSimConfig) -> PVSystem:
    if PVSystem is None:
        raise RuntimeError("pvlib is required for PV simulation")
    temp_params = None
    if TEMPERATURE_MODEL_PARAMETERS is not None:
        temp_params = TEMPERATURE_MODEL_PARAMETERS["sapm"].get("open_rack_glass_glass")
    module_params = {"pdc0": config.dc_capacity_kw * 1000, "gamma_pdc": config.gamma_pdc}
    inverter_params = {"pdc0": config.ac_capacity_kw * 1000}
    return PVSystem(
        surface_tilt=config.surface_tilt,
        surface_azimuth=config.surface_azimuth,
        module_parameters=module_params,
        inverter_parameters=inverter_params,
        temperature_model_parameters=temp_params,
        albedo=config.albedo,
    )


def build_location(config: SolarPVSimConfig) -> Location:
    if Location is None:
        raise RuntimeError("pvlib is required for PV simulation")
    return Location(
        latitude=config.latitude,
        longitude=config.longitude,
        tz=config.timezone,
        altitude=config.altitude,
    )


def _extract_mc_series(mc: ModelChain, attr: str, index: pd.DatetimeIndex) -> pd.Series:
    value = getattr(mc, attr, None)
    if value is None and hasattr(mc, "results"):
        value = getattr(mc.results, attr, None)
    if value is None:
        raise RuntimeError(f"pvlib ModelChain did not produce '{attr}' results")
    if isinstance(value, pd.Series):
        return value
    return pd.Series(value, index=index)


def simulate_pv_power(
    system: PVSystem,
    location: Location,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    if ModelChain is None:
        raise RuntimeError("pvlib is required for PV simulation")
    mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss', losses_model='pvwatts')
    mc.run_model(weather)
    ac = _extract_mc_series(mc, "ac", weather.index)
    df = pd.DataFrame({"solarpv_forecast_kw": ac.astype(float) / 1000.0}, index=weather.index)
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def write_forecasts(
    df: pd.DataFrame,
    *,
    client: InfluxDBClient,
    bucket: str,
    org: str,
    measurement: str,
) -> int:
    if InfluxDBClient is None or Point is None:
        raise RuntimeError("influxdb-client is required to write PV forecasts")
    if df.empty:
        LOGGER.warning("PV simulation produced no rows")
        return 0

    records = []
    for _, row in df.iterrows():
        point = (
            Point(measurement)
            .time(pd.Timestamp(row["timestamp"]).to_pydatetime(), WritePrecision.NS)
            .field("solarpv_forecast_kw", float(row["solarpv_forecast_kw"]))
        )
        records.append(point)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org=org, record=records)
    return len(records)


def run_simulation(config: SolarPVSimConfig) -> int:
    if InfluxDBClient is None:
        raise RuntimeError("influxdb-client is required for PV simulation")
    # Determine window: 48 half-hour slots starting at now or override
    start_override = os.getenv("PV_START_DATE") or os.getenv("START_DATE")
    window_start = _resolve_start(start_override, config.timezone)
    horizon_end = window_start + pd.Timedelta(minutes=30 * HALF_HOURLY_POINTS)

    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=config.verify_ssl,
    ) as client:
        weather = fetch_weather(client, config, start=window_start, stop=horizon_end)
        # Ensure pvlib gets local-time indexed weather
        weather = weather.tz_convert(config.timezone)
        LOGGER.info(
            "PV window UTC %s -> %s | local %s -> %s",
            window_start,
            horizon_end,
            weather.index.min(),
            weather.index.max(),
        )
        location = build_location(config)
        system = build_pv_system(config)
        pv_df = simulate_pv_power(system, location, weather)
        pv_df = pv_df.iloc[:HALF_HOURLY_POINTS]
        written = write_forecasts(
            pv_df,
            client=client,
            bucket=config.influx_bucket,
            org=config.influx_org,
            measurement=config.forecast_measurement,
        )
        LOGGER.info("PV simulation wrote %s rows", written)
        return written


def main() -> None:
    configure_logging(os.getenv("LOG_LEVEL", "INFO"))
    config = build_config_from_env()
    run_simulation(config)


if __name__ == "__main__":
    main()
