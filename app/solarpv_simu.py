"""Solar PV simulation agent for the EPC1522 demo.

Reads the latest weather forecast from InfluxDB, runs a pvlib-based PV system
simulation, and writes AC/DC forecasts into the shared `forecasts` measurement.
"""
from __future__ import annotations

import argparse
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


@dataclass
class SolarPVSimConfig:
    site: str
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
    tag_filter = f' and r["site"] == "{site}"' if site else ""
    flux = f"""
from(bucket: \"{bucket}\")
  |> range(start: time(v: \"{start.isoformat()}\"), stop: time(v: \"{stop.isoformat()}\"))
  |> filter(fn: (r) => r[\"_measurement\"] == \"{measurement}\")
  |> filter(fn: (r) => {field_filter}){tag_filter}
  |> keep(columns: [\"_time\", \"_field\", \"_value\"])
  |> pivot(rowKey: [\"_time\"], columnKey: [\"_field\"], valueColumn: \"_value\")
  |> sort(columns: [\"_time\"])
"""
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


def fetch_weather(
    client: InfluxDBClient,
    config: SolarPVSimConfig,
    *,
    start: pd.Timestamp,
    stop: pd.Timestamp,
) -> pd.DataFrame:
    weather = query_influx_frame(
        client,
        config.influx_bucket,
        config.weather_measurement,
        fields=WEATHER_FIELDS,
        start=start,
        stop=stop,
        site=config.site,
    )
    if weather.empty:
        raise RuntimeError("No weather data available for PV simulation")
    weather = weather.tz_convert(config.timezone)
    weather = weather.ffill().bfill()
    return weather


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


def simulate_pv_power(
    system: PVSystem,
    location: Location,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    if ModelChain is None:
        raise RuntimeError("pvlib is required for PV simulation")
    mc = ModelChain(system, location, aoi_model='physical', spectral_model='no_loss', losses_model='pvwatts')
    mc.run_model(weather)
    ac = mc.ac if isinstance(mc.ac, pd.Series) else pd.Series(mc.ac, index=weather.index)
    dc = mc.dc if isinstance(mc.dc, pd.Series) else pd.Series(mc.dc, index=weather.index)
    df = pd.DataFrame(
        {
            "solarpv_forecast_kw": ac.astype(float) / 1000.0,
        },
        index=weather.index,
    )
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df


def write_forecasts(
    df: pd.DataFrame,
    *,
    client: InfluxDBClient,
    bucket: str,
    org: str,
    measurement: str,
    tags: Optional[Dict[str, str]] = None,
) -> int:
    if InfluxDBClient is None or Point is None:
        raise RuntimeError("influxdb-client is required to write PV forecasts")
    if df.empty:
        LOGGER.warning("PV simulation produced no rows")
        return 0

    tags = tags or {}
    issued_at = pd.Timestamp.now(tz="UTC")
    records = []
    for _, row in df.iterrows():
        point = (
            Point(measurement)
            .time(pd.Timestamp(row["timestamp"]).to_pydatetime(), WritePrecision.NS)
            .field("solarpv_forecast_kw", float(row["solarpv_forecast_kw"]))
            .field("issued_at", issued_at.isoformat())
        )
        for key, value in tags.items():
            if value is None:
                continue
            point = point.tag(key, value)
        records.append(point)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org=org, record=records)
    return len(records)


def build_config_from_args(args: argparse.Namespace) -> SolarPVSimConfig:
    return SolarPVSimConfig(
        site=args.site,
        horizon_hours=args.horizon_hours,
        weather_measurement=args.weather_measurement,
        forecast_measurement=args.forecast_measurement,
        influx_url=args.influx_url,
        influx_token=args.influx_token,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        verify_ssl=args.verify_ssl,
        latitude=args.latitude,
        longitude=args.longitude,
        altitude=args.altitude,
        timezone=args.timezone,
        surface_tilt=args.surface_tilt,
        surface_azimuth=args.surface_azimuth,
        dc_capacity_kw=args.dc_capacity_kw,
        ac_capacity_kw=args.ac_capacity_kw,
        gamma_pdc=args.gamma_pdc,
        albedo=args.albedo,
        model_name=args.model_name,
    )


def run_simulation(config: SolarPVSimConfig) -> int:
    if InfluxDBClient is None:
        raise RuntimeError("influxdb-client is required for PV simulation")
    now = pd.Timestamp.now(tz="UTC")
    horizon_end = now + pd.Timedelta(hours=config.horizon_hours)

    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=config.verify_ssl,
    ) as client:
        weather = fetch_weather(client, config, start=now, stop=horizon_end)
        location = build_location(config)
        system = build_pv_system(config)
        pv_df = simulate_pv_power(system, location, weather)
        pv_df = pv_df[pv_df["timestamp"] >= weather.index[0]]
        tags = {"site": config.site, "model": config.model_name}
        written = write_forecasts(
            pv_df,
            client=client,
            bucket=config.influx_bucket,
            org=config.influx_org,
            measurement=config.forecast_measurement,
            tags=tags,
        )
        LOGGER.info("PV simulation wrote %s rows", written)
        return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pvlib simulation and store forecasts")
    parser.add_argument("--site", default=os.getenv("SITE_NAME", "default"), help="Site tag")
    parser.add_argument(
        "--horizon-hours",
        type=int,
        default=int(os.getenv("PV_FORECAST_HOURS", 24)),
        dest="horizon_hours",
        help="Forecast horizon in hours",
    )
    parser.add_argument(
        "--weather-measurement",
        default=os.getenv("WEATHER_MEASUREMENT", "weather_forecast"),
        help="Measurement storing weather forecasts",
    )
    parser.add_argument(
        "--forecast-measurement",
        default=os.getenv("FORECAST_MEASUREMENT", "forecasts"),
        help="Measurement where PV forecasts will be written",
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
        help="InfluxDB bucket",)
    parser.add_argument(
        "--verify-ssl",
        default=os.getenv("INFLUX_VERIFY_SSL", "true"),
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="Toggle TLS verification",
    )
    parser.add_argument(
        "--latitude",
        type=float,
        default=float(os.getenv("SITE_LATITUDE", 35.9801)),
        help="PV site latitude",
    )
    parser.add_argument(
        "--longitude",
        type=float,
        default=float(os.getenv("SITE_LONGITUDE", 139.8866)),
        help="PV site longitude",
    )
    parser.add_argument(
        "--altitude",
        type=float,
        default=float(os.getenv("SITE_ALTITUDE", 0)),
        help="Site altitude in meters",
    )
    parser.add_argument(
        "--timezone",
        default=os.getenv("SITE_TIMEZONE", "Asia/Tokyo"),
        help="IANA timezone string",
    )
    parser.add_argument(
        "--surface-tilt",
        type=float,
        default=float(os.getenv("PV_SURFACE_TILT", 20)),
        dest="surface_tilt",
        help="Array tilt in degrees",
    )
    parser.add_argument(
        "--surface-azimuth",
        type=float,
        default=float(os.getenv("PV_SURFACE_AZIMUTH", 180)),
        dest="surface_azimuth",
        help="Array azimuth in degrees",
    )
    parser.add_argument(
        "--dc-capacity-kw",
        type=float,
        default=float(os.getenv("PV_DC_CAPACITY_KW", 165)),
        dest="dc_capacity_kw",
        help="Installed DC capacity in kW",
    )
    parser.add_argument(
        "--ac-capacity-kw",
        type=float,
        default=float(os.getenv("PV_AC_CAPACITY_KW", 150)),
        dest="ac_capacity_kw",
        help="Inverter AC capacity in kW",
    )
    parser.add_argument(
        "--gamma-pdc",
        type=float,
        default=float(os.getenv("PV_GAMMA_PDC", -0.004)),
        dest="gamma_pdc",
        help="Temperature coefficient for DC power",
    )
    parser.add_argument(
        "--albedo",
        type=float,
        default=float(os.getenv("PV_ALBEDO", 0.2)),
        help="Ground reflectance",
    )
    parser.add_argument(
        "--model-name",
        default=os.getenv("PV_MODEL_NAME", "pvlib_pvwatts"),
        help="Tag identifying the PV simulation model",
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
    run_simulation(config)


if __name__ == "__main__":
    main()
