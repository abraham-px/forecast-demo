"""One-off script to backfill historical load/PV data into InfluxDB."""
from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from influxdb_client import InfluxDBClient, Point, WritePrecision
    from influxdb_client.client.write_api import SYNCHRONOUS
except ImportError:  # pragma: no cover
    InfluxDBClient = None  # type: ignore[assignment]
    Point = None  # type: ignore[assignment]
    WritePrecision = None  # type: ignore[assignment]
    SYNCHRONOUS = None  # type: ignore[assignment]

LOGGER = logging.getLogger("backfill_historical")

ORIGINAL_FIELD_MAP = {
    "poiActvPwr": "poi_kw",
    "pvActvPwr": "pv_kw",
    "essPcsActvPwr": "ess_kw",
    "NETLOAD": "netload_kw",
}


@dataclass
class BackfillConfig:
    csv_path: Path
    measurement: str
    site: str
    influx_url: str
    influx_token: str
    influx_org: str
    influx_bucket: str
    verify_ssl: bool


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns=ORIGINAL_FIELD_MAP)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize("UTC")
    df = df.sort_values("timestamp")
    return df


def write_frame(
    df: pd.DataFrame,
    *,
    client: InfluxDBClient,
    bucket: str,
    org: str,
    measurement: str,
    site: str,
) -> int:
    records = []
    for _, row in df.iterrows():
        point = Point(measurement).time(row["timestamp"].to_pydatetime(), WritePrecision.NS)
        for influx_field in ORIGINAL_FIELD_MAP.values():
            if influx_field in row and pd.notna(row[influx_field]):
                point = point.field(influx_field, float(row[influx_field]))
        point = point.tag("site", site)
        records.append(point)
    if not records:
        LOGGER.warning("No records to write")
        return 0
    write_api = client.write_api(write_options=SYNCHRONOUS)
    write_api.write(bucket=bucket, org=org, record=records)
    return len(records)


def build_config_from_args(args: argparse.Namespace) -> BackfillConfig:
    return BackfillConfig(
        csv_path=Path(args.csv_path),
        measurement=args.measurement,
        site=args.site,
        influx_url=args.influx_url,
        influx_token=args.influx_token,
        influx_org=args.influx_org,
        influx_bucket=args.influx_bucket,
        verify_ssl=args.verify_ssl,
    )


def run_backfill(config: BackfillConfig) -> int:
    if InfluxDBClient is None:
        raise RuntimeError("influxdb-client package is required for backfill")
    df = load_csv(config.csv_path)
    with InfluxDBClient(
        url=config.influx_url,
        token=config.influx_token,
        org=config.influx_org,
        verify_ssl=config.verify_ssl,
    ) as client:
        written = write_frame(
            df,
            client=client,
            bucket=config.influx_bucket,
            org=config.influx_org,
            measurement=config.measurement,
            site=config.site,
        )
    LOGGER.info("Backfill completed; rows written: %s", written)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill CSV historical data into Influx")
    parser.add_argument(
        "--csv-path",
        default=os.getenv("BACKFILL_CSV", "historical_data/technos-1118.csv"),
        help="CSV file to ingest",
    )
    parser.add_argument(
        "--measurement",
        default=os.getenv("HISTORICAL_MEASUREMENT", "historical_actuals"),
        help="Target Influx measurement",
    )
    parser.add_argument("--site", default=os.getenv("SITE_NAME", "default"), help="Site tag")
    parser.add_argument("--influx-url", default=os.getenv("INFLUX_URL"), help="Influx URL")
    parser.add_argument("--influx-token", default=os.getenv("INFLUX_TOKEN"), help="Influx token")
    parser.add_argument("--influx-org", default=os.getenv("INFLUX_ORG"), help="Influx org")
    parser.add_argument(
        "--influx-bucket",
        default=os.getenv("INFLUX_BUCKET", "AIML"),
        help="Influx bucket",
    )
    parser.add_argument(
        "--verify-ssl",
        default=os.getenv("INFLUX_VERIFY_SSL", "true"),
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="Verify TLS certificates",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="Logging level",
    )
    args = parser.parse_args()
    configure_logging(args.log_level)
    required = {
        "csv_path": args.csv_path,
        "influx_url": args.influx_url,
        "influx_token": args.influx_token,
        "influx_org": args.influx_org,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(f"Missing required arguments: {', '.join(missing)}")
    return args


def main() -> None:
    args = parse_args()
    config = build_config_from_args(args)
    run_backfill(config)


if __name__ == "__main__":
    main()
