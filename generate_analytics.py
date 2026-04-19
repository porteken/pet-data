"""Generate PET analytics from PET parquet shards or a materialized PET CSV."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

DataFrame: TypeAlias = Any
np: Any = cast("Any", import_module("numpy"))
pd: Any = cast("Any", import_module("pandas"))

try:
    prophet_module: Any = import_module("prophet")
    Prophet: Any = prophet_module.Prophet
except (ImportError, AttributeError):
    Prophet = None

PET_CSV_NAME = "pet.csv"
PET_BATCH_GLOB = "pet_batch_*.parquet"
ANALYTICS_ROOT = Path("analytics_data_csv")
PET_ROOT = Path("pet_data_csv")
FORECAST_END_YEAR = 2100
GENERATING_LOG_MESSAGE = "Generating %s..."

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class ForecastRecord(TypedDict):
    """A single PET forecast row."""

    location_id: object
    year: int
    pet: float
    lower: float
    upper: float


def _yearly_average_pet(df: DataFrame) -> DataFrame:
    return df.groupby(["location_id", "year"])["pet"].mean().reset_index()


def _empty_forecast_frame() -> DataFrame:
    return pd.DataFrame(
        columns=["location_id", "year", "pet", "lower", "upper"],
    )


def _build_forecast_frame(df: DataFrame) -> DataFrame:
    if Prophet is None:
        logger.warning("Prophet not found. Forecast outputs will be empty.")
        return _empty_forecast_frame()

    logging.getLogger("prophet").setLevel(logging.WARNING)
    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

    yearly_avg = _yearly_average_pet(df)
    all_forecasts: list[DataFrame] = []
    location_ids = sorted(yearly_avg["location_id"].unique().tolist())

    for loc_id in location_ids:
        loc_df = yearly_avg[yearly_avg["location_id"] == loc_id].copy()
        if len(loc_df) <= 1:
            continue

        loc_df = loc_df.rename(columns={"year": "ds", "pet": "y"})
        loc_df["ds"] = pd.to_datetime(loc_df["ds"], format="%Y")

        model = Prophet(
            growth="linear",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.01,
        )
        model.fit(loc_df)

        last_year = int(loc_df["ds"].dt.year.max())
        periods_to_forecast = FORECAST_END_YEAR - last_year
        if periods_to_forecast <= 0:
            continue

        future_df = model.make_future_dataframe(periods=periods_to_forecast, freq="YS")
        forecast = model.predict(future_df)
        forecast["location_id"] = loc_id

        forecast_subset = forecast[
            ["ds", "location_id", "yhat", "yhat_lower", "yhat_upper"]
        ].copy()
        forecast_subset = forecast_subset.rename(
            columns={
                "ds": "year",
                "yhat": "pet",
                "yhat_lower": "lower",
                "yhat_upper": "upper",
            },
        )
        forecast_subset["year"] = forecast_subset["year"].dt.year
        forecast_subset = forecast_subset[forecast_subset["year"] > last_year]
        all_forecasts.append(forecast_subset)

    if not all_forecasts:
        logger.warning("No forecasts were generated.")
        return _empty_forecast_frame()

    final_forecast_df = pd.concat(all_forecasts, ignore_index=True).round(1)
    float_cols = final_forecast_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        final_forecast_df[float_cols] = final_forecast_df[float_cols].astype("float32")
    return final_forecast_df


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PET analytics from partitioned PET parquet shards or a "
            "materialized pet.csv."
        ),
    )
    parser.add_argument("--pet-root", default=str(PET_ROOT))
    parser.add_argument("--pet-csv", default=PET_CSV_NAME)
    parser.add_argument("--out-dir", default=str(ANALYTICS_ROOT))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=20)
    return parser.parse_args()


def _quantile(values: object, quantile: float) -> float:
    """Calculate a quantile from a pandas Series-like object."""
    return float(cast("Any", values).quantile(quantile))


def _p10(values: object) -> float:
    """Calculate the 10th percentile from a pandas Series-like object."""
    return _quantile(values, 0.1)


def _p90(values: object) -> float:
    """Calculate the 90th percentile from a pandas Series-like object."""
    return _quantile(values, 0.9)


def generate_percentiles(df: DataFrame, output_dir: Path) -> Path:
    """Calculate the 10th and 90th percentile of PET per year per location."""
    output_path = output_dir / "percentiles.parquet"
    logger.info(GENERATING_LOG_MESSAGE, output_path)
    agg_df = (
        df.groupby(["year", "location_id"])["pet"].agg(p10=_p10, p90=_p90).reset_index()
    )

    rounded_df = agg_df.round(1)
    float_cols = rounded_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        rounded_df[float_cols] = rounded_df[float_cols].astype("float32")
    tmp_output_path = output_path.with_suffix(".tmp")
    rounded_df.to_parquet(tmp_output_path, index=False, compression="snappy")
    tmp_output_path.rename(output_path)
    return output_path


def generate_forecast(df: DataFrame, output_dir: Path) -> Path:
    """Generate PET forecasts with lower and upper bounds using Prophet (up to 2100)."""
    output_path = output_dir / "forecast.parquet"
    logger.info(GENERATING_LOG_MESSAGE, output_path)
    final_forecast_df = _build_forecast_frame(df)

    tmp_output_path = output_path.with_suffix(".tmp")
    final_forecast_df.to_parquet(tmp_output_path, index=False, compression="snappy")
    tmp_output_path.rename(output_path)
    return output_path


def generate_change_per_decade(df: DataFrame, output_dir: Path) -> Path:
    """Calculate the change in average PET between decades."""
    output_path = output_dir / "change_per_decade.parquet"
    logger.info(GENERATING_LOG_MESSAGE, output_path)
    yearly_avg = _yearly_average_pet(df)
    forecast_df = _build_forecast_frame(df)
    future_yearly_avg = forecast_df[["location_id", "year", "pet"]].copy()
    frames = [yearly_avg]
    if not future_yearly_avg.empty:
        frames.append(future_yearly_avg)
    combined_yearly_avg = pd.concat(frames, ignore_index=True)
    combined_yearly_avg = combined_yearly_avg.drop_duplicates(
        subset=["location_id", "year"],
        keep="first",
    )
    combined_yearly_avg["decade_start"] = (combined_yearly_avg["year"] // 10) * 10

    decade_avg = (
        combined_yearly_avg.groupby(["location_id", "decade_start"])["pet"]
        .mean()
        .reset_index()
    )

    decade_avg = decade_avg.sort_values(["location_id", "decade_start"])
    decade_avg["change_value"] = decade_avg.groupby("location_id")["pet"].diff()
    decade_avg = decade_avg.dropna(subset=["change_value"])

    final_df = pd.DataFrame(
        decade_avg[["location_id", "decade_start", "change_value"]].round(2),
        columns=["location_id", "decade_start", "change_value"],
    ).rename(columns={"change_value": "change", "decade_start": "year"})

    float_cols = final_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        final_df[float_cols] = final_df[float_cols].astype("float32")

    tmp_output_path = output_path.with_suffix(".tmp")
    final_df.to_parquet(tmp_output_path, index=False, compression="snappy")
    tmp_output_path.rename(output_path)
    return output_path


def _discover_pet_files(pet_root: Path) -> list[Path]:
    """Return all pet batch CSV files sorted by path."""
    if not pet_root.exists():
        return []
    return sorted(pet_root.rglob(PET_BATCH_GLOB))


def _select_pet_files(
    all_files: list[Path],
    *,
    shard_index: int,
    shard_count: int,
) -> list[Path]:
    """Distribute pet files evenly across shards by position."""
    return [f for i, f in enumerate(all_files) if i % shard_count == shard_index]


def _load_pet_frame_from_csv(
    pet_csv_path: Path,
    *,
    shard_index: int,
    shard_count: int,
) -> DataFrame:
    logger.info("Loading PET rows from %s...", pet_csv_path)
    df = pd.read_csv(pet_csv_path, usecols=["location_id", "date", "pet"])
    if shard_count > 1:
        location_ids = pd.to_numeric(df["location_id"], errors="coerce")
        df = df[location_ids.mod(shard_count).eq(shard_index)].copy()

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


def _load_pet_frame(args: argparse.Namespace) -> DataFrame:
    pet_root = Path(args.pet_root)
    pet_csv_path = Path(args.pet_csv)

    all_files = _discover_pet_files(pet_root)
    if all_files:
        logger.info(
            "Loading %s PET files for analytics shard %s/%s.",
            len(all_files),
            args.shard_index,
            args.shard_count,
        )
        frames = [
            pd.read_parquet(f, columns=["location_id", "date", "pet"])
            for f in all_files
        ]
        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year

        if args.shard_count > 1:
            location_ids = pd.to_numeric(df["location_id"], errors="coerce")
            df = df[location_ids.mod(args.shard_count).eq(args.shard_index)].copy()

        return df

    if pet_csv_path.exists():
        return _load_pet_frame_from_csv(
            pet_csv_path,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )

    logger.warning("No PET data found in %s or %s.", pet_root, pet_csv_path)
    return pd.DataFrame(columns=["location_id", "date", "pet", "year"])


def _output_dir(out_dir: Path, *, shard_index: int, shard_count: int) -> Path:
    return out_dir / f"shard_count={shard_count:05d}" / f"shard_index={shard_index:05d}"


def main() -> None:
    """Load PET data and generate analytical CSV files for one shard."""
    args = _parse_args()
    output_dir = _output_dir(
        Path(args.out_dir),
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_pet_frame(args)

    if df.empty:
        logger.warning(
            "No PET rows matched shard %s/%s. Skipping.",
            args.shard_index,
            args.shard_count,
        )
        return

    generate_percentiles(df, output_dir)
    generate_forecast(df, output_dir)
    generate_change_per_decade(df, output_dir)
    logger.info("Analytics generation complete for %s.", output_dir)


if __name__ == "__main__":
    main()
