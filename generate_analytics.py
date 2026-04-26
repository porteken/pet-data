"""Generate PET analytics from parquet or CSV data."""

from __future__ import annotations

import argparse
import logging
import math
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

DataFrame: TypeAlias = Any
np: Any = cast("Any", import_module("numpy"))
pd: Any = cast("Any", import_module("pandas"))

PET_CSV_NAME = "pet.csv"
PET_BATCH_GLOB = "pet_batch_*.parquet"
ANALYTICS_ROOT = Path("analytics_data_csv")
PET_ROOT = Path("pet_data_csv")
FORECAST_END_YEAR = 2100
GENERATING_LOG_MESSAGE = "Generating %s..."
DEFAULT_MAX_WORKERS = -1

MIN_FULL_YEARS_TO_FORECAST = 10
MIN_FULL_YEARS_FOR_ACCELERATION = 25
MIN_BACKTEST_TRAIN_YEARS_LINEAR = 12
MIN_BACKTEST_TRAIN_YEARS_QUADRATIC = 20
MIN_BACKTEST_TEST_POINTS = 5
QUADRATIC_RMSE_IMPROVEMENT = 0.98
ACCELERATION_DAMPING_START_YEARS = 15
ACCELERATION_DAMPING_END_YEARS = 35
PREDICTION_INTERVAL_Z = 1.2816
UNCERTAINTY_GROWTH_DENOMINATOR = 10.0
UNCERTAINTY_GROWTH_CAP_YEARS = 25.0
QUADRATIC_UNCERTAINTY_DENOMINATOR = 30.0

QUADRATIC_DEGREE = 2

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class ForecastRecord(TypedDict):
    """A single PET forecast row."""

    location_id: object
    year: int
    pet: float
    lower: float
    upper: float
    model_type: str
    full_years_used: int
    warming_rate: float
    acceleration: float


class TrendModel(TypedDict):
    """Polynomial trend model metadata."""

    degree: int
    coef: object
    reference_year: float
    rmse: float
    cv_rmse: float


def _expected_days_for_years(years: object) -> object:
    years_arr = np.asarray(years, dtype="int32")
    leap_mask = (years_arr % 4 == 0) & ((years_arr % 100 != 0) | (years_arr % 400 == 0))
    return np.where(leap_mask, 366, 365)


def _yearly_average_pet(df: DataFrame, *, allow_incomplete: bool = False) -> DataFrame:
    """Return annual mean PET. If allow_incomplete=False, strictly require complete days."""
    if df.empty:
        return pd.DataFrame(
            columns=["location_id", "year", "pet", "days_present", "expected_days"],
        )

    daily = (
        df[["location_id", "date", "pet"]]
        .dropna(subset=["location_id", "date", "pet"])
        .copy()
    )
    daily["date"] = pd.to_datetime(daily["date"]).dt.floor("D")

    daily = daily.groupby(["location_id", "date"], as_index=False)["pet"].mean()
    daily["year"] = daily["date"].dt.year.astype("int32")

    yearly = daily.groupby(["location_id", "year"], as_index=False).agg(
        days_present=("date", "nunique"),
        pet=("pet", "mean"),
    )
    yearly["expected_days"] = _expected_days_for_years(yearly["year"].to_numpy())
    if not allow_incomplete:
        yearly = yearly[yearly["days_present"].eq(yearly["expected_days"])].copy()

    return yearly.sort_values(["location_id", "year"]).reset_index(drop=True)


def _empty_forecast_frame() -> DataFrame:
    return pd.DataFrame(
        columns=[
            "location_id",
            "year",
            "pet",
            "lower",
            "upper",
            "model_type",
            "full_years_used",
            "warming_rate",
            "acceleration",
        ],
    )


def _fit_polynomial_model(
    years: object,
    values: object,
    degree: int,
) -> TrendModel | None:
    years_arr = np.asarray(years, dtype="float64")
    values_arr = np.asarray(values, dtype="float64")

    if len(years_arr) <= degree:
        return None

    reference_year = float(np.median(years_arr))
    x = years_arr - reference_year

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coef = np.polyfit(x, values_arr, degree)

    fitted = np.polyval(coef, x)
    residuals = values_arr - fitted
    rmse = float(np.sqrt(np.mean(np.square(residuals)))) if len(residuals) else 0.0

    return TrendModel(
        degree=degree,
        coef=coef,
        reference_year=reference_year,
        rmse=rmse,
        cv_rmse=math.inf,
    )


def _predict_from_model(model: TrendModel, years: object) -> object:
    years_arr = np.asarray(years, dtype="float64")
    x = years_arr - float(model["reference_year"])
    return np.polyval(model["coef"], x)


def _slope_at_year(model: TrendModel, year: float) -> float:
    coef = np.asarray(model["coef"], dtype="float64")
    x = float(year) - float(model["reference_year"])
    if int(model["degree"]) == 1:
        return float(coef[0])
    return float((2.0 * coef[0] * x) + coef[1])


def _acceleration_from_model(model: TrendModel) -> float:
    coef = np.asarray(model["coef"], dtype="float64")
    if model["degree"] == QUADRATIC_DEGREE:
        return float(2.0 * coef[0])
    return 0.0


def _rolling_origin_rmse(
    years: object,
    values: object,
    *,
    degree: int,
    min_train_years: int,
) -> float:
    years_arr = np.asarray(years, dtype="float64")
    values_arr = np.asarray(values, dtype="float64")

    if len(years_arr) < (min_train_years + 3):
        return math.inf

    errors: list[float] = []
    for end_idx in range(min_train_years, len(years_arr)):
        train_years = years_arr[:end_idx]
        train_values = values_arr[:end_idx]
        model = _fit_polynomial_model(train_years, train_values, degree)
        if model is None:
            continue
        predicted = float(
            cast("Any", _predict_from_model(model, np.asarray([years_arr[end_idx]])))[
                0
            ],
        )
        errors.append(float(values_arr[end_idx] - predicted))

    if len(errors) < MIN_BACKTEST_TEST_POINTS:
        return math.inf

    return float(np.sqrt(np.mean(np.square(errors))))


def _select_best_model(loc_df: DataFrame) -> TrendModel | None:
    years = loc_df["year"].to_numpy(dtype="float64")
    values = loc_df["pet"].to_numpy(dtype="float64")

    linear_model = _fit_polynomial_model(years, values, degree=1)
    if linear_model is None:
        return None
    linear_model["cv_rmse"] = _rolling_origin_rmse(
        years,
        values,
        degree=1,
        min_train_years=MIN_BACKTEST_TRAIN_YEARS_LINEAR,
    )
    return linear_model


def _apply_quadratic_damping(
    model: TrendModel,
    future_years: object,
    *,
    last_year: int,
) -> object:
    raw = np.asarray(_predict_from_model(model, future_years), dtype="float64")

    if model["degree"] != QUADRATIC_DEGREE:
        return raw

    start_year = min(last_year + ACCELERATION_DAMPING_START_YEARS, FORECAST_END_YEAR)
    end_year = min(last_year + ACCELERATION_DAMPING_END_YEARS, FORECAST_END_YEAR)

    if start_year >= end_year:
        return raw

    future_years_arr = np.asarray(future_years, dtype="float64")
    y_start = float(
        cast(
            "Any",
            _predict_from_model(model, np.asarray([start_year], dtype="float64")),
        )[0],
    )
    slope_start = _slope_at_year(model, float(start_year))
    linear_tail = y_start + (future_years_arr - float(start_year)) * slope_start

    weights = np.clip(
        (future_years_arr - float(start_year)) / float(end_year - start_year),
        0.0,
        1.0,
    )
    return (raw * (1.0 - weights)) + (linear_tail * weights)


def _forecast_single_location(
    loc_id: object,
    loc_df: DataFrame,
) -> DataFrame | None:
    """Fit a city-specific historical PET trend model and return forecast rows."""
    loc_df = loc_df.sort_values("year").reset_index(drop=True)

    if len(loc_df) < MIN_FULL_YEARS_TO_FORECAST:
        return None

    model = _select_best_model(loc_df)
    if model is None:
        return None

    last_year = int(loc_df["year"].max())
    periods_to_forecast = FORECAST_END_YEAR - last_year
    if periods_to_forecast <= 0:
        return None

    future_years = np.arange(last_year + 1, FORECAST_END_YEAR + 1, dtype="int32")
    future_pet = _apply_quadratic_damping(model, future_years, last_year=last_year)

    base_rmse = max(
        float(model["rmse"]),
        float(model["cv_rmse"]) if math.isfinite(float(model["cv_rmse"])) else 0.0,
    )
    horizons = (future_years - last_year).astype("float64")
    capped_horizons = np.minimum(horizons, UNCERTAINTY_GROWTH_CAP_YEARS)

    is_quadratic = model["degree"] == QUADRATIC_DEGREE
    quadratic_term = (
        (capped_horizons / QUADRATIC_UNCERTAINTY_DENOMINATOR) ** 2
        if is_quadratic
        else 0.0
    )
    sigma = base_rmse * np.sqrt(
        1.0 + (capped_horizons / UNCERTAINTY_GROWTH_DENOMINATOR) + quadratic_term,
    )
    lower = future_pet - (PREDICTION_INTERVAL_Z * sigma)
    upper = future_pet + (PREDICTION_INTERVAL_Z * sigma)

    warming_rate = np.asarray(
        [_slope_at_year(model, float(year)) for year in future_years],
        dtype="float64",
    )
    acceleration = np.full(
        shape=len(future_years),
        fill_value=_acceleration_from_model(model),
        dtype="float64",
    )

    model_type = "quadratic" if model["degree"] == QUADRATIC_DEGREE else "linear"
    logger.debug(
        "Location %s: selected %s model (warming_rate=%.4f °C/yr, "
        "acceleration=%.5f, full_years=%d, cv_rmse=%.3f)",
        loc_id,
        model_type,
        float(warming_rate[0]),
        float(_acceleration_from_model(model)),
        len(loc_df),
        float(model["cv_rmse"])
        if math.isfinite(float(model["cv_rmse"]))
        else float("nan"),
    )

    return pd.DataFrame(
        {
            "location_id": loc_id,
            "year": future_years.astype("int32"),
            "pet": cast("Any", future_pet).astype("float64"),
            "lower": cast("Any", lower).astype("float64"),
            "upper": cast("Any", upper).astype("float64"),
            "model_type": model_type,
            "full_years_used": len(loc_df),
            "warming_rate": warming_rate,
            "acceleration": acceleration,
        },
    )


def _run_serial_forecasts(
    loc_frames: list[tuple[object, DataFrame]],
) -> list[DataFrame]:
    results: list[DataFrame] = []
    for loc_id, loc_df in loc_frames:
        result = _forecast_single_location(loc_id, loc_df)
        if result is not None:
            results.append(result)
    return results


def _run_parallel_forecasts(
    loc_frames: list[tuple[object, DataFrame]],
    n_workers: int,
) -> list[DataFrame]:
    results: list[DataFrame] = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(_forecast_single_location, loc_id, loc_df): loc_id
            for loc_id, loc_df in loc_frames
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


def _build_forecast_frame(
    df: DataFrame,
    *,
    max_workers: int = 1,
    allow_incomplete: bool = False,
) -> DataFrame:
    """Fit location-specific historical trend models and return forecast rows."""
    yearly_avg = _yearly_average_pet(df, allow_incomplete=allow_incomplete)
    if yearly_avg.empty:
        logger.warning(
            "No complete location-years found. Forecast outputs will be empty.",
        )
        return _empty_forecast_frame()

    loc_frames = [
        (
            loc_id,
            loc_df[["location_id", "year", "pet"]].copy(),
        )
        for loc_id, loc_df in yearly_avg.groupby("location_id", sort=True)
    ]

    if max_workers == 1:
        all_forecasts = _run_serial_forecasts(loc_frames)
    else:
        n_workers = max_workers if max_workers > 0 else os.cpu_count()
        all_forecasts = _run_parallel_forecasts(loc_frames, n_workers or 1)

    if not all_forecasts:
        logger.warning("No forecasts were generated.")
        return _empty_forecast_frame()

    final_forecast_df = pd.concat(all_forecasts, ignore_index=True).round(
        {
            "pet": 2,
            "lower": 2,
            "upper": 2,
            "warming_rate": 4,
            "acceleration": 5,
        },
    )

    float_cols = final_forecast_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        final_forecast_df[float_cols] = final_forecast_df[float_cols].astype("float32")

    int_cols = ["year", "full_years_used"]
    for col in int_cols:
        if col in final_forecast_df.columns:
            final_forecast_df[col] = final_forecast_df[col].astype("int32")

    return final_forecast_df.sort_values(["location_id", "year"]).reset_index(drop=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PET analytics from partitioned PET parquet shards or a "
            "materialized pet.csv using city-specific historical PET trends."
        ),
    )
    parser.add_argument("--pet-root", default=str(PET_ROOT))
    parser.add_argument("--pet-csv", default=PET_CSV_NAME)
    parser.add_argument(
        "--prefer-pet-csv",
        action="store_true",
        help=(
            "Load the explicit --pet-csv input even when PET parquet shards are "
            "also present under --pet-root."
        ),
    )
    parser.add_argument("--out-dir", default=str(ANALYTICS_ROOT))
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=20)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=DEFAULT_MAX_WORKERS,
        help=(
            "Worker processes for parallel city trend fitting. "
            "-1 = os.cpu_count() (default). 1 = single-process."
        ),
    )
    parser.add_argument(
        "--allow-incomplete-years",
        action="store_true",
        help="Allow computing annual average PET with incomplete years.",
    )
    return parser.parse_args()


def _quantile(values: object, quantile: float) -> float:
    return float(cast("Any", values).quantile(quantile))


def _p10(values: object) -> float:
    return _quantile(values, 0.1)


def _p90(values: object) -> float:
    return _quantile(values, 0.9)


def generate_percentiles(df: DataFrame, output_dir: Path) -> Path:
    """Calculate the 10th and 90th percentile of PET per year per location."""
    output_path = output_dir / "percentiles.parquet"
    logger.info(GENERATING_LOG_MESSAGE, output_path)

    working = (
        df[["location_id", "date", "pet"]]
        .dropna(subset=["location_id", "date", "pet"])
        .copy()
    )
    working["date"] = pd.to_datetime(working["date"])
    working["year"] = working["date"].dt.year.astype("int32")

    agg_df = (
        working.groupby(["year", "location_id"])["pet"]
        .agg(p10=_p10, p90=_p90)
        .reset_index()
    )

    rounded_df = agg_df.round(1)
    float_cols = rounded_df.select_dtypes(include=["float64"]).columns
    if len(float_cols) > 0:
        rounded_df[float_cols] = rounded_df[float_cols].astype("float32")

    tmp_output_path = output_path.with_suffix(".tmp")
    rounded_df.to_parquet(tmp_output_path, index=False, compression="snappy")
    tmp_output_path.rename(output_path)
    return output_path


def generate_forecast(forecast_df: DataFrame, output_dir: Path) -> Path:
    """Write the pre-computed city trend forecast frame to parquet."""
    output_path = output_dir / "forecast.parquet"
    logger.info(GENERATING_LOG_MESSAGE, output_path)

    tmp_output_path = output_path.with_suffix(".tmp")
    forecast_df.to_parquet(tmp_output_path, index=False, compression="snappy")
    tmp_output_path.rename(output_path)
    return output_path


def generate_change_per_decade(
    df: DataFrame,
    forecast_df: DataFrame,
    output_dir: Path,
    *,
    allow_incomplete: bool = False,
) -> Path:
    """Calculate the change in average PET between decades."""
    output_path = output_dir / "change_per_decade.parquet"
    logger.info(GENERATING_LOG_MESSAGE, output_path)

    yearly_avg = _yearly_average_pet(df, allow_incomplete=allow_incomplete)[
        ["location_id", "year", "pet"]
    ].copy()
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
    """Return all PET parquet batch files sorted by path."""
    if not pet_root.exists():
        return []
    return sorted(pet_root.rglob(PET_BATCH_GLOB))


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
    prefer_pet_csv = bool(getattr(args, "prefer_pet_csv", False))

    if prefer_pet_csv and pet_csv_path.exists():
        return _load_pet_frame_from_csv(
            pet_csv_path,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )

    all_files = _discover_pet_files(pet_root)
    if all_files:
        logger.info(
            "Loading %s PET files for analytics shard %s/%s.",
            len(all_files),
            args.shard_index,
            args.shard_count,
        )

        frames = []
        for f in all_files:
            df_part = pd.read_parquet(f, columns=["location_id", "date", "pet"])
            if args.shard_count > 1:
                loc_ids = pd.to_numeric(df_part["location_id"], errors="coerce")
                df_part = df_part[loc_ids.mod(args.shard_count).eq(args.shard_index)]
            frames.append(df_part)

        if not frames:
            return pd.DataFrame(columns=["location_id", "date", "pet", "year"])

        df = pd.concat(frames, ignore_index=True)
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year
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
    """Generate analytical files for a PET shard."""
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

    effective_workers = (
        args.max_workers if args.max_workers > 0 else (os.cpu_count() or 1)
    )

    forecast_df = _build_forecast_frame(
        df,
        max_workers=effective_workers,
        allow_incomplete=args.allow_incomplete_years,
    )

    generate_percentiles(df, output_dir)
    generate_forecast(forecast_df, output_dir)
    generate_change_per_decade(
        df,
        forecast_df,
        output_dir,
        allow_incomplete=args.allow_incomplete_years,
    )
    logger.info("Analytics generation complete for %s.", output_dir)


if __name__ == "__main__":
    main()
