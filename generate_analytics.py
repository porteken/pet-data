"""Generate PET analytics from PET CSV shards or a materialized PET CSV."""

from __future__ import annotations

import argparse
import logging
from importlib import import_module
from pathlib import Path
from typing import Any, TypeAlias, TypedDict, cast

DataFrame: TypeAlias = Any
np: Any = cast("Any", import_module("numpy"))
pd: Any = cast("Any", import_module("pandas"))

PET_CSV_NAME = "pet.csv"
ANALYTICS_ROOT = Path("analytics_data_csv")
PET_ROOT = Path("pet_data_csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class ForecastRecord(TypedDict):
    """A single PET forecast row."""

    location_id: object
    year: int
    pet: float


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate PET analytics from partitioned PET CSV shards or a "
            "materialized pet.csv."
        ),
    )
    parser.add_argument("--pet-root", default=str(PET_ROOT))
    parser.add_argument("--pet-csv", default=PET_CSV_NAME)
    parser.add_argument("--out-dir", default=str(ANALYTICS_ROOT))
    parser.add_argument("--tile-id", dest="tile_ids", action="append", type=int)
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
    output_path = output_dir / "percentiles.csv"
    logger.info("Generating %s...", output_path)
    agg_df = (
        df.groupby(["year", "location_id"])["pet"].agg(p10=_p10, p90=_p90).reset_index()
    )

    rounded_df = agg_df.round(1)
    rounded_df.to_csv(output_path, index=False)
    return output_path


def generate_forecast(df: DataFrame, output_dir: Path) -> Path:
    """Generate linear trend forecasts for future decades (2030, 2040, 2050)."""
    output_path = output_dir / "forecast.csv"
    logger.info("Generating %s...", output_path)
    yearly_avg = df.groupby(["location_id", "year"])["pet"].mean().reset_index()

    forecast_records: list[ForecastRecord] = []
    for loc_id, group in yearly_avg.groupby("location_id"):
        if len(group) <= 1:
            continue

        x = np.asarray(group["year"].to_numpy(), dtype=float)
        y = np.asarray(group["pet"].to_numpy(), dtype=float)
        slope, intercept = (float(value) for value in np.polyfit(x, y, 1))

        for future_year in [2030, 2040, 2050]:
            projected_pet = (slope * future_year) + intercept
            forecast_records.append(
                {
                    "location_id": loc_id,
                    "year": future_year,
                    "pet": round(projected_pet, 2),
                },
            )

    forecast_df = pd.DataFrame(
        forecast_records,
        columns=["location_id", "year", "pet"],
    )
    forecast_df.to_csv(output_path, index=False)
    return output_path


def generate_change_per_decade(df: DataFrame, output_dir: Path) -> Path:
    """Calculate the change in average PET between decades."""
    output_path = output_dir / "change_per_decade.csv"
    logger.info("Generating %s...", output_path)
    yearly_avg = df.groupby(["location_id", "year"])["pet"].mean().reset_index()

    yearly_avg["decade_start"] = (yearly_avg["year"] // 10) * 10

    decade_avg = (
        yearly_avg.groupby(["location_id", "decade_start"])["pet"].mean().reset_index()
    )

    decade_avg = decade_avg.sort_values(["location_id", "decade_start"])
    decade_avg["change_value"] = decade_avg.groupby("location_id")["pet"].diff()
    decade_avg["decade"] = decade_avg["decade_start"].astype(str) + "s"
    decade_avg = decade_avg.dropna(subset=["change_value"])

    final_df = pd.DataFrame(
        decade_avg[["location_id", "decade", "change_value"]].round(2),
        columns=["location_id", "decade", "change_value"],
    ).rename(columns={"change_value": "change"})
    final_df.to_csv(output_path, index=False)
    return output_path


def _discover_pet_shards(pet_root: Path) -> dict[int, list[Path]]:
    shard_mapping: dict[int, list[Path]] = {}
    if not pet_root.exists():
        return shard_mapping

    for shard_path in sorted(pet_root.rglob(PET_CSV_NAME)):
        tile_id = _parse_partition_value(shard_path, pet_root, "tile_id")
        if tile_id is None:
            continue
        shard_mapping.setdefault(tile_id, []).append(shard_path)

    return shard_mapping


def _parse_partition_value(
    shard_path: Path,
    root_path: Path,
    partition_key: str,
) -> int | None:
    try:
        relative_parts = shard_path.relative_to(root_path).parts
    except ValueError:
        return None

    for part in relative_parts:
        prefix = f"{partition_key}="
        if part.startswith(prefix):
            try:
                return int(part.removeprefix(prefix))
            except ValueError:
                return None

    return None


def _select_tile_ids(
    available_tile_ids: list[int],
    *,
    requested_tile_ids: list[int] | None,
    shard_index: int,
    shard_count: int,
) -> list[int]:
    if shard_count < 1:
        msg = "shard_count must be >= 1"
        raise ValueError(msg)
    if shard_index < 0 or shard_index >= shard_count:
        msg = f"shard_index must be between 0 and {shard_count - 1}"
        raise ValueError(msg)

    allowed_tiles = None if requested_tile_ids is None else set(requested_tile_ids)
    filtered_tile_ids = [
        tile_id
        for tile_id in sorted(available_tile_ids)
        if allowed_tiles is None or tile_id in allowed_tiles
    ]
    return [
        tile_id
        for position, tile_id in enumerate(filtered_tile_ids)
        if position % shard_count == shard_index
    ]


def _load_pet_frame_from_shards(
    pet_root: Path,
    *,
    tile_ids: list[int],
) -> DataFrame:
    shard_mapping = _discover_pet_shards(pet_root)
    selected_paths = [
        shard_path
        for tile_id in tile_ids
        for shard_path in shard_mapping.get(tile_id, [])
    ]
    if not selected_paths:
        return pd.DataFrame(columns=["location_id", "date", "pet", "year"])

    logger.info(
        "Loading PET rows from %s shard CSV files across %s tiles...",
        len(selected_paths),
        len(tile_ids),
    )
    shard_frames = [
        pd.read_csv(shard_path, usecols=["location_id", "date", "pet"])
        for shard_path in selected_paths
    ]
    df = pd.concat(shard_frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    return df


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
    shard_mapping = _discover_pet_shards(pet_root)

    if shard_mapping:
        selected_tile_ids = _select_tile_ids(
            sorted(shard_mapping),
            requested_tile_ids=args.tile_ids,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
        if not selected_tile_ids:
            logger.warning("No PET tiles matched the requested shard filters.")
            return pd.DataFrame(columns=["location_id", "date", "pet", "year"])

        logger.info(
            "Selected %s PET tiles for analytics shard %s/%s.",
            len(selected_tile_ids),
            args.shard_index,
            args.shard_count,
        )
        return _load_pet_frame_from_shards(pet_root, tile_ids=selected_tile_ids)

    if not pet_csv_path.exists():
        msg = "No PET shard CSV files or materialized pet.csv were found."
        raise FileNotFoundError(msg)

    logger.info(
        "PET shard CSV files not found under %s. Falling back to %s.",
        pet_root,
        pet_csv_path,
    )
    return _load_pet_frame_from_csv(
        pet_csv_path,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )


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

    try:
        df = _load_pet_frame(args)
    except FileNotFoundError:
        logger.exception("PET inputs not found. Ensure calculate_pet.py has run.")
        return

    if df.empty:
        logger.warning("No PET rows matched the requested analytics shard.")
        return

    generate_percentiles(df, output_dir)
    generate_forecast(df, output_dir)
    generate_change_per_decade(df, output_dir)
    logger.info("Analytics generation complete for %s.", output_dir)


if __name__ == "__main__":
    main()
