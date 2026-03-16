"""Download and shard UTCI mean radiant temperature data from CDS."""

from __future__ import annotations

import argparse
import importlib
import multiprocessing
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

from tqdm.auto import tqdm

from boxes import GRID_DEG, OUTPUT_DIR, generate_tile_outputs
from pull_cds_shared import (
    LOGGER,
    CDSClient,
    DataFrame,
    SeriesLike,
    create_cds_client,
    extract_files,
    pa,
    partition_exists,
    pd,
    pq,
    retrieve_with_retry,
)
from shards import resolve_filesystem
from shared_config import build_year_months

MRT_DATASET = "derived-utci-historical"
MRT_CDS_REQUEST_CONCURRENCY = 2


def process_mrt(
    client: CDSClient,
    year: str,
    out_dir: str,
    *,
    month: int | None = None,
    tile_ids: list[int] | None = None,
    tile_shard_index: int = 0,
    tile_shard_count: int = 1,
    max_workers: int = 4,
) -> None:
    """Download and process UTCI mean radiant temperature data grouped by tile."""
    mrt_root = f"{out_dir}/utci_data_parquet"
    selected_tiles, selected_cells = _load_selected_tiles(
        tile_ids=tile_ids,
        tile_shard_index=tile_shard_index,
        tile_shard_count=tile_shard_count,
    )
    tile_jobs = [
        (
            tile._asdict(),
            selected_cells[selected_cells["tile_id"] == tile.tile_id][
                ["grid_lat", "grid_lon"]
            ].reset_index(drop=True),
        )
        for tile in selected_tiles.itertuples(index=False)
    ]
    if not tile_jobs:
        LOGGER.warning("No MRT tiles selected for processing.")
        return

    _run_mrt_tile_jobs(
        tile_jobs=tile_jobs,
        max_workers=max_workers,
        mrt_root=mrt_root,
        year=year,
        month=month,
        client=client,
    )

    LOGGER.info(
        "MRT processing complete for %s selected tiles.",
        len(selected_tiles),
    )


def _load_selected_tiles(
    *,
    tile_ids: list[int] | None,
    tile_shard_index: int,
    tile_shard_count: int,
) -> tuple[DataFrame, DataFrame]:
    _ensure_tile_outputs()
    tile_boxes_df = pd.read_csv(Path(OUTPUT_DIR) / "tile_boxes.csv")
    unique_cells_df = pd.read_csv(Path(OUTPUT_DIR) / "unique_grid_cells.csv")
    merge_columns = ["tile_lat_min", "tile_lon_min"]
    if "tile_deg" in tile_boxes_df.columns and "tile_deg" in unique_cells_df.columns:
        merge_columns.append("tile_deg")

    unique_cells_df = unique_cells_df.merge(
        tile_boxes_df[["tile_id", *merge_columns]],
        how="inner",
        on=merge_columns,
        validate="many_to_one",
    )

    available_tile_ids = sorted(
        int(tile_id) for tile_id in tile_boxes_df["tile_id"].tolist()
    )
    allowed_tile_ids = (
        available_tile_ids
        if tile_ids is None
        else sorted({int(tile_id) for tile_id in tile_ids})
    )
    selected_tile_ids = [
        tile_id
        for position, tile_id in enumerate(allowed_tile_ids)
        if position % tile_shard_count == tile_shard_index
    ]
    selected_tiles = tile_boxes_df[
        tile_boxes_df["tile_id"].isin(selected_tile_ids)
    ].copy()
    selected_cells = unique_cells_df[
        unique_cells_df["tile_id"].isin(selected_tile_ids)
    ].copy()
    return selected_tiles.sort_values("tile_id"), selected_cells.sort_values(
        ["tile_id", "grid_lat", "grid_lon"],
    )


def _ensure_tile_outputs() -> None:
    required_paths = [
        Path(OUTPUT_DIR) / "tile_boxes.csv",
        Path(OUTPUT_DIR) / "unique_grid_cells.csv",
    ]
    if all(path.exists() for path in required_paths):
        LOGGER.info("Using existing tile metadata under %s.", OUTPUT_DIR)
        return

    LOGGER.info("Tile metadata missing. Regenerating under %s.", OUTPUT_DIR)
    generate_tile_outputs()


def _run_mrt_tile_jobs(
    *,
    tile_jobs: list[tuple[dict[str, Any], DataFrame]],
    max_workers: int,
    mrt_root: str,
    year: str,
    month: int | None,
    client: CDSClient,
) -> None:
    progress_desc = "MRT tiles"
    worker_count = max(
        1,
        min(max_workers, len(tile_jobs), MRT_CDS_REQUEST_CONCURRENCY),
    )
    if worker_count == 1:
        for tile_row, tile_cells in tqdm(
            tile_jobs,
            desc=progress_desc,
            unit="tile",
        ):
            _process_mrt_tile(
                client=client,
                year=year,
                month=month,
                mrt_root=mrt_root,
                tile_row=tile_row,
                tile_cells=tile_cells,
            )
        return

    LOGGER.info(
        "Processing %s MRT tiles with %s worker processes.",
        len(tile_jobs),
        worker_count,
    )
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future_to_tile_id = {
            executor.submit(
                _process_mrt_tile_with_new_client,
                year=year,
                month=month,
                mrt_root=mrt_root,
                tile_row=tile_row,
                tile_cells=tile_cells,
            ): int(tile_row["tile_id"])
            for tile_row, tile_cells in tile_jobs
        }
        for future in tqdm(
            as_completed(future_to_tile_id),
            total=len(future_to_tile_id),
            desc=progress_desc,
            unit="tile",
        ):
            tile_id = future_to_tile_id[future]
            future.result()
            LOGGER.info("MRT tile %s completed for %s.", tile_id, year)


def _process_mrt_tile(
    *,
    client: CDSClient,
    year: str,
    month: int | None,
    mrt_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    tile_id = int(tile_row["tile_id"])
    partition_path = _mrt_partition_path(year, tile_id, month=month)
    if partition_exists(mrt_root, partition_path):
        LOGGER.info(
            "MRT tile %s already exists for %s%s. Skipping.",
            tile_id,
            year,
            "" if month is None else f"-{month:02d}",
        )
        return

    with tempfile.TemporaryDirectory() as tmpdir_name:
        tmpdir = Path(tmpdir_name)
        zip_label = f"{year}_tile_{tile_id}"
        if month is not None:
            zip_label = f"{year}_{month:02d}_tile_{tile_id}"
        zip_path = tmpdir / f"mrt_{zip_label}.zip"
        LOGGER.info(
            "Starting MRT tile %s for %s%s.",
            tile_id,
            year,
            "" if month is None else f"-{month:02d}",
        )
        retrieve_with_retry(
            client,
            MRT_DATASET,
            {
                "variable": ["mean_radiant_temperature"],
                "version": "1_1",
                "product_type": "consolidated_dataset",
                "year": year,
                "month": build_year_months(int(year), month=month),
                "day": [f"{day:02d}" for day in range(1, 32)],
                "area": [
                    float(tile_row["north"]),
                    float(tile_row["west"]),
                    float(tile_row["south"]),
                    float(tile_row["east"]),
                ],
            },
            str(zip_path),
        )

        normalized_tile_cells = tile_cells.rename(
            columns={"grid_lat": "lat", "grid_lon": "lng"},
        )
        mrt_df = _load_mrt_frame(zip_path)
        mrt_df = mrt_df.merge(
            normalized_tile_cells,
            how="inner",
            on=["lat", "lng"],
            validate="many_to_one",
        )
        if mrt_df.empty:
            LOGGER.warning(
                "No MRT rows remained after grid-cell filtering for tile %s.",
                tile_id,
            )
            return

        _write_mrt_partition(
            mrt_root=mrt_root,
            partition_path=partition_path,
            mrt_df=mrt_df,
        )


def _process_mrt_tile_with_new_client(
    *,
    year: str,
    month: int | None,
    mrt_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    _process_mrt_tile(
        client=create_cds_client(),
        year=year,
        month=month,
        mrt_root=mrt_root,
        tile_row=tile_row,
        tile_cells=tile_cells,
    )


def _mrt_partition_path(
    year: str,
    tile_id: int,
    *,
    month: int | None = None,
) -> str:
    if month is not None:
        return f"year={year}/month={month:02d}/tile_id={tile_id}"
    return f"year={year}/tile_id={tile_id}"


def _write_mrt_partition(
    *,
    mrt_root: str,
    partition_path: str,
    mrt_df: DataFrame,
) -> None:
    filesystem, base_path = resolve_filesystem(mrt_root)
    partition_dir = f"{base_path}/{partition_path}"
    filesystem.create_dir(partition_dir, recursive=True)
    output_path = f"{partition_dir}/mrt.parquet"
    table: Any = pa.Table.from_pandas(mrt_df, preserve_index=False)
    with filesystem.open_output_stream(output_path) as output_stream:
        pq.write_table(table, output_stream)


def _load_mrt_frame(zip_path: Path) -> DataFrame:
    xr = cast("Any", importlib.import_module("xarray"))
    nc_files = extract_files(zip_path, suffix=".nc")
    frames: list[DataFrame] = []
    for nc_file in nc_files:
        with xr.open_dataset(
            nc_file,
            engine="netcdf4",
            decode_timedelta=True,
        ) as dataset:
            frame = dataset.to_dataframe().reset_index()
        frames.append(_normalize_mrt_columns(frame))

    if not frames:
        msg = f"No NetCDF files found in MRT download: {zip_path}"
        raise FileNotFoundError(msg)

    return pd.concat(frames, ignore_index=True)


def _normalize_mrt_columns(df: DataFrame) -> DataFrame:
    df = df.rename(
        columns={
            "lon": "lng",
            "longitude": "lng",
            "latitude": "lat",
            "time": "timestamp",
            "mrt": "mean_radiant_temperature_c",
        },
    )

    required_columns = {"timestamp", "lat", "lng", "mean_radiant_temperature_c"}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        msg = (
            "MRT download missing expected columns: "
            f"{', '.join(sorted(missing_columns))}"
        )
        raise ValueError(msg)

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["lat"] = _snap_grid_series(df["lat"])
    df["lng"] = _snap_grid_series(df["lng"])
    df["mean_radiant_temperature_c"] = (
        pd.to_numeric(
            df["mean_radiant_temperature_c"],
            errors="coerce",
        )
        - 273.15
    )

    return (
        df.dropna(subset=["timestamp", "lat", "lng", "mean_radiant_temperature_c"])
        .drop(columns=["height"], errors="ignore")
        .reset_index(drop=True)
    )


def _snap_grid_series(series: SeriesLike) -> SeriesLike:
    return (pd.to_numeric(series, errors="coerce") / GRID_DEG).round() * GRID_DEG


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull MRT data from CDS.")
    parser.add_argument(
        "--year",
        required=True,
        type=str,
        help="Year to download (e.g., 2023).",
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        help="Optional month to download within the requested MRT year (1-12).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=".",
        help="Output directory base path or S3 URI (e.g., s3://my-bucket/data)",
    )
    parser.add_argument(
        "--tile-id",
        dest="tile_ids",
        action="append",
        type=int,
        help="Optional tile_id filter. Pass multiple times to target specific tiles.",
    )
    parser.add_argument(
        "--tile-shard-index",
        type=int,
        default=0,
        help="Zero-based shard index for optional MRT/manual tile splitting.",
    )
    parser.add_argument(
        "--tile-shard-count",
        type=int,
        default=1,
        help="Total shard count for optional MRT/manual tile splitting.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Maximum number of concurrent workers for MRT pulls.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the MRT CDS download and parquet conversion steps."""
    args = _parse_args()

    try:
        client = create_cds_client()
    except Exception as exc:
        LOGGER.exception(
            "Failed to initialize CDS API client. Ensure your credentials are set.",
        )
        raise SystemExit(1) from exc

    try:
        process_mrt(
            client,
            args.year,
            args.out_dir,
            month=args.month,
            tile_ids=args.tile_ids,
            tile_shard_index=args.tile_shard_index,
            tile_shard_count=args.tile_shard_count,
            max_workers=args.max_workers,
        )
    except Exception as exc:
        LOGGER.exception("MRT pipeline failed for year=%s", args.year)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
