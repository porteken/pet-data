"""Download and shard UTCI mean radiant temperature data from CDS."""

from __future__ import annotations

import argparse
import importlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
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
    partition_file_max_timestamp,
    pd,
    pq,
    retrieve_with_retry,
)
from shards import resolve_filesystem
from shared_config import (
    build_mrt_date_bounds,
    build_mrt_days,
    build_mrt_months,
    mrt_product_type,
)

MRT_DATASET = "derived-utci-historical"
MRT_CDS_REQUEST_CONCURRENCY = max(
    1,
    int(os.getenv("MRT_CDS_REQUEST_CONCURRENCY", "4")),
)

MRT_CDS_REQUEST_SEMAPHORE = threading.BoundedSemaphore(MRT_CDS_REQUEST_CONCURRENCY)
MRT_THREAD_LOCAL = threading.local()
MRT_NETCDF_READ_LOCK = threading.Lock()


def process_mrt(
    client: CDSClient,
    year: str,
    out_dir: str,
    *,
    month: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    tile_ids: list[int] | None = None,
    tile_shard_index: int = 0,
    tile_shard_count: int = 1,
    max_workers: int = 4,
    disable_tqdm: bool = False,
) -> None:
    """Download and process UTCI mean radiant temperature data grouped by tile."""
    request_months, request_days, expected_end_date = _resolve_mrt_request_window(
        year=int(year),
        month=month,
        start_date=start_date,
        end_date=end_date,
    )
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
        request_months=request_months,
        request_days=request_days,
        expected_end_date=expected_end_date,
        client=client,
        disable_tqdm=disable_tqdm,
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
    request_months: list[str],
    request_days: list[str],
    expected_end_date: date,
    client: CDSClient,
    disable_tqdm: bool = False,
) -> None:
    progress_desc = "MRT tiles"
    worker_count = max(1, min(max_workers, len(tile_jobs), MRT_CDS_REQUEST_CONCURRENCY))
    if worker_count == 1:
        for tile_row, tile_cells in tqdm(
            tile_jobs,
            desc=progress_desc,
            unit="tile",
            disable=disable_tqdm,
        ):
            _process_mrt_tile(
                client=client,
                year=year,
                month=month,
                request_months=request_months,
                request_days=request_days,
                expected_end_date=expected_end_date,
                mrt_root=mrt_root,
                tile_row=tile_row,
                tile_cells=tile_cells,
            )
        return

    LOGGER.info(
        "Processing %s MRT tiles with %s worker threads.",
        len(tile_jobs),
        worker_count,
    )
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_tile_id = {
            executor.submit(
                _process_mrt_tile_with_thread_client,
                year=year,
                month=month,
                request_months=request_months,
                request_days=request_days,
                expected_end_date=expected_end_date,
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
            disable=disable_tqdm,
        ):
            tile_id = future_to_tile_id[future]
            future.result()
            LOGGER.info("MRT tile %s completed for %s.", tile_id, year)


def _compute_tight_area(tile_cells: DataFrame) -> list[float]:
    """Compute the tightest CDS area box [N, W, S, E] around actual grid cells."""
    north = float(tile_cells["grid_lat"].max())
    south = float(tile_cells["grid_lat"].min())
    east = float(tile_cells["grid_lon"].max())
    west = float(tile_cells["grid_lon"].min())
    return [north, west, south, east]


def _process_mrt_tile(
    *,
    client: CDSClient,
    year: str,
    month: int | None,
    request_months: list[str],
    request_days: list[str],
    expected_end_date: date,
    mrt_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    tile_id = int(tile_row["tile_id"])
    partition_path = _mrt_partition_path(year, tile_id, month=month)
    if partition_exists(mrt_root, partition_path) and _mrt_partition_is_current(
        mrt_root,
        partition_path,
        expected_end_date=expected_end_date,
    ):
        LOGGER.info(
            "MRT tile %s already exists for %s%s. Skipping.",
            tile_id,
            year,
            "" if month is None else f"-{month:02d}",
        )
        return

    with TemporaryDirectory() as tmpdir_name:
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
        with MRT_CDS_REQUEST_SEMAPHORE:
            retrieve_with_retry(
                client,
                MRT_DATASET,
                {
                    "variable": ["mean_radiant_temperature"],
                    "version": "1_1",
                    "product_type": mrt_product_type(int(year), month=month),
                    "year": year,
                    "month": request_months,
                    "day": request_days,
                    "area": _compute_tight_area(tile_cells),
                },
                str(zip_path),
            )

        normalized_tile_cells = tile_cells.rename(
            columns={"grid_lat": "lat", "grid_lon": "lng"},
        )
        with MRT_NETCDF_READ_LOCK:
            mrt_df = _load_mrt_frame(zip_path)
        mrt_df = mrt_df.merge(
            normalized_tile_cells,
            how="inner",
            on=["lat", "lng"],
            validate="many_to_one",
        )
        if mrt_df.empty:
            msg = f"No MRT rows remained after grid-cell filtering for tile {tile_id}."
            LOGGER.error(msg)
            raise RuntimeError(msg)

        _write_mrt_partition(
            mrt_root=mrt_root,
            partition_path=partition_path,
            mrt_df=mrt_df,
        )


def _process_mrt_tile_with_thread_client(
    *,
    year: str,
    month: int | None,
    request_months: list[str],
    request_days: list[str],
    expected_end_date: date,
    mrt_root: str,
    tile_row: dict[str, Any],
    tile_cells: DataFrame,
) -> None:
    client = getattr(MRT_THREAD_LOCAL, "client", None)
    if client is None:
        client = create_cds_client()
        MRT_THREAD_LOCAL.client = client
    _process_mrt_tile(
        client=client,
        year=year,
        month=month,
        request_months=request_months,
        request_days=request_days,
        expected_end_date=expected_end_date,
        mrt_root=mrt_root,
        tile_row=tile_row,
        tile_cells=tile_cells,
    )


def _resolve_mrt_request_window(
    *,
    year: int,
    month: int | None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[list[str], list[str], date]:
    if (start_date is None) != (end_date is None):
        msg = "--start-date and --end-date must be provided together."
        raise ValueError(msg)

    if start_date is None and end_date is None:
        _, allowed_end = build_mrt_date_bounds(year, month=month)
        return (
            build_mrt_months(year, month=month),
            (
                build_mrt_days(year, month=month)
                if month is not None
                else [f"{day:02d}" for day in range(1, 32)]
            ),
            allowed_end,
        )

    parsed_start = date.fromisoformat(cast("str", start_date))
    parsed_end = date.fromisoformat(cast("str", end_date))
    if parsed_start > parsed_end:
        msg = "--start-date must be on or before --end-date."
        raise ValueError(msg)
    if parsed_start.year != year or parsed_end.year != year:
        msg = "Explicit MRT date overrides must stay within the requested --year."
        raise ValueError(msg)
    if parsed_start.month != parsed_end.month:
        msg = "Explicit MRT date overrides must stay within a single month."
        raise ValueError(msg)

    effective_month = month if month is not None else parsed_start.month
    if parsed_start.month != effective_month or parsed_end.month != effective_month:
        msg = "Explicit MRT date overrides must match the requested --month."
        raise ValueError(msg)

    allowed_start, allowed_end = build_mrt_date_bounds(year, month=effective_month)
    if parsed_start < allowed_start or parsed_end > allowed_end:
        msg = (
            "Explicit MRT date overrides fall outside the supported pull window "
            f"{allowed_start.isoformat()} to {allowed_end.isoformat()}."
        )
        raise ValueError(msg)

    return (
        [f"{effective_month:02d}"],
        [
            f"{day_value:02d}"
            for day_value in range(parsed_start.day, parsed_end.day + 1)
        ],
        parsed_end,
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


def _mrt_partition_is_current(
    base_uri: str,
    partition_path: str,
    *,
    expected_end_date: date,
) -> bool:
    max_date = partition_file_max_timestamp(
        base_uri,
        partition_path,
        "mrt.parquet",
    )
    if max_date is None:
        return False

    is_current = max_date >= expected_end_date
    if not is_current:
        LOGGER.info(
            "MRT partition %s only extends through %s but the expected end date is %s. "
            "Refreshing partition.",
            partition_path,
            max_date.isoformat(),
            expected_end_date.isoformat(),
        )
    return is_current


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

    tmp_output_path = output_path + ".tmp"
    with filesystem.open_output_stream(tmp_output_path) as output_stream:
        pq.write_table(table, output_stream, compression="ZSTD")
    filesystem.move(tmp_output_path, output_path)


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
        help="Year to download (e.g., 2025).",
    )
    parser.add_argument(
        "--month",
        type=int,
        choices=range(1, 13),
        help="Optional month to download within the requested MRT year (1-12).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional inclusive ISO start date override (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional inclusive ISO end date override (YYYY-MM-DD).",
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
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Disable internal tqdm progress bars.",
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
            start_date=args.start_date,
            end_date=args.end_date,
            tile_ids=args.tile_ids,
            tile_shard_index=args.tile_shard_index,
            tile_shard_count=args.tile_shard_count,
            max_workers=args.max_workers,
            disable_tqdm=args.disable_tqdm,
        )
    except Exception as exc:
        LOGGER.exception("MRT pipeline failed for year=%s", args.year)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
