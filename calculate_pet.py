"""Compute daily maximum PET for discovered combined parquet shards."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import time
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol, TypeAlias, cast

from shards import ShardKey, discover_parquet_shards, read_parquet_files, select_shards

DataFrame: TypeAlias = Any
LOGGER = logging.getLogger(__name__)
CHUNK_SIZE = 50_000
PET_ROUNDING_FACTOR = 2.0


class _PetCorrectedCallable(Protocol):
    def __call__(
        self,
        tair: object,
        t_mrt: object,
        v_air: object,
        rh: object,
        *,
        icl: float,
    ) -> Any: ...


pd: Any = cast("Any", import_module("pandas"))

try:
    pet_module: Any = import_module("pet_corrected")
    pet_corrected: _PetCorrectedCallable = cast(
        "_PetCorrectedCallable",
        pet_module.pet_corrected,
    )
except (ImportError, AttributeError) as error:
    LOGGER.exception("Could not import pet_corrected from pet_corrected.py.")
    raise SystemExit(1) from error


def compute_pet_chunk(df_chunk: DataFrame) -> DataFrame:
    """Compute PET values for one chunk of distinct weather combinations.

    The legacy R pipeline evaluates PET row-by-row for each distinct rounded
    weather combination. Doing one large vectorized solver call changes branch
    decisions inside the translated thermodynamic model and can shift the final
    daily maxima. We therefore preserve scalar semantics here.
    """
    pet_results = [
        _compute_pet_value_scalar(tair=t, t_mrt=mrt, v_air=v, rh=rh)
        for t, mrt, v, rh in df_chunk[["t", "mrt", "v", "rh"]].itertuples(
            index=False,
            name=None,
        )
    ]

    df_result = df_chunk.copy()
    df_result["pet"] = pet_results
    df_result["pet"] = (
        df_result["pet"] * PET_ROUNDING_FACTOR
    ).round() / PET_ROUNDING_FACTOR
    return df_result


def _compute_pet_value_scalar(
    *,
    tair: float,
    t_mrt: float,
    v_air: float,
    rh: float,
) -> float:
    result = pet_corrected(
        float(tair),
        float(t_mrt),
        float(v_air),
        float(rh),
        icl=0.5,
    )
    if result is None:
        msg = (
            "PET solver returned no value for "
            f"t={tair}, mrt={t_mrt}, v={v_air}, rh={rh}."
        )
        raise RuntimeError(msg)
    return float(result)


def calculate_pet_frame(df: DataFrame) -> DataFrame:
    """Run PET calculation for one combined shard DataFrame."""
    df = df[["location_id", "time", "v", "t", "rh", "mrt"]].copy()
    df = df[(df["rh"] >= 1) & (df["v"] > 0)].copy()

    for column_name in ["v", "t", "rh", "mrt"]:
        df[column_name] = (
            df[column_name] * PET_ROUNDING_FACTOR
        ).round() / PET_ROUNDING_FACTOR

    df_distinct: DataFrame = (
        df[["v", "t", "rh", "mrt"]].drop_duplicates().reset_index(drop=True)
    )
    row_count = int(df_distinct.shape[0])
    chunks: list[DataFrame] = [
        df_distinct.iloc[index : index + CHUNK_SIZE]
        for index in range(0, row_count, CHUNK_SIZE)
    ]

    LOGGER.info("Computing PET for %s unique combinations.", row_count)
    start_time = time.time()

    results: list[DataFrame] = list(map(compute_pet_chunk, chunks))

    if results:
        df_pet_unique: DataFrame = pd.concat(results, ignore_index=True)
    else:
        df_pet_unique = df_distinct.copy()
        df_pet_unique["pet"] = pd.Series(dtype=float)

    LOGGER.info(
        "PET core computation finished in %.2f seconds.",
        time.time() - start_time,
    )

    df_joined: DataFrame = df.merge(
        df_pet_unique,
        on=["v", "t", "rh", "mrt"],
        how="inner",
    )
    df_joined["date"] = pd.to_datetime(df_joined["time"]).dt.date
    return df_joined.groupby(["location_id", "date"])["pet"].max().reset_index()


def _pet_output_path(output_dir: Path, shard_key: ShardKey) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_name = f"pet_batch_{shard_key.tile_id:03d}.parquet"
    return output_dir / batch_name


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PET for discovered combined parquet shards.",
    )
    parser.add_argument("--combined-root", default="combined_data_parquet")
    parser.add_argument("--out-dir", default="pet_data_csv")
    parser.add_argument("--year", type=int)
    parser.add_argument("--tile-id", dest="tile_ids", action="append", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    """Compute and save PET parquet outputs for the selected combined shards."""
    args = _parse_args()
    shard_mapping = discover_parquet_shards(args.combined_root)
    selected_shards = select_shards(
        shard_mapping,
        year=args.year,
        tile_ids=args.tile_ids,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )

    if not selected_shards:
        msg = "No combined shards found for PET calculation."
        raise RuntimeError(msg)

    for shard_key in selected_shards:
        LOGGER.info("Processing PET for shard %s.", shard_key.label)
        combined_df = read_parquet_files(
            args.combined_root,
            shard_mapping[shard_key],
            columns=["location_id", "time", "v", "t", "rh", "mrt"],
        )
        if combined_df.empty:
            msg = f"Shard {shard_key.label}: combined parquet is empty."
            raise RuntimeError(msg)

        pet_df = calculate_pet_frame(combined_df)
        float_cols = pet_df.select_dtypes(include=["float64"]).columns
        if len(float_cols) > 0:
            pet_df[float_cols] = pet_df[float_cols].astype("float32")
        output_dir = Path(args.out_dir) / shard_key.partition_path
        output_path = _pet_output_path(output_dir, shard_key)
        tmp_output_path = output_path.with_suffix(".tmp")
        pet_df.to_parquet(tmp_output_path, index=False, compression="snappy")
        tmp_output_path.rename(output_path)
        LOGGER.info("Saved %s rows to %s.", len(pet_df), output_path)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAX_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    multiprocessing.freeze_support()
    main()
