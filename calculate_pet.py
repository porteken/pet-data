"""Compute daily maximum PET per location from a combined weather shard."""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import time
from importlib import import_module
from typing import Any, Protocol, TypeAlias, cast

DataFrame: TypeAlias = Any
LOGGER = logging.getLogger(__name__)
CHUNK_SIZE = 10000
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
    ) -> object: ...


pd: Any = cast("Any", import_module("pandas"))
tqdm_module: Any = cast("Any", import_module("tqdm"))
tqdm_progress = tqdm_module.tqdm

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
    """Compute PET values for one chunk of distinct weather combinations."""
    t_vals = df_chunk["t"].to_numpy()
    mrt_vals = df_chunk["mrt"].to_numpy()
    v_vals = df_chunk["v"].to_numpy()
    rh_vals = df_chunk["rh"].to_numpy()

    pet_results = pet_corrected(t_vals, mrt_vals, v_vals, rh_vals, icl=0.5)

    df_result = df_chunk.copy()
    df_result["pet"] = pet_results
    df_result["pet"] = (
        df_result["pet"] * PET_ROUNDING_FACTOR
    ).round() / PET_ROUNDING_FACTOR
    return df_result


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for PET calculation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", required=True, type=str)
    parser.add_argument("--month", required=True, type=str)
    return parser.parse_args()


def main() -> None:
    """Run the PET pipeline for a specific year and month shard."""
    args = _parse_args()

    input_file = f"combined_data_{args.year}_{args.month}.parquet"
    LOGGER.info("1. Loading and preprocessing data for %s-%s.", args.year, args.month)

    try:
        df: DataFrame = pd.read_parquet(input_file)
    except FileNotFoundError:
        LOGGER.warning(
            "File %s not found. Skipping PET calc for %s-%s.",
            input_file,
            args.year,
            args.month,
        )
        return

    cols_to_keep = ["location_id", "time", "v", "t", "rh", "mrt"]
    df = df[cols_to_keep]
    df = df[(df["rh"] >= 1) & (df["v"] > 0)].copy()

    cols_to_round = ["v", "t", "rh", "mrt"]
    for column_name in cols_to_round:
        df[column_name] = (
            df[column_name] * PET_ROUNDING_FACTOR
        ).round() / PET_ROUNDING_FACTOR

    LOGGER.info("2. Extracting distinct weather combinations.")
    df_distinct: DataFrame = (
        df[["v", "t", "rh", "mrt"]].drop_duplicates().reset_index(drop=True)
    )
    row_count = int(df_distinct.shape[0])

    chunks: list[DataFrame] = [
        df_distinct.iloc[index : index + CHUNK_SIZE]
        for index in range(0, row_count, CHUNK_SIZE)
    ]

    LOGGER.info(
        "3. Parallel computation for %s unique combinations.",
        row_count,
    )
    start_time = time.time()
    n_cores = max(1, multiprocessing.cpu_count() - 1)

    with multiprocessing.Pool(processes=n_cores) as pool:
        results: list[DataFrame] = list(
            tqdm_progress(
                pool.imap_unordered(compute_pet_chunk, chunks),
                total=len(chunks),
            ),
        )

    if results:
        df_pet_unique: DataFrame = pd.concat(results, ignore_index=True)
    else:
        df_pet_unique = df_distinct.copy()
        df_pet_unique["pet"] = pd.Series(dtype=float)

    elapsed = time.time() - start_time
    LOGGER.info("Finished PET computation in %.2f seconds.", elapsed)

    LOGGER.info("4. Joining calculated PET back and finding daily max.")
    df_joined: DataFrame = df.merge(
        df_pet_unique,
        on=["v", "t", "rh", "mrt"],
        how="inner",
    )
    df_joined["date"] = pd.to_datetime(df_joined["time"]).dt.date

    df_final: DataFrame = (
        df_joined.groupby(["location_id", "date"])["pet"].max().reset_index()
    )

    output_csv = f"pet_{args.year}_{args.month}.csv"
    df_final.to_csv(output_csv, index=False)
    LOGGER.info("Saved %s.", output_csv)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    multiprocessing.freeze_support()
    main()
