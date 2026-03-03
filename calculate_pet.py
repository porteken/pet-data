"""Compute daily maximum PET per location from combined weather data."""

from __future__ import annotations

import logging
import multiprocessing
import time
from importlib import import_module
from typing import Any, Protocol, TypeAlias, cast

DataFrame: TypeAlias = Any
LOGGER = logging.getLogger(__name__)
CHUNK_SIZE = 1000
PET_ROUNDING_FACTOR = 2.0


class _PetCorrectedCallable(Protocol):
    def __call__(
        self,
        tair: float,
        t_mrt: float,
        v_air: float,
        rh: float,
        *,
        icl: float,
    ) -> float: ...


pd: Any = cast("Any", import_module("pandas"))
tqdm_module: Any = cast("Any", import_module("tqdm"))
tqdm_progress: Any = tqdm_module.tqdm

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

    pet_results = [
        pet_corrected(float(t), float(mrt), float(v_air), float(rh), icl=0.5)
        for t, mrt, v_air, rh in zip(
            t_vals,
            mrt_vals,
            v_vals,
            rh_vals,
            strict=False,
        )
    ]

    df_result = df_chunk.copy()
    df_result["pet"] = pet_results
    df_result["pet"] = (
        df_result["pet"] * PET_ROUNDING_FACTOR
    ).round() / PET_ROUNDING_FACTOR
    return df_result


def main() -> None:
    """Run the PET pipeline and save `pet.csv`."""
    LOGGER.info("1. Loading and preprocessing data.")
    df: DataFrame = pd.read_parquet("combined_data.parquet")

    cols_to_keep = ["location_id", "time", "v", "t", "rh", "mrt"]
    df = df[cols_to_keep]

    # Keep rows with valid humidity and positive wind speed.
    df = df[(df["rh"] >= 1) & (df["v"] > 0)].copy()

    cols_to_round = ["v", "t", "rh", "mrt"]
    for col in cols_to_round:
        df[col] = (df[col] * PET_ROUNDING_FACTOR).round() / PET_ROUNDING_FACTOR

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
        "3. Starting parallel computation for %s unique combinations.",
        row_count,
    )
    start_time = time.time()
    n_cores = max(1, multiprocessing.cpu_count() - 1)

    with multiprocessing.Pool(processes=n_cores) as pool:
        results: list[DataFrame] = list(
            tqdm_progress(
                pool.imap(compute_pet_chunk, chunks),
                total=len(chunks),
            ),
        )

    if results:
        df_pet_unique: DataFrame = pd.concat(results, ignore_index=True)
    else:
        df_pet_unique = df_distinct.copy()
        df_pet_unique["pet"] = pd.Series(dtype=float)

    elapsed = time.time() - start_time
    LOGGER.info("Computation finished in %.2f seconds.", elapsed)

    LOGGER.info("4. Joining calculated PET back to the main dataset.")
    df_joined: DataFrame = df.merge(
        df_pet_unique,
        on=["v", "t", "rh", "mrt"],
        how="inner",
    )

    LOGGER.info("5. Aggregating daily max PET per location.")
    df_joined["date"] = pd.to_datetime(df_joined["time"]).dt.date
    df_final: DataFrame = (
        df_joined.groupby(["location_id", "date"])["pet"].max().reset_index()
    )
    df_final["id"] = df_final.index + 1

    LOGGER.info("6. Saving final output.")
    df_final.to_csv("pet.csv", index=False)
    LOGGER.info("Pipeline complete.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    multiprocessing.freeze_support()
    main()
