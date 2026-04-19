"""Validation script for PET math implementation."""

import sys
from typing import Any, cast

import numpy as np
import pandas as pd

sys.path.append("/home/kenneth-porter/pet-data")

from pet_corrected import pet_corrected

TOLERANCE = 0.01


def main() -> None:
    """Load sample data and compare Python PET implementation with reference values."""
    # Load rows to test the math
    df = pd.read_parquet(
        "/home/kenneth-porter/pet_files/pet.parquet",
        columns=["time", "v", "t", "rh", "mrt", "pet"],
    ).head(10000)

    t = df["t"].to_numpy()
    mrt = df["mrt"].to_numpy()
    v = df["v"].to_numpy()
    rh = df["rh"].to_numpy()

    pet_py_05 = cast("Any", pet_corrected(t, mrt, v, rh, icl=0.5))

    pet_py_09 = cast("Any", pet_corrected(t, mrt, v, rh, icl=0.9))

    df["pet_py_05"] = np.round(pet_py_05 * 2) / 2
    df["pet_py_09"] = np.round(pet_py_09 * 2) / 2

    mismatches_05 = np.abs(df["pet_py_05"] - df["pet"]) > TOLERANCE
    mismatches_09 = np.abs(df["pet_py_09"] - df["pet"]) > TOLERANCE

    if mismatches_05.sum() == 0 or mismatches_09.sum() == 0:
        pass
    else:
        pass


if __name__ == "__main__":
    main()
