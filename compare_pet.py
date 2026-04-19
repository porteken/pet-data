"""Compare Python and R implementations of PET calculation."""

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from pet_corrected import pet_corrected

# Load inputs
df = pd.read_csv("mismatch_inputs.csv").sample(n=min(100, 2448), random_state=42)


# 1. Compute Python PET
def run_py_pet(row: pd.Series) -> float:
    """Run Python PET calculation for a given row of data."""
    try:
        return pet_corrected(
            row["t"] - 273.15,
            row["v"],
            row["rh"],
            row["mrt"] - 273.15,
            icl=0.5,
        )
    except Exception:  # noqa: BLE001
        return np.nan


df["pet_py"] = df.apply(run_py_pet, axis=1)

# 2. Compute R PET
df[["t", "v", "rh", "mrt"]].to_csv("r_inputs.csv", index=False)

r_script = """
C2K <- function(K) K - 273.15
SVP.Murray <- function(T) 6.1078 * exp(17.269 * T / (T + 237.3))
source("/home/kenneth-porter/pet_files/pet_corrected.R")
inputs <- read.csv("r_inputs.csv")
results <- apply(inputs, 1, function(row) {
  PETcorrected(row["t"] - 273.15, row["v"], row["rh"], row["mrt"] - 273.15, icl=0.5)
})
write.csv(results, "r_results.csv", row.names=FALSE)
"""
with Path("shim_pet.R").open("w") as f:
    f.write(r_script)

subprocess.run(["/usr/bin/Rscript", "shim_pet.R"], check=True)
df["pet_r"] = pd.read_csv("r_results.csv")["x"].to_numpy()

# 3. Compare
df["pet_py_round"] = (df["pet_py"] * 2).round() / 2
df["pet_r_round"] = (df["pet_r"] * 2).round() / 2

df["diff"] = (df["pet_py_round"] - df["pet_r_round"]).abs()
mismatches = df[df["diff"] > 0]

if len(mismatches) > 0:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
else:
    pass

# Also show raw diffs for the first 10
