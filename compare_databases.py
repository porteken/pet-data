"""Comparison script for PET data across different database environments."""

import os
import sys

import pandas as pd
import psycopg2
from dotenv import load_dotenv

load_dotenv()

uri1 = os.environ.get("SUPABASE_DB_URI")
uri2 = os.environ.get("SUPABASE_DB_URI_PRD")

query = """
SELECT location_id, date, pet
FROM pet
WHERE extract(month from date) BETWEEN 5 AND 9
  AND extract(year from date) IN (2024, 2025)
"""

with psycopg2.connect(uri1, connect_timeout=10) as conn1:
    df1 = pd.read_sql_query(query, conn1)

with psycopg2.connect(uri2, connect_timeout=10) as conn2:
    df2 = pd.read_sql_query(query, conn2)


if df1.empty and df2.empty:
    sys.exit(0)

# Merge datasets to compare
df_merged = df1.merge(
    df2,
    on=["location_id", "date"],
    how="outer",
    suffixes=("_dev", "_prd"),
    validate="many_to_many",
)

# Find differences
missing_in_prd = df_merged[df_merged["pet_prd"].isna()]
missing_in_dev = df_merged[df_merged["pet_dev"].isna()]
diff_pet = df_merged[
    df_merged["pet_dev"].notna()
    & df_merged["pet_prd"].notna()
    & (round(df_merged["pet_dev"], 4) != round(df_merged["pet_prd"], 4))
]


df_merged["diff"] = (df_merged["pet_dev"] - df_merged["pet_prd"]).abs()

# Check the MAE for just the 7000 rows we expect to be updated
# Since we only updated 1 week in May, let's filter for dates in May
df_may = df_merged[df_merged["date"].astype(str).str.contains("-05-")]
