#!/usr/bin/env bash
# Local pipeline relying EXCLUSIVELY on Google ARCO ERA5 -> Direct PET calculation
set -euo pipefail

if [ -f .env ]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

if [ -z "${SUPABASE_DB_URI:-}" ]; then
    echo "WARNING: SUPABASE_DB_URI is not set. Database upload steps will be skipped."
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_LOCAL_MODE=${RUN_LOCAL_MODE:-full}
RUN_LOCAL_SKIP_ERA5_PULL=${RUN_LOCAL_SKIP_ERA5_PULL:-0}

_PIDS=()

_kill_tree() {
    local pid=$1 sig=${2:-TERM} children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    if [ -n "$children" ]; then for c in $children; do _kill_tree "$c" "$sig"; done; fi
    kill -"$sig" "$pid" 2>/dev/null || true
}

_cleanup() {
    local exit_code=$?
    trap - SIGINT SIGTERM ERR EXIT
    if (( exit_code != 0 && ${#_PIDS[@]} > 0 )); then
        for pid in "${_PIDS[@]}"; do _kill_tree "$pid" TERM; done
        sleep 1
        for pid in "${_PIDS[@]}"; do _kill_tree "$pid" KILL; done
    fi
    exit "$exit_code"
}
# Catch all abort signals AND script exits to ensure cleanup fires safely
trap _cleanup SIGINT SIGTERM ERR EXIT

_launch() {
    local max=$1; shift
    "$@" &
    _PIDS+=($!)

    while (( ${#_PIDS[@]} >= max )); do
        # Wait for any job to finish, catch failure immediately
        wait -n || { echo "A background job failed!"; exit 1; }
        
        local active_pids=()
        for pid in "${_PIDS[@]}"; do
            # kill -0 checks if the process is still running
            if kill -0 "$pid" 2>/dev/null; then
                active_pids+=("$pid")
            else
                # Explicitly reap the finished PID to guarantee we capture its exit status
                wait "$pid" || { echo "Shard job $pid failed!"; exit 1; }
            fi
        done
        _PIDS=("${active_pids[@]}")
    done
}

_wait_phase() {
    for pid in "${_PIDS[@]}"; do
        wait "$pid" || { echo "Shard job $pid failed!"; exit 1; }
    done
    _PIDS=()
}

_cpu_count() { getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1; }
CPU_COUNT=$(_cpu_count)
AVAILABLE_CPUS=$(( CPU_COUNT > 1 ? CPU_COUNT - 1 : 1 ))
COMPUTE_JOB_LIMIT=$(( AVAILABLE_CPUS < 8 ? AVAILABLE_CPUS : 8 ))

if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
    ALL_YEARS='2024 2025'
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-168}
    ERA5_TIME_SHARD_COUNT=${ERA5_TIME_SHARD_COUNT:-53}
    echo "Running local pipeline in smoke mode: 2024-2025 ERA5 week only."
else
    ALL_YEARS='2024 2025'
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-168}
    MONTHS_FILTER="--months 5 6 7 8 9"
    echo "Running local pipeline in full mode: $ALL_YEARS (May-Sept)."
fi

export ALL_YEARS

echo "====== Step 1: Compute year ranges + setup locations ======"
mkdir -p output_tiles pet_data_csv analytics_data_csv
if [[ "$RUN_LOCAL_SKIP_ERA5_PULL" != "1" ]]; then
    for YEAR in $ALL_YEARS; do rm -rf "pet_data_csv/year=$YEAR"; done
fi
rm -rf analytics_data_csv/shard_count=*

python cities.py

CITY_COUNT=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' cities.csv)
DEFAULT_ERA5_CITY_SHARD_COUNT=$(( CITY_COUNT < 2 ? CITY_COUNT : 2 ))
ERA5_CITY_SHARD_COUNT=${RUN_LOCAL_ERA5_CITY_SHARD_COUNT:-$DEFAULT_ERA5_CITY_SHARD_COUNT}
if (( ERA5_CITY_SHARD_COUNT < 1 )); then ERA5_CITY_SHARD_COUNT=1; fi
if (( ERA5_CITY_SHARD_COUNT > CITY_COUNT )); then ERA5_CITY_SHARD_COUNT=$CITY_COUNT; fi

if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
    ERA5_JOB_LIMIT=${RUN_LOCAL_ERA5_JOB_LIMIT:-$ERA5_CITY_SHARD_COUNT}
    ERA5_CONCURRENCY_PROFILE=${RUN_LOCAL_ERA5_CONCURRENCY_PROFILE:-aggressive}
    ERA5_BATCH_WORKERS=${RUN_LOCAL_ERA5_BATCH_WORKERS:-$(( AVAILABLE_CPUS / ERA5_CITY_SHARD_COUNT ))}
else
    DEFAULT_FULL_ERA5_JOB_LIMIT=$(( ERA5_CITY_SHARD_COUNT * 2 ))
    if (( DEFAULT_FULL_ERA5_JOB_LIMIT > AVAILABLE_CPUS )); then
        DEFAULT_FULL_ERA5_JOB_LIMIT=$AVAILABLE_CPUS
    fi
    ERA5_JOB_LIMIT=${RUN_LOCAL_ERA5_JOB_LIMIT:-$DEFAULT_FULL_ERA5_JOB_LIMIT}
    ERA5_CONCURRENCY_PROFILE=${RUN_LOCAL_ERA5_CONCURRENCY_PROFILE:-balanced}
    ERA5_BATCH_WORKERS=${RUN_LOCAL_ERA5_BATCH_WORKERS:-1}
fi

ANALYTICS_SHARD_COUNT=$(( CITY_COUNT < COMPUTE_JOB_LIMIT ? CITY_COUNT : COMPUTE_JOB_LIMIT ))
(( ERA5_BATCH_WORKERS < 1 )) && ERA5_BATCH_WORKERS=1
if [[ "$RUN_LOCAL_MODE" == "smoke" && "$ERA5_BATCH_WORKERS" -gt 2 ]]; then
    ERA5_BATCH_WORKERS=2
fi

echo "====== Step 2: Compute ERA5 + PET (parallel, CPU-aware) ======"
if [[ "$RUN_LOCAL_SKIP_ERA5_PULL" != "1" ]]; then
    for YEAR in $ALL_YEARS; do
        for (( CITY_SHARD=0; CITY_SHARD<ERA5_CITY_SHARD_COUNT; CITY_SHARD++ )); do
            ERA5_ARGS=(--year "$YEAR" --city-shard-count "$ERA5_CITY_SHARD_COUNT" --city-shard-index "$CITY_SHARD" --max-workers "$ERA5_BATCH_WORKERS" --concurrency-profile "$ERA5_CONCURRENCY_PROFILE" --out-dir .)
            
            if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
                ERA5_ARGS+=(--batch-hours "$ERA5_BATCH_HOURS" --time-shard-index 17 --time-shard-count "$ERA5_TIME_SHARD_COUNT")
            else
                ERA5_ARGS+=(--batch-hours "$ERA5_BATCH_HOURS" --time-shard-index 0 --time-shard-count 1)
                # Apply the months filter if it is set
                if [ -n "${MONTHS_FILTER:-}" ]; then
                    ERA5_ARGS+=($MONTHS_FILTER)
                fi
            fi
            
            _launch "$ERA5_JOB_LIMIT" python google_era5.py "${ERA5_ARGS[@]}"
        done
    done
    _wait_phase "pull-google-era5-pet"
fi

echo "====== Step 3: Generate Analytics (parallel, CPU-aware) ======"
for (( ANA_SHARD=0; ANA_SHARD<ANALYTICS_SHARD_COUNT; ANA_SHARD++ )); do
    _launch "$COMPUTE_JOB_LIMIT" python generate_analytics.py --shard-index "$ANA_SHARD" --shard-count "$ANALYTICS_SHARD_COUNT"
done
_wait_phase "generate-analytics"

echo "====== Step 3.5: Save combined pet.csv reference before upload ======"
python - <<'PY'
import pandas as pd
from pathlib import Path
pet_root = Path("pet_data_csv")
all_files = sorted(pet_root.rglob("pet_batch_*.parquet"))
if all_files:
    dfs = [pd.read_parquet(f, columns=["location_id", "date", "pet"]) for f in all_files]
    combined = pd.concat(dfs, ignore_index=True).sort_values(["location_id", "date"])
    combined.to_csv("pet.csv", index=False)
    print(f"Saved {len(combined)} rows to pet.csv")
else:
    print("No pet batch parquet files found; pet.csv not written.")
PY

echo "====== Step 4: Prepare DB load ======"
python - <<'PY'
import os, sys, psycopg2
from pathlib import Path
db_uri = os.environ.get("SUPABASE_DB_URI")
if not db_uri: sys.exit(0)
conn = psycopg2.connect(db_uri)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(Path("drop_views.sql").read_text(encoding="utf-8"))
    cur.execute(Path("create_tables.sql").read_text(encoding="utf-8"))
    cur.execute("TRUNCATE TABLE locations, pet, pet_percentiles, pet_forecast, pet_change CASCADE")
PY

python load.py --append-only --skip-drop-views --skip-create-views --skip-table pet --skip-table pet_percentiles --skip-table pet_forecast --skip-table pet_change

echo "====== Step 5: Load shards to DB (parallel, CPU-aware) ======"
for (( LOAD_SHARD=0; LOAD_SHARD<ANALYTICS_SHARD_COUNT; LOAD_SHARD++ )); do
    _launch "$COMPUTE_JOB_LIMIT" python load.py --append-only --skip-drop-views --skip-create-views --skip-table locations --analytics-shard-count "$ANALYTICS_SHARD_COUNT" --load-shard-index "$LOAD_SHARD" --load-shard-count "$ANALYTICS_SHARD_COUNT"
done
_wait_phase "load-to-db"

echo "====== Step 6: Recreate views ======"
python load.py --append-only --skip-drop-views --skip-table locations --skip-table pet --skip-table pet_percentiles --skip-table pet_forecast --skip-table pet_change
echo "====== Pipeline complete! ======"