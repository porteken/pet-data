#!/usr/bin/env bash
# Local hybrid PET pipeline: Google ARCO weather + CDS MRT -> combined hourly -> PET -> analytics
set -euo pipefail

if [ -f .env ]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

PYTHON_BIN=${PYTHON_BIN:-python}

if [ -z "${SUPABASE_DB_URI:-}" ]; then
    echo "WARNING: SUPABASE_DB_URI is not set. Database upload steps will be skipped."
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PET_ANALYTICS_USE_DASK=${PET_ANALYTICS_USE_DASK:-1}

RUN_LOCAL_MODE=${RUN_LOCAL_MODE:-smoke}
RUN_LOCAL_SKIP_WEATHER_PULL=${RUN_LOCAL_SKIP_WEATHER_PULL:-0}
RUN_LOCAL_SKIP_MRT_PULL=${RUN_LOCAL_SKIP_MRT_PULL:-0}
RUN_LOCAL_SKIP_COMBINE=${RUN_LOCAL_SKIP_COMBINE:-0}
RUN_LOCAL_SKIP_PET=${RUN_LOCAL_SKIP_PET:-0}
RUN_LOCAL_SKIP_ANALYTICS=${RUN_LOCAL_SKIP_ANALYTICS:-0}
RUN_LOCAL_SKIP_DB_LOAD=${RUN_LOCAL_SKIP_DB_LOAD:-0}
RUN_LOCAL_YEARS=${RUN_LOCAL_YEARS:-"2024 2025"}

_PIDS=()

_kill_tree() {
    local pid=$1 sig=${2:-TERM} children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    if [ -n "$children" ]; then
        for c in $children; do
            _kill_tree "$c" "$sig"
        done
    fi
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
trap _cleanup SIGINT SIGTERM ERR EXIT

_launch() {
    local max=$1; shift
    "$@" &
    _PIDS+=($!)

    while (( ${#_PIDS[@]} >= max )); do
        wait -n || { echo "A background job failed!"; exit 1; }

        local active_pids=()
        for pid in "${_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                active_pids+=("$pid")
            else
                wait "$pid" || { echo "Shard job $pid failed!"; exit 1; }
            fi
        done
        _PIDS=("${active_pids[@]}")
    done
}

_wait_phase() {
    local label=${1:-phase}
    for pid in "${_PIDS[@]}"; do
        wait "$pid" || { echo "Phase '$label' failed in job $pid!"; exit 1; }
    done
    _PIDS=()
}

_cpu_count() { getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1; }
CPU_COUNT=$(_cpu_count)
AVAILABLE_CPUS=$(( CPU_COUNT > 1 ? CPU_COUNT - 1 : 1 ))
COMPUTE_JOB_LIMIT=$(( AVAILABLE_CPUS < 8 ? AVAILABLE_CPUS : 8 ))

ALL_YEARS="$RUN_LOCAL_YEARS"
SMOKE_MONTH=5
SMOKE_START_DAY=1
SMOKE_END_DAY=7
SMOKE_MONTH_PAD=$(printf '%02d' "$SMOKE_MONTH")

if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
    PIPELINE_MONTHS=(5)
    WEATHER_BATCH_HOURS=${RUN_LOCAL_WEATHER_BATCH_HOURS:-168}
    echo "Running local pipeline in smoke mode: ${ALL_YEARS} first week of May."
else
    PIPELINE_MONTHS=(5 6 7 8 9)
    WEATHER_BATCH_HOURS=${RUN_LOCAL_WEATHER_BATCH_HOURS:-720}
    echo "Running local pipeline in full mode: ${ALL_YEARS} (May-Sept)."
fi

echo "====== Step 1: Setup locations, tiles, and clean outputs ======"
rm -rf output_tiles
if [[ "$RUN_LOCAL_SKIP_ANALYTICS" != "1" ]]; then rm -rf analytics_data_csv; fi
if [[ "$RUN_LOCAL_SKIP_COMBINE" != "1" ]]; then rm -rf combined_data_parquet; fi
if [[ "$RUN_LOCAL_SKIP_PET" != "1" ]]; then rm -rf pet_data_csv; fi
mkdir -p output_tiles analytics_data_csv combined_data_parquet pet_data_csv
if [[ "$RUN_LOCAL_SKIP_WEATHER_PULL" != "1" ]]; then rm -rf weather_data_parquet; fi
if [[ "$RUN_LOCAL_SKIP_MRT_PULL" != "1" ]]; then rm -rf utci_data_parquet; fi
mkdir -p weather_data_parquet utci_data_parquet

"$PYTHON_BIN" cities.py

CITY_COUNT=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' cities.csv)
TILE_COUNT=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' output_tiles/tile_boxes.csv)

DEFAULT_MRT_TILE_SHARD_COUNT=$(( TILE_COUNT < 3 ? TILE_COUNT : 3 ))
MRT_TILE_SHARD_COUNT=${RUN_LOCAL_MRT_TILE_SHARD_COUNT:-$DEFAULT_MRT_TILE_SHARD_COUNT}
(( MRT_TILE_SHARD_COUNT < 1 )) && MRT_TILE_SHARD_COUNT=1
(( MRT_TILE_SHARD_COUNT > TILE_COUNT )) && MRT_TILE_SHARD_COUNT=$TILE_COUNT

if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
    DEFAULT_WEATHER_TILE_SHARD_COUNT=$MRT_TILE_SHARD_COUNT
else
    DEFAULT_WEATHER_TILE_SHARD_COUNT=$(( TILE_COUNT < 4 ? TILE_COUNT : 4 ))
fi
WEATHER_TILE_SHARD_COUNT=${RUN_LOCAL_WEATHER_TILE_SHARD_COUNT:-$DEFAULT_WEATHER_TILE_SHARD_COUNT}
(( WEATHER_TILE_SHARD_COUNT < 1 )) && WEATHER_TILE_SHARD_COUNT=1
(( WEATHER_TILE_SHARD_COUNT > TILE_COUNT )) && WEATHER_TILE_SHARD_COUNT=$TILE_COUNT

if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
    DEFAULT_WEATHER_JOB_LIMIT=$WEATHER_TILE_SHARD_COUNT
else
    DEFAULT_WEATHER_JOB_LIMIT=$(( WEATHER_TILE_SHARD_COUNT < 2 ? WEATHER_TILE_SHARD_COUNT : 2 ))
fi
WEATHER_JOB_LIMIT=${RUN_LOCAL_WEATHER_JOB_LIMIT:-$DEFAULT_WEATHER_JOB_LIMIT}
MRT_JOB_LIMIT=${RUN_LOCAL_MRT_JOB_LIMIT:-$(( MRT_TILE_SHARD_COUNT < 2 ? MRT_TILE_SHARD_COUNT : 2 ))}
COMBINE_JOB_LIMIT=${RUN_LOCAL_COMBINE_JOB_LIMIT:-$MRT_TILE_SHARD_COUNT}
PET_JOB_LIMIT=${RUN_LOCAL_PET_JOB_LIMIT:-$MRT_TILE_SHARD_COUNT}
ANALYTICS_SHARD_COUNT=$(( CITY_COUNT < COMPUTE_JOB_LIMIT ? CITY_COUNT : COMPUTE_JOB_LIMIT ))

if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
    DEFAULT_WEATHER_TILE_WORKERS=2
    DEFAULT_WEATHER_CONCURRENCY_PROFILE=balanced
else
    DEFAULT_WEATHER_TILE_WORKERS=1
    DEFAULT_WEATHER_CONCURRENCY_PROFILE=conservative
fi
WEATHER_TILE_WORKERS=${RUN_LOCAL_WEATHER_TILE_WORKERS:-$DEFAULT_WEATHER_TILE_WORKERS}
MRT_BATCH_WORKERS=${RUN_LOCAL_MRT_BATCH_WORKERS:-2}
WEATHER_CONCURRENCY_PROFILE=${RUN_LOCAL_WEATHER_CONCURRENCY_PROFILE:-$DEFAULT_WEATHER_CONCURRENCY_PROFILE}
export MRT_CDS_REQUEST_CONCURRENCY=${MRT_CDS_REQUEST_CONCURRENCY:-4}

echo "Weather tile shards: $WEATHER_TILE_SHARD_COUNT | weather job limit: $WEATHER_JOB_LIMIT | weather tile workers: $WEATHER_TILE_WORKERS | weather profile: $WEATHER_CONCURRENCY_PROFILE"
echo "MRT tile shards: $MRT_TILE_SHARD_COUNT | MRT job limit: $MRT_JOB_LIMIT | analytics shards: $ANALYTICS_SHARD_COUNT"

echo "====== Step 2: Pull Google weather by tile bounding boxes ======"
if [[ "$RUN_LOCAL_SKIP_WEATHER_PULL" != "1" ]]; then
    for YEAR in $ALL_YEARS; do
        if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
            for (( TILE_SHARD=0; TILE_SHARD<WEATHER_TILE_SHARD_COUNT; TILE_SHARD++ )); do
                _launch "$WEATHER_JOB_LIMIT" "$PYTHON_BIN" pull_weather_tiles.py \
                    --year "$YEAR" \
                    --month "$SMOKE_MONTH" \
                    --start-date "${YEAR}-${SMOKE_MONTH_PAD}-$(printf '%02d' "$SMOKE_START_DAY")" \
                    --end-date "${YEAR}-${SMOKE_MONTH_PAD}-$(printf '%02d' "$SMOKE_END_DAY")" \
                    --tile-shard-index "$TILE_SHARD" \
                    --tile-shard-count "$WEATHER_TILE_SHARD_COUNT" \
                    --max-workers "$WEATHER_TILE_WORKERS" \
                    --concurrency-profile "$WEATHER_CONCURRENCY_PROFILE" \
                    --out-dir .
            done
        else
            for MONTH in "${PIPELINE_MONTHS[@]}"; do
                for (( TILE_SHARD=0; TILE_SHARD<WEATHER_TILE_SHARD_COUNT; TILE_SHARD++ )); do
                    _launch "$WEATHER_JOB_LIMIT" "$PYTHON_BIN" pull_weather_tiles.py \
                        --year "$YEAR" \
                        --month "$MONTH" \
                        --tile-shard-index "$TILE_SHARD" \
                        --tile-shard-count "$WEATHER_TILE_SHARD_COUNT" \
                        --max-workers "$WEATHER_TILE_WORKERS" \
                        --concurrency-profile "$WEATHER_CONCURRENCY_PROFILE" \
                        --out-dir .
                done
            done
        fi
    done
    _wait_phase "pull-weather"
fi

echo "====== Step 3: Pull CDS MRT using tile bounding boxes ======"
if [[ "$RUN_LOCAL_SKIP_MRT_PULL" != "1" ]]; then
    for YEAR in $ALL_YEARS; do
        if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
            for (( TILE_SHARD=0; TILE_SHARD<MRT_TILE_SHARD_COUNT; TILE_SHARD++ )); do
                _launch "$MRT_JOB_LIMIT" "$PYTHON_BIN" pull_mrt.py \
                    --year "$YEAR" \
                    --month "$SMOKE_MONTH" \
                    --start-date "${YEAR}-${SMOKE_MONTH_PAD}-$(printf '%02d' "$SMOKE_START_DAY")" \
                    --end-date "${YEAR}-${SMOKE_MONTH_PAD}-$(printf '%02d' "$SMOKE_END_DAY")" \
                    --tile-shard-index "$TILE_SHARD" \
                    --tile-shard-count "$MRT_TILE_SHARD_COUNT" \
                    --max-workers "$MRT_BATCH_WORKERS" \
                    --out-dir .
            done
        else
            for MONTH in "${PIPELINE_MONTHS[@]}"; do
                for (( TILE_SHARD=0; TILE_SHARD<MRT_TILE_SHARD_COUNT; TILE_SHARD++ )); do
                    _launch "$MRT_JOB_LIMIT" "$PYTHON_BIN" pull_mrt.py \
                        --year "$YEAR" \
                        --month "$MONTH" \
                        --tile-shard-index "$TILE_SHARD" \
                        --tile-shard-count "$MRT_TILE_SHARD_COUNT" \
                        --max-workers "$MRT_BATCH_WORKERS" \
                        --out-dir .
                done
            done
        fi
    done
    _wait_phase "pull-mrt"
fi

echo "====== Step 4: Combine hourly weather + MRT shards ======"
if [[ "$RUN_LOCAL_SKIP_COMBINE" != "1" ]]; then
    for YEAR in $ALL_YEARS; do
        for (( TILE_SHARD=0; TILE_SHARD<MRT_TILE_SHARD_COUNT; TILE_SHARD++ )); do
            _launch "$COMBINE_JOB_LIMIT" "$PYTHON_BIN" combine.py --year "$YEAR" --shard-index "$TILE_SHARD" --shard-count "$MRT_TILE_SHARD_COUNT"
        done
    done
    _wait_phase "combine-hourly"
fi

echo "====== Step 5: Calculate PET from combined hourly shards ======"
if [[ "$RUN_LOCAL_SKIP_PET" != "1" ]]; then
    for YEAR in $ALL_YEARS; do
        for (( TILE_SHARD=0; TILE_SHARD<MRT_TILE_SHARD_COUNT; TILE_SHARD++ )); do
            _launch "$PET_JOB_LIMIT" "$PYTHON_BIN" calculate_pet.py --year "$YEAR" --shard-index "$TILE_SHARD" --shard-count "$MRT_TILE_SHARD_COUNT"
        done
    done
    _wait_phase "calculate-pet"
fi

echo "====== Step 6: Generate Analytics (parallel, CPU-aware) ======"
if [[ "$RUN_LOCAL_SKIP_ANALYTICS" != "1" ]]; then
    for (( ANA_SHARD=0; ANA_SHARD<ANALYTICS_SHARD_COUNT; ANA_SHARD++ )); do
        _launch "$COMPUTE_JOB_LIMIT" "$PYTHON_BIN" generate_analytics.py --shard-index "$ANA_SHARD" --shard-count "$ANALYTICS_SHARD_COUNT"
    done
    _wait_phase "generate-analytics"
fi

echo "====== Step 6.5: Save combined pet.csv reference before upload ======"
"$PYTHON_BIN" - <<'PY'
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

if [[ -z "${SUPABASE_DB_URI:-}" || "$RUN_LOCAL_SKIP_DB_LOAD" == "1" ]]; then
    echo "====== Pipeline complete! Skipped DB load. ======"
    exit 0
fi

echo "====== Step 7: Prepare DB load ======"
"$PYTHON_BIN" - <<'PY'
import os
import psycopg2
from pathlib import Path

conn = psycopg2.connect(os.environ["SUPABASE_DB_URI"])
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(Path("drop_views.sql").read_text(encoding="utf-8"))
    cur.execute(Path("create_tables.sql").read_text(encoding="utf-8"))
    cur.execute("TRUNCATE TABLE locations, pet, pet_percentiles, pet_forecast, pet_change CASCADE")
conn.close()
PY

"$PYTHON_BIN" load.py --append-only --skip-drop-views --skip-create-views --skip-table pet --skip-table pet_percentiles --skip-table pet_forecast --skip-table pet_change

echo "====== Step 8: Load shards to DB (parallel, CPU-aware) ======"
for (( LOAD_SHARD=0; LOAD_SHARD<ANALYTICS_SHARD_COUNT; LOAD_SHARD++ )); do
    _launch "$COMPUTE_JOB_LIMIT" "$PYTHON_BIN" load.py --append-only --skip-drop-views --skip-create-views --skip-table locations --analytics-shard-count "$ANALYTICS_SHARD_COUNT" --load-shard-index "$LOAD_SHARD" --load-shard-count "$ANALYTICS_SHARD_COUNT"
done
_wait_phase "load-to-db"

echo "====== Step 9: Recreate views ======"
"$PYTHON_BIN" load.py --append-only --skip-drop-views --skip-table locations --skip-table pet --skip-table pet_percentiles --skip-table pet_forecast --skip-table pet_change
echo "====== Pipeline complete! ======"