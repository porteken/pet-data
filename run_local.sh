#!/usr/bin/env bash
# Local mirror of .github/workflows/pull_all.yml
# Skips S3 sync and AWS CloudFormation steps; all data lives on local disk.
set -euo pipefail

export SUPABASE_DB_URI="postgresql://postgres:postgres@localhost:5432/postgres"

# ---------------------------------------------------------------------------
# Parallel job helpers
# ---------------------------------------------------------------------------

# _pids: global accumulator reused per phase
_PIDS=()

_kill_tree() {
    local pid=$1
    local sig=${2:-TERM}
    local children
    children=$(pgrep -P "$pid" 2>/dev/null || true)
    if [ -n "$children" ]; then
        for child in $children; do
            _kill_tree "$child" "$sig"
        done
    fi
    kill -"$sig" "$pid" 2>/dev/null || true
}

_cleanup() {
    trap - SIGINT SIGTERM ERR EXIT
    echo -e "\n[Shutdown] Caught signal. Stopping all background jobs..." >&2

    local active=()
    for pid in "${_PIDS[@]+"${_PIDS[@]}"}"; do
        if kill -0 "$pid" 2>/dev/null; then
            active+=("$pid")
        fi
    done

    if (( ${#active[@]} > 0 )); then
        echo "[Shutdown] Sending SIGTERM to PIDs: ${active[*]}" >&2
        for pid in "${active[@]}"; do
            _kill_tree "$pid" TERM
        done

        # Give them up to 5 seconds to gracefully exit
        for _ in {1..25}; do
            local still_alive=0
            for pid in "${active[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    still_alive=1
                    break
                fi
            done
            if (( still_alive == 0 )); then
                break
            fi
            sleep 0.2
        done

        # Force kill any remaining
        for pid in "${active[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                echo "[Shutdown] Force killing PID $pid..." >&2
                _kill_tree "$pid" KILL
            fi
        done
    fi
    exit 1
}

trap _cleanup SIGINT SIGTERM

_retry() {
    local max_attempts=$1
    shift
    local attempt=1
    until "$@"; do
        if (( attempt == max_attempts )); then
            echo "ERROR: Command '$*' failed after $attempt attempts." >&2
            return 1
        fi
        echo "Command '$*' failed. Retrying... (Attempt $((attempt + 1)) of $max_attempts)" >&2
        sleep 5
        ((attempt++))
    done
}

# Background a command, add its PID to _PIDS, and block until fewer than
# MAX_PARALLEL jobs are running (mirrors GitHub's max-parallel setting).
_launch() {
    local max=$1
    shift
    "$@" &
    _PIDS+=($!)
    # throttle: wait until a slot is free
    while (( ${#_PIDS[@]} >= max )); do
        local alive=()
        for pid in "${_PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null && alive+=("$pid")
        done
        _PIDS=("${alive[@]+"${alive[@]}"}")
        if (( ${#_PIDS[@]} >= max )); then
            sleep 0.3
        fi
    done
}

# Wait for all accumulated PIDs; fail if any job failed.
_wait_phase() {
    local label=${1:-phase}
    local failed=0
    for pid in "${_PIDS[@]+"${_PIDS[@]}"}"; do
        wait "$pid" || { echo "ERROR: job $pid failed in [$label]" >&2; failed=1; }
    done
    _PIDS=()
    if (( failed )); then
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Step 1: Compute year ranges  (build-year-range + setup-locations jobs)
# ---------------------------------------------------------------------------
echo "====== Step 1: Compute year ranges + setup locations ======"
mkdir -p output_tiles era5_data_parquet weather_data_parquet utci_data_parquet \
         pet_data_csv analytics_data_csv combined_data_parquet

# build_year_range.py --shell emits: ERA5_YEARS, CDS_YEARS, CDS_YEAR_MONTHS, ALL_YEARS, etc.
eval "$(python build_year_range.py --shell)"
# ERA5_YEARS   — space-separated list of years to pull from Google ARCO
# CDS_YEARS    — space-separated list of years to pull from CDS (weather)
# CDS_YEAR_MONTHS — JSON array of {year, month, month_pad} objects (for MRT)
# ALL_YEARS    — every year from ERA5_START through previous year (for PET/analytics)

echo "  ERA5 years : $ERA5_YEARS"
echo "  CDS years  : $CDS_YEARS"
echo "  All years  : $ALL_YEARS"

# setup-locations
python cities.py
python boxes.py \
    --cities-csv cities.csv \
    --cells-out output_tiles/unique_grid_cells.csv \
    --boxes-out output_tiles/tile_boxes.csv \
    --snapped-out output_tiles/snapped_cities.csv \
    --city-tile-out output_tiles/city_to_tile.csv

# ---------------------------------------------------------------------------
# Step 2: Pull ERA5  (8 processes in parallel, 1 internal thread each = 8 concurrent GCS streams)
# ---------------------------------------------------------------------------
echo "====== Step 2: Pull ERA5 (8 processes in parallel, 1 worker each) ======"
for YEAR in $ERA5_YEARS; do
    _launch 8 _retry 3 python google_era5.py \
        --year "$YEAR" \
        --city-shard-count 1 \
        --time-shard-index 0 \
        --time-shard-count 1 \
        --concurrency-profile balanced \
        --batch-hours 720 \
        --max-workers 1
done
_wait_phase "pull-google-era5"

# ---------------------------------------------------------------------------
# Step 3: Pull Weather  (process-weather: 10 city-shards, max-parallel 2)
#         Weather runs in parallel with ERA5 in CI but depends on CDS quota;
#         here we run it after ERA5 to avoid CDS conflicts, matching CI intent.
# ---------------------------------------------------------------------------
echo "====== Step 3: Pull Weather (parallel: 10 city-shards, max 2) ======"
for CITY_SHARD in 0 1 2 3 4 5 6 7 8 9; do
    _launch 2 _retry 3 python pull_weather.py \
        --start-year "$START_YEAR" \
        --weather-city-shard-index "$CITY_SHARD" \
        --weather-city-shard-count 10 \
        --max-workers 2
done
_wait_phase "process-weather"

# ---------------------------------------------------------------------------
# Step 4: Pull MRT  (process-mrt: year × month, max-parallel 8)
#         Runs after weather in CI (full CDS quota available).
# ---------------------------------------------------------------------------
echo "====== Step 4: Pull MRT (parallel: year-month pairs, max 8) ======"
# CDS_YEAR_MONTHS is a JSON array; parse year/month pairs with python
while IFS=$'\t' read -r YM_YEAR YM_MONTH; do
    _launch 8 _retry 3 python pull_mrt.py \
        --year "$YM_YEAR" \
        --month "$YM_MONTH" \
        --max-workers 6
done < <(python - <<'PY'
import json, sys, os
for ym in json.loads(os.environ["CDS_YEAR_MONTHS"]):
    print(f"{ym['year']}\t{ym['month']}")
PY
)
_wait_phase "process-mrt"

# ---------------------------------------------------------------------------
# Step 5: Combine + Calculate PET  (process-pet: year × 4 shards, max 8)
# ---------------------------------------------------------------------------
echo "====== Step 5: Combine + Calculate PET (parallel: year × 4 shards, max 8) ======"
for YEAR in $ALL_YEARS; do
    for PET_SHARD in 0 1 2 3; do
        _launch 8 bash -c "
            python combine.py \
                --era5-root ./era5_data_parquet \
                --year $YEAR \
                --shard-index $PET_SHARD \
                --shard-count 4 && \
            python calculate_pet.py \
                --year $YEAR \
                --shard-index $PET_SHARD \
                --shard-count 4
        "
    done
done
_wait_phase "process-pet"

# ---------------------------------------------------------------------------
# Step 6: Generate Analytics  (generate-analytics: 20 shards, max 10)
# ---------------------------------------------------------------------------
echo "====== Step 6: Generate Analytics (parallel: 20 shards, max 10) ======"
for ANA_SHARD in $(seq 0 19); do
    _launch 10 python generate_analytics.py \
        --shard-index "$ANA_SHARD" \
        --shard-count 20
done
_wait_phase "generate-analytics"

# ---------------------------------------------------------------------------
# Step 7: Prepare DB load  (prepare-db-load: drop views, truncate, load locations)
# ---------------------------------------------------------------------------
echo "====== Step 7: Prepare DB (drop views, truncate tables, load locations) ======"
python - <<'PY'
import os, sys
from pathlib import Path

import psycopg2

db_uri = os.environ.get("SUPABASE_DB_URI")
if not db_uri:
    print("SUPABASE_DB_URI not set, skipping database cleanup.")
    sys.exit(0)

conn = psycopg2.connect(db_uri)
conn.autocommit = True
try:
    with conn.cursor() as cur:
        cur.execute(Path("drop_views.sql").read_text(encoding="utf-8"))
        cur.execute("""
            TRUNCATE TABLE
                locations,
                pet,
                pet_percentiles,
                pet_forecast,
                pet_change
            CASCADE
        """)
finally:
    conn.close()
PY

python load.py \
    --append-only \
    --skip-drop-views \
    --skip-create-views \
    --skip-table pet \
    --skip-table pet_percentiles \
    --skip-table pet_forecast \
    --skip-table pet_change

# ---------------------------------------------------------------------------
# Step 8: Load shards to DB  (load-to-db: 20 shards, max 10)
# ---------------------------------------------------------------------------
echo "====== Step 8: Load shards to DB (parallel: 20 shards, max 10) ======"
for LOAD_SHARD in $(seq 0 19); do
    _launch 10 python load.py \
        --append-only \
        --skip-drop-views \
        --skip-create-views \
        --skip-table locations \
        --analytics-shard-count 20 \
        --load-shard-index "$LOAD_SHARD" \
        --load-shard-count 20
done
_wait_phase "load-to-db"

# ---------------------------------------------------------------------------
# Step 9: Finalize  (finalize-db-load: recreate views)
# ---------------------------------------------------------------------------
echo "====== Step 9: Recreate views ======"
python load.py \
    --append-only \
    --skip-drop-views \
    --skip-table locations \
    --skip-table pet \
    --skip-table pet_percentiles \
    --skip-table pet_forecast \
    --skip-table pet_change

echo "====== Pipeline complete! ======"
