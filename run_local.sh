#!/usr/bin/env bash
# Local mirror of .github/workflows/pull_all.yml
# Uses Google Cloud Run for ERA5 data, local compute for CDS.
set -euo pipefail

if [ -f .env ]; then
    # Load environment variables from .env file
    # shellcheck disable=SC2046
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "${SUPABASE_DB_URI:-}" ]; then
    echo "ERROR: SUPABASE_DB_URI is not set. Please defined it in .env or your environment." >&2
    # exit 1
    # Not exiting here as some steps might not need it, though Step 7/8/9 do.
fi


# ---------------------------------------------------------------------------
# Parallel job helpers (FAIL FAST ENABLED)
# ---------------------------------------------------------------------------
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
    echo -e "\n[Shutdown] Caught signal or failure. Stopping all background jobs..." >&2

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

_launch() {
    local max=$1
    shift
    "$@" &
    _PIDS+=($!)
    while (( ${#_PIDS[@]} >= max )); do
        local alive=()
        for pid in "${_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            else
                # Fail fast logic: Check exit code instantly when job dies
                wait "$pid" || {
                    echo "ERROR: Background job $pid failed during launch throttling." >&2
                    exit 1
                }
            fi
        done
        _PIDS=("${alive[@]+"${alive[@]}"}")
        if (( ${#_PIDS[@]} >= max )); then
            sleep 0.3
        fi
    done
}

_wait_phase() {
    local label=${1:-phase}
    while (( ${#_PIDS[@]} > 0 )); do
        local alive=()
        for pid in "${_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                alive+=("$pid")
            else
                # Fail fast logic: if any job fails in this phase, crash pipeline immediately
                wait "$pid" || {
                    echo "ERROR: job $pid failed in [$label]" >&2
                    exit 1
                }
            fi
        done
        _PIDS=("${alive[@]+"${alive[@]}"}")
        if (( ${#_PIDS[@]} > 0 )); then
            sleep 0.3
        fi
    done
}

# ---------------------------------------------------------------------------
# Step 1: Compute year ranges
# ---------------------------------------------------------------------------
echo "====== Step 1: Compute year ranges + setup locations ======"
mkdir -p output_tiles era5_data_parquet weather_data_parquet utci_data_parquet \
         pet_data_csv analytics_data_csv combined_data_parquet

eval "$(python build_year_range.py --shell)"

# TEMPORARY OVERRIDE FOR DRY RUN
ERA5_YEARS="2023 2024"
CDS_YEARS="2025"
ALL_YEARS="2023 2024 2025"
export CDS_YEAR_MONTHS='[{"year": 2025, "month": 1, "month_pad": "01"}, {"year": 2025, "month": 2, "month_pad": "02"}]'


echo "  ERA5 years : $ERA5_YEARS"
echo "  CDS years  : $CDS_YEARS"
echo "  All years  : $ALL_YEARS"

python cities.py
python boxes.py \
    --cities-csv cities.csv \
    --cells-out output_tiles/unique_grid_cells.csv \
    --boxes-out output_tiles/tile_boxes.csv \
    --snapped-out output_tiles/snapped_cities.csv \
    --city-tile-out output_tiles/city_to_tile.csv

# ---------------------------------------------------------------------------
# Step 2: Pull ERA5 (Offloaded to Google Cloud Run Jobs)
# ---------------------------------------------------------------------------
echo "====== Step 2: Pull ERA5 (parallel via Cloud Run, max 8) ======"
for YEAR in $ERA5_YEARS; do
    _launch 8 _retry 3 gcloud run jobs execute era5-worker \
        --region=us-east1 \
        --wait \
        --args="--year=$YEAR,--city-shard-count=1,--time-shard-index=0,--time-shard-count=1,--concurrency-profile=balanced,--out-dir=s3://pet-parquet-files"
done
_wait_phase "pull-google-era5"

echo "====== Syncing ERA5 Data Locally from S3 ======"
# Cloud Run wrote directly to S3. We must sync it locally so Steps 5 and 6 can compute PET.
aws s3 sync s3://pet-parquet-files/era5_data_parquet/ ./era5_data_parquet/

# ---------------------------------------------------------------------------
# Step 3: Pull Weather (CDS API - STRICT 4-job concurrency)
# ---------------------------------------------------------------------------
echo "====== Step 3: Pull Weather (parallel: 10 city-shards, max 4) ======"
for CITY_SHARD in {0..9}; do
    _launch 4 _retry 3 python pull_weather.py \
        --start-year "2024" \
        --weather-city-shard-index "$CITY_SHARD" \
        --weather-city-shard-count 10 \
        --max-workers 1 # Forces 1 thread per job to map 1:1 with _launch bounds
done
_wait_phase "process-weather"

# ---------------------------------------------------------------------------
# Step 4: Pull MRT (CDS API - STRICT 4-job concurrency)
# ---------------------------------------------------------------------------
echo "====== Step 4: Pull MRT (parallel: year-month pairs, max 4) ======"
while IFS=$'\t' read -r YM_YEAR YM_MONTH; do
    _launch 4 _retry 3 python pull_mrt.py \
        --year "$YM_YEAR" \
        --month "$YM_MONTH" \
        --max-workers 1 # Forces 1 thread per job to map 1:1 with _launch bounds
done < <(python - <<'PY'
import json, sys, os
for ym in json.loads(os.environ["CDS_YEAR_MONTHS"]):
    print(f"{ym['year']}\t{ym['month']}")
PY
)
_wait_phase "process-mrt"

# ---------------------------------------------------------------------------
# Step 5: Combine + Calculate PET (Compute Intensive - Maxed out)
# ---------------------------------------------------------------------------
echo "====== Step 5: Combine + Calculate PET (parallel: year × 4 shards, max 16) ======"
for YEAR in $ALL_YEARS; do
    for PET_SHARD in {0..3}; do
        _launch 16 bash -c "
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
# Step 6: Generate Analytics (Compute Intensive - Maxed out)
# ---------------------------------------------------------------------------
echo "====== Step 6: Generate Analytics (parallel: 20 shards, max 20) ======"
for ANA_SHARD in {0..19}; do
    _launch 20 python generate_analytics.py \
        --shard-index "$ANA_SHARD" \
        --shard-count 20
done
_wait_phase "generate-analytics"

# ---------------------------------------------------------------------------
# Step 7: Prepare DB load
# ---------------------------------------------------------------------------
echo "====== Step 7: Prepare DB (drop views, truncate tables, load locations) ======"
python - <<'PY'
import os, sys, psycopg2
from pathlib import Path
db_uri = os.environ.get("SUPABASE_DB_URI")
if not db_uri: sys.exit(0)
conn = psycopg2.connect(db_uri)
conn.autocommit = True
try:
    with conn.cursor() as cur:
        cur.execute(Path("drop_views.sql").read_text(encoding="utf-8"))
        cur.execute("TRUNCATE TABLE locations, pet, pet_percentiles, pet_forecast, pet_change CASCADE")
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
# Step 8: Load shards to DB (Database I/O bound - Max 16)
# ---------------------------------------------------------------------------
echo "====== Step 8: Load shards to DB (parallel: 20 shards, max 16) ======"
for LOAD_SHARD in {0..19}; do
    _launch 16 python load.py \
        --append-only \
        --skip-drop-views \
        --skip-create-views \
        --skip-table locations \
        --analytics-shard
