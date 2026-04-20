#!/usr/bin/env bash
# Local pipeline using the Cloud Run ERA5 worker for ERA5->PET, then local analytics/db steps.
set -euo pipefail

if [ -f .env ]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

PYTHON_BIN=${PYTHON_BIN:-python}
GCLOUD_BIN=${GCLOUD_BIN:-gcloud}
AWS_BIN=${AWS_BIN:-aws}

if [ -z "${SUPABASE_DB_URI:-}" ]; then
    echo "WARNING: SUPABASE_DB_URI is not set. Database upload steps will be skipped."
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1

RUN_LOCAL_MODE=${RUN_LOCAL_MODE:-full}
RUN_LOCAL_YEARS=${RUN_LOCAL_YEARS:-}
RUN_LOCAL_SKIP_ERA5_PULL=${RUN_LOCAL_SKIP_ERA5_PULL:-0}
RUN_LOCAL_SKIP_DB_LOAD=${RUN_LOCAL_SKIP_DB_LOAD:-0}
RUN_LOCAL_USE_CLOUD_RUN=${RUN_LOCAL_USE_CLOUD_RUN:-1}
RUN_LOCAL_PROVISION_CLOUD_RUN=${RUN_LOCAL_PROVISION_CLOUD_RUN:-0}
RUN_LOCAL_SYNC_PET_FROM_S3=${RUN_LOCAL_SYNC_PET_FROM_S3:-0}
RUN_LOCAL_SKIP_REMOTE_CLEAR=${RUN_LOCAL_SKIP_REMOTE_CLEAR:-0}
RUN_LOCAL_GCP_REGION=${RUN_LOCAL_GCP_REGION:-us-east1}
RUN_LOCAL_GCP_PROJECT=${RUN_LOCAL_GCP_PROJECT:-}
RUN_LOCAL_CLOUD_RUN_JOB=${RUN_LOCAL_CLOUD_RUN_JOB:-era5-worker}
RUN_LOCAL_S3_BUCKET=${RUN_LOCAL_S3_BUCKET:-pet-parquet-files}
RUN_LOCAL_S3_PREFIX=${RUN_LOCAL_S3_PREFIX:-local-run/${USER:-unknown-user}}
RUN_LOCAL_CLEAR_MAX_WORKERS=${RUN_LOCAL_CLEAR_MAX_WORKERS:-8}

REMOTE_BASE_PREFIX=""
REMOTE_OUT_DIR=""
REMOTE_PET_PREFIX=""

if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
    REMOTE_BASE_PREFIX=${RUN_LOCAL_S3_PREFIX#/}
    REMOTE_BASE_PREFIX=${REMOTE_BASE_PREFIX%/}
    if [[ -z "$REMOTE_BASE_PREFIX" ]]; then
        echo "RUN_LOCAL_S3_PREFIX must not be empty when RUN_LOCAL_USE_CLOUD_RUN=1." >&2
        exit 1
    fi
    REMOTE_OUT_DIR="s3://${RUN_LOCAL_S3_BUCKET}/${REMOTE_BASE_PREFIX}"
    REMOTE_PET_PREFIX="${REMOTE_BASE_PREFIX}/pet_data_csv"
fi

_PIDS=()
ERA5_MONTHS=()
ERA5_TIME_SHARDS=()

_require_executable() {
    local executable=$1
    if [[ "$executable" == */* ]]; then
        if [[ -x "$executable" ]]; then
            return
        fi
    elif command -v "$executable" >/dev/null 2>&1; then
        return
    fi

    echo "Required executable not found: $executable" >&2
    exit 1
}

_join_csv_args() {
    local IFS=,
    printf '%s' "$*"
}

_s3_prefix_has_objects() {
    local uri=$1
    local first_line
    first_line=$(
        "$AWS_BIN" s3 ls "$uri" --recursive 2>/dev/null | awk 'NR==1 {print; exit}' || true
    )
    [[ -n "$first_line" ]]
}

_local_pet_output_exists() {
    find pet_data_csv -type f -name 'pet_batch_*.parquet' -print -quit | grep -q .
}

_assert_pet_output_available() {
    if _local_pet_output_exists; then
        return
    fi

    if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
        local legacy_uri="s3://${RUN_LOCAL_S3_BUCKET}/${REMOTE_BASE_PREFIX}/era5_data_parquet/"
        if _s3_prefix_has_objects "$legacy_uri"; then
            echo "Cloud Run job ${RUN_LOCAL_CLOUD_RUN_JOB} only produced legacy era5_data_parquet output under ${legacy_uri}. Redeploy the job from this checkout by setting RUN_LOCAL_PROVISION_CLOUD_RUN=1 or running ./cloudrun_provision.sh manually." >&2
        else
            echo "No pet batch parquet files were written under ${REMOTE_OUT_DIR}/pet_data_csv." >&2
        fi
    else
        echo "No pet batch parquet files were written under ./pet_data_csv." >&2
    fi

    exit 1
}

_materialize_pet_csv() {
    echo "====== Step 2.5: Save combined pet.csv reference before analytics ======"
    "$PYTHON_BIN" - <<'PY'
import pandas as pd
from pathlib import Path

pet_root = Path("pet_data_csv")
all_files = sorted(pet_root.rglob("pet_batch_*.parquet"))
if not all_files:
    raise SystemExit("No pet batch parquet files found; cannot materialize pet.csv.")

dfs = [pd.read_parquet(f, columns=["location_id", "date", "pet"]) for f in all_files]
combined = pd.concat(dfs, ignore_index=True).sort_values(["location_id", "date"])
combined.to_csv("pet.csv", index=False)
print(f"Saved {len(combined)} rows to pet.csv")
PY
}

_clear_remote_pet_prefix() {
    echo "Clearing remote Cloud Run output at s3://${RUN_LOCAL_S3_BUCKET}/${REMOTE_PET_PREFIX}/"
    "$PYTHON_BIN" clear_s3_prefix.py \
        --bucket "$RUN_LOCAL_S3_BUCKET" \
        --prefix "${REMOTE_PET_PREFIX}/" \
        --max-workers "$RUN_LOCAL_CLEAR_MAX_WORKERS"
}

_sync_pet_from_s3() {
    echo "Syncing PET parquet from s3://${RUN_LOCAL_S3_BUCKET}/${REMOTE_PET_PREFIX}/"
    mkdir -p pet_data_csv
    "$AWS_BIN" s3 sync \
        "s3://${RUN_LOCAL_S3_BUCKET}/${REMOTE_PET_PREFIX}/" \
        ./pet_data_csv/ \
        --delete
}

_build_cloud_run_args_csv() {
    local year=$1
    local city_shard_index=$2
    local city_shard_count=$3
    local time_shard_index=$4
    local time_shard_count=$5

    local era5_args=(
        "--year=${year}"
        "--city-shard-index=${city_shard_index}"
        "--city-shard-count=${city_shard_count}"
        "--time-shard-index=${time_shard_index}"
        "--time-shard-count=${time_shard_count}"
        "--max-workers=${ERA5_BATCH_WORKERS}"
        "--batch-hours=${ERA5_BATCH_HOURS}"
        "--concurrency-profile=${ERA5_CONCURRENCY_PROFILE}"
        "--out-dir=${REMOTE_OUT_DIR}"
    )

    if (( ${#ERA5_MONTHS[@]} > 0 )); then
        era5_args+=("--months" "${ERA5_MONTHS[@]}")
    fi

    _join_csv_args "${era5_args[@]}"
}

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

if [[ -n "$RUN_LOCAL_YEARS" ]]; then
    read -r -a ALL_YEARS_ARRAY <<< "$RUN_LOCAL_YEARS"
else
    read -r -a ALL_YEARS_ARRAY <<< "2024 2025"
fi

if (( ${#ALL_YEARS_ARRAY[@]} == 0 )); then
    echo "RUN_LOCAL_YEARS must contain at least one year." >&2
    exit 1
fi

ALL_YEARS=${ALL_YEARS_ARRAY[*]}
YEAR_COUNT=${#ALL_YEARS_ARRAY[@]}

if [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-168}
    ERA5_TIME_SHARD_COUNT=${RUN_LOCAL_ERA5_TIME_SHARD_COUNT:-53}
    ERA5_TIME_SHARDS=("${RUN_LOCAL_SMOKE_TIME_SHARD_INDEX:-17}")
    echo "Running local pipeline in smoke mode: ${ALL_YEARS} ERA5 week only."
elif [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-168}
    ERA5_TIME_SHARD_COUNT=${RUN_LOCAL_ERA5_TIME_SHARD_COUNT:-4}
    ERA5_MONTHS=(5 6 7 8 9)
    for (( TIME_SHARD=0; TIME_SHARD<ERA5_TIME_SHARD_COUNT; TIME_SHARD++ )); do
        ERA5_TIME_SHARDS+=("$TIME_SHARD")
    done
    echo "Running local pipeline in full mode via Cloud Run: ${ALL_YEARS} (May-Sept)."
else
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-168}
    ERA5_TIME_SHARD_COUNT=${RUN_LOCAL_ERA5_TIME_SHARD_COUNT:-1}
    ERA5_TIME_SHARDS=(0)
    ERA5_MONTHS=(5 6 7 8 9)
    echo "Running local pipeline in full mode via local compute fallback: ${ALL_YEARS} (May-Sept)."
fi

export ALL_YEARS

echo "====== Step 1: Compute year ranges + setup locations ======"
mkdir -p output_tiles pet_data_csv analytics_data_csv
if [[ "$RUN_LOCAL_SKIP_ERA5_PULL" != "1" ]]; then
    for YEAR in "${ALL_YEARS_ARRAY[@]}"; do rm -rf "pet_data_csv/year=$YEAR"; done
fi
rm -rf analytics_data_csv/shard_count=*

"$PYTHON_BIN" cities.py

CITY_COUNT=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' cities.csv)
if (( CITY_COUNT < 1 )); then
    echo "cities.py produced zero locations; cannot continue." >&2
    exit 1
fi

if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
    DEFAULT_ERA5_CITY_SHARD_COUNT=1
else
    DEFAULT_ERA5_CITY_SHARD_COUNT=$(( CITY_COUNT < 2 ? CITY_COUNT : 2 ))
fi
ERA5_CITY_SHARD_COUNT=${RUN_LOCAL_ERA5_CITY_SHARD_COUNT:-$DEFAULT_ERA5_CITY_SHARD_COUNT}
if (( ERA5_CITY_SHARD_COUNT < 1 )); then ERA5_CITY_SHARD_COUNT=1; fi
if (( ERA5_CITY_SHARD_COUNT > CITY_COUNT )); then ERA5_CITY_SHARD_COUNT=$CITY_COUNT; fi

if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
    DEFAULT_CLOUD_RUN_JOB_LIMIT=$(( ERA5_CITY_SHARD_COUNT * ${#ERA5_TIME_SHARDS[@]} ))
    if (( YEAR_COUNT > 1 && DEFAULT_CLOUD_RUN_JOB_LIMIT < 2 )); then
        DEFAULT_CLOUD_RUN_JOB_LIMIT=2
    fi
    if (( DEFAULT_CLOUD_RUN_JOB_LIMIT > 4 )); then
        DEFAULT_CLOUD_RUN_JOB_LIMIT=4
    fi
    ERA5_JOB_LIMIT=${RUN_LOCAL_ERA5_JOB_LIMIT:-$DEFAULT_CLOUD_RUN_JOB_LIMIT}
    ERA5_CONCURRENCY_PROFILE=${RUN_LOCAL_ERA5_CONCURRENCY_PROFILE:-aggressive}
    ERA5_BATCH_WORKERS=${RUN_LOCAL_ERA5_BATCH_WORKERS:-1}
elif [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
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
if (( ERA5_JOB_LIMIT < 1 )); then ERA5_JOB_LIMIT=1; fi
if (( ERA5_BATCH_WORKERS < 1 )); then ERA5_BATCH_WORKERS=1; fi
if [[ "$RUN_LOCAL_MODE" == "smoke" && "$RUN_LOCAL_USE_CLOUD_RUN" != "1" && "$ERA5_BATCH_WORKERS" -gt 2 ]]; then
    ERA5_BATCH_WORKERS=2
fi

echo "====== Step 2: Compute ERA5 + PET ======"
if [[ "$RUN_LOCAL_SKIP_ERA5_PULL" != "1" ]]; then
    if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
        _require_executable "$PYTHON_BIN"
        _require_executable "$GCLOUD_BIN"
        _require_executable "$AWS_BIN"
        echo "Using Cloud Run job ${RUN_LOCAL_CLOUD_RUN_JOB} in ${RUN_LOCAL_GCP_REGION} with output ${REMOTE_OUT_DIR}"
        if [[ "$RUN_LOCAL_PROVISION_CLOUD_RUN" == "1" ]]; then
            echo "Refreshing Cloud Run job ${RUN_LOCAL_CLOUD_RUN_JOB} from the current checkout"
            /bin/bash ./cloudrun_provision.sh
        fi
        if [[ "$RUN_LOCAL_SKIP_REMOTE_CLEAR" != "1" ]]; then
            _clear_remote_pet_prefix
        fi
    fi

    for YEAR in "${ALL_YEARS_ARRAY[@]}"; do
        for (( CITY_SHARD=0; CITY_SHARD<ERA5_CITY_SHARD_COUNT; CITY_SHARD++ )); do
            for TIME_SHARD in "${ERA5_TIME_SHARDS[@]}"; do
                if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
                    CLOUD_RUN_ARGS=$(_build_cloud_run_args_csv \
                        "$YEAR" \
                        "$CITY_SHARD" \
                        "$ERA5_CITY_SHARD_COUNT" \
                        "$TIME_SHARD" \
                        "$ERA5_TIME_SHARD_COUNT")
                    GCLOUD_CMD=(
                        "$GCLOUD_BIN" run jobs execute "$RUN_LOCAL_CLOUD_RUN_JOB"
                        --region "$RUN_LOCAL_GCP_REGION"
                        --wait
                        "--args=${CLOUD_RUN_ARGS}"
                    )
                    if [[ -n "$RUN_LOCAL_GCP_PROJECT" ]]; then
                        GCLOUD_CMD+=(--project "$RUN_LOCAL_GCP_PROJECT")
                    fi
                    _launch "$ERA5_JOB_LIMIT" "${GCLOUD_CMD[@]}"
                else
                    ERA5_ARGS=(
                        --year "$YEAR"
                        --city-shard-count "$ERA5_CITY_SHARD_COUNT"
                        --city-shard-index "$CITY_SHARD"
                        --time-shard-index "$TIME_SHARD"
                        --time-shard-count "$ERA5_TIME_SHARD_COUNT"
                        --batch-hours "$ERA5_BATCH_HOURS"
                        --max-workers "$ERA5_BATCH_WORKERS"
                        --concurrency-profile "$ERA5_CONCURRENCY_PROFILE"
                        --out-dir .
                    )
                    if (( ${#ERA5_MONTHS[@]} > 0 )); then
                        ERA5_ARGS+=(--months "${ERA5_MONTHS[@]}")
                    fi
                    _launch "$ERA5_JOB_LIMIT" "$PYTHON_BIN" google_era5.py "${ERA5_ARGS[@]}"
                fi
            done
        done
    done
    _wait_phase "pull-google-era5-pet"

    if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
        _sync_pet_from_s3
    fi
elif [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" && "$RUN_LOCAL_SYNC_PET_FROM_S3" == "1" ]]; then
    _require_executable "$AWS_BIN"
    _sync_pet_from_s3
fi

_assert_pet_output_available
_materialize_pet_csv

echo "====== Step 3: Generate Analytics (parallel, CPU-aware) ======"
for (( ANA_SHARD=0; ANA_SHARD<ANALYTICS_SHARD_COUNT; ANA_SHARD++ )); do
    _launch "$COMPUTE_JOB_LIMIT" "$PYTHON_BIN" generate_analytics.py --shard-index "$ANA_SHARD" --shard-count "$ANALYTICS_SHARD_COUNT"
done
_wait_phase "generate-analytics"

if [[ "$RUN_LOCAL_SKIP_DB_LOAD" == "1" || -z "${SUPABASE_DB_URI:-}" ]]; then
    echo "====== Step 4-6: Skipping DB load ======"
    echo "====== Pipeline complete! ======"
    exit 0
fi

echo "====== Step 4: Prepare DB load ======"
"$PYTHON_BIN" - <<'PY'
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

"$PYTHON_BIN" load.py --append-only --skip-drop-views --skip-create-views --skip-table pet --skip-table pet_percentiles --skip-table pet_forecast --skip-table pet_change

echo "====== Step 5: Load shards to DB (parallel, CPU-aware) ======"
for (( LOAD_SHARD=0; LOAD_SHARD<ANALYTICS_SHARD_COUNT; LOAD_SHARD++ )); do
    _launch "$COMPUTE_JOB_LIMIT" "$PYTHON_BIN" load.py --append-only --skip-drop-views --skip-create-views --skip-table locations --analytics-shard-count "$ANALYTICS_SHARD_COUNT" --load-shard-index "$LOAD_SHARD" --load-shard-count "$ANALYTICS_SHARD_COUNT"
done
_wait_phase "load-to-db"

echo "====== Step 6: Recreate views ======"
"$PYTHON_BIN" load.py --append-only --skip-drop-views --skip-table locations --skip-table pet --skip-table pet_percentiles --skip-table pet_forecast --skip-table pet_change
echo "====== Pipeline complete! ======"