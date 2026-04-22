#!/usr/bin/env bash
# Local pipeline using the Cloud Run ERA5 worker for ERA5->PET, then local analytics/db steps.
set -euo pipefail

if [ -f .env ]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

RUN_LOCAL_USE_UV=${RUN_LOCAL_USE_UV:-1}
UV_BIN=${UV_BIN:-uv}
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
RUN_LOCAL_MERGE_EXISTING_PET_HISTORY=${RUN_LOCAL_MERGE_EXISTING_PET_HISTORY:-1}
RUN_LOCAL_HISTORY_DB_URI_ENV=${RUN_LOCAL_HISTORY_DB_URI_ENV:-SUPABASE_DB_URI}
RUN_LOCAL_HISTORY_FALLBACK_DB_URI_ENV=${RUN_LOCAL_HISTORY_FALLBACK_DB_URI_ENV:-SUPABASE_DB_URI_PRD}
RUN_LOCAL_ALLOW_EMPTY_ANALYTICS=${RUN_LOCAL_ALLOW_EMPTY_ANALYTICS:-0}
RUN_LOCAL_PROGRESS=${RUN_LOCAL_PROGRESS:-1}
RUN_LOCAL_ERA5_PARALLEL_STRATEGY=${RUN_LOCAL_ERA5_PARALLEL_STRATEGY:-auto}

if [[ "$RUN_LOCAL_SKIP_DB_LOAD" == "1" ]]; then
    echo "WARNING: RUN_LOCAL_SKIP_DB_LOAD=1 means this run will not update Supabase."
fi

if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
    echo "WARNING: RUN_LOCAL_USE_CLOUD_RUN=1 uses the deployed Cloud Run image, not your local checkout."
    echo "         Use RUN_LOCAL_PROVISION_CLOUD_RUN=1 to redeploy it first, or RUN_LOCAL_USE_CLOUD_RUN=0 to run local google_era5.py."
fi

REMOTE_BASE_PREFIX=""
REMOTE_OUT_DIR=""
REMOTE_PET_PREFIX=""
HISTORY_EXPORT_CSV="existing_pet.csv"
FULL_PET_CSV="pet_full.csv"
DB_LOAD_PET_CSV="pet.csv"
DB_LOAD_PREFER_PET_CSV=0
ANALYTICS_GENERATE_ARGS=()

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
declare -A _PID_PROGRESS_LABELS=()

PROGRESS_TOTAL=0
PROGRESS_CURRENT=0
PROGRESS_BAR_WIDTH=28
PROGRESS_LAST_PERCENT=-1
PROGRESS_IS_TTY=0
if [[ -t 2 ]]; then
    PROGRESS_IS_TTY=1
fi

_progress_enabled() {
    [[ "$RUN_LOCAL_PROGRESS" == "1" ]]
}

_progress_bar() {
    local current=$1
    local total=$2
    local width=$3
    local filled=0
    local empty=0

    if (( total > 0 )); then
        filled=$(( current * width / total ))
    fi
    if (( filled > width )); then
        filled=$width
    fi
    empty=$(( width - filled ))

    printf '%*s' "$filled" '' | tr ' ' '#'
    printf '%*s' "$empty" '' | tr ' ' '-'
}

_progress_display() {
    local message=$1
    local force=${2:-0}
    local total=$PROGRESS_TOTAL
    local percent=0

    if ! _progress_enabled; then
        return
    fi

    if (( total < 1 )); then
        total=1
    fi
    percent=$(( PROGRESS_CURRENT * 100 / total ))

    if (( PROGRESS_IS_TTY )); then
        printf '\r[%s] %3d%% (%d/%d) %s' \
            "$(_progress_bar "$PROGRESS_CURRENT" "$total" "$PROGRESS_BAR_WIDTH")" \
            "$percent" \
            "$PROGRESS_CURRENT" \
            "$total" \
            "$message" >&2
        if (( force )) || (( PROGRESS_CURRENT >= total )); then
            printf '\n' >&2
        fi
        return
    fi

    if (( force )) || (( percent >= PROGRESS_LAST_PERCENT + 5 )) || (( PROGRESS_CURRENT >= total )); then
        echo "Progress [$(_progress_bar "$PROGRESS_CURRENT" "$total" 20)] ${percent}% (${PROGRESS_CURRENT}/${total}) ${message}" >&2
        PROGRESS_LAST_PERCENT=$percent
    fi
}

_progress_note() {
    _progress_display "$1" 1
}

_progress_advance() {
    local increment=${1:-1}
    local message=${2:-Working...}

    (( PROGRESS_CURRENT += increment ))
    if (( PROGRESS_CURRENT > PROGRESS_TOTAL )); then
        PROGRESS_CURRENT=$PROGRESS_TOTAL
    fi

    _progress_display "$message"
}

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

_require_python_runner() {
    if [[ "$RUN_LOCAL_USE_UV" == "1" ]]; then
        _require_executable "$UV_BIN"
    else
        _require_executable "$PYTHON_BIN"
    fi
}

_sync_python_environment() {
    if [[ "$RUN_LOCAL_USE_UV" != "1" ]]; then
        return
    fi

    echo "Syncing local project environment with uv..."
    "$UV_BIN" sync --locked --extra gcs
}

_run_python() {
    if [[ "$RUN_LOCAL_USE_UV" == "1" ]]; then
        "$UV_BIN" run python "$@"
    else
        "$PYTHON_BIN" "$@"
    fi
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
    _run_python - <<'PY'
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

_write_empty_pet_csv() {
    printf 'location_id,date,pet\n' > "$1"
}

_count_pet_csv_years() {
    local csv_path=$1
    if [[ ! -f "$csv_path" ]]; then
        echo 0
        return
    fi

    awk -F, 'NR > 1 {print substr($2, 1, 4)}' "$csv_path" | sort -u | awk 'END {print NR + 0}'
}

_export_history_pet_csv() {
    local env_name=$1
    local output_csv=$2

    if [[ -z "$env_name" || -z "${!env_name:-}" ]]; then
        _write_empty_pet_csv "$output_csv"
        return
    fi

    SUPABASE_DB_URI="${!env_name}" _run_python historical_pet_update.py export-all "$output_csv"
}

_prepare_analytics_pet_inputs() {
    ANALYTICS_GENERATE_ARGS=()
    DB_LOAD_PET_CSV="pet.csv"
    DB_LOAD_PREFER_PET_CSV=0

    if [[ "$RUN_LOCAL_SKIP_DB_LOAD" == "1" || -z "${SUPABASE_DB_URI:-}" ]]; then
        return
    fi

    if [[ "$RUN_LOCAL_MERGE_EXISTING_PET_HISTORY" != "1" ]]; then
        return
    fi

    echo "====== Step 2.75: Merge historical PET for analytics + DB load ======"
    rm -f "$HISTORY_EXPORT_CSV" "$FULL_PET_CSV"

    local primary_env=$RUN_LOCAL_HISTORY_DB_URI_ENV
    local fallback_env=$RUN_LOCAL_HISTORY_FALLBACK_DB_URI_ENV

    _export_history_pet_csv "$primary_env" "$HISTORY_EXPORT_CSV"
    _run_python historical_pet_update.py merge "$FULL_PET_CSV" "$HISTORY_EXPORT_CSV" --dirs pet_data_csv

    local merged_year_count
    merged_year_count=$(_count_pet_csv_years "$FULL_PET_CSV")
    if (( merged_year_count < 10 )) && [[ -n "$fallback_env" ]] && [[ "$fallback_env" != "$primary_env" ]] && [[ -n "${!fallback_env:-}" ]]; then
        echo "Primary history source did not provide enough PET history; retrying with ${fallback_env}."
        _export_history_pet_csv "$fallback_env" "$HISTORY_EXPORT_CSV"
        _run_python historical_pet_update.py merge "$FULL_PET_CSV" "$HISTORY_EXPORT_CSV" --dirs pet_data_csv
        merged_year_count=$(_count_pet_csv_years "$FULL_PET_CSV")
    fi

    if (( merged_year_count < 10 )); then
        echo "Need at least 10 PET years to generate pet_forecast (and the derived pet_change view). Found ${merged_year_count} year(s) in ${FULL_PET_CSV}." >&2
        echo "Either widen the historical source or set RUN_LOCAL_MERGE_EXISTING_PET_HISTORY=0 and accept empty analytics." >&2
        exit 1
    fi

    ANALYTICS_GENERATE_ARGS=(--pet-csv "$FULL_PET_CSV" --prefer-pet-csv)
    DB_LOAD_PET_CSV="$FULL_PET_CSV"
    DB_LOAD_PREFER_PET_CSV=1
    _progress_advance 1 "Merged historical PET inputs"
}

_validate_analytics_outputs() {
    if [[ "$RUN_LOCAL_SKIP_DB_LOAD" == "1" || -z "${SUPABASE_DB_URI:-}" || "$RUN_LOCAL_ALLOW_EMPTY_ANALYTICS" == "1" ]]; then
        return
    fi

    local analytics_counts
    analytics_counts=$(_run_python - <<'PY'
from pathlib import Path

import pyarrow.parquet as pq

analytics_root = Path("analytics_data_csv")
forecast_rows = sum(
    pq.ParquetFile(str(path)).metadata.num_rows
    for path in analytics_root.rglob("forecast.parquet")
)
change_rows = sum(
    pq.ParquetFile(str(path)).metadata.num_rows
    for path in analytics_root.rglob("change_per_decade.parquet")
)
print(f"{forecast_rows} {change_rows}")
PY
)

    local forecast_rows change_rows
    read -r forecast_rows change_rows <<< "$analytics_counts"

    if (( forecast_rows == 0 || change_rows == 0 )); then
        echo "Refusing to load empty analytics into Supabase: forecast rows=${forecast_rows}, change rows=${change_rows}." >&2
        echo "Provide additional PET history or set RUN_LOCAL_ALLOW_EMPTY_ANALYTICS=1 to override." >&2
        exit 1
    fi

    _progress_advance 1 "Validated analytics outputs"
}

_clear_remote_pet_prefix() {
    echo "Clearing remote Cloud Run output at s3://${RUN_LOCAL_S3_BUCKET}/${REMOTE_PET_PREFIX}/"
    _run_python clear_s3_prefix.py \
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

_cancel_running_cloud_run_executions() {
    local cancel_args=(
        cancel_cloud_run_job_executions.py
        --job "$RUN_LOCAL_CLOUD_RUN_JOB"
        --region "$RUN_LOCAL_GCP_REGION"
    )

    if [[ -n "$RUN_LOCAL_GCP_PROJECT" ]]; then
        cancel_args+=(--project "$RUN_LOCAL_GCP_PROJECT")
    fi

    _run_python "${cancel_args[@]}"
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

_remove_pid_from_tracking() {
    local finished_pid=$1
    local active_pids=()

    for pid in "${_PIDS[@]}"; do
        if [[ "$pid" != "$finished_pid" ]]; then
            active_pids+=("$pid")
        fi
    done

    _PIDS=("${active_pids[@]}")
}

_wait_for_any_pid() {
    local finished_pid=""
    local exit_status=0
    local label="background task"

    if wait -n -p finished_pid; then
        exit_status=0
    else
        exit_status=$?
    fi

    if [[ -n "$finished_pid" ]]; then
        label=${_PID_PROGRESS_LABELS[$finished_pid]:-background task}
        _remove_pid_from_tracking "$finished_pid"
        unset "_PID_PROGRESS_LABELS[$finished_pid]"
    fi

    if (( exit_status != 0 )); then
        echo "Background job failed: ${label}" >&2
        exit "$exit_status"
    fi

    _progress_advance 1 "$label"
}

_launch() {
    local max=$1
    local label=$2
    shift 2

    "$@" &
    local pid=$!
    _PIDS+=("$pid")
    _PID_PROGRESS_LABELS[$pid]="$label"

    while (( ${#_PIDS[@]} >= max )); do
        _wait_for_any_pid
    done
}

_wait_phase() {
    while (( ${#_PIDS[@]} > 0 )); do
        _wait_for_any_pid
    done
}

_resolve_era5_parallel_strategy() {
    if [[ "$RUN_LOCAL_ERA5_PARALLEL_STRATEGY" == "auto" ]]; then
        if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
            echo "cloud-run-scale-out"
        elif [[ "$RUN_LOCAL_MODE" == "smoke" ]]; then
            echo "local-smoke"
        else
            echo "local-balanced"
        fi
        return
    fi

    case "$RUN_LOCAL_ERA5_PARALLEL_STRATEGY" in
        cloud-run-scale-out|local-balanced|local-smoke|serial)
            echo "$RUN_LOCAL_ERA5_PARALLEL_STRATEGY"
            ;;
        *)
            echo "RUN_LOCAL_ERA5_PARALLEL_STRATEGY must be one of auto, cloud-run-scale-out, local-balanced, local-smoke, serial." >&2
            exit 1
            ;;
    esac
}

_calculate_progress_total() {
    local total=1
    local era5_task_count=0

    if [[ "$RUN_LOCAL_SKIP_ERA5_PULL" != "1" ]]; then
        if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
            (( total += 1 ))
            if [[ "$RUN_LOCAL_PROVISION_CLOUD_RUN" == "1" ]]; then
                (( total += 1 ))
            fi
            if [[ "$RUN_LOCAL_SKIP_REMOTE_CLEAR" != "1" ]]; then
                (( total += 1 ))
            fi
        fi

        era5_task_count=$(( YEAR_COUNT * ERA5_CITY_SHARD_COUNT * ${#ERA5_TIME_SHARDS[@]} ))
        (( total += era5_task_count ))

        if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
            (( total += 1 ))
        fi
    elif [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" && "$RUN_LOCAL_SYNC_PET_FROM_S3" == "1" ]]; then
        (( total += 1 ))
    fi

    (( total += 1 ))

    if [[ "$RUN_LOCAL_SKIP_DB_LOAD" != "1" && -n "${SUPABASE_DB_URI:-}" && "$RUN_LOCAL_MERGE_EXISTING_PET_HISTORY" == "1" ]]; then
        (( total += 1 ))
    fi

    (( total += ANALYTICS_SHARD_COUNT ))

    if [[ "$RUN_LOCAL_SKIP_DB_LOAD" != "1" && -n "${SUPABASE_DB_URI:-}" && "$RUN_LOCAL_ALLOW_EMPTY_ANALYTICS" != "1" ]]; then
        (( total += 1 ))
    fi

    if [[ "$RUN_LOCAL_SKIP_DB_LOAD" != "1" && -n "${SUPABASE_DB_URI:-}" ]]; then
        (( total += 1 ))
        (( total += ANALYTICS_SHARD_COUNT ))
        (( total += 1 ))
    fi

    echo "$total"
}

_cpu_count() { getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1; }
CPU_COUNT=$(_cpu_count)
AVAILABLE_CPUS=$(( CPU_COUNT > 1 ? CPU_COUNT - 1 : 1 ))
COMPUTE_JOB_LIMIT=$(( AVAILABLE_CPUS < 8 ? AVAILABLE_CPUS : 8 ))

if [[ -n "$RUN_LOCAL_YEARS" ]]; then
    read -r -a ALL_YEARS_ARRAY <<< "$RUN_LOCAL_YEARS"
else
    DEFAULT_ERA5_END_YEAR=$(( $(date -u +%Y) - 1 ))
    if (( DEFAULT_ERA5_END_YEAR < 2000 )); then
        echo "Computed default ERA5 end year ${DEFAULT_ERA5_END_YEAR} is earlier than 2000; set RUN_LOCAL_YEARS explicitly." >&2
        exit 1
    fi

    ALL_YEARS_ARRAY=()
    for (( YEAR=2000; YEAR<=DEFAULT_ERA5_END_YEAR; YEAR++ )); do
        ALL_YEARS_ARRAY+=("$YEAR")
    done
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
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-720}
    ERA5_TIME_SHARD_COUNT=${RUN_LOCAL_ERA5_TIME_SHARD_COUNT:-13}
    for (( TIME_SHARD=0; TIME_SHARD<ERA5_TIME_SHARD_COUNT; TIME_SHARD++ )); do
        ERA5_TIME_SHARDS+=("$TIME_SHARD")
    done
    echo "Running local pipeline in full mode via Cloud Run: ${ALL_YEARS} (full year)."
else
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-720}
    ERA5_TIME_SHARD_COUNT=${RUN_LOCAL_ERA5_TIME_SHARD_COUNT:-13}
    for (( TIME_SHARD=0; TIME_SHARD<ERA5_TIME_SHARD_COUNT; TIME_SHARD++ )); do
        ERA5_TIME_SHARDS+=("$TIME_SHARD")
    done
    echo "Running local pipeline in full mode via local compute fallback: ${ALL_YEARS} (full year)."
fi

export ALL_YEARS

_require_python_runner
_sync_python_environment

echo "====== Step 1: Compute year ranges + setup locations ======"
mkdir -p output_tiles pet_data_csv analytics_data_csv
if [[ "$RUN_LOCAL_SKIP_ERA5_PULL" != "1" ]]; then
    for YEAR in "${ALL_YEARS_ARRAY[@]}"; do rm -rf "pet_data_csv/year=$YEAR"; done
fi
rm -rf analytics_data_csv/shard_count=*

_run_python cities.py

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

ANALYTICS_SHARD_COUNT=$(( CITY_COUNT < COMPUTE_JOB_LIMIT ? CITY_COUNT : COMPUTE_JOB_LIMIT ))
ERA5_PARALLEL_STRATEGY=$(_resolve_era5_parallel_strategy)

case "$ERA5_PARALLEL_STRATEGY" in
    cloud-run-scale-out)
        DEFAULT_CLOUD_RUN_JOB_LIMIT=$(( ERA5_CITY_SHARD_COUNT * ${#ERA5_TIME_SHARDS[@]} ))
        if (( DEFAULT_CLOUD_RUN_JOB_LIMIT > 8 )); then
            DEFAULT_CLOUD_RUN_JOB_LIMIT=8
        fi
        ERA5_JOB_LIMIT=${RUN_LOCAL_ERA5_JOB_LIMIT:-$DEFAULT_CLOUD_RUN_JOB_LIMIT}
        ERA5_CONCURRENCY_PROFILE=${RUN_LOCAL_ERA5_CONCURRENCY_PROFILE:-aggressive}
        ERA5_BATCH_WORKERS=${RUN_LOCAL_ERA5_BATCH_WORKERS:-1}
        ;;
    local-smoke)
        DEFAULT_SMOKE_BATCH_WORKERS=$(( AVAILABLE_CPUS / ERA5_CITY_SHARD_COUNT ))
        if (( DEFAULT_SMOKE_BATCH_WORKERS < 1 )); then
            DEFAULT_SMOKE_BATCH_WORKERS=1
        fi
        if (( DEFAULT_SMOKE_BATCH_WORKERS > 2 )); then
            DEFAULT_SMOKE_BATCH_WORKERS=2
        fi
        ERA5_JOB_LIMIT=${RUN_LOCAL_ERA5_JOB_LIMIT:-$ERA5_CITY_SHARD_COUNT}
        ERA5_CONCURRENCY_PROFILE=${RUN_LOCAL_ERA5_CONCURRENCY_PROFILE:-aggressive}
        ERA5_BATCH_WORKERS=${RUN_LOCAL_ERA5_BATCH_WORKERS:-$DEFAULT_SMOKE_BATCH_WORKERS}
        ;;
    local-balanced)
        ERA5_JOB_LIMIT=${RUN_LOCAL_ERA5_JOB_LIMIT:-$ERA5_CITY_SHARD_COUNT}
        ERA5_CONCURRENCY_PROFILE=${RUN_LOCAL_ERA5_CONCURRENCY_PROFILE:-balanced}
        ERA5_BATCH_WORKERS=${RUN_LOCAL_ERA5_BATCH_WORKERS:-1}
        ;;
    serial)
        ERA5_JOB_LIMIT=${RUN_LOCAL_ERA5_JOB_LIMIT:-1}
        ERA5_CONCURRENCY_PROFILE=${RUN_LOCAL_ERA5_CONCURRENCY_PROFILE:-conservative}
        ERA5_BATCH_WORKERS=${RUN_LOCAL_ERA5_BATCH_WORKERS:-1}
        ;;
esac

if (( ERA5_JOB_LIMIT < 1 )); then ERA5_JOB_LIMIT=1; fi
if (( ERA5_BATCH_WORKERS < 1 )); then ERA5_BATCH_WORKERS=1; fi

PROGRESS_TOTAL=$(_calculate_progress_total)
_progress_note "Initialized pipeline progress tracking for ${PROGRESS_TOTAL} work unit(s)."
echo "Using ERA5 parallel strategy: ${ERA5_PARALLEL_STRATEGY} (job limit=${ERA5_JOB_LIMIT}, batch workers=${ERA5_BATCH_WORKERS}, profile=${ERA5_CONCURRENCY_PROFILE})"
_progress_advance 1 "Prepared locations"

echo "====== Step 2: Compute ERA5 + PET ======"
if [[ "$RUN_LOCAL_SKIP_ERA5_PULL" != "1" ]]; then
    if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
        _require_python_runner
        _require_executable "$GCLOUD_BIN"
        _require_executable "$AWS_BIN"
        _cancel_running_cloud_run_executions
        _progress_advance 1 "Cancelled stale Cloud Run executions"
        echo "Using Cloud Run job ${RUN_LOCAL_CLOUD_RUN_JOB} in ${RUN_LOCAL_GCP_REGION} with output ${REMOTE_OUT_DIR}"
        if [[ "$RUN_LOCAL_PROVISION_CLOUD_RUN" == "1" ]]; then
            echo "Refreshing Cloud Run job ${RUN_LOCAL_CLOUD_RUN_JOB} from the current checkout"
            /bin/bash ./cloudrun_provision.sh
            _progress_advance 1 "Provisioned Cloud Run worker"
        fi
        if [[ "$RUN_LOCAL_SKIP_REMOTE_CLEAR" != "1" ]]; then
            _clear_remote_pet_prefix
            _progress_advance 1 "Cleared remote PET prefix"
        fi
    fi

    for YEAR in "${ALL_YEARS_ARRAY[@]}"; do
        for (( CITY_SHARD=0; CITY_SHARD<ERA5_CITY_SHARD_COUNT; CITY_SHARD++ )); do
            for TIME_SHARD in "${ERA5_TIME_SHARDS[@]}"; do
                SHARD_LABEL="ERA5 year=${YEAR} city-shard=$(( CITY_SHARD + 1 ))/${ERA5_CITY_SHARD_COUNT} time-shard=${TIME_SHARD}/${ERA5_TIME_SHARD_COUNT}"
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
                    _launch "$ERA5_JOB_LIMIT" "$SHARD_LABEL" "${GCLOUD_CMD[@]}"
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
                    _launch "$ERA5_JOB_LIMIT" "$SHARD_LABEL" _run_python google_era5.py "${ERA5_ARGS[@]}"
                fi
            done
        done
    done
    _wait_phase "pull-google-era5-pet"

    if [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" ]]; then
        _sync_pet_from_s3
        _progress_advance 1 "Synced PET shards from S3"
    fi
elif [[ "$RUN_LOCAL_USE_CLOUD_RUN" == "1" && "$RUN_LOCAL_SYNC_PET_FROM_S3" == "1" ]]; then
    _require_executable "$AWS_BIN"
    _sync_pet_from_s3
    _progress_advance 1 "Synced PET shards from S3"
fi

_assert_pet_output_available
_materialize_pet_csv
_progress_advance 1 "Materialized pet.csv"
_prepare_analytics_pet_inputs

echo "====== Step 3: Generate Analytics (parallel, CPU-aware) ======"
for (( ANA_SHARD=0; ANA_SHARD<ANALYTICS_SHARD_COUNT; ANA_SHARD++ )); do
    _launch "$COMPUTE_JOB_LIMIT" "Analytics shard $(( ANA_SHARD + 1 ))/${ANALYTICS_SHARD_COUNT}" _run_python generate_analytics.py "${ANALYTICS_GENERATE_ARGS[@]}" --shard-index "$ANA_SHARD" --shard-count "$ANALYTICS_SHARD_COUNT" --max-workers 1 --allow-incomplete-years
done
_wait_phase "generate-analytics"
_validate_analytics_outputs

if [[ "$RUN_LOCAL_SKIP_DB_LOAD" == "1" || -z "${SUPABASE_DB_URI:-}" ]]; then
    echo "====== Step 4-6: Skipping DB load ======"
    echo "====== Pipeline complete! ======"
    exit 0
fi

echo "====== Step 4: Prepare DB load ======"
_run_python - <<'PY'
import os, sys, psycopg2
from pathlib import Path
db_uri = os.environ.get("SUPABASE_DB_URI")
if not db_uri: sys.exit(0)
conn = psycopg2.connect(db_uri)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute(Path("drop_views.sql").read_text(encoding="utf-8"))
    cur.execute(Path("create_tables.sql").read_text(encoding="utf-8"))
    cur.execute("TRUNCATE TABLE locations, pet, pet_forecast CASCADE")
PY

_run_python load.py --append-only --skip-drop-views --skip-create-views --skip-table pet --skip-table pet_forecast
_progress_advance 1 "Prepared database and loaded locations"

echo "====== Step 5: Load shards to DB (parallel, CPU-aware) ======"
for (( LOAD_SHARD=0; LOAD_SHARD<ANALYTICS_SHARD_COUNT; LOAD_SHARD++ )); do
    LOAD_ARGS=(
        load.py
        --append-only
        --skip-drop-views
        --skip-create-views
        --skip-table locations
        --pet-csv "$DB_LOAD_PET_CSV"
        --analytics-shard-count "$ANALYTICS_SHARD_COUNT"
        --load-shard-index "$LOAD_SHARD"
        --load-shard-count "$ANALYTICS_SHARD_COUNT"
    )
    if [[ "$DB_LOAD_PREFER_PET_CSV" == "1" ]]; then
        LOAD_ARGS+=(--prefer-pet-csv)
    fi
    _launch "$COMPUTE_JOB_LIMIT" "DB load shard $(( LOAD_SHARD + 1 ))/${ANALYTICS_SHARD_COUNT}" _run_python "${LOAD_ARGS[@]}"
done
_wait_phase "load-to-db"

echo "====== Step 6: Recreate views ======"
_run_python load.py --append-only --skip-table locations --skip-table pet --skip-table pet_forecast
_progress_advance 1 "Recreated database views"
echo "====== Pipeline complete! ======"