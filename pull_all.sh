#!/usr/bin/env bash
# Local pipeline using the Cloud Run ERA5 worker for ERA5->PET, then local analytics/db steps.
set -euo pipefail

if [[ -f .env ]]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

USE_UV=${USE_UV:-1}
UV_BIN=${UV_BIN:-uv}
PYTHON_BIN=${PYTHON_BIN:-python}
GCLOUD_BIN=${GCLOUD_BIN:-gcloud}
AWS_BIN=${AWS_BIN:-aws}

_database_configured() {
    if [[ -n "${POSTGRES_DB_URI:-}" || -n "${DATABASE_URL:-}" || -n "${SUPABASE_DB_URI:-}" ]]; then
        return 0
    fi

    [[ -n "${PGHOST:-}" && -n "${PGDATABASE:-}" && -n "${PGUSER:-}" && -n "${PGPASSWORD:-}" ]]
    return $?
}

if ! _database_configured; then
    echo "WARNING: Postgres database credentials are not set. Database upload steps will be skipped."
fi

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAX_THREADS=1
export NUMEXPR_NUM_THREADS=1

MODE=${MODE:-full}
YEARS=${YEARS:-}
SKIP_ERA5_PULL=${SKIP_ERA5_PULL:-0}
SKIP_DB_LOAD=${SKIP_DB_LOAD:-0}
USE_CLOUD_RUN=${USE_CLOUD_RUN:-1}
PROVISION_CLOUD_RUN=${PROVISION_CLOUD_RUN:-0}
SYNC_PET_FROM_S3=${SYNC_PET_FROM_S3:-0}
SKIP_REMOTE_CLEAR=${SKIP_REMOTE_CLEAR:-0}
GCP_REGION=${GCP_REGION:-us-east1}
GCP_PROJECT=${GCP_PROJECT:-}
CLOUD_RUN_JOB=${CLOUD_RUN_JOB:-era5-worker}
S3_BUCKET=${S3_BUCKET:-pet-parquet-files}
S3_PREFIX=${S3_PREFIX:-local-run/${USER:-unknown-user}}
CLEAR_MAX_WORKERS=${CLEAR_MAX_WORKERS:-8}
MERGE_EXISTING_PET_HISTORY=${MERGE_EXISTING_PET_HISTORY:-1}
HISTORY_DB_URI_ENV=${HISTORY_DB_URI_ENV:-}
HISTORY_FALLBACK_DB_URI_ENV=${HISTORY_FALLBACK_DB_URI_ENV:-POSTGRES_DB_URI_PRD}
ALLOW_EMPTY_ANALYTICS=${ALLOW_EMPTY_ANALYTICS:-0}
PROGRESS=${PROGRESS:-1}
ERA5_PARALLEL_STRATEGY=${ERA5_PARALLEL_STRATEGY:-auto}

if [[ "${SKIP_DB_LOAD}" == "1" ]]; then
    echo "WARNING: SKIP_DB_LOAD=1 means this run will not update Postgres."
fi

if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
    echo "WARNING: USE_CLOUD_RUN=1 uses the deployed Cloud Run image, not your local checkout."
    echo "         Use PROVISION_CLOUD_RUN=1 to redeploy it first, or USE_CLOUD_RUN=0 to run local google_era5.py."
fi

REMOTE_BASE_PREFIX=""
REMOTE_OUT_DIR=""
REMOTE_PET_PREFIX=""
HISTORY_EXPORT_CSV="existing_pet.csv"
FULL_PET_CSV="pet_full.csv"
DB_LOAD_PET_CSV="pet.csv"
DB_LOAD_PREFER_PET_CSV=0
MIN_PET_YEARS_FOR_FORECAST=10

if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
    REMOTE_BASE_PREFIX=${S3_PREFIX#/}
    REMOTE_BASE_PREFIX=${REMOTE_BASE_PREFIX%/}
    if [[ -z "${REMOTE_BASE_PREFIX}" ]]; then
        echo "S3_PREFIX must not be empty when USE_CLOUD_RUN=1." >&2
        exit 1
    fi
    REMOTE_OUT_DIR="s3://${S3_BUCKET}/${REMOTE_BASE_PREFIX}"
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
    [[ "${PROGRESS}" == "1" ]]
    return $?
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
        filled=${width}
    fi
    empty=$(( width - filled ))

    printf '%*s' "${filled}" '' | tr ' ' '#'
    printf '%*s' "${empty}" '' | tr ' ' '-'
    return 0
}

_progress_display() {
    local message=$1
    local force=${2:-0}
    local total=${PROGRESS_TOTAL}
    local percent=0

    # shellcheck disable=SC2310
    if ! _progress_enabled; then
        return 0
    fi

    if (( total < 1 )); then
        total=1
    fi
    percent=$(( PROGRESS_CURRENT * 100 / total ))

    if (( PROGRESS_IS_TTY )); then
        # shellcheck disable=SC2312
        printf '\r[%s] %3d%% (%d/%d) %s' \
            "$(_progress_bar "${PROGRESS_CURRENT}" "${total}" "${PROGRESS_BAR_WIDTH}")" \
            "${percent}" \
            "${PROGRESS_CURRENT}" \
            "${total}" \
            "${message}" >&2
        if (( force )) || (( PROGRESS_CURRENT >= total )); then
            printf '\n' >&2
        fi
        return 0
    fi

    if (( force )) || (( percent >= PROGRESS_LAST_PERCENT + 5 )) || (( PROGRESS_CURRENT >= total )); then
        # shellcheck disable=SC2312
        echo "Progress [$(_progress_bar "${PROGRESS_CURRENT}" "${total}" 20)] ${percent}% (${PROGRESS_CURRENT}/${total}) ${message}" >&2
        PROGRESS_LAST_PERCENT=${percent}
    fi
    return 0
}

_progress_note() {
    local message=$1
    _progress_display "${message}" 1
    return $?
}

_progress_advance() {
    local increment=${1:-1}
    local message=${2:-Working...}

    (( PROGRESS_CURRENT += increment ))
    if (( PROGRESS_CURRENT > PROGRESS_TOTAL )); then
        PROGRESS_CURRENT=${PROGRESS_TOTAL}
    fi

    _progress_display "${message}"
    return $?
}

_require_executable() {
    local executable=$1
    if [[ "${executable}" == */* ]]; then
        if [[ -x "${executable}" ]]; then
            return 0
        fi
    elif command -v "${executable}" >/dev/null 2>&1; then
        return 0
    fi

    echo "Required executable not found: ${executable}" >&2
    exit 1
}

_require_python_runner() {
    if [[ "${USE_UV}" == "1" ]]; then
        _require_executable "${UV_BIN}"
    else
        _require_executable "${PYTHON_BIN}"
    fi
    return $?
}

_sync_python_environment() {
    if [[ "${USE_UV}" != "1" ]]; then
        return 0
    fi

    echo "Syncing local project environment with uv..."
    "${UV_BIN}" sync --locked --extra gcs
    return $?
}

_run_python() {
    if [[ "${USE_UV}" == "1" ]]; then
        "${UV_BIN}" run python "$@"
    else
        "${PYTHON_BIN}" "$@"
    fi
    return $?
}

_join_csv_args() {
    local first=1
    local arg
    for arg in "$@"; do
        if (( first )); then
            printf '%s' "${arg}"
            first=0
        else
            printf ',%s' "${arg}"
        fi
    done
    return 0
}

_s3_prefix_has_objects() {
    local uri=$1
    local stripped_uri bucket prefix first_key

    stripped_uri=${uri#s3://}
    if [[ "${stripped_uri}" == "${uri}" || -z "${stripped_uri}" ]]; then
        return 1
    fi

    bucket=${stripped_uri%%/*}
    prefix=""
    if [[ "${stripped_uri}" == */* ]]; then
        prefix=${stripped_uri#*/}
    fi

    first_key=$(
        "${AWS_BIN}" s3api list-objects-v2 \
            --bucket "${bucket}" \
            --prefix "${prefix}" \
            --max-keys 1 \
            --query 'Contents[0].Key' \
            --output text \
            2>/dev/null || true
    )
    [[ -n "${first_key}" && "${first_key}" != "None" ]]
    return $?
}

_local_pet_output_exists() {
    find pet_data_csv -type f -name 'pet_batch_*.parquet' -print -quit | grep -q .
    return $?
}

_assert_pet_output_available() {
    # shellcheck disable=SC2310
    if _local_pet_output_exists; then
        return 0
    fi

    if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
        local legacy_uri="s3://${S3_BUCKET}/${REMOTE_BASE_PREFIX}/era5_data_parquet/"
        # shellcheck disable=SC2310
        if _s3_prefix_has_objects "${legacy_uri}"; then
            echo "Cloud Run job ${CLOUD_RUN_JOB} only produced legacy era5_data_parquet output under ${legacy_uri}. Redeploy the job from this checkout by setting PROVISION_CLOUD_RUN=1 or running ./cloudrun_provision.sh manually." >&2
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
if not any(pet_root.rglob("pet_batch_*.parquet")):
    raise SystemExit("No pet batch parquet files found; cannot materialize pet.csv.")

combined = pd.read_parquet(
    pet_root,
    columns=["location_id", "date", "pet"],
).sort_values(["location_id", "date"])
combined.to_csv("pet.csv", index=False)
print(f"Saved {len(combined)} rows to pet.csv")
PY
    return $?
}

_write_empty_pet_csv() {
    local output_path=$1
    printf 'location_id,date,pet\n' > "${output_path}"
    return $?
}

_count_pet_csv_years() {
    local csv_path=$1
    if [[ ! -f "${csv_path}" ]]; then
        echo 0
        return 0
    fi

    awk -F, '
        NR == 1 {
            for (i = 1; i <= NF; i++) {
                if ($i == "date") {
                    date_col = i
                    break
                }
            }
            next
        }
        date_col && $date_col ~ /^[0-9]{4}-[0-9]{2}-[0-9]{2}$/ {
            print substr($date_col, 1, 4)
        }
    ' "${csv_path}" | sort -u | awk 'END {print NR + 0}'
    return $?
}

_export_history_pet_csv() {
    local env_name=$1
    local output_csv=$2

    if [[ -z "${env_name}" ]]; then
        if ! _database_configured; then
            _write_empty_pet_csv "${output_csv}"
            return 0
        fi

        _run_python historical_pet_update.py export-all "${output_csv}"
        return $?
    fi

    if [[ -z "${!env_name:-}" ]]; then
        _write_empty_pet_csv "${output_csv}"
        return 0
    fi

    POSTGRES_DB_URI="${!env_name}" _run_python historical_pet_update.py export-all "${output_csv}"
    return $?
}

_prepare_analytics_pet_inputs() {
    DB_LOAD_PET_CSV="pet.csv"
    DB_LOAD_PREFER_PET_CSV=0

    if [[ "${SKIP_DB_LOAD}" == "1" ]] || ! _database_configured; then
        return 0
    fi

    if [[ "${MERGE_EXISTING_PET_HISTORY}" != "1" ]]; then
        return 0
    fi

    echo "====== Step 2.75: Merge historical PET for DB load ======"
    rm -f "${HISTORY_EXPORT_CSV}" "${FULL_PET_CSV}"

    local primary_env=${HISTORY_DB_URI_ENV}
    local fallback_env=${HISTORY_FALLBACK_DB_URI_ENV}

    _export_history_pet_csv "${primary_env}" "${HISTORY_EXPORT_CSV}"
    _run_python historical_pet_update.py merge "${FULL_PET_CSV}" "${HISTORY_EXPORT_CSV}" --dirs pet_data_csv

    local merged_year_count
    merged_year_count=$(_count_pet_csv_years "${FULL_PET_CSV}")
    if (( merged_year_count < MIN_PET_YEARS_FOR_FORECAST )) && [[ -n "${fallback_env}" ]] && [[ "${fallback_env}" != "${primary_env}" ]] && [[ -n "${!fallback_env:-}" ]]; then
        echo "Primary history source did not provide enough PET history; retrying with ${fallback_env}."
        _export_history_pet_csv "${fallback_env}" "${HISTORY_EXPORT_CSV}"
        _run_python historical_pet_update.py merge "${FULL_PET_CSV}" "${HISTORY_EXPORT_CSV}" --dirs pet_data_csv
        merged_year_count=$(_count_pet_csv_years "${FULL_PET_CSV}")
    fi

    if (( merged_year_count < MIN_PET_YEARS_FOR_FORECAST )); then
        echo "Need at least ${MIN_PET_YEARS_FOR_FORECAST} PET years to derive pet_forecast (and the downstream city_rankings_view change fields). Found ${merged_year_count} year(s) in ${FULL_PET_CSV}." >&2
        echo "Either widen the historical source or set MERGE_EXISTING_PET_HISTORY=0 and accept empty analytics." >&2
        exit 1
    fi

    DB_LOAD_PET_CSV="${FULL_PET_CSV}"
    DB_LOAD_PREFER_PET_CSV=1
    _progress_advance 1 "Merged historical PET inputs"
    return 0
}

_clear_remote_pet_prefix() {
    echo "Clearing remote Cloud Run output at s3://${S3_BUCKET}/${REMOTE_PET_PREFIX}/"
    _run_python clear_s3_prefix.py \
        --bucket "${S3_BUCKET}" \
        --prefix "${REMOTE_PET_PREFIX}/" \
        --max-workers "${CLEAR_MAX_WORKERS}"
    return $?
}

_sync_pet_from_s3() {
    echo "Syncing PET parquet from s3://${S3_BUCKET}/${REMOTE_PET_PREFIX}/"
    mkdir -p pet_data_csv
    "${AWS_BIN}" s3 sync \
        "s3://${S3_BUCKET}/${REMOTE_PET_PREFIX}/" \
        ./pet_data_csv/ \
        --delete
    return $?
}

_cancel_running_cloud_run_executions() {
    local cancel_args=(
        cancel_cloud_run_job_executions.py
        --job "${CLOUD_RUN_JOB}"
        --region "${GCP_REGION}"
    )

    if [[ -n "${GCP_PROJECT}" ]]; then
        cancel_args+=(--project "${GCP_PROJECT}")
    fi

    _run_python "${cancel_args[@]}"
    return $?
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
    return 0
}

_kill_tree() {
    local pid=$1 sig=${2:-TERM} children
    children=$(pgrep -P "${pid}" 2>/dev/null || true)
    if [[ -n "${children}" ]]; then for c in ${children}; do _kill_tree "${c}" "${sig}"; done; fi
    kill -"${sig}" "${pid}" 2>/dev/null || true
    return 0
}

_cleanup() {
    local exit_code=$?
    trap - SIGINT SIGTERM ERR EXIT
    if (( exit_code != 0 && ${#_PIDS[@]} > 0 )); then
        for pid in "${_PIDS[@]}"; do _kill_tree "${pid}" TERM; done
        sleep 1
        for pid in "${_PIDS[@]}"; do _kill_tree "${pid}" KILL; done
    fi
    if (( exit_code != 0 )); then
        exit "${exit_code}"
    fi
    return 0
}
trap _cleanup SIGINT SIGTERM ERR EXIT

_remove_pid_from_tracking() {
    local finished_pid=$1
    local active_pids=()

    for pid in "${_PIDS[@]}"; do
        if [[ "${pid}" != "${finished_pid}" ]]; then
            active_pids+=("${pid}")
        fi
    done

    _PIDS=()
    if (( ${#active_pids[@]} > 0 )); then
        _PIDS=("${active_pids[@]}")
    fi
    return 0
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

    if [[ -n "${finished_pid}" ]]; then
        label=${_PID_PROGRESS_LABELS[${finished_pid}]:-background task}
        _remove_pid_from_tracking "${finished_pid}"
        unset "_PID_PROGRESS_LABELS[${finished_pid}]"
    fi

    if (( exit_status != 0 )); then
        echo "Background job failed: ${label}" >&2
        exit "${exit_status}"
    fi

    _progress_advance 1 "${label}"
    return 0
}

_launch() {
    local max=$1
    local label=$2
    shift 2

    "$@" &
    local pid=$!
    _PIDS+=("${pid}")
    _PID_PROGRESS_LABELS[${pid}]="${label}"

    while (( ${#_PIDS[@]} >= max )); do
        _wait_for_any_pid
    done
    return 0
}

_wait_phase() {
    local label=${1:-background jobs}

    if (( ${#_PIDS[@]} > 0 )); then
        echo "Waiting for ${label} (${#_PIDS[@]} task(s) remaining)..."
    fi

    while (( ${#_PIDS[@]} > 0 )); do
        _wait_for_any_pid
    done
    return 0
}

_resolve_era5_parallel_strategy() {
    if [[ "${ERA5_PARALLEL_STRATEGY}" == "auto" ]]; then
        if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
            echo "cloud-run-scale-out"
        elif [[ "${MODE}" == "smoke" ]]; then
            echo "local-smoke"
        else
            echo "local-balanced"
        fi
        return 0
    fi

    case "${ERA5_PARALLEL_STRATEGY}" in
        cloud-run-scale-out|local-balanced|local-smoke|serial)
            echo "${ERA5_PARALLEL_STRATEGY}"
            ;;
        *)
            echo "ERA5_PARALLEL_STRATEGY must be one of auto, cloud-run-scale-out, local-balanced, local-smoke, serial." >&2
            exit 1
            ;;
    esac
    return 0
}

_calculate_progress_total() {
    local total=1
    local era5_task_count=0

    if [[ "${SKIP_ERA5_PULL}" != "1" ]]; then
        if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
            (( total += 1 ))
            if [[ "${PROVISION_CLOUD_RUN}" == "1" ]]; then
                (( total += 1 ))
            fi
            if [[ "${SKIP_REMOTE_CLEAR}" != "1" ]]; then
                (( total += 1 ))
            fi
        fi

        era5_task_count=$(( YEAR_COUNT * ERA5_CITY_SHARD_COUNT * ${#ERA5_TIME_SHARDS[@]} ))
        (( total += era5_task_count ))

        if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
            (( total += 1 ))
        fi
    elif [[ "${USE_CLOUD_RUN}" == "1" && "${SYNC_PET_FROM_S3}" == "1" ]]; then
        (( total += 1 ))
    fi

    (( total += 1 ))

    if [[ "${SKIP_DB_LOAD}" != "1" ]] && _database_configured && [[ "${MERGE_EXISTING_PET_HISTORY}" == "1" ]]; then
        (( total += 1 ))
    fi

    if [[ "${SKIP_DB_LOAD}" != "1" ]] && _database_configured; then
        (( total += 1 ))
        (( total += ANALYTICS_SHARD_COUNT ))
        (( total += 1 ))
    fi

    echo "${total}"
    return 0
}

_cpu_count() {
    getconf _NPROCESSORS_ONLN 2>/dev/null || nproc 2>/dev/null || echo 1
    return $?
}
CPU_COUNT=$(_cpu_count)
AVAILABLE_CPUS=$(( CPU_COUNT > 1 ? CPU_COUNT - 1 : 1 ))
COMPUTE_JOB_LIMIT=$(( AVAILABLE_CPUS < 8 ? AVAILABLE_CPUS : 8 ))

if [[ -n "${YEARS}" ]]; then
    read -r -a ALL_YEARS_ARRAY <<< "${YEARS}"
else
    DEFAULT_ERA5_END_YEAR=$(( $(date -u +%Y) - 1 ))
    if (( DEFAULT_ERA5_END_YEAR < 2000 )); then
        echo "Computed default ERA5 end year ${DEFAULT_ERA5_END_YEAR} is earlier than 2000; set YEARS explicitly." >&2
        exit 1
    fi

    ALL_YEARS_ARRAY=()
    for (( YEAR=2000; YEAR<=DEFAULT_ERA5_END_YEAR; YEAR++ )); do
        ALL_YEARS_ARRAY+=("${YEAR}")
    done
fi

if (( ${#ALL_YEARS_ARRAY[@]} == 0 )); then
    echo "YEARS must contain at least one year." >&2
    exit 1
fi

ALL_YEARS=${ALL_YEARS_ARRAY[*]}
YEAR_COUNT=${#ALL_YEARS_ARRAY[@]}

if [[ "${MODE}" == "smoke" ]]; then
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-168}
    ERA5_TIME_SHARD_COUNT=${ERA5_TIME_SHARD_COUNT:-53}
    ERA5_TIME_SHARDS=("${SMOKE_TIME_SHARD_INDEX:-17}")
    echo "Running local pipeline in smoke mode: ${ALL_YEARS} ERA5 week only."
else
    ERA5_BATCH_HOURS=${ERA5_BATCH_HOURS:-720}
    ERA5_TIME_SHARD_COUNT=${ERA5_TIME_SHARD_COUNT:-13}
    for (( TIME_SHARD=0; TIME_SHARD<ERA5_TIME_SHARD_COUNT; TIME_SHARD++ )); do
        ERA5_TIME_SHARDS+=("${TIME_SHARD}")
    done
    full_mode_label="local compute fallback"
    if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
        full_mode_label="Cloud Run"
    fi
    echo "Running local pipeline in full mode via ${full_mode_label}: ${ALL_YEARS} (full year)."
fi

export ALL_YEARS

_require_python_runner
_sync_python_environment

echo "====== Step 1: Compute year ranges + setup locations ======"
mkdir -p output_tiles pet_data_csv analytics_data_csv
if [[ "${SKIP_ERA5_PULL}" != "1" ]]; then
    for YEAR in "${ALL_YEARS_ARRAY[@]}"; do rm -rf "pet_data_csv/year=${YEAR}"; done
fi
rm -rf analytics_data_csv/shard_count=*

_run_python cities.py

CITY_COUNT=$(awk 'END {print (NR > 0 ? NR - 1 : 0)}' cities.csv)
if (( CITY_COUNT < 1 )); then
    echo "cities.py produced zero locations; cannot continue." >&2
    exit 1
fi

if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
    DEFAULT_ERA5_CITY_SHARD_COUNT=1
else
    DEFAULT_ERA5_CITY_SHARD_COUNT=$(( CITY_COUNT < 2 ? CITY_COUNT : 2 ))
fi
ERA5_CITY_SHARD_COUNT=${ERA5_CITY_SHARD_COUNT:-${DEFAULT_ERA5_CITY_SHARD_COUNT}}
if (( ERA5_CITY_SHARD_COUNT < 1 )); then ERA5_CITY_SHARD_COUNT=1; fi
if (( ERA5_CITY_SHARD_COUNT > CITY_COUNT )); then ERA5_CITY_SHARD_COUNT=${CITY_COUNT}; fi

ANALYTICS_SHARD_COUNT=$(( CITY_COUNT < COMPUTE_JOB_LIMIT ? CITY_COUNT : COMPUTE_JOB_LIMIT ))
ERA5_PARALLEL_STRATEGY=$(_resolve_era5_parallel_strategy)

case "${ERA5_PARALLEL_STRATEGY}" in
    cloud-run-scale-out)
        DEFAULT_CLOUD_RUN_JOB_LIMIT=$(( ERA5_CITY_SHARD_COUNT * ${#ERA5_TIME_SHARDS[@]} ))
        if (( DEFAULT_CLOUD_RUN_JOB_LIMIT > 8 )); then
            DEFAULT_CLOUD_RUN_JOB_LIMIT=8
        fi
        ERA5_JOB_LIMIT=${ERA5_JOB_LIMIT:-${DEFAULT_CLOUD_RUN_JOB_LIMIT}}
        ERA5_CONCURRENCY_PROFILE=${ERA5_CONCURRENCY_PROFILE:-aggressive}
        ERA5_BATCH_WORKERS=${ERA5_BATCH_WORKERS:-1}
        ;;
    local-smoke)
        DEFAULT_SMOKE_BATCH_WORKERS=$(( AVAILABLE_CPUS / ERA5_CITY_SHARD_COUNT ))
        if (( DEFAULT_SMOKE_BATCH_WORKERS < 1 )); then
            DEFAULT_SMOKE_BATCH_WORKERS=1
        fi
        if (( DEFAULT_SMOKE_BATCH_WORKERS > 2 )); then
            DEFAULT_SMOKE_BATCH_WORKERS=2
        fi
        ERA5_JOB_LIMIT=${ERA5_JOB_LIMIT:-${ERA5_CITY_SHARD_COUNT}}
        ERA5_CONCURRENCY_PROFILE=${ERA5_CONCURRENCY_PROFILE:-aggressive}
        ERA5_BATCH_WORKERS=${ERA5_BATCH_WORKERS:-${DEFAULT_SMOKE_BATCH_WORKERS}}
        ;;
    local-balanced)
        ERA5_JOB_LIMIT=${ERA5_JOB_LIMIT:-${ERA5_CITY_SHARD_COUNT}}
        ERA5_CONCURRENCY_PROFILE=${ERA5_CONCURRENCY_PROFILE:-balanced}
        ERA5_BATCH_WORKERS=${ERA5_BATCH_WORKERS:-1}
        ;;
    serial)
        ERA5_JOB_LIMIT=${ERA5_JOB_LIMIT:-1}
        ERA5_CONCURRENCY_PROFILE=${ERA5_CONCURRENCY_PROFILE:-conservative}
        ERA5_BATCH_WORKERS=${ERA5_BATCH_WORKERS:-1}
        ;;
    *)
        echo "Unexpected ERA5_PARALLEL_STRATEGY: ${ERA5_PARALLEL_STRATEGY}" >&2
        exit 1
        ;;
esac

if (( ERA5_JOB_LIMIT < 1 )); then ERA5_JOB_LIMIT=1; fi
if (( ERA5_BATCH_WORKERS < 1 )); then ERA5_BATCH_WORKERS=1; fi

PROGRESS_TOTAL=$(_calculate_progress_total)
_progress_note "Initialized pipeline progress tracking for ${PROGRESS_TOTAL} work unit(s)."
echo "Using ERA5 parallel strategy: ${ERA5_PARALLEL_STRATEGY} (job limit=${ERA5_JOB_LIMIT}, batch workers=${ERA5_BATCH_WORKERS}, profile=${ERA5_CONCURRENCY_PROFILE})"
_progress_advance 1 "Prepared locations"

echo "====== Step 2: Compute ERA5 + PET ======"
if [[ "${SKIP_ERA5_PULL}" != "1" ]]; then
    if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
        _require_executable "${GCLOUD_BIN}"
        _require_executable "${AWS_BIN}"
        _cancel_running_cloud_run_executions
        _progress_advance 1 "Cancelled stale Cloud Run executions"
        echo "Using Cloud Run job ${CLOUD_RUN_JOB} in ${GCP_REGION} with output ${REMOTE_OUT_DIR}"
        if [[ "${PROVISION_CLOUD_RUN}" == "1" ]]; then
            echo "Refreshing Cloud Run job ${CLOUD_RUN_JOB} from the current checkout"
            /bin/bash ./cloudrun_provision.sh
            _progress_advance 1 "Provisioned Cloud Run worker"
        fi
        if [[ "${SKIP_REMOTE_CLEAR}" != "1" ]]; then
            _clear_remote_pet_prefix
            _progress_advance 1 "Cleared remote PET prefix"
        fi
    fi

    for YEAR in "${ALL_YEARS_ARRAY[@]}"; do
        for (( CITY_SHARD=0; CITY_SHARD<ERA5_CITY_SHARD_COUNT; CITY_SHARD++ )); do
            for TIME_SHARD in "${ERA5_TIME_SHARDS[@]}"; do
                SHARD_LABEL="ERA5 year=${YEAR} city-shard=$(( CITY_SHARD + 1 ))/${ERA5_CITY_SHARD_COUNT} time-shard=${TIME_SHARD}/${ERA5_TIME_SHARD_COUNT}"
                if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
                    CLOUD_RUN_ARGS=$(_build_cloud_run_args_csv \
                        "${YEAR}" \
                        "${CITY_SHARD}" \
                        "${ERA5_CITY_SHARD_COUNT}" \
                        "${TIME_SHARD}" \
                        "${ERA5_TIME_SHARD_COUNT}")
                    GCLOUD_CMD=(
                        "${GCLOUD_BIN}" run jobs execute "${CLOUD_RUN_JOB}"
                        --region "${GCP_REGION}"
                        --wait
                        "--args=${CLOUD_RUN_ARGS}"
                    )
                    if [[ -n "${GCP_PROJECT}" ]]; then
                        GCLOUD_CMD+=(--project "${GCP_PROJECT}")
                    fi
                    _launch "${ERA5_JOB_LIMIT}" "${SHARD_LABEL}" "${GCLOUD_CMD[@]}"
                else
                    ERA5_ARGS=(
                        --year "${YEAR}"
                        --city-shard-count "${ERA5_CITY_SHARD_COUNT}"
                        --city-shard-index "${CITY_SHARD}"
                        --time-shard-index "${TIME_SHARD}"
                        --time-shard-count "${ERA5_TIME_SHARD_COUNT}"
                        --batch-hours "${ERA5_BATCH_HOURS}"
                        --max-workers "${ERA5_BATCH_WORKERS}"
                        --concurrency-profile "${ERA5_CONCURRENCY_PROFILE}"
                        --out-dir .
                    )
                    if (( ${#ERA5_MONTHS[@]} > 0 )); then
                        ERA5_ARGS+=(--months "${ERA5_MONTHS[@]}")
                    fi
                    _launch "${ERA5_JOB_LIMIT}" "${SHARD_LABEL}" _run_python google_era5.py "${ERA5_ARGS[@]}"
                fi
            done
        done
    done
    _wait_phase "pull-google-era5-pet"

    if [[ "${USE_CLOUD_RUN}" == "1" ]]; then
        _sync_pet_from_s3
        _progress_advance 1 "Synced PET shards from S3"
    fi
elif [[ "${USE_CLOUD_RUN}" == "1" && "${SYNC_PET_FROM_S3}" == "1" ]]; then
    _require_executable "${AWS_BIN}"
    _sync_pet_from_s3
    _progress_advance 1 "Synced PET shards from S3"
fi

_assert_pet_output_available
_materialize_pet_csv
_progress_advance 1 "Materialized pet.csv"
_prepare_analytics_pet_inputs

if [[ "${SKIP_DB_LOAD}" == "1" ]] || ! _database_configured; then
    echo "====== Step 3-5: Skipping DB load ======"
    echo "====== Pipeline complete! ======"
    exit 0
fi

echo "====== Step 3: Prepare DB load ======"
_run_python - <<'PY'
import sys
from pathlib import Path

import psycopg

from shared_config import resolve_database_uri

db_uri = resolve_database_uri()
if not db_uri: sys.exit(0)
with psycopg.connect(db_uri) as conn:
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute(Path("drop_views.sql").read_text(encoding="utf-8"))
        cur.execute(Path("create_tables.sql").read_text(encoding="utf-8"))
        cur.execute("TRUNCATE TABLE locations, pet CASCADE")
PY

_run_python load.py --append-only --skip-drop-views --skip-create-views --skip-table pet
_progress_advance 1 "Prepared database and loaded locations"

echo "====== Step 4: Load shards to DB (parallel, CPU-aware) ======"
for (( LOAD_SHARD=0; LOAD_SHARD<ANALYTICS_SHARD_COUNT; LOAD_SHARD++ )); do
    LOAD_ARGS=(
        load.py
        --append-only
        --skip-drop-views
        --skip-create-views
        --skip-table locations
        --pet-csv "${DB_LOAD_PET_CSV}"
        --load-shard-index "${LOAD_SHARD}"
        --load-shard-count "${ANALYTICS_SHARD_COUNT}"
    )
    if [[ "${DB_LOAD_PREFER_PET_CSV}" == "1" ]]; then
        LOAD_ARGS+=(--prefer-pet-csv)
    fi
    _launch "${COMPUTE_JOB_LIMIT}" "DB load shard $(( LOAD_SHARD + 1 ))/${ANALYTICS_SHARD_COUNT}" _run_python "${LOAD_ARGS[@]}"
done
_wait_phase "load-to-db"

echo "====== Step 5: Recreate views ======"
_run_python load.py --append-only --skip-table locations --skip-table pet
_progress_advance 1 "Recreated database views"
echo "====== Pipeline complete! ======"