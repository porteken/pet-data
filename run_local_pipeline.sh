#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKFLOW_PATH=${ACT_WORKFLOW_PATH:-.github/workflows/act_integration_test.yml}
JOB_NAME=${ACT_JOB_NAME:-local-full-pipeline}
RUNNER_IMAGE=${ACT_RUNNER_IMAGE:-ghcr.io/catthehacker/ubuntu:act-latest}
LOG_PATH=${ACT_LOG_PATH:-/tmp/pet-data-act-$(date +%Y%m%d-%H%M%S).log}

usage() {
    cat <<'EOF'
Usage: ./run_local_pipeline.sh [--log PATH] [--workflow PATH] [--job NAME]

Runs the local act-based integration test workflow for the PET pipeline.
Uses synthetic data, MinIO (S3), and PostgreSQL test containers — no
external API credentials required.

Options:
  --log PATH        Write act output to PATH.
  --workflow PATH   Workflow file to run. Default: .github/workflows/act_integration_test.yml
  --job NAME        Job name to run. Default: local-full-pipeline
  --help            Show this help text.

Environment overrides:
  ACT_WORKFLOW_PATH   Override workflow path.
  ACT_JOB_NAME        Override job name.
  ACT_RUNNER_IMAGE    Override the act runner image.
  ACT_LOG_PATH        Override the log path.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --log)
            LOG_PATH=$2
            shift 2
            ;;
        --workflow)
            WORKFLOW_PATH=$2
            shift 2
            ;;
        --job)
            JOB_NAME=$2
            shift 2
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

require_command() {
    local command_name=$1
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "Missing required command: $command_name" >&2
        exit 1
    fi
}

require_command docker
require_command act

cd "$ROOT_DIR"

if [[ ! -f "$WORKFLOW_PATH" ]]; then
    echo "Workflow file not found: $WORKFLOW_PATH" >&2
    exit 1
fi

# Ensure Docker daemon is running
if ! docker info >/dev/null 2>&1; then
    echo "Docker daemon is not running." >&2
    exit 1
fi

# Pre-pull container images to avoid timeouts inside act
echo "Pre-pulling required container images..."
for image in postgres:16 minio/minio:latest; do
    if ! docker image inspect "$image" >/dev/null 2>&1; then
        echo "  Pulling $image..."
        docker pull "$image"
    else
        echo "  $image already cached."
    fi
done

# Clean up any leftover containers from previous runs
for container in act-local-pg act-local-minio; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${container}$"; then
        echo "Cleaning up leftover container: $container"
        docker rm -f "$container" >/dev/null 2>&1 || true
    fi
done

echo ""
echo "Running act workflow: $WORKFLOW_PATH (job: $JOB_NAME)"
echo "Log: $LOG_PATH"
echo ""

act \
    --workflows "$WORKFLOW_PATH" \
    --job "$JOB_NAME" \
    --platform "ubuntu-latest=$RUNNER_IMAGE" \
    --bind \
    --container-daemon-socket /var/run/docker.sock \
    --artifact-server-path /tmp/act-artifacts \
    2>&1 | tee "$LOG_PATH"

EXIT_CODE=${PIPESTATUS[0]}

if [[ $EXIT_CODE -eq 0 ]]; then
    echo ""
    echo "Pipeline test PASSED."
else
    echo ""
    echo "Pipeline test FAILED (exit code: $EXIT_CODE). See log: $LOG_PATH" >&2
fi

exit $EXIT_CODE

mkdir -p "$(dirname "$LOG_PATH")"

echo "Running $JOB_NAME from $WORKFLOW_PATH"
echo "Log: $LOG_PATH"

docker rm -f act-local-pg act-local-minio >/dev/null 2>&1 || true

act workflow_dispatch \
    -W "$WORKFLOW_PATH" \
    -j "$JOB_NAME" \
    -P "ubuntu-latest=$RUNNER_IMAGE" \
    -s "CDSAPI_URL=$CDSAPI_URL" \
    -s "CDSAPI_KEY=$CDSAPI_KEY" \
    2>&1 | tee "$LOG_PATH"

echo "Finished. Log saved to $LOG_PATH"
