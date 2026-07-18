#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  . .env
  set +a
fi

GCP_PROJECT_ID=${GCP_PROJECT_ID:-weather-ai-478502}
CLOUD_RUN_JOB_NAME=${CLOUD_RUN_JOB_NAME:-era5-worker}
CLOUD_RUN_REGION=${CLOUD_RUN_REGION:-us-east1}
CLOUD_RUN_JOB_CPU=${CLOUD_RUN_JOB_CPU:-2}
CLOUD_RUN_JOB_MEMORY=${CLOUD_RUN_JOB_MEMORY:-8Gi}
CLOUD_RUN_TASK_TIMEOUT=${CLOUD_RUN_TASK_TIMEOUT:-3600s}
CLOUD_RUN_MAX_RETRIES=${CLOUD_RUN_MAX_RETRIES:-1}
CLOUD_RUN_IMAGE=${CLOUD_RUN_IMAGE:-gcr.io/${GCP_PROJECT_ID}/${CLOUD_RUN_JOB_NAME}}
AWS_CLOUD_RUN_REGION=${AWS_DEFAULT_REGION:-${AWS_REGION:-}}
AWS_KEY_SECRET_NAME=${AWS_KEY_SECRET_NAME:-${CLOUD_RUN_JOB_NAME}-aws-access-key-id}
AWS_SECRET_SECRET_NAME=${AWS_SECRET_SECRET_NAME:-${CLOUD_RUN_JOB_NAME}-aws-secret-access-key}

: "${AWS_ACCESS_KEY_ID:?AWS_ACCESS_KEY_ID must be set}"
: "${AWS_SECRET_ACCESS_KEY:?AWS_SECRET_ACCESS_KEY must be set}"
: "${AWS_CLOUD_RUN_REGION:?AWS_DEFAULT_REGION or AWS_REGION must be set}"

_upsert_secret() {
    local secret_name=$1
    local secret_value=$2
    if gcloud secrets describe "${secret_name}" --project "${GCP_PROJECT_ID}" >/dev/null 2>&1; then
        printf '%s' "${secret_value}" | gcloud secrets versions add "${secret_name}" \
            --project "${GCP_PROJECT_ID}" --data-file=-
    else
        printf '%s' "${secret_value}" | gcloud secrets create "${secret_name}" \
            --project "${GCP_PROJECT_ID}" --replication-policy=automatic --data-file=-
    fi
    return 0
}

_grant_secret_access() {
    local secret_name=$1
    local service_account=$2
    if ! gcloud secrets add-iam-policy-binding "${secret_name}" \
        --project "${GCP_PROJECT_ID}" \
        --member="serviceAccount:${service_account}" \
        --role=roles/secretmanager.secretAccessor >/dev/null; then
        echo "WARNING: could not grant secretAccessor on ${secret_name} to ${service_account}." >&2
        echo "         Grant it manually or the job will fail to start." >&2
    fi
    return 0
}

echo "Upserting AWS credential secrets in Secret Manager"
_upsert_secret "${AWS_KEY_SECRET_NAME}" "${AWS_ACCESS_KEY_ID}"
_upsert_secret "${AWS_SECRET_SECRET_NAME}" "${AWS_SECRET_ACCESS_KEY}"

PROJECT_NUMBER=$(gcloud projects describe "${GCP_PROJECT_ID}" --format='value(projectNumber)')
JOB_SERVICE_ACCOUNT=${JOB_SERVICE_ACCOUNT:-"${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"}
_grant_secret_access "${AWS_KEY_SECRET_NAME}" "${JOB_SERVICE_ACCOUNT}"
_grant_secret_access "${AWS_SECRET_SECRET_NAME}" "${JOB_SERVICE_ACCOUNT}"

echo "Building Cloud Run image ${CLOUD_RUN_IMAGE}"
gcloud auth configure-docker --quiet
docker build --pull -t "${CLOUD_RUN_IMAGE}" .
docker push "${CLOUD_RUN_IMAGE}"

echo "Deploying Cloud Run job ${CLOUD_RUN_JOB_NAME} in ${CLOUD_RUN_REGION}"
gcloud run jobs deploy "${CLOUD_RUN_JOB_NAME}" \
  --project "${GCP_PROJECT_ID}" \
  --image "${CLOUD_RUN_IMAGE}" \
  --region "${CLOUD_RUN_REGION}" \
  --cpu "${CLOUD_RUN_JOB_CPU}" \
  --memory "${CLOUD_RUN_JOB_MEMORY}" \
  --max-retries "${CLOUD_RUN_MAX_RETRIES}" \
  --task-timeout "${CLOUD_RUN_TASK_TIMEOUT}" \
  --set-env-vars="AWS_DEFAULT_REGION=${AWS_CLOUD_RUN_REGION},AWS_REGION=${AWS_CLOUD_RUN_REGION}" \
  --set-secrets="AWS_ACCESS_KEY_ID=${AWS_KEY_SECRET_NAME}:latest,AWS_SECRET_ACCESS_KEY=${AWS_SECRET_SECRET_NAME}:latest"
