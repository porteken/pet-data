#!/usr/bin/env bash
# 1. Build and push your docker image
gcloud builds submit --tag gcr.io/weather-ai-478502/era5-worker

# 2. Create the job, giving it AWS credentials to write to S3
gcloud run jobs create era5-worker \
  --image=gcr.io/weather-ai-478502/era5-worker \
  --region=us-east1 \
  --cpu=2 \
  --memory=4Gi \
  --max-retries=1 \
  --task-timeout=3600s \
  --set-env-vars=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID,AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY,AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION