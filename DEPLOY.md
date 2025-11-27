# Deployment Guide

This guide explains how to redeploy the project, replacing the old Python backend with the new Rust backend.

## Prerequisites

- Google Cloud SDK (`gcloud`) installed and authenticated.
- Firebase CLI (`firebase`) installed and authenticated.
- Docker (optional, for local testing).

## 1. Deploy Backend (Cloud Run)

We will use Cloud Build to build the Rust Docker image and deploy it to Cloud Run.

### Step 1: Submit Build

Run the following command from the project root. Replace `YOUR_REGION` and `YOUR_REPO_NAME` with your actual values (e.g., `asia-northeast1`, `my-repo`).

```bash
gcloud builds submit --config cloudbuild.yaml . \
    --substitutions=_REGION="asia-northeast1",_REPO_NAME="fibonacci-repo"
```

_Note: If you haven't created an Artifact Registry repository yet, create one:_

```bash
gcloud artifacts repositories create fibonacci-repo \
    --repository-format=docker \
    --location=asia-northeast1 \
    --description="Docker repository for Fibonacci Spiral"
```

### Step 2: Get Backend URL

After the build and deployment complete successfully, get the URL of the deployed service:

```bash
gcloud run services describe rust-backend --platform managed --region asia-northeast1 --format "value(status.url)"
```

## Cleanup (Optional)

If you want to remove the old Python service:

```bash
gcloud run services delete python-backend --region asia-northeast1
```

L
