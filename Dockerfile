FROM python:3.10-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY uv.lock ./

RUN python -m venv "$VIRTUAL_ENV"

RUN python - <<'PY' > /tmp/cloudrun-worker-requirements.txt
from pathlib import Path

WORKER_PACKAGES = (
    "dask",
    "gcsfs",
    "numpy",
    "pandas",
    "pyarrow",
    "thermofeel",
    "tqdm",
    "xarray",
    "zarr",
)

versions: dict[str, str] = {}
current_name: str | None = None
for raw_line in Path("uv.lock").read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line == "[[package]]":
        current_name = None
        continue
    if line.startswith('name = "'):
        current_name = line.split('"', maxsplit=2)[1]
        continue
    if current_name in WORKER_PACKAGES and line.startswith('version = "'):
        versions[current_name] = line.split('"', maxsplit=2)[1]
        current_name = None

missing = [name for name in WORKER_PACKAGES if name not in versions]
if missing:
    raise SystemExit(f"Missing Cloud Run worker packages in uv.lock: {missing}")

for package_name in WORKER_PACKAGES:
    print(f"{package_name}=={versions[package_name]}")
PY

RUN uv pip install --python "$VIRTUAL_ENV/bin/python" --requirement /tmp/cloudrun-worker-requirements.txt # NOSONAR

RUN "$VIRTUAL_ENV/bin/python" - <<'PY'
import dask
import gcsfs
import numpy
import pandas
import pyarrow
import thermofeel
import tqdm
import xarray
import zarr

print("Cloud Run worker dependencies verified.")
PY

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY entrypoint.sh cities.py google_era5.py pet_corrected.py shards.py ./

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app \
    && chmod +x entrypoint.sh

USER appuser

ENTRYPOINT ["./entrypoint.sh"]