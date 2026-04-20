# Use an official, lightweight Python 3.10 image
FROM python:3.10-slim

# Keep Python from buffering stdout/stderr so logs appear instantly in Google Cloud
ENV PYTHONUNBUFFERED=1
# Keep Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/app/.venv/bin:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Install uv from Astral's published container image.
COPY --from=ghcr.io/astral-sh/uv:0.9.9 /uv /uvx /bin/

# Install system dependencies (sometimes required for xarray/pandas C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency metadata first to leverage Docker layer caching
COPY pyproject.toml uv.lock README.md ./

# Install Python dependencies into the project virtual environment
RUN uv sync --frozen --no-dev --extra gcs --no-install-project

# Copy the rest of your local project files into the container
COPY . .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Launch the shell script instead of python directly
ENTRYPOINT ["./entrypoint.sh"]