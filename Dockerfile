# Use an official, lightweight Python 3.10 image
FROM python:3.10-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Keep Python from buffering stdout/stderr so logs appear instantly in Google Cloud
ENV PYTHONUNBUFFERED=1
# Keep Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies (sometimes required for xarray/pandas C-extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml first to leverage Docker layer caching
COPY pyproject.toml ./

# Install Python dependencies using uv
# We use --system to install into the system python since we're in a container
RUN uv pip install --system --no-cache .

# Copy the rest of your local project files into the container
COPY . .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Launch the shell script instead of python directly
ENTRYPOINT ["./entrypoint.sh"]
