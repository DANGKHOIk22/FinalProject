# Stage 1: Build Stage
# Use a Python 3.11 slim image for a smaller footprint
FROM python:3.11-slim-bookworm AS builder

# Set environment variables to prevent Python from buffering stdout/stderr and writing .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# - build-essential for compiling some Python packages
# - libpq-dev for PostgreSQL client development files (needed by psycopg[binary])
# - ffmpeg for video processing
RUN apt-get update && apt-get install -y build-essential libpq-dev ffmpeg git && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install uv

# Set the working directory inside the container
WORKDIR /app

# Copy dependency definition files first to leverage Docker's build cache
COPY pyproject.toml ./

# Install Python dependencies using uv
# --system installs into the system site-packages
# uv.lock ensures reproducible builds
RUN uv pip install --system .

# Stage 2: Production Stage
# Use a fresh, minimal Python 3.11 slim image
FROM python:3.11-slim-bookworm

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install runtime system dependencies
# - libpq5 for PostgreSQL client libraries (runtime dependency)
# - ffmpeg for video processing (runtime dependency)
RUN apt-get update && apt-get install -y libpq5 ffmpeg && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy Python executables from the builder stage
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Copy the entire application code into the container
# Ensure .dockerignore is used to exclude unnecessary files like .git, .venv, __pycache__, .env
COPY . /app

# Expose the port that FastAPI listens on
EXPOSE 8000

# Command to run the application using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
