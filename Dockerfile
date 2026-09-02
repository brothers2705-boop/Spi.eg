# =============================================================================
# VisionForge AI -- Production Dockerfile
# Portable across Windows, macOS, and Linux hosts via Docker Desktop / Engine.
# =============================================================================

FROM python:3.11-slim AS base

# Prevents Python from writing .pyc files and buffers stdout (cleaner logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# System deps kept minimal: scikit-learn/numpy wheels are prebuilt for slim images,
# so no compiler toolchain is required.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core/ ./core/
COPY dataset/ ./dataset/
COPY .streamlit/ ./.streamlit/
COPY app.py .

# Local SQLite database lives here; declared as a volume in docker-compose.yml
RUN mkdir -p /app/data

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Runs as a non-root user for defense-in-depth
RUN useradd --create-home --uid 1000 visionforge && \
    chown -R visionforge:visionforge /app
USER visionforge

ENTRYPOINT ["streamlit", "run", "app.py"]
CMD ["--server.port=8501", "--server.address=0.0.0.0"]
