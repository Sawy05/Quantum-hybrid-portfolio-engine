# Production Dockerfile for Hybrid Quantum-Classical Quantitative Engine
FROM python:3.11-slim

# Environment setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_HOME=/app

WORKDIR $APP_HOME

# Install build essentials for C++ extensions if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration and pyproject
COPY pyproject.toml README.md ./

# Install python dependencies
RUN pip install --upgrade pip && \
    pip install .

# Copy source code, configs, and scripts
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Create unprivileged user for security
RUN useradd -m -u 1000 quantumuser && \
    chown -R quantumuser:quantumuser $APP_HOME
USER quantumuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run API server
CMD ["uvicorn", "src.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
