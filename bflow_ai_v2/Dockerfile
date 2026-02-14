# =============================================================================
# bflow_ai_v2 - Dockerfile
# =============================================================================
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# Copy requirements first (for Docker cache)
COPY requirements.txt ./requirements.txt

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Switch to non-root user
USER appuser

EXPOSE 8010

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8010/api/coa/health || exit 1

# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
