FROM python:3.11-slim

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# Tạo non-root user cho security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# 1. Copy file requirements trước
COPY requirements.txt ./requirements.txt

# 2. Cài đặt thư viện Python
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 3. Copy toàn bộ source code sau cùng (để tận dụng Docker Cache cho các bước trên)
COPY . .

# Chuyển sang non-root user
USER appuser

EXPOSE 8010

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8010/ || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]