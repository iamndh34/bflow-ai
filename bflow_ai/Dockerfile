FROM python:3.11-slim

WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# 1. Copy file requirements trước
COPY requirements.txt ./requirements.txt

# 2. Cài đặt thư viện Python
# THÊM: --default-timeout=1000 để không bị lỗi ReadTimeoutError khi mạng chậm
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# 3. Copy toàn bộ source code sau cùng (để tận dụng Docker Cache cho các bước trên)
COPY . .

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]