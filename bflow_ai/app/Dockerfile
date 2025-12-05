FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Cài dependencies hệ thống cần cho torch / sklearn
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ✅ Copy requirements từ ROOT
COPY requirements.txt /app/requirements.txt

# ✅ Cài thư viện
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# ✅ Copy source trong thư mục app vào container
COPY app /app

EXPOSE 8010

# ✅ Nếu file của bạn là app/main.py
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
