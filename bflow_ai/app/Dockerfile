FROM python:3.11-slim

WORKDIR /app

# Cài dependencies hệ thống cần thiết (nếu có thư viện build)
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Cài thư viện
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code
COPY . .

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
