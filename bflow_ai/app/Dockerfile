FROM python:3.11-slim

WORKDIR /app

# Cài dependency hệ thống
RUN apt-get update && apt-get install -y gcc curl && rm -rf /var/lib/apt/lists/*

# Copy requirements từ THƯ MỤC CHA
COPY ../requirements.txt ./requirements.txt

# Cài thư viện
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code trong thư mục app/
COPY . .

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]
