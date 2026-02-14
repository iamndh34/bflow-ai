# Production Docker Build Guide

Hướng dẫn build và deploy BFLOW AI bằng Docker Compose cho môi trường production.

---

## Table of Contents

- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cấu trúc project](#cấu-trúc-project)
- [Chuẩn bị môi trường](#chuẩn-bị-môi-trường)
- [Build & Deploy](#build--deploy)
- [Health Check & Monitoring](#health-check--monitoring)
- [Troubleshooting](#troubleshooting)
- [Backup & Restore](#backup--restore)

---

## Yêu cầu hệ thống

### Minimum Requirements

| Component | CPU | RAM | Disk |
|-----------|-----|-----|------|
| App (mis_ai) | 2 cores | 2GB | 5GB |
| MongoDB | 2 cores | 4GB | 20GB |
| Redis | 1 core | 1GB | 5GB |
| Ollama | 4+ cores | 8GB+ | 30GB+ (models) |
| **Total** | **8+ cores** | **16GB+** | **60GB+** |

### Software Requirements

- Docker Engine 24.0+
- Docker Compose 2.20+
- Linux/macOS/Windows với WSL2

---

## Cấu trúc project

```
bflow_ai/
├── Dockerfile              # Build image cho app
├── docker-compose.yaml     # Orchestrate 4 services
├── requirements.txt        # Python dependencies
├── .env.production         # Environment variables (production)
├── app/
│   ├── main.py            # FastAPI entry point
│   ├── core/
│   │   └── config.py      # App configuration
│   ├── db/
│   │   └── mongodb.py     # MongoDB client
│   └── api/
└── README_HELP_BUILD.md   # File này
```

---

## Chuẩn bị môi trường

### 1. Tạo file .env.production

Copy từ template hoặc tạo mới:

```bash
cd bflow_ai
cp .env.development .env.production
# Hoặc tạo mới
nano .env.production
```

### 2. Cấu hình .env.production

```bash
# =============================================================================
# BFLOW AI - Production Environment
# =============================================================================

# Project
PROJECT_NAME=BFLOW_AI
ENVIRONMENT=production

# =============================================================================
# Database - MongoDB
# =============================================================================
MONGO_URL=mongodb://mis_mongo:27017/bflow_db

# =============================================================================
# AI Server - Ollama
# =============================================================================
OLLAMA_HOST=http://mis_ollama:11434
CLASSIFIER_MODEL=qwen2.5:0.5b
GENERATION_MODEL=qwen2.5:7b

# =============================================================================
# Cache - Redis
# =============================================================================
REDIS_HOST=mis_redis
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=your_secure_password_here
USE_REDIS=true

# =============================================================================
# LLM Cache Settings
# =============================================================================
ENABLE_LLM_CACHE=true
CACHE_TTL=3600
MAX_CACHE_SIZE=100

# =============================================================================
# Semantic History Matching
# =============================================================================
ENABLE_SEMANTIC_HISTORY=true
SEMANTIC_MODE=hybrid
SEMANTIC_ALPHA=0.7
SEMANTIC_SIMILARITY_THRESHOLD=0.85
```

---

## Build & Deploy

### Option 1: Build & Run (Recommended for first time)

```bash
cd bflow_ai

# Build images
docker-compose build

# Start all services
docker-compose up -d

# Check status
docker-compose ps
```

### Option 2: Production-ready detached mode

```bash
cd bflow_ai

# Production mode với env file
docker-compose --env-file .env.production up -d --build

# Xem logs (all services)
docker-compose logs -f

# Xem logs của 1 service
docker-compose logs -f mis_ai
docker-compose logs -f mis_ollama
```

### Option 3: Rebuild individual service

```bash
# Chỉ rebuild app
docker-compose build mis_ai
docker-compose up -d mis_ai

# Pull latest images
docker-compose pull
```

---

## Health Check & Monitoring

### Check services status

```bash
# Xem tất cả containers
docker-compose ps

# Health check detail
docker inspect mis_ai | grep -A 10 Health

# Test app endpoint
curl http://localhost:8010/

# Test MongoDB
docker exec mis_mongo mongosh --eval "db.adminCommand('ping')"

# Test Redis
docker exec mis_redis redis-cli ping
```

### View logs

```bash
# Real-time logs
docker-compose logs -f

# Logs last 100 lines
docker-compose logs --tail=100

# Logs since specific time
docker-compose logs --since="2024-01-01T10:00:00"

# Logs cho service cụ thể
docker-compose logs -f mis_ai
docker-compose logs -f mis_ollama
docker-compose logs -f mis_mongo
docker-compose logs -f mis_redis
```

### Resource monitoring

```bash
# Resource usage của tất cả containers
docker stats

# Resource usage của cụ thể
docker stats mis_ai mis_ollama
```

---

## Troubleshooting

### 1. Container không start

```bash
# Check logs
docker-compose logs <service_name>

# Restart service
docker-compose restart <service_name>

# Rebuild & restart
docker-compose up -d --build <service_name>
```

### 2. Ollama không tải được model

```bash
# Vào container ollama
docker exec -it mis_ollama bash

# Tải model thủ công
ollama pull qwen2.5:7b
ollama pull qwen2.5:0.5b

# Test
ollama run qwen2.5:7b "Hello"
```

### 3. MongoDB connection refused

```bash
# Check mongo container status
docker-compose ps mis_mongo

# Test connection từ app container
docker exec mis_ai ping -c 3 mis_mongo

# Check mongo logs
docker-compose logs mis_mongo
```

### 4. Redis connection refused

```bash
# Test redis từ app container
docker exec mis_ai redis-cli -h mis_redis ping

# Check redis logs
docker-compose logs mis_redis
```

### 5. App health check failed

```bash
# Check app logs
docker-compose logs mis_ai

# Check environment variables
docker exec mis_ai env | grep -E "(MONGO|OLLAMA|REDIS)"

# Test endpoint từ inside container
docker exec mis_ai curl http://localhost:8010/
```

### 6. Clean & Restart (Last resort)

```bash
# Stop và remove containers
docker-compose down

# Xóa volumes (CAUTION: Mất dữ liệu!)
docker-compose down -v

# Rebuild from scratch
docker-compose up -d --build
```

---

## Backup & Restore

### Backup MongoDB data

```bash
# Backup từ container đang chạy
docker exec mis_mongo mongodump --archive=/data/db/backup_$(date +%Y%m%d).archive

# Copy ra host
docker cp mis_mongo:/data/db/backup_$(date +%Y%m%d).archive ./backups/
```

### Restore MongoDB data

```bash
# Copy file backup vào container
docker cp ./backups/backup_20240101.archive mis_mongo:/data/db/

# Restore
docker exec mis_mongo mongorestore --archive=/data/db/backup_20240101.archive
```

### Backup Redis data

```bash
# Save snapshot
docker exec mis_redis redis-cli BGSAVE

# Copy RDB file
docker cp mis_redis:/data/dump.rdb ./backups/redis_dump_$(date +%Y%m%d).rdb
```

### Backup Ollama models

```bash
# Copy toàn bộ ollama data
docker cp mis_ollama:/root/.ollama ./backups/ollama_backup_$(date +%Y%m%d)/
```

---

## Production Tips

### 1. Security

- Đổi `REDIS_PASSWORD` trong production
- Restrict ports exposure (chỉ expose 8010, 11434 nếu cần)
- Use secrets manager cho sensitive data

### 2. Performance

- Tăng `num_ctx` trong config nếu model hỗ trợ
- Enable GPU cho Ollama (cần nvidia-docker)
- Use Redis persistence mode

### 3. High Availability

- Use Docker Swarm hoặc Kubernetes cho multi-node
- Enable MongoDB replica set
- Use Redis Sentinel hoặc Cluster

### 4. Monitoring

- Integrate với Prometheus/Grafana
- Setup alerts cho health checks
- Log aggregation (ELK, Loki)

---

## Quick Reference

```bash
# Build & Start
docker-compose up -d --build

# Stop
docker-compose stop

# Stop & Remove containers
docker-compose down

# View logs
docker-compose logs -f

# Restart service
docker-compose restart <service>

# Exec vào container
docker exec -it <container_name> bash

# Prune unused resources
docker system prune -a
```

---

## Support

Nếu gặp vấn đề:

1. Check logs: `docker-compose logs -f`
2. Check health: `docker-compose ps`
3. Check resources: `docker stats`
4. Refer to troubleshooting section above
