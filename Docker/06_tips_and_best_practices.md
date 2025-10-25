# 06. Tips & Best Practices

> **Professional Docker workflows for production-ready applications**

---

## Table of Contents

1. [Golden Rules](#golden-rules)
2. [Dockerfile Best Practices](#dockerfile-best-practices)
3. [Security Best Practices](#security-best-practices)
4. [Image Optimization](#image-optimization)
5. [Docker Compose Best Practices](#docker-compose-best-practices)
6. [Production Deployment](#production-deployment)
7. [Development Workflow](#development-workflow)
8. [Monitoring and Logging](#monitoring-and-logging)
9. [CI/CD Integration](#cicd-integration)

---

## Golden Rules

### Rule 1: One Process Per Container

**Bad:**
```dockerfile
# ❌ Multiple services in one container
CMD service nginx start && service mysql start && python app.py
```

**Good:**
```yaml
# ✅ Separate containers for each service
services:
  web:
    image: nginx
  db:
    image: mysql
  app:
    build: .
```

**Why?** Easier to scale, update, and debug.

---

### Rule 2: Keep Images Small

**Bad:**
```dockerfile
# ❌ 1.5 GB image
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y python3 python3-pip
```

**Good:**
```dockerfile
# ✅ 50 MB image
FROM python:3.9-alpine
```

**Why?** Faster builds, pulls, deployments, and less attack surface.

---

### Rule 3: Use Official Images

**Bad:**
```dockerfile
# ❌ Unknown source
FROM randomuser/python-custom
```

**Good:**
```dockerfile
# ✅ Official, maintained, secure
FROM python:3.9-slim
```

**Why?** Security updates, best practices, reliability.

---

### Rule 4: Pin Versions

**Bad:**
```dockerfile
# ❌ Unpredictable
FROM python:latest
RUN pip install flask
```

**Good:**
```dockerfile
# ✅ Reproducible
FROM python:3.9.7-slim
RUN pip install flask==2.3.0
```

**Why?** Reproducible builds, no surprises.

---

### Rule 5: Don't Run as Root

**Bad:**
```dockerfile
# ❌ Runs as root (dangerous)
FROM python:3.9
COPY app.py .
CMD ["python", "app.py"]
```

**Good:**
```dockerfile
# ✅ Runs as non-root user
FROM python:3.9
RUN useradd -m -u 1000 appuser
USER appuser
COPY app.py .
CMD ["python", "app.py"]
```

**Why?** Security - limits damage if container is compromised.

---

## Dockerfile Best Practices

### 1. Order Layers by Change Frequency

**Bad:**
```dockerfile
# ❌ Code changes invalidate dependency cache
FROM python:3.9
COPY . .
RUN pip install -r requirements.txt
```

**Good:**
```dockerfile
# ✅ Dependencies cached separately
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

---

### 2. Combine RUN Commands

**Bad:**
```dockerfile
# ❌ Creates 3 layers
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
```

**Good:**
```dockerfile
# ✅ Creates 1 layer, includes cleanup
RUN apt-get update && \
    apt-get install -y \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*
```

---

### 3. Use Multi-Stage Builds

**Single-stage (large):**
```dockerfile
# ❌ 500 MB - includes build tools
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

**Multi-stage (small):**
```dockerfile
# ✅ 150 MB - only runtime dependencies
FROM python:3.9 AS builder
COPY requirements.txt .
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
COPY . .
ENV PATH=/root/.local/bin:$PATH
CMD ["python", "app.py"]
```

---

### 4. Use .dockerignore

```
# .dockerignore
.git
.gitignore
README.md
.env
.vscode/
.idea/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
*.log
.DS_Store
node_modules/
dist/
build/
*.egg-info/
.pytest_cache/
.coverage
htmlcov/
```

**Benefits:**
- Faster builds
- Smaller context
- Fewer secrets in images

---

### 5. COPY vs ADD

**Use COPY (preferred):**
```dockerfile
# ✅ Explicit, simple
COPY requirements.txt .
COPY app/ /app/
```

**Use ADD only when needed:**
```dockerfile
# ✅ Only for auto-extraction
ADD archive.tar.gz /app/
```

**Rule:** Use `COPY` unless you specifically need `ADD`'s features.

---

### 6. Use HEALTHCHECK

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .

EXPOSE 8000

# Add healthcheck
HEALTHCHECK --interval=30s \
            --timeout=3s \
            --start-period=5s \
            --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

CMD ["python", "app.py"]
```

---

### 7. Use ENV for Configuration

```dockerfile
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "app.py"]
```

---

### 8. Template: Production-Ready Dockerfile

```dockerfile
# syntax=docker/dockerfile:1

# Stage 1: Builder
FROM python:3.9-slim AS builder

# Install build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

# Create non-root user
RUN groupadd -r appuser && \
    useradd -r -g appuser -u 1000 appuser

# Copy dependencies from builder
COPY --from=builder /root/.local /home/appuser/.local

# Set environment
ENV PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Copy application
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["python", "app.py"]
```

---

## Security Best Practices

### 1. Never Store Secrets in Images

**Bad:**
```dockerfile
# ❌ API key baked into image!
ENV API_KEY=super_secret_key_123
```

**Good:**
```bash
# ✅ Pass at runtime
docker run -e API_KEY=$API_KEY myapp

# Or use secrets management
docker secret create api_key api_key.txt
docker service create --secret api_key myapp
```

---

### 2. Scan Images for Vulnerabilities

```bash
# Use Docker Scout
docker scout cves myapp:latest

# Or Trivy
trivy image myapp:latest

# Or Snyk
snyk container test myapp:latest
```

---

### 3. Use Minimal Base Images

**Security ranking (best to worst):**

1. **Distroless** (best - no shell, minimal packages)
```dockerfile
FROM gcr.io/distroless/python3
```

2. **Alpine** (good - minimal)
```dockerfile
FROM python:3.9-alpine
```

3. **Slim** (okay - reduced packages)
```dockerfile
FROM python:3.9-slim
```

4. **Full** (avoid - many packages, larger attack surface)
```dockerfile
FROM python:3.9  # ❌
```

---

### 4. Drop Capabilities

```bash
# Remove all capabilities, add only needed ones
docker run --cap-drop ALL --cap-add NET_BIND_SERVICE myapp
```

---

### 5. Use Read-Only Root Filesystem

```bash
docker run --read-only --tmpfs /tmp myapp
```

```yaml
# docker-compose.yml
services:
  app:
    image: myapp
    read_only: true
    tmpfs:
      - /tmp
```

---

### 6. Set Resource Limits

```bash
docker run \
  --memory=512m \
  --memory-swap=512m \
  --cpus=1 \
  --pids-limit=100 \
  myapp
```

```yaml
# docker-compose.yml
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M
        reservations:
          cpus: '0.5'
          memory: 256M
```

---

## Image Optimization

### 1. Choose Right Base Image

```dockerfile
# Scenario 1: Python app (no compilation)
FROM python:3.9-slim  # ✅ 150 MB

# Scenario 2: Python app (with C extensions)
FROM python:3.9-alpine  # ✅ 50 MB (but slower builds)

# Scenario 3: Static binary
FROM scratch  # ✅ 10 MB (ultimate minimal)
COPY myapp /
CMD ["/myapp"]
```

---

### 2. Minimize Layers

**Bad (10 layers):**
```dockerfile
FROM python:3.9
RUN apt-get update
RUN apt-get install -y curl
RUN apt-get install -y git
RUN pip install flask
RUN pip install requests
COPY file1.py .
COPY file2.py .
COPY file3.py .
CMD ["python", "app.py"]
```

**Good (4 layers):**
```dockerfile
FROM python:3.9
RUN apt-get update && \
    apt-get install -y curl git && \
    rm -rf /var/lib/apt/lists/*
RUN pip install flask requests
COPY *.py .
CMD ["python", "app.py"]
```

---

### 3. Remove Build Dependencies

```dockerfile
FROM python:3.9-slim

# Install build deps, build, remove build deps - all in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc python3-dev && \
    pip install numpy && \
    apt-get purge -y --auto-remove gcc python3-dev && \
    rm -rf /var/lib/apt/lists/*
```

---

### 4. Use --no-cache-dir for pip

```dockerfile
# ❌ Stores cache (larger image)
RUN pip install -r requirements.txt

# ✅ No cache (smaller image)
RUN pip install --no-cache-dir -r requirements.txt
```

---

### 5. Benchmark Image Size

```bash
# Check image size
docker images myapp

# See layer sizes
docker history myapp

# Deep dive
dive myapp  # Install: https://github.com/wagoodman/dive
```

---

## Docker Compose Best Practices

### 1. Use Environment Files

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    env_file:
      - .env
      - .env.local  # Override with local settings
```

```bash
# .env
DATABASE_URL=postgresql://db:5432/mydb
DEBUG=false
API_KEY=changeme

# .env.local (gitignored)
API_KEY=real_production_key
```

---

### 2. Use Version-Specific Images

```yaml
# ❌ Bad
services:
  db:
    image: postgres:latest

# ✅ Good
services:
  db:
    image: postgres:13.7-alpine
```

---

### 3. Define Health Checks

```yaml
version: '3.8'

services:
  web:
    build: .
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 3s
      retries: 3
      start_period: 40s

  db:
    image: postgres:13
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
```

---

### 4. Use Restart Policies

```yaml
services:
  web:
    restart: unless-stopped  # Recommended for production

  db:
    restart: always  # Critical services

  worker:
    restart: on-failure  # Retry on crashes
```

---

### 5. Proper Dependencies

```yaml
version: '3.8'

services:
  web:
    depends_on:
      db:
        condition: service_healthy  # Wait for health check
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost/health"]

  db:
    healthcheck:
      test: ["CMD-SHELL", "pg_isready"]
```

---

### 6. Use Named Volumes

```yaml
# ❌ Bad - anonymous volumes
services:
  db:
    volumes:
      - /var/lib/postgresql/data

# ✅ Good - named volumes
services:
  db:
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
    driver: local
```

---

### 7. Separate Networks

```yaml
version: '3.8'

services:
  frontend:
    networks:
      - frontend-net

  backend:
    networks:
      - frontend-net
      - backend-net

  db:
    networks:
      - backend-net  # Not accessible from frontend

networks:
  frontend-net:
  backend-net:
```

---

### 8. Template: Production docker-compose.yml

```yaml
version: '3.8'

services:
  nginx:
    image: nginx:1.21-alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    networks:
      - frontend
    depends_on:
      - web
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "--quiet", "--tries=1", "--spider", "http://localhost/health"]
      interval: 30s

  web:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        VERSION: ${VERSION:-latest}
    env_file:
      - .env
    networks:
      - frontend
      - backend
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
    deploy:
      resources:
        limits:
          cpus: '1'
          memory: 512M

  db:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: ${DB_NAME}
      POSTGRES_USER: ${DB_USER}
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./db/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${DB_USER}"]
      interval: 10s

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data
    networks:
      - backend
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s

volumes:
  postgres_data:
  redis_data:

networks:
  frontend:
  backend:
```

---

## Production Deployment

### 1. Use Docker Swarm or Kubernetes

**Docker Swarm (simpler):**
```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.yml myapp

# Scale service
docker service scale myapp_web=3
```

**Kubernetes (more features):**
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myapp
spec:
  replicas: 3
  selector:
    matchLabels:
      app: myapp
  template:
    metadata:
      labels:
        app: myapp
    spec:
      containers:
      - name: myapp
        image: myapp:v1.0
        resources:
          limits:
            memory: "512Mi"
            cpu: "1"
```

---

### 2. Implement CI/CD

**GitHub Actions example:**

```yaml
# .github/workflows/docker.yml
name: Docker Build and Push

on:
  push:
    branches: [main]
    tags: ['v*']

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v2

      - name: Login to DockerHub
        uses: docker/login-action@v2
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Build and push
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: |
            username/myapp:latest
            username/myapp:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

---

### 3. Use Private Registry

```bash
# Set up private registry
docker run -d -p 5000:5000 --name registry registry:2

# Tag image
docker tag myapp localhost:5000/myapp:v1.0

# Push to private registry
docker push localhost:5000/myapp:v1.0

# Pull from private registry
docker pull localhost:5000/myapp:v1.0
```

---

### 4. Implement Rolling Updates

```bash
# Docker Swarm
docker service update --image myapp:v2 myapp_web

# Kubernetes
kubectl set image deployment/myapp myapp=myapp:v2
kubectl rollout status deployment/myapp
```

---

## Development Workflow

### 1. Hot Reloading for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  web:
    build: .
    volumes:
      - ./app:/app  # Bind mount for live reload
    environment:
      - FLASK_ENV=development
      - DEBUG=true
    command: flask run --host=0.0.0.0 --reload
```

```bash
docker-compose -f docker-compose.dev.yml up
```

---

### 2. Separate Dev and Prod Configs

```bash
# Development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up

# Production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up
```

```yaml
# docker-compose.yml (base)
version: '3.8'
services:
  web:
    build: .

# docker-compose.dev.yml (dev overrides)
version: '3.8'
services:
  web:
    volumes:
      - ./app:/app
    environment:
      - DEBUG=true

# docker-compose.prod.yml (prod overrides)
version: '3.8'
services:
  web:
    restart: unless-stopped
    environment:
      - DEBUG=false
```

---

### 3. Use Docker BuildKit

```bash
# Enable BuildKit (faster builds, better caching)
export DOCKER_BUILDKIT=1

# Or set in daemon.json
{
  "features": {
    "buildkit": true
  }
}
```

---

## Monitoring and Logging

### 1. Centralized Logging

```yaml
version: '3.8'

services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  # Or use logging driver
  app2:
    logging:
      driver: "syslog"
      options:
        syslog-address: "tcp://log-server:514"
```

---

### 2. Monitoring with Prometheus

```yaml
version: '3.8'

services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
```

---

### 3. Export Metrics

```python
# app.py
from flask import Flask
from prometheus_client import Counter, Histogram, generate_latest

app = Flask(__name__)

REQUEST_COUNT = Counter('requests_total', 'Total requests')
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency')

@app.route('/metrics')
def metrics():
    return generate_latest()

@app.route('/')
@REQUEST_LATENCY.time()
def index():
    REQUEST_COUNT.inc()
    return "Hello!"
```

---

## CI/CD Integration

### GitLab CI Example

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

build:
  stage: build
  script:
    - docker build -t $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA .
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

test:
  stage: test
  script:
    - docker run $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA pytest

deploy:
  stage: deploy
  script:
    - docker stack deploy -c docker-compose.yml myapp
  only:
    - main
```

---

## Quick Reference: Best Practices Checklist

**Dockerfile:**
- [ ] Use official base images
- [ ] Pin specific versions
- [ ] Order layers by change frequency
- [ ] Use multi-stage builds
- [ ] Create non-root user
- [ ] Add health check
- [ ] Use .dockerignore
- [ ] Combine RUN commands
- [ ] Clean up in same layer

**Security:**
- [ ] Don't run as root
- [ ] Don't store secrets in image
- [ ] Scan for vulnerabilities
- [ ] Use minimal base images
- [ ] Set resource limits
- [ ] Use read-only filesystem where possible

**Production:**
- [ ] Use orchestration (Swarm/K8s)
- [ ] Implement health checks
- [ ] Set restart policies
- [ ] Use named volumes
- [ ] Implement logging
- [ ] Monitor resources
- [ ] Set up CI/CD
- [ ] Use private registry

---

## Summary

**Key Takeaways:**

1. **Security First** - Non-root users, no secrets, scan images
2. **Keep It Small** - Multi-stage builds, minimal base images
3. **Make It Reproducible** - Pin versions, proper dependencies
4. **Monitor Everything** - Logs, metrics, health checks
5. **Automate** - CI/CD, testing, deployment

---

**You're now ready to use Docker professionally!**

For more learning:
- Docker Official Docs: https://docs.docker.com
- Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/
- Security: https://cheatsheetseries.owasp.org/cheatsheets/Docker_Security_Cheat_Sheet.html

---
