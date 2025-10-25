# 02. Key Concepts

> **Deep dive into Docker's core concepts and architecture**

---

## 1. Images vs Containers

### The Recipe Analogy

| Concept | Real World | Docker |
|---------|-----------|---------|
| **Recipe** | Instructions to make a cake | **Image** (Dockerfile) |
| **Cake** | The actual cake you baked | **Container** (running instance) |

### Images

**Definition:** A read-only template with instructions for creating a container.

**Characteristics:**
- Immutable (doesn't change)
- Built in layers
- Can be shared via registries
- Lightweight (shares common layers)

**Example:**
```bash
# List images
docker images

REPOSITORY    TAG       IMAGE ID       SIZE
python        3.9-slim  a1b2c3d4e5f6   122MB
ubuntu        20.04     f6g7h8i9j0k1   72.8MB
```

### Containers

**Definition:** A runnable instance of an image.

**Characteristics:**
- Mutable (can change during runtime)
- Isolated from other containers
- Can be started, stopped, deleted
- Has its own filesystem, network, process tree

**Example:**
```bash
# List running containers
docker ps

CONTAINER ID   IMAGE     COMMAND           STATUS
abc123def456   python    "python app.py"   Up 2 hours
```

### Relationship

```
Image (Template)  →  Container (Instance)
   one               →  many

python:3.9        →  container1 (running app A)
                  →  container2 (running app B)
                  →  container3 (for testing)
```

---

## 2. Dockerfile: The Blueprint

### What is a Dockerfile?

A text file containing instructions to build a Docker image.

### Basic Structure

```dockerfile
# 1. Base image
FROM python:3.9-slim

# 2. Metadata
LABEL maintainer="you@example.com"

# 3. Set working directory
WORKDIR /app

# 4. Copy files
COPY requirements.txt .

# 5. Run commands (during build)
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy application code
COPY . .

# 7. Expose port
EXPOSE 8000

# 8. Default command (when container starts)
CMD ["python", "app.py"]
```

### Key Instructions

| Instruction | Purpose | Example |
|------------|---------|---------|
| **FROM** | Base image | `FROM python:3.9` |
| **WORKDIR** | Set working directory | `WORKDIR /app` |
| **COPY** | Copy files from host | `COPY app.py .` |
| **ADD** | Like COPY, but can extract archives | `ADD archive.tar.gz /app` |
| **RUN** | Execute command during build | `RUN pip install flask` |
| **CMD** | Default command when container starts | `CMD ["python", "app.py"]` |
| **ENTRYPOINT** | Command that always runs | `ENTRYPOINT ["python"]` |
| **EXPOSE** | Document which port to use | `EXPOSE 8000` |
| **ENV** | Set environment variable | `ENV API_KEY=secret` |
| **ARG** | Build-time variable | `ARG VERSION=1.0` |
| **VOLUME** | Create mount point | `VOLUME /data` |

---

## 3. Layers and Caching

### How Layers Work

Each instruction in a Dockerfile creates a **layer**.

**Example Dockerfile:**
```dockerfile
FROM python:3.9-slim        # Layer 1
WORKDIR /app                # Layer 2
COPY requirements.txt .     # Layer 3
RUN pip install -r req.txt  # Layer 4
COPY . .                    # Layer 5
CMD ["python", "app.py"]    # Layer 6
```

**Visualization:**
```
┌─────────────────────────┐
│ CMD ["python", "app.py"]│ ← Layer 6
├─────────────────────────┤
│ COPY . .                │ ← Layer 5
├─────────────────────────┤
│ RUN pip install...      │ ← Layer 4
├─────────────────────────┤
│ COPY requirements.txt   │ ← Layer 3
├─────────────────────────┤
│ WORKDIR /app            │ ← Layer 2
├─────────────────────────┤
│ FROM python:3.9-slim    │ ← Layer 1
└─────────────────────────┘
```

### Layer Caching

Docker caches each layer. If nothing changed, it reuses the cached layer.

**Smart ordering:**
```dockerfile
# ✅ GOOD: Dependencies first (changes rarely)
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .  # Code changes often
```

**Bad ordering:**
```dockerfile
# ❌ BAD: Code first
FROM python:3.9-slim
COPY . .  # Changes often → invalidates cache
RUN pip install -r requirements.txt  # Reinstalls every time!
```

### Why It Matters

```
First build:  5 minutes
Second build: 10 seconds (with good caching)
             5 minutes (with bad ordering)
```

---

## 4. Docker Registry and Hub

### What is a Registry?

A storage and distribution system for Docker images (like GitHub for code).

### Docker Hub

The default public registry (hub.docker.com)

**Common operations:**
```bash
# Pull image from Docker Hub
docker pull nginx:latest

# Tag your image
docker tag myapp:latest username/myapp:v1.0

# Push to Docker Hub
docker push username/myapp:v1.0

# Search for images
docker search python
```

### Image Naming

```
registry/repository:tag
```

**Examples:**
```
nginx:latest              → Docker Hub, nginx repo, latest tag
python:3.9-slim           → Docker Hub, python repo, 3.9-slim tag
ghcr.io/user/app:v1.0     → GitHub registry, user's app, v1.0 tag
myregistry.com/app:dev    → Private registry, app, dev tag
```

### Official vs User Images

```bash
# Official images (maintained by Docker/vendors)
docker pull python:3.9
docker pull postgres:13
docker pull nginx:latest

# User images (maintained by community)
docker pull username/myapp:latest
```

---

## 5. Volumes: Data Persistence

### The Problem

Containers are **ephemeral** (temporary). When deleted, data inside is lost.

```bash
docker run --name db postgres  # Create container
# ... add data to database ...
docker rm db                   # Delete container
# DATA IS GONE! 😱
```

### The Solution: Volumes

Volumes persist data outside the container.

### Types of Mounts

#### 1. Named Volumes (Recommended)

```bash
# Create volume
docker volume create mydata

# Use volume
docker run -v mydata:/var/lib/postgresql/data postgres

# Data persists even if container is deleted!
docker rm container
docker run -v mydata:/var/lib/postgresql/data postgres
# Same data is back!
```

#### 2. Bind Mounts

Mount a host directory into container:

```bash
# Current directory → /app in container
docker run -v $(pwd):/app myapp

# Absolute path
docker run -v /home/user/data:/data myapp
```

**Use cases:**
- Development (live code reloading)
- Sharing files between host and container

#### 3. tmpfs Mounts

Temporary in-memory storage (gone when container stops):

```bash
docker run --tmpfs /tmp myapp
```

### Volume Commands

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect mydata

# Remove volume
docker volume rm mydata

# Remove unused volumes
docker volume prune
```

### Volume in Dockerfile

```dockerfile
FROM postgres:13
VOLUME /var/lib/postgresql/data
```

---

## 6. Networks: Container Communication

### Default Networks

```bash
# List networks
docker network ls

NETWORK ID     NAME      DRIVER
abc123         bridge    bridge    # Default
def456         host      host
ghi789         none      null
```

### Network Types

#### 1. Bridge Network (Default)

Containers on same bridge can talk to each other.

```bash
# Create network
docker network create mynetwork

# Run containers on same network
docker run --network mynetwork --name web nginx
docker run --network mynetwork --name api python

# 'web' can reach 'api' using hostname 'api'
```

#### 2. Host Network

Container uses host's network directly (no isolation):

```bash
docker run --network host nginx
# nginx binds to host's port 80 directly
```

#### 3. None Network

No network (complete isolation):

```bash
docker run --network none myapp
```

### Port Mapping

```bash
# Map container port to host port
docker run -p HOST_PORT:CONTAINER_PORT image

# Examples:
docker run -p 8080:80 nginx        # Host 8080 → Container 80
docker run -p 5432:5432 postgres   # Host 5432 → Container 5432
docker run -p 80:8000 myapp        # Host 80 → Container 8000
```

**Visualization:**
```
Host Machine                Container
Port 8080  ───────────────> Port 80 (nginx)
```

### Container Communication Example

```bash
# Create network
docker network create app-network

# Run database
docker run -d \
  --name db \
  --network app-network \
  postgres

# Run app (can access db via hostname 'db')
docker run -d \
  --name web \
  --network app-network \
  -e DATABASE_URL=postgresql://db:5432/mydb \
  myapp
```

---

## 7. Docker Compose: Multi-Container Apps

### What is Docker Compose?

A tool to define and run multi-container applications using YAML.

### Why Use It?

**Without Compose:**
```bash
# Create network
docker network create mynetwork

# Run database
docker run -d --name db --network mynetwork \
  -e POSTGRES_PASSWORD=secret \
  -v pgdata:/var/lib/postgresql/data \
  postgres:13

# Run web app
docker run -d --name web --network mynetwork \
  -p 8000:8000 \
  -e DATABASE_URL=postgresql://db:5432/mydb \
  myapp

# Run Redis
docker run -d --name redis --network mynetwork redis:6
```

**With Compose:**
```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "8000:8000"
    environment:
      DATABASE_URL: postgresql://db:5432/mydb
    depends_on:
      - db
      - redis

  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

  redis:
    image: redis:6

volumes:
  pgdata:
```

```bash
# One command to run everything!
docker-compose up
```

### Key Compose Concepts

#### Services

A container definition:

```yaml
services:
  web:
    image: nginx:latest
    ports:
      - "80:80"
```

#### Build Context

Build from Dockerfile:

```yaml
services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
```

#### Environment Variables

```yaml
services:
  app:
    environment:
      - DEBUG=true
      - API_KEY=secret
    # Or from file:
    env_file:
      - .env
```

#### Depends On

Define startup order:

```yaml
services:
  web:
    depends_on:
      - db
  db:
    image: postgres:13
```

#### Volumes

```yaml
services:
  db:
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql

volumes:
  pgdata:
```

---

## 8. Container Lifecycle

### States

```
    docker run
       ↓
   [Created]
       ↓
    docker start
       ↓
   [Running]  ←─────┐
       ↓            │
    docker stop     │ docker restart
       ↓            │
   [Stopped] ───────┘
       ↓
    docker rm
       ↓
   [Deleted]
```

### Commands

```bash
# Create and start
docker run -d --name myapp nginx

# Start stopped container
docker start myapp

# Stop running container
docker stop myapp

# Restart container
docker restart myapp

# Pause container (freeze processes)
docker pause myapp

# Unpause
docker unpause myapp

# Remove container
docker rm myapp

# Force remove running container
docker rm -f myapp
```

### Auto-restart Policies

```bash
# Never restart (default)
docker run --restart no myapp

# Always restart
docker run --restart always myapp

# Restart on failure
docker run --restart on-failure myapp

# Restart unless manually stopped
docker run --restart unless-stopped myapp
```

---

## 9. Environment Variables

### Setting Variables

**In docker run:**
```bash
docker run -e API_KEY=secret -e DEBUG=true myapp
```

**In Dockerfile:**
```dockerfile
ENV API_KEY=default_key
ENV DEBUG=false
```

**In docker-compose.yml:**
```yaml
services:
  app:
    environment:
      API_KEY: secret
      DEBUG: true
    # Or from file:
    env_file: .env
```

**From .env file:**
```bash
# .env
API_KEY=secret
DEBUG=true
```

### Using in Code

**Python:**
```python
import os
api_key = os.getenv('API_KEY')
debug = os.getenv('DEBUG', 'false') == 'true'
```

---

## 10. .dockerignore

Like `.gitignore`, but for Docker builds.

### Why Use It?

Exclude files from build context (faster builds, smaller images).

**Example .dockerignore:**
```
# Git
.git
.gitignore

# Python
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# IDEs
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Project specific
*.log
tmp/
node_modules/
```

---

## 11. Image Tags and Versions

### Tag Strategies

```bash
# Latest (default, not recommended for production)
myapp:latest

# Semantic versioning
myapp:1.0.0
myapp:1.0
myapp:1

# Git commit
myapp:abc123f

# Environment
myapp:dev
myapp:staging
myapp:prod

# Timestamp
myapp:2025-10-25
```

### Best Practices

```bash
# ✅ GOOD: Pin specific versions
FROM python:3.9.7-slim

# ❌ BAD: Use 'latest' (unpredictable)
FROM python:latest
```

---

## Summary Table

| Concept | Purpose | Key Command |
|---------|---------|-------------|
| **Image** | Template for containers | `docker build`, `docker images` |
| **Container** | Running instance | `docker run`, `docker ps` |
| **Dockerfile** | Build instructions | `docker build -t name .` |
| **Layers** | Image building blocks | Efficient with caching |
| **Registry** | Image storage | `docker pull`, `docker push` |
| **Volume** | Data persistence | `docker volume create` |
| **Network** | Container communication | `docker network create` |
| **Compose** | Multi-container apps | `docker-compose up` |

---

## Key Takeaways

1. **Images are templates**, containers are instances
2. **Layers enable caching** - order Dockerfile wisely
3. **Volumes persist data** outside containers
4. **Networks connect** containers together
5. **Docker Compose simplifies** multi-container apps
6. **Use .dockerignore** to exclude unnecessary files
7. **Pin versions** for reproducibility

---

## What's Next?

Now that you understand the core concepts:

1. **Next**: Check [03_commands.md](03_commands.md) for detailed command reference
2. **Then**: Build real projects in [04_mini_projects.md](04_mini_projects.md)

---

## Quick Reference

### Essential Dockerfile
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```

### Essential docker-compose.yml
```yaml
version: '3.8'
services:
  app:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: secret
```

---

**Ready for commands?** → [03. Commands Reference](03_commands.md)

---
