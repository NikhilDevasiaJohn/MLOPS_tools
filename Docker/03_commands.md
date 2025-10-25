# 03. Commands Reference

> **Complete reference for Docker commands and Dockerfile instructions**

---

## Installation & Setup

### Install Docker

**Ubuntu/Debian:**
```bash
# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

**macOS:**
```bash
# Install via Homebrew
brew install --cask docker

# Or download Docker Desktop from:
# https://www.docker.com/products/docker-desktop
```

**Windows:**
- Download Docker Desktop from https://www.docker.com/products/docker-desktop
- Enable WSL 2 for better performance

### Post-Installation

```bash
# Run Docker without sudo (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker run hello-world

# Install Docker Compose (if not included)
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
docker-compose --version
```

---

## Image Commands

### Building Images

```bash
# Build from Dockerfile in current directory
docker build .

# Build with tag
docker build -t myapp:latest .
docker build -t myapp:v1.0 .

# Build from specific Dockerfile
docker build -f Dockerfile.prod -t myapp:prod .

# Build with build arguments
docker build --build-arg VERSION=1.0 -t myapp .

# Build without cache
docker build --no-cache -t myapp .

# Show build output
docker build -t myapp . --progress=plain
```

### Listing Images

```bash
# List all images
docker images
docker image ls

# List with details
docker images -a

# Filter images
docker images python
docker images "python:3.*"

# Show image sizes
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
```

### Pulling Images

```bash
# Pull from Docker Hub
docker pull ubuntu:20.04
docker pull python:3.9-slim
docker pull nginx:latest

# Pull from specific registry
docker pull ghcr.io/username/myapp:latest

# Pull all tags
docker pull -a ubuntu
```

### Pushing Images

```bash
# Login to registry
docker login
docker login ghcr.io

# Tag image for registry
docker tag myapp:latest username/myapp:v1.0

# Push to registry
docker push username/myapp:v1.0

# Logout
docker logout
```

### Inspecting Images

```bash
# Show detailed info
docker inspect python:3.9

# Show image history (layers)
docker history python:3.9

# Show image layers
docker history --no-trunc python:3.9
```

### Removing Images

```bash
# Remove image
docker rmi myapp:latest

# Remove multiple images
docker rmi image1 image2 image3

# Force remove (even if containers exist)
docker rmi -f myapp:latest

# Remove dangling images (untagged)
docker image prune

# Remove all unused images
docker image prune -a

# Remove images by filter
docker images -f "dangling=true" -q | xargs docker rmi
```

### Tagging Images

```bash
# Tag image
docker tag myapp:latest myapp:v1.0
docker tag myapp:latest username/myapp:latest

# Tag for different registry
docker tag myapp:latest ghcr.io/username/myapp:v1.0
```

### Saving and Loading Images

```bash
# Save image to tar file
docker save myapp:latest -o myapp.tar
docker save myapp:latest | gzip > myapp.tar.gz

# Load image from tar file
docker load -i myapp.tar
docker load < myapp.tar.gz

# Export container to tar (flattened)
docker export container_name > container.tar

# Import from tar
docker import container.tar myapp:imported
```

---

## Container Commands

### Running Containers

```bash
# Basic run
docker run nginx

# Run in background (detached)
docker run -d nginx

# Run with name
docker run -d --name mynginx nginx

# Run with port mapping
docker run -d -p 8080:80 nginx

# Run with multiple ports
docker run -d -p 8080:80 -p 8443:443 nginx

# Run with environment variables
docker run -d -e API_KEY=secret -e DEBUG=true myapp

# Run with volume
docker run -d -v mydata:/data myapp
docker run -d -v $(pwd):/app myapp

# Run with network
docker run -d --network mynetwork nginx

# Run with resource limits
docker run -d --memory=512m --cpus=1 myapp

# Run interactively
docker run -it ubuntu bash

# Run and remove after exit
docker run --rm -it ubuntu bash

# Run with custom entrypoint
docker run --entrypoint /bin/sh myapp

# Run with restart policy
docker run -d --restart unless-stopped nginx
```

### Listing Containers

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# List with specific format
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# List only IDs
docker ps -q

# Filter containers
docker ps --filter "status=running"
docker ps --filter "name=web"
```

### Managing Container State

```bash
# Stop container
docker stop container_name
docker stop container_id

# Stop with timeout
docker stop -t 30 container_name

# Start stopped container
docker start container_name

# Restart container
docker restart container_name

# Pause container (freeze processes)
docker pause container_name

# Unpause container
docker unpause container_name

# Kill container (force stop)
docker kill container_name

# Rename container
docker rename old_name new_name
```

### Executing Commands in Containers

```bash
# Execute command
docker exec container_name ls -la

# Interactive bash shell
docker exec -it container_name bash
docker exec -it container_name sh

# Run as specific user
docker exec -u root -it container_name bash

# Set working directory
docker exec -w /app container_name ls

# Run with environment variable
docker exec -e VAR=value container_name env
```

### Viewing Logs

```bash
# View logs
docker logs container_name

# Follow logs (like tail -f)
docker logs -f container_name

# Show timestamps
docker logs -t container_name

# Show last N lines
docker logs --tail 100 container_name

# Show logs since specific time
docker logs --since 2024-01-01 container_name
docker logs --since 10m container_name

# Combine options
docker logs -f --tail 50 container_name
```

### Inspecting Containers

```bash
# Show detailed info
docker inspect container_name

# Get specific value
docker inspect --format '{{.State.Status}}' container_name
docker inspect --format '{{.NetworkSettings.IPAddress}}' container_name

# Show resource usage
docker stats container_name

# Show running processes
docker top container_name

# Show port mappings
docker port container_name

# Show changes to filesystem
docker diff container_name
```

### Copying Files

```bash
# Copy from container to host
docker cp container_name:/path/in/container /path/on/host

# Copy from host to container
docker cp /path/on/host container_name:/path/in/container

# Copy directory
docker cp container_name:/app/logs ./logs
```

### Removing Containers

```bash
# Remove stopped container
docker rm container_name

# Force remove running container
docker rm -f container_name

# Remove multiple containers
docker rm container1 container2

# Remove all stopped containers
docker container prune

# Remove containers by filter
docker ps -a -f status=exited -q | xargs docker rm
```

---

## Dockerfile Instructions

### FROM - Base Image

```dockerfile
# Official image
FROM python:3.9

# Specific version (recommended)
FROM python:3.9.7-slim

# Alpine variant (smaller)
FROM python:3.9-alpine

# Multi-stage build
FROM python:3.9 AS builder
FROM python:3.9-slim AS runtime
```

### WORKDIR - Set Working Directory

```dockerfile
# Set working directory (creates if doesn't exist)
WORKDIR /app

# All subsequent commands run here
WORKDIR /app/src
```

### COPY - Copy Files

```dockerfile
# Copy single file
COPY app.py /app/

# Copy to working directory
COPY app.py .

# Copy multiple files
COPY app.py config.py utils.py /app/

# Copy directory
COPY ./src /app/src

# Copy with wildcard
COPY *.py /app/

# Copy with ownership
COPY --chown=user:group app.py /app/
```

### ADD - Advanced Copy

```dockerfile
# Like COPY
ADD app.py /app/

# Extract tar archive
ADD archive.tar.gz /app/

# Download from URL (not recommended)
ADD https://example.com/file.txt /app/
```

**Note:** Prefer `COPY` over `ADD` unless you need extraction.

### RUN - Execute Commands

```dockerfile
# Run command
RUN apt-get update

# Chain commands (one layer)
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# Multiple RUN instructions (multiple layers)
RUN apt-get update
RUN apt-get install -y curl

# Run with shell
RUN /bin/bash -c 'echo hello'

# Exec form (no shell)
RUN ["apt-get", "update"]
```

### CMD - Default Command

```dockerfile
# Shell form
CMD python app.py

# Exec form (preferred)
CMD ["python", "app.py"]

# As parameters to ENTRYPOINT
CMD ["--port", "8000"]
```

**Note:** Only last `CMD` is used. Can be overridden by `docker run`.

### ENTRYPOINT - Container Executable

```dockerfile
# Exec form (preferred)
ENTRYPOINT ["python", "app.py"]

# Shell form
ENTRYPOINT python app.py

# Combined with CMD
ENTRYPOINT ["python", "app.py"]
CMD ["--port", "8000"]

# Override at runtime:
# docker run myapp --port 9000
```

### ENV - Environment Variables

```dockerfile
# Single variable
ENV API_KEY=secret

# Multiple variables
ENV API_KEY=secret \
    DEBUG=true \
    PORT=8000

# Use variables
ENV APP_HOME=/app
WORKDIR $APP_HOME
```

### ARG - Build Arguments

```dockerfile
# Define build argument
ARG VERSION=1.0
ARG PORT

# Use in Dockerfile
FROM python:${VERSION}
EXPOSE ${PORT}

# Build with argument:
# docker build --build-arg VERSION=3.9 --build-arg PORT=8000 .
```

### EXPOSE - Document Ports

```dockerfile
# Single port
EXPOSE 8000

# Multiple ports
EXPOSE 8000 8443

# With protocol
EXPOSE 8000/tcp
EXPOSE 53/udp
```

**Note:** This is documentation only. Use `-p` flag to actually publish.

### VOLUME - Create Mount Point

```dockerfile
# Single volume
VOLUME /data

# Multiple volumes
VOLUME ["/data", "/logs"]
```

### USER - Set User

```dockerfile
# Switch to user
USER appuser

# Create and switch to user
RUN useradd -m appuser
USER appuser
```

### LABEL - Add Metadata

```dockerfile
# Single label
LABEL version="1.0"

# Multiple labels
LABEL maintainer="you@example.com" \
      version="1.0" \
      description="My app"
```

### HEALTHCHECK - Container Health

```dockerfile
# Basic healthcheck
HEALTHCHECK CMD curl -f http://localhost/ || exit 1

# With options
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD curl -f http://localhost/ || exit 1

# Disable inherited healthcheck
HEALTHCHECK NONE
```

### SHELL - Change Shell

```dockerfile
# Change default shell
SHELL ["/bin/bash", "-c"]

# Use PowerShell (Windows)
SHELL ["powershell", "-Command"]
```

### ONBUILD - Trigger on Child Images

```dockerfile
# Runs when image is used as base
ONBUILD COPY . /app
ONBUILD RUN pip install -r requirements.txt
```

---

## Volume Commands

### Managing Volumes

```bash
# Create volume
docker volume create mydata

# Create with driver options
docker volume create --driver local --opt type=none --opt device=/path --opt o=bind mydata

# List volumes
docker volume ls

# Inspect volume
docker volume inspect mydata

# Remove volume
docker volume rm mydata

# Remove unused volumes
docker volume prune

# Remove all volumes
docker volume prune -a
```

### Using Volumes

```bash
# Named volume
docker run -v mydata:/app/data myapp

# Bind mount
docker run -v $(pwd):/app myapp
docker run -v /host/path:/container/path myapp

# Read-only mount
docker run -v mydata:/app/data:ro myapp

# Multiple volumes
docker run -v vol1:/data1 -v vol2:/data2 myapp

# Mount with options
docker run --mount type=volume,source=mydata,target=/data myapp
docker run --mount type=bind,source=$(pwd),target=/app myapp
```

---

## Network Commands

### Managing Networks

```bash
# Create network
docker network create mynetwork

# Create with driver
docker network create --driver bridge mynetwork

# Create with subnet
docker network create --subnet=172.20.0.0/16 mynetwork

# List networks
docker network ls

# Inspect network
docker network inspect mynetwork

# Remove network
docker network rm mynetwork

# Remove unused networks
docker network prune

# Connect container to network
docker network connect mynetwork container_name

# Disconnect container from network
docker network disconnect mynetwork container_name
```

### Using Networks

```bash
# Run container on network
docker run --network mynetwork --name web nginx

# Run with custom hostname
docker run --network mynetwork --hostname myhost nginx

# Run with network alias
docker run --network mynetwork --network-alias db postgres

# Expose ports
docker run --network mynetwork -p 8080:80 nginx

# Link containers (legacy)
docker run --link db:database webapp
```

---

## Docker Compose Commands

### Basic Commands

```bash
# Start services
docker-compose up

# Start in background
docker-compose up -d

# Build and start
docker-compose up --build

# Start specific service
docker-compose up web

# Stop services
docker-compose stop

# Stop and remove
docker-compose down

# Stop, remove, and delete volumes
docker-compose down -v

# Restart services
docker-compose restart

# Restart specific service
docker-compose restart web
```

### Service Management

```bash
# List services
docker-compose ps

# View logs
docker-compose logs

# Follow logs
docker-compose logs -f

# Logs for specific service
docker-compose logs web

# Execute command in service
docker-compose exec web bash

# Run one-off command
docker-compose run web python manage.py migrate

# Scale services
docker-compose up --scale web=3
```

### Building and Pulling

```bash
# Build services
docker-compose build

# Build specific service
docker-compose build web

# Build without cache
docker-compose build --no-cache

# Pull images
docker-compose pull

# Push images
docker-compose push
```

### Configuration

```bash
# Validate compose file
docker-compose config

# View config with resolved values
docker-compose config --resolve-image-digests

# Use specific compose file
docker-compose -f docker-compose.prod.yml up

# Use multiple compose files
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up

# Set project name
docker-compose -p myproject up
```

---

## System Commands

### System Information

```bash
# Show Docker info
docker info

# Show Docker version
docker version

# Show disk usage
docker system df

# Detailed disk usage
docker system df -v

# Show events
docker events

# Monitor events
docker events --since 1h
```

### Cleaning Up

```bash
# Remove all unused objects
docker system prune

# Remove with volumes
docker system prune --volumes

# Remove everything (careful!)
docker system prune -a

# Remove stopped containers
docker container prune

# Remove unused images
docker image prune

# Remove dangling images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune
```

---

## Docker Compose File Reference

### Basic Structure

```yaml
version: '3.8'

services:
  web:
    image: nginx:latest
    ports:
      - "80:80"

  app:
    build: .
    depends_on:
      - db

  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:

networks:
  frontend:
  backend:
```

### Service Configuration

```yaml
services:
  web:
    # Use existing image
    image: nginx:latest

    # Or build from Dockerfile
    build:
      context: .
      dockerfile: Dockerfile.prod
      args:
        VERSION: "1.0"

    # Container name
    container_name: my-web-server

    # Ports
    ports:
      - "8080:80"
      - "443:443"

    # Expose (for inter-container)
    expose:
      - "8000"

    # Environment
    environment:
      - API_KEY=secret
      - DEBUG=true
    # Or from file
    env_file:
      - .env

    # Volumes
    volumes:
      - ./data:/app/data
      - logs:/var/log

    # Networks
    networks:
      - frontend
      - backend

    # Depends on
    depends_on:
      - db
      - redis

    # Command override
    command: python app.py --port 8000

    # Entrypoint override
    entrypoint: /app/entrypoint.sh

    # Restart policy
    restart: unless-stopped

    # Resource limits
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M

    # Healthcheck
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost"]
      interval: 30s
      timeout: 10s
      retries: 3

    # Logging
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Quick Reference Card

### Most Used Commands

```bash
# Images
docker build -t name .
docker images
docker rmi image_name

# Containers
docker run -d -p 8080:80 --name web nginx
docker ps
docker stop container_name
docker rm container_name
docker logs -f container_name
docker exec -it container_name bash

# Volumes
docker volume create name
docker volume ls
docker volume rm name

# Networks
docker network create name
docker network ls

# Compose
docker-compose up -d
docker-compose down
docker-compose logs -f

# Cleanup
docker system prune
docker system prune -a --volumes
```

### Common Dockerfile

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["python", "app.py"]
```

### Common docker-compose.yml

```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
    environment:
      - DATABASE_URL=postgresql://db:5432/mydb
    depends_on:
      - db
  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: secret
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

---

## What's Next?

Now that you have the command reference:

1. **Next**: Try [04_mini_projects.md](04_mini_projects.md) for hands-on practice
2. **Reference**: Keep this file open while working
3. **Troubleshooting**: Check [05_troubleshooting.md](05_troubleshooting.md) when stuck

---

**Ready to build?** → [04. Mini Projects](04_mini_projects.md)

---
