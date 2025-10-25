# 05. Troubleshooting

> **Common Docker problems and how to fix them**

---

## How to Use This Guide

1. **Find your error** - Use Ctrl+F to search for error message
2. **Try the solution** - Follow steps carefully
3. **Understand why** - Learn what caused it
4. **Prevent it** - Know how to avoid it next time

---

## Installation Issues

### Problem: "Cannot connect to Docker daemon"

**Error:**
```
Cannot connect to the Docker daemon at unix:///var/run/docker.sock.
Is the docker daemon running?
```

**Causes:**
- Docker service not running
- Insufficient permissions

**Solutions:**

```bash
# Check if Docker is running
sudo systemctl status docker

# Start Docker
sudo systemctl start docker

# Enable Docker to start on boot
sudo systemctl enable docker

# Add user to docker group (Linux)
sudo usermod -aG docker $USER
newgrp docker

# Restart for changes to take effect
sudo systemctl restart docker
```

---

### Problem: "Permission denied" on Linux

**Error:**
```
Got permission denied while trying to connect to the Docker daemon socket
```

**Solution:**

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Log out and log back in, or run:
newgrp docker

# Verify
docker ps
```

---

### Problem: Docker Desktop won't start (macOS/Windows)

**Solutions:**

**macOS:**
```bash
# Reset Docker Desktop
rm -rf ~/Library/Group\ Containers/group.com.docker
rm -rf ~/Library/Containers/com.docker.docker

# Reinstall Docker Desktop
```

**Windows:**
- Enable WSL 2
- Enable Hyper-V
- Update Windows
- Reinstall Docker Desktop

---

## Build Issues

### Problem: "no such file or directory" during COPY

**Error:**
```dockerfile
COPY app.py /app/
# Error: app.py: no such file or directory
```

**Cause:** File not in build context

**Solution:**

```bash
# Check what's in build context
ls -la

# Ensure file exists
ls app.py

# Check .dockerignore isn't excluding it
cat .dockerignore

# Build from correct directory
cd /path/to/dockerfile
docker build -t myapp .
```

---

### Problem: Build cache not working

**Symptom:** Every build reinstalls everything

**Cause:** Poor Dockerfile ordering

**Bad:**
```dockerfile
FROM python:3.9
COPY . .                        # ❌ Changes often
RUN pip install -r requirements.txt  # Reinstalls every time
```

**Good:**
```dockerfile
FROM python:3.9
COPY requirements.txt .         # ✅ Changes rarely
RUN pip install -r requirements.txt  # Cached unless requirements change
COPY . .                        # Code changes don't invalidate cache
```

**Force rebuild without cache:**
```bash
docker build --no-cache -t myapp .
```

---

### Problem: "Layer already being pulled by another client"

**Error:**
```
failed to copy: httpReadSeeker: failed to read: layer already being pulled
```

**Solution:**

```bash
# Wait a moment and retry
docker build -t myapp .

# Or pull base image first
docker pull python:3.9-slim
docker build -t myapp .

# Clean up if persistent
docker system prune -a
```

---

### Problem: Build arguments not working

**Dockerfile:**
```dockerfile
ARG VERSION=1.0
FROM python:${VERSION}
```

**Wrong:**
```bash
docker build -t myapp .  # Uses default VERSION=1.0
```

**Correct:**
```bash
docker build --build-arg VERSION=3.9 -t myapp .
```

---

### Problem: "Unable to locate package" in Dockerfile

**Error:**
```dockerfile
RUN apt-get install curl
# E: Unable to locate package curl
```

**Solution:**
```dockerfile
# Always update package index first
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*
```

---

## Container Runtime Issues

### Problem: Container exits immediately

**Symptom:**
```bash
docker ps -a
# STATUS: Exited (0) 2 seconds ago
```

**Debugging:**

```bash
# View logs
docker logs container_name

# Run interactively to see what happens
docker run -it myapp bash

# Check what command is running
docker inspect container_name | grep -A 5 Cmd
```

**Common causes:**

1. **No foreground process:**
```dockerfile
# ❌ Bad - script exits immediately
CMD ["python", "script.py"]  # Script completes and exits

# ✅ Good - keeps running
CMD ["python", "app.py"]     # Web server keeps running
```

2. **Application crashes:**
```bash
# Check logs for errors
docker logs container_name
```

---

### Problem: "Port already in use"

**Error:**
```
Bind for 0.0.0.0:8080 failed: port is already allocated
```

**Solutions:**

```bash
# Find what's using the port
sudo lsof -i :8080
sudo netstat -tulpn | grep 8080

# Stop conflicting container
docker ps
docker stop container_name

# Use different port
docker run -p 8081:8080 myapp  # Host 8081 → Container 8080
```

---

### Problem: Cannot access container from browser

**Symptoms:**
- Container is running
- curl from inside container works
- Browser shows "Connection refused"

**Solutions:**

```bash
# 1. Check port mapping
docker ps
# Ensure PORTS shows 0.0.0.0:8080->8080/tcp

# 2. Check app binds to 0.0.0.0, not 127.0.0.1
# Wrong:
app.run(host='127.0.0.1', port=8080)  # ❌ Only localhost

# Correct:
app.run(host='0.0.0.0', port=8080)    # ✅ All interfaces

# 3. Check firewall
sudo ufw allow 8080

# 4. Test from inside container
docker exec container_name curl localhost:8080
```

---

### Problem: Container can't connect to host

**Scenario:** Container needs to access service on host machine

**Solution:**

```bash
# Linux/macOS: Use host.docker.internal
docker run -e API_URL=http://host.docker.internal:8000 myapp

# Linux alternative: Use host network
docker run --network host myapp

# Or get host IP
ip addr show docker0 | grep inet
# Use that IP in container
```

---

### Problem: "OCI runtime create failed"

**Error:**
```
docker: Error response from daemon: OCI runtime create failed
```

**Solutions:**

```bash
# 1. Restart Docker
sudo systemctl restart docker

# 2. Check disk space
df -h

# 3. Clean up
docker system prune -a

# 4. Check Docker logs
sudo journalctl -u docker.service -n 50

# 5. Reinstall Docker if persistent
```

---

## Volume and Mount Issues

### Problem: Volume data not persisting

**Debugging:**

```bash
# Check if volume exists
docker volume ls

# Inspect volume
docker volume inspect volume_name

# Check container's mounts
docker inspect container_name | grep -A 10 Mounts
```

**Common mistakes:**

```bash
# ❌ Wrong - creates anonymous volume
docker run -v /data myapp

# ✅ Correct - named volume
docker run -v mydata:/data myapp
```

---

### Problem: Permission denied with bind mounts

**Error:**
```
Permission denied: '/app/data/file.txt'
```

**Cause:** User inside container doesn't have permissions

**Solutions:**

```dockerfile
# Option 1: Match host user ID
RUN useradd -m -u 1000 appuser
USER appuser
```

```bash
# Option 2: Run as host user
docker run -u $(id -u):$(id -g) -v $(pwd):/app myapp

# Option 3: Change permissions
chmod -R 777 ./data  # Not recommended for production
```

---

### Problem: Changes on host not reflected in container

**Cause:** Using volume instead of bind mount

**Wrong:**
```bash
docker run -v myvolume:/app myapp  # Named volume, isolated
```

**Correct for development:**
```bash
docker run -v $(pwd):/app myapp    # Bind mount, synced with host
```

---

### Problem: Cannot delete volume

**Error:**
```
Error response from daemon: remove volume: volume is in use
```

**Solution:**

```bash
# Find containers using volume
docker ps -a --filter volume=volume_name

# Stop and remove containers
docker stop container_name
docker rm container_name

# Now remove volume
docker volume rm volume_name

# Force remove (careful!)
docker volume rm -f volume_name
```

---

## Network Issues

### Problem: Containers can't communicate

**Scenario:** Container A can't reach Container B

**Debugging:**

```bash
# Check if on same network
docker inspect container_a | grep NetworkMode
docker inspect container_b | grep NetworkMode

# Test connectivity
docker exec container_a ping container_b

# Check DNS resolution
docker exec container_a nslookup container_b
```

**Solution:**

```bash
# Create network
docker network create mynetwork

# Run both containers on same network
docker run -d --name container_a --network mynetwork image_a
docker run -d --name container_b --network mynetwork image_b

# Now they can reach each other by name
docker exec container_a curl http://container_b:8080
```

---

### Problem: DNS not working in containers

**Symptom:**
```bash
docker exec container ping google.com
# ping: bad address 'google.com'
```

**Solutions:**

```bash
# Option 1: Specify DNS servers
docker run --dns 8.8.8.8 --dns 8.8.4.4 myapp

# Option 2: Configure Docker daemon
# Edit /etc/docker/daemon.json
{
  "dns": ["8.8.8.8", "8.8.4.4"]
}

sudo systemctl restart docker

# Option 3: Use host network
docker run --network host myapp
```

---

## Docker Compose Issues

### Problem: "Service 'X' failed to build"

**Debugging:**

```bash
# See full error
docker-compose up --build

# Build specific service
docker-compose build web

# Build without cache
docker-compose build --no-cache web
```

---

### Problem: Services can't connect

**Error in logs:**
```
Connection refused: db:5432
```

**Causes:**

1. **Wrong hostname:**
```yaml
# ❌ Wrong
environment:
  - DATABASE_URL=postgresql://localhost:5432/db

# ✅ Correct - use service name
environment:
  - DATABASE_URL=postgresql://db:5432/db
```

2. **Service not ready:**
```yaml
services:
  web:
    depends_on:
      - db  # ❌ Starts after db, but db might not be ready

# ✅ Better - use healthcheck
  db:
    healthcheck:
      test: ["CMD", "pg_isready", "-U", "postgres"]
      interval: 5s
  web:
    depends_on:
      db:
        condition: service_healthy
```

---

### Problem: "Port is already allocated"

**Solution:**

```bash
# Find conflicting container
docker ps

# Stop it
docker-compose down

# Or use different port in docker-compose.yml
ports:
  - "8081:8080"  # Host 8081 instead of 8080
```

---

### Problem: Environment variables not working

**Wrong:**
```yaml
services:
  web:
    environment:
      - API_KEY=$API_KEY  # ❌ Not interpolated
```

**Correct:**

```yaml
# Option 1: Use .env file (recommended)
# Create .env file:
# API_KEY=secret123

services:
  web:
    env_file:
      - .env

# Option 2: Direct substitution
services:
  web:
    environment:
      - API_KEY=${API_KEY}  # ✅ Reads from shell/env

# Option 3: Hard-coded (not for secrets!)
services:
  web:
    environment:
      - DEBUG=true
```

---

### Problem: Changes not reflected after rebuild

**Solution:**

```bash
# Stop and remove everything
docker-compose down

# Rebuild and restart
docker-compose up --build

# Force recreate
docker-compose up --force-recreate

# Nuclear option
docker-compose down -v  # Deletes volumes!
docker-compose up --build
```

---

## Performance Issues

### Problem: Slow builds

**Solutions:**

1. **Use .dockerignore:**
```
# .dockerignore
node_modules/
.git/
*.log
__pycache__/
```

2. **Optimize layer caching:**
```dockerfile
# ✅ Dependencies first (changes rarely)
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code last (changes often)
COPY . .
```

3. **Use multi-stage builds:**
```dockerfile
FROM python:3.9 AS builder
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
```

4. **Build with BuildKit:**
```bash
DOCKER_BUILDKIT=1 docker build -t myapp .
```

---

### Problem: Container using too much memory

**Debugging:**

```bash
# Check resource usage
docker stats

# Inspect container memory limit
docker inspect container_name | grep Memory
```

**Solutions:**

```bash
# Set memory limit
docker run -m 512m myapp

# In docker-compose.yml:
services:
  web:
    deploy:
      resources:
        limits:
          memory: 512M
```

---

### Problem: Disk space full

**Error:**
```
no space left on device
```

**Solutions:**

```bash
# Check Docker disk usage
docker system df

# Remove unused containers
docker container prune

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything unused
docker system prune -a --volumes

# Free up space manually
docker rm $(docker ps -aq)
docker rmi $(docker images -q)
```

---

## Image Issues

### Problem: Image too large

**Debugging:**

```bash
# Check image size
docker images myapp

# See layer sizes
docker history myapp
```

**Solutions:**

1. **Use slim/alpine base:**
```dockerfile
# ❌ Large (1GB+)
FROM python:3.9

# ✅ Medium (200MB)
FROM python:3.9-slim

# ✅ Small (50MB)
FROM python:3.9-alpine
```

2. **Clean up in same layer:**
```dockerfile
# ❌ Creates 2 layers
RUN apt-get update
RUN apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# ✅ One layer, cleaned up
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*
```

3. **Use multi-stage builds:**
```dockerfile
FROM python:3.9 AS builder
RUN pip install --user -r requirements.txt

FROM python:3.9-slim
COPY --from=builder /root/.local /root/.local
```

---

### Problem: Cannot pull image

**Error:**
```
Error response from daemon: pull access denied
```

**Solutions:**

```bash
# 1. Login to registry
docker login

# 2. Check image name is correct
docker pull python:3.9  # Not python3.9

# 3. For private registry
docker login registry.example.com
docker pull registry.example.com/myapp:latest

# 4. Check network connection
curl -I https://hub.docker.com
```

---

### Problem: Wrong platform/architecture

**Error:**
```
WARNING: The requested image's platform (linux/amd64) does not match
the detected host platform (linux/arm64/v8)
```

**Solutions:**

```bash
# Build for specific platform
docker build --platform linux/amd64 -t myapp .

# Build multi-platform (requires buildx)
docker buildx build --platform linux/amd64,linux/arm64 -t myapp .

# Run specific platform
docker run --platform linux/amd64 myapp
```

---

## Debugging Strategies

### General Debugging Workflow

```bash
# 1. Check if container is running
docker ps -a

# 2. View logs
docker logs container_name
docker logs -f container_name  # Follow
docker logs --tail 100 container_name  # Last 100 lines

# 3. Execute into container
docker exec -it container_name bash
docker exec -it container_name sh  # If bash not available

# 4. Inspect container
docker inspect container_name

# 5. Check resources
docker stats container_name

# 6. View processes
docker top container_name

# 7. Check network
docker network inspect network_name

# 8. Check volumes
docker volume inspect volume_name
```

### Debugging Dockerfile

```bash
# Build and stop at specific layer
docker build --target builder -t debug .

# Run intermediate image
docker run -it debug bash

# Build with verbose output
docker build --progress=plain --no-cache -t myapp .
```

### Debugging Container Crashes

```bash
# Keep container running even if app crashes
docker run -it --entrypoint bash myapp

# Or override CMD
docker run -it myapp bash

# Check exit code
docker inspect container_name --format='{{.State.ExitCode}}'

# Common exit codes:
# 0   = Success
# 1   = Application error
# 137 = Killed (OOM)
# 139 = Segmentation fault
# 143 = Terminated (SIGTERM)
```

---

## Quick Fix Checklist

When something doesn't work:

- [ ] Check logs: `docker logs container_name`
- [ ] Verify container is running: `docker ps`
- [ ] Check port mappings: `docker ps` (PORTS column)
- [ ] Inspect container: `docker inspect container_name`
- [ ] Test from inside container: `docker exec -it container_name bash`
- [ ] Check disk space: `docker system df`
- [ ] Restart Docker: `sudo systemctl restart docker`
- [ ] Clean up: `docker system prune`
- [ ] Rebuild: `docker-compose up --build`

---

## Getting Help

### Docker Commands for Debugging

```bash
docker version      # Docker version
docker info        # System info
docker events      # Real-time events
docker system df   # Disk usage
docker stats       # Resource usage
```

### Where to Get Help

1. **Docker Documentation**: https://docs.docker.com
2. **Stack Overflow**: https://stackoverflow.com/questions/tagged/docker
3. **Docker Forums**: https://forums.docker.com
4. **GitHub Issues**: https://github.com/docker/docker-ce/issues

---

## What's Next?

Now that you know how to fix issues:

1. **Best Practices**: Read [06_tips_and_best_practices.md](06_tips_and_best_practices.md)
2. **Keep this handy**: Bookmark for quick reference
3. **Learn from errors**: Each error teaches something new

---

**Ready for best practices?** → [06. Tips & Best Practices](06_tips_and_best_practices.md)

---
