# 01. Introduction to Docker

> **Learn what Docker is, why it exists, and how it revolutionizes software development**

---

## What is Docker?

Docker is an **open-source platform** that automates the deployment of applications inside lightweight, portable containers.

### Simple Analogy: Shipping Containers

Think of Docker like shipping containers in the real world:

- **Before containers**: Goods were loaded onto ships in different ways - boxes, barrels, bags. Loading/unloading was slow and error-prone.
- **After containers**: Everything goes into standardized shipping containers. Any container can go on any ship, truck, or train.

Similarly:
- **Before Docker**: Applications ran differently on different machines. "It works on my machine" was a common problem.
- **After Docker**: Applications run in standardized containers. The same container works on any machine with Docker.

---

## Why Docker Exists: Problems It Solves

### Problem 1: "It Works on My Machine"

**Before Docker:**
```
Developer: "My code works perfectly!"
Tester: "It crashes on my machine..."
DevOps: "It won't even start in production!"
```

**Reason:** Different Python versions, missing dependencies, different OS settings

**With Docker:**
```dockerfile
FROM python:3.9
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
```
Same container runs everywhere: dev, test, production.

---

### Problem 2: Dependency Hell

**Scenario:**
- App A needs Python 3.7
- App B needs Python 3.9
- App C needs a specific version of PostgreSQL

**Before Docker:** Install multiple versions, manage conflicts, break things

**With Docker:** Each app runs in its own container with its own dependencies

```bash
docker run -d app-a  # Has Python 3.7
docker run -d app-b  # Has Python 3.9
docker run -d app-c  # Has PostgreSQL
```

No conflicts!

---

### Problem 3: Slow Onboarding

**Before Docker:**
```
Day 1: Install Python, Node.js, PostgreSQL, Redis...
Day 2: Debug version conflicts
Day 3: Still configuring environment
Day 4: Finally ready to code!
```

**With Docker:**
```bash
git clone project
docker-compose up
# Ready to code in 5 minutes!
```

---

### Problem 4: Deployment Complexity

**Before Docker:**
1. SSH into server
2. Install dependencies
3. Configure services
4. Pray it works
5. Debug when it doesn't

**With Docker:**
```bash
docker pull myapp:latest
docker run -d myapp
# Done!
```

---

## Containers vs Virtual Machines

### Virtual Machines (The Old Way)

```
┌─────────────────────────────────────┐
│        Application A                │
│  ┌──────────────────────────────┐   │
│  │     Guest OS (Ubuntu)        │   │
│  └──────────────────────────────┘   │
├─────────────────────────────────────┤
│        Application B                │
│  ┌──────────────────────────────┐   │
│  │     Guest OS (CentOS)        │   │
│  └──────────────────────────────┘   │
├─────────────────────────────────────┤
│         Hypervisor                  │
├─────────────────────────────────────┤
│         Host OS                     │
├─────────────────────────────────────┤
│         Hardware                    │
└─────────────────────────────────────┘
```

**Problems:**
- Each VM needs a full OS (GBs of space)
- Slow to start (minutes)
- Heavy on resources
- Expensive

---

### Containers (The Docker Way)

```
┌─────────────────────────────────────┐
│  Container A │ Container B │ Container C
│  (App + Deps)│ (App + Deps)│ (App + Deps)
├─────────────────────────────────────┤
│         Docker Engine               │
├─────────────────────────────────────┤
│         Host OS                     │
├─────────────────────────────────────┤
│         Hardware                    │
└─────────────────────────────────────┘
```

**Benefits:**
- Share the host OS (MBs instead of GBs)
- Start in seconds
- Lightweight
- Cheap to run

---

### Comparison Table

| Feature | Virtual Machine | Docker Container |
|---------|----------------|------------------|
| **Size** | GBs (1-10 GB) | MBs (50-500 MB) |
| **Startup** | Minutes | Seconds |
| **Performance** | Slower | Near-native |
| **Isolation** | Complete | Process-level |
| **Portability** | Limited | Excellent |
| **Resource Usage** | Heavy | Light |
| **Use Case** | Different OSes | Same OS, different apps |

---

## Docker Architecture: The Big Picture

```
┌──────────────────────────────────────────────┐
│               Docker Client                   │
│  (You type: docker run, docker build, etc.)  │
└──────────────────┬───────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────┐
│             Docker Daemon                     │
│  (Docker Engine - does the actual work)      │
│  - Builds images                              │
│  - Runs containers                            │
│  - Manages networks/volumes                   │
└──────────────────┬───────────────────────────┘
                   │
                   ↓
┌──────────────────────────────────────────────┐
│         Images & Containers                   │
│  - Images: Templates                          │
│  - Containers: Running instances              │
└──────────────────────────────────────────────┘
```

### Components:

1. **Docker Client**: Command-line tool you use (`docker run`, etc.)
2. **Docker Daemon**: Background service that manages everything
3. **Docker Images**: Read-only templates (like a recipe)
4. **Docker Containers**: Running instances of images (like a cake made from recipe)
5. **Docker Registry**: Storage for images (Docker Hub, like GitHub for images)

---

## How Docker Works: A Simple Example

### Step 1: Write a Dockerfile (Recipe)

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY app.py .
CMD ["python", "app.py"]
```

Think: "Recipe for making my app container"

### Step 2: Build an Image (Bake the Recipe)

```bash
docker build -t myapp .
```

Think: "Create a template from the recipe"

### Step 3: Run a Container (Use the Template)

```bash
docker run myapp
```

Think: "Create a running instance from the template"

---

## Real-World Example: Python Web App

### Without Docker

**Setup steps:**
1. Install Python 3.9
2. Create virtual environment
3. Install Flask
4. Install PostgreSQL
5. Configure database
6. Set environment variables
7. Run app

**Result:** Works on your machine, might break elsewhere

---

### With Docker

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  web:
    build: .
    ports:
      - "5000:5000"
  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: password
```

**Run:**
```bash
docker-compose up
```

**Result:** Guaranteed to work the same way everywhere!

---

## When to Use Docker

### Perfect For:

✅ **Microservices**: Run many small services independently
✅ **Development environments**: Consistent setup across team
✅ **CI/CD pipelines**: Test in isolated environments
✅ **ML model deployment**: Package model + dependencies
✅ **Legacy apps**: Isolate old dependencies
✅ **Multi-tenant applications**: Isolate customer data

### Not Ideal For:

❌ **GUI desktop applications**: Better as native apps
❌ **High-performance computing**: VMs might be better
❌ **Different OS kernels**: Need VMs (e.g., run Windows on Linux host)

---

## Docker in MLOps Context

### Problem: ML Model Deployment
```python
# Data scientist's code
import tensorflow as tf  # version 2.8
model = load_model('model.h5')
```

**On production:**
- Different TensorFlow version → Model fails
- Missing CUDA drivers → Can't use GPU
- Different Python version → Import errors

### Solution: Docker Container
```dockerfile
FROM tensorflow/tensorflow:2.8.0-gpu

WORKDIR /app
COPY model.h5 .
COPY predict.py .
COPY requirements.txt .

RUN pip install -r requirements.txt

CMD ["python", "predict.py"]
```

Now: Same environment everywhere, from laptop to production!

---

## Key Takeaways

1. **Docker = Standardized Containers** for applications
   - Write once, run anywhere

2. **Solves Real Problems**:
   - "Works on my machine"
   - Dependency conflicts
   - Slow setup
   - Deployment complexity

3. **Containers ≠ VMs**:
   - Lighter (MBs vs GBs)
   - Faster (seconds vs minutes)
   - More portable

4. **Core Concepts**:
   - **Image**: Template/blueprint
   - **Container**: Running instance
   - **Dockerfile**: Recipe to build image

5. **Use Cases**:
   - Development environments
   - Microservices
   - ML deployment
   - CI/CD

---

## What's Next?

Now that you understand **why** Docker exists and **what** problems it solves:

1. **Next**: Read [02_key_concepts.md](02_key_concepts.md) to understand the core concepts in depth
2. **Then**: Check [03_commands.md](03_commands.md) for hands-on commands
3. **Finally**: Build projects in [04_mini_projects.md](04_mini_projects.md)

---

## Quick Quiz

Test your understanding:

1. **What's the main difference between containers and VMs?**
   - Containers share the host OS, VMs have their own OS

2. **What is a Dockerfile?**
   - A recipe/instructions to build a Docker image

3. **What problem does "docker-compose" solve?**
   - Running multi-container applications easily

4. **When would you use Docker?**
   - When you need consistent environments across machines

---

**Ready to dive deeper?** → [02. Key Concepts](02_key_concepts.md)

---
