# 04. Mini Projects

> **Learn Docker by building real projects**

---

## How to Use This Guide

1. **Type, don't copy-paste** - Builds muscle memory
2. **Do projects in order** - Each builds on previous concepts
3. **Experiment** - Try changing things, break stuff, fix it
4. **Take breaks** - Docker is vast, pace yourself

**Time commitment:** 2.5-3 hours total

---

## Project 1: Hello World Container (10 min)

**Goal:** Create your first Docker container

**Skills:** Basic Dockerfile, building, running containers

### Step 1: Create Project Directory

```bash
mkdir docker-hello && cd docker-hello
```

### Step 2: Create Application

```python
# app.py
print("Hello from Docker!")
print("Container ID:", __import__('socket').gethostname())
```

### Step 3: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY app.py .

CMD ["python", "app.py"]
```

### Step 4: Build and Run

```bash
# Build image
docker build -t hello-docker .

# Run container
docker run hello-docker

# Run with custom name
docker run --name my-hello hello-docker

# List containers
docker ps -a

# Remove container
docker rm my-hello
```

### Challenge

Modify to:
- Print current timestamp
- Print environment variable
- Use different base image (alpine)

---

## Project 2: Python Web App with Flask (20 min)

**Goal:** Containerize a simple web application

**Skills:** Port mapping, requirements, environment variables

### Step 1: Setup

```bash
mkdir flask-app && cd flask-app
```

### Step 2: Create Flask App

```python
# app.py
from flask import Flask
import os

app = Flask(__name__)

@app.route('/')
def hello():
    return f"""
    <h1>Hello from Docker!</h1>
    <p>Environment: {os.getenv('ENV', 'development')}</p>
    <p>Version: {os.getenv('VERSION', '1.0')}</p>
    """

@app.route('/health')
def health():
    return {'status': 'healthy'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### Step 3: Create Requirements

```txt
# requirements.txt
Flask==2.3.0
```

### Step 4: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY app.py .

# Expose port
EXPOSE 5000

# Run app
CMD ["python", "app.py"]
```

### Step 5: Create .dockerignore

```
# .dockerignore
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.git
.gitignore
```

### Step 6: Build and Run

```bash
# Build
docker build -t flask-app .

# Run with port mapping
docker run -d -p 5000:5000 --name webapp flask-app

# Test
curl http://localhost:5000
curl http://localhost:5000/health

# Run with environment variables
docker run -d -p 5000:5000 \
  -e ENV=production \
  -e VERSION=2.0 \
  --name webapp-prod \
  flask-app

# View logs
docker logs -f webapp-prod

# Stop and remove
docker stop webapp-prod
docker rm webapp-prod
```

### Step 7: Optimize Dockerfile

```dockerfile
# Dockerfile.optimized
FROM python:3.9-slim

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

# Switch to non-root user
USER appuser

EXPOSE 5000

CMD ["python", "app.py"]
```

### Challenge

Add:
- Healthcheck to Dockerfile
- Volume for logs
- Another route that returns JSON

---

## Project 3: Multi-Container App with Docker Compose (30 min)

**Goal:** Build a web app with database using Docker Compose

**Skills:** Docker Compose, networking, volumes, multi-container orchestration

### Step 1: Setup

```bash
mkdir todo-app && cd todo-app
```

### Step 2: Create Application

```python
# app.py
from flask import Flask, request, jsonify
import psycopg2
import os

app = Flask(__name__)

# Database connection
def get_db():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'db'),
        database=os.getenv('DB_NAME', 'todos'),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', 'secret')
    )

# Initialize database
def init_db():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS todos (
                id SERIAL PRIMARY KEY,
                task TEXT NOT NULL,
                done BOOLEAN DEFAULT FALSE
            )
        ''')
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

@app.route('/')
def home():
    return '''
    <h1>Todo API</h1>
    <p>GET /todos - List all todos</p>
    <p>POST /todos - Create todo (JSON: {"task": "..."})</p>
    <p>PUT /todos/&lt;id&gt; - Toggle done status</p>
    '''

@app.route('/todos', methods=['GET'])
def get_todos():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT id, task, done FROM todos')
    todos = [{'id': row[0], 'task': row[1], 'done': row[2]} for row in cur.fetchall()]
    cur.close()
    conn.close()
    return jsonify(todos)

@app.route('/todos', methods=['POST'])
def create_todo():
    data = request.get_json()
    conn = get_db()
    cur = conn.cursor()
    cur.execute('INSERT INTO todos (task) VALUES (%s) RETURNING id', (data['task'],))
    todo_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'id': todo_id, 'task': data['task'], 'done': False}), 201

@app.route('/todos/<int:todo_id>', methods=['PUT'])
def toggle_todo(todo_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('UPDATE todos SET done = NOT done WHERE id = %s', (todo_id,))
    conn.commit()
    cur.close()
    conn.close()
    return jsonify({'success': True})

if __name__ == '__main__':
    init_db()
    app.run(host='0.0.0.0', port=5000)
```

### Step 3: Create Requirements

```txt
# requirements.txt
Flask==2.3.0
psycopg2-binary==2.9.6
```

### Step 4: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

### Step 5: Create Docker Compose File

```yaml
# docker-compose.yml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DB_HOST=db
      - DB_NAME=todos
      - DB_USER=postgres
      - DB_PASSWORD=secret
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:13-alpine
    environment:
      - POSTGRES_DB=todos
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=secret
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### Step 6: Run with Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Test API
curl http://localhost:5000
curl -X POST -H "Content-Type: application/json" \
  -d '{"task":"Learn Docker"}' \
  http://localhost:5000/todos

curl http://localhost:5000/todos

# Execute command in web service
docker-compose exec web python -c "print('Hello')"

# Execute command in db service
docker-compose exec db psql -U postgres -d todos -c "SELECT * FROM todos;"

# Stop services
docker-compose stop

# Start again (data persists!)
docker-compose start

# Stop and remove (keeps volumes)
docker-compose down

# Stop, remove, and delete volumes
docker-compose down -v
```

### Step 7: Add Redis for Caching

Update `docker-compose.yml`:

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    environment:
      - DB_HOST=db
      - DB_NAME=todos
      - DB_USER=postgres
      - DB_PASSWORD=secret
      - REDIS_HOST=redis
    depends_on:
      - db
      - redis

  db:
    image: postgres:13-alpine
    environment:
      - POSTGRES_DB=todos
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=secret
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  postgres_data:
  redis_data:
```

### Challenge

Add:
- DELETE endpoint for todos
- Environment file (.env)
- Nginx reverse proxy
- Health checks for all services

---

## Project 4: ML Model Serving with Docker (25 min)

**Goal:** Deploy a machine learning model in a container

**Skills:** Python dependencies, model files, API serving

### Step 1: Setup

```bash
mkdir ml-docker && cd ml-docker
```

### Step 2: Train and Save Model

```python
# train_model.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
iris = load_iris()
X, y = iris.data, iris.target

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved!")
```

Run locally first:
```bash
python train_model.py
```

### Step 3: Create Prediction API

```python
# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

SPECIES = ['setosa', 'versicolor', 'virginica']

@app.route('/')
def home():
    return '''
    <h1>Iris Model API</h1>
    <p>POST /predict</p>
    <p>Body: {"features": [5.1, 3.5, 1.4, 0.2]}</p>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = np.array(data['features']).reshape(1, -1)

    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]

    return jsonify({
        'prediction': SPECIES[prediction],
        'probabilities': {
            species: float(prob)
            for species, prob in zip(SPECIES, probability)
        }
    })

@app.route('/health')
def health():
    return {'status': 'healthy', 'model': 'RandomForest'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
```

### Step 4: Create Requirements

```txt
# requirements.txt
Flask==2.3.0
scikit-learn==1.3.0
numpy==1.24.3
```

### Step 5: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model and app
COPY model.pkl .
COPY app.py .

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run app
CMD ["python", "app.py"]
```

### Step 6: Build and Run

```bash
# Build image
docker build -t ml-api .

# Run container
docker run -d -p 8000:8000 --name ml-api ml-api

# Test prediction
curl -X POST -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}' \
  http://localhost:8000/predict

# Check health
curl http://localhost:8000/health

# View logs
docker logs ml-api

# Check health status
docker inspect ml-api | grep -A 10 Health
```

### Step 7: Multi-Stage Build (Optimized)

```dockerfile
# Dockerfile.multi-stage
# Stage 1: Build
FROM python:3.9-slim AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary files
COPY --from=builder /root/.local /root/.local
COPY model.pkl .
COPY app.py .

# Update PATH
ENV PATH=/root/.local/bin:$PATH

EXPOSE 8000

CMD ["python", "app.py"]
```

```bash
# Build optimized image
docker build -f Dockerfile.multi-stage -t ml-api:optimized .

# Compare sizes
docker images | grep ml-api
```

### Challenge

Add:
- Batch prediction endpoint
- Model versioning
- Model metrics endpoint
- Prometheus metrics

---

## Project 5: Database with Persistent Volumes (20 min)

**Goal:** Understand data persistence with volumes

**Skills:** Volumes, data management, backups

### Step 1: Run PostgreSQL with Volume

```bash
# Create volume
docker volume create postgres_data

# Run PostgreSQL
docker run -d \
  --name postgres \
  -e POSTGRES_PASSWORD=mysecret \
  -e POSTGRES_DB=testdb \
  -v postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:13
```

### Step 2: Add Data

```bash
# Connect to database
docker exec -it postgres psql -U postgres -d testdb

# In psql:
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob', 'bob@example.com');

SELECT * FROM users;

\q
```

### Step 3: Test Persistence

```bash
# Stop and remove container
docker stop postgres
docker rm postgres

# Run new container with same volume
docker run -d \
  --name postgres-new \
  -e POSTGRES_PASSWORD=mysecret \
  -e POSTGRES_DB=testdb \
  -v postgres_data:/var/lib/postgresql/data \
  -p 5432:5432 \
  postgres:13

# Check data is still there!
docker exec -it postgres-new psql -U postgres -d testdb -c "SELECT * FROM users;"
```

### Step 4: Backup and Restore

```bash
# Backup database
docker exec postgres-new pg_dump -U postgres testdb > backup.sql

# Or backup entire data directory
docker run --rm \
  -v postgres_data:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/postgres_backup.tar.gz /data

# Restore from backup
docker exec -i postgres-new psql -U postgres testdb < backup.sql
```

### Step 5: Bind Mount for Config

```bash
# Create custom config
cat > postgresql.conf << EOF
max_connections = 200
shared_buffers = 256MB
EOF

# Run with config file
docker run -d \
  --name postgres-custom \
  -e POSTGRES_PASSWORD=mysecret \
  -v postgres_data:/var/lib/postgresql/data \
  -v $(pwd)/postgresql.conf:/etc/postgresql/postgresql.conf \
  -p 5432:5432 \
  postgres:13 -c config_file=/etc/postgresql/postgresql.conf
```

### Step 6: Docker Compose Version

```yaml
# docker-compose.yml
version: '3.8'

services:
  db:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: mysecret
      POSTGRES_DB: testdb
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init.sql:/docker-entrypoint-initdb.d/init.sql
      - ./postgresql.conf:/etc/postgresql/postgresql.conf
    ports:
      - "5432:5432"
    command: -c config_file=/etc/postgresql/postgresql.conf

volumes:
  postgres_data:
```

```sql
-- init.sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100)
);

INSERT INTO users (name, email) VALUES
    ('Alice', 'alice@example.com'),
    ('Bob', 'bob@example.com');
```

```bash
docker-compose up -d
```

### Challenge

Try:
- MongoDB with persistent storage
- MySQL with initialization script
- Redis with AOF persistence

---

## Project 6: Complete ML Pipeline with Docker (60 min)

**Goal:** Build a production-ready ML system

**Skills:** Everything learned so far + orchestration

### Architecture

```
├── Frontend (React/HTML)
├── API (Flask)
├── Model Training Service
├── PostgreSQL (Model metadata)
├── Redis (Caching)
└── Nginx (Reverse proxy)
```

### Step 1: Project Structure

```bash
mkdir ml-pipeline && cd ml-pipeline
mkdir -p frontend backend training nginx
```

### Step 2: Training Service

```python
# training/train.py
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import json
from datetime import datetime
import psycopg2
import os

def train_model():
    # Load data
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # Evaluate
    accuracy = accuracy_score(y_test, model.predict(X_test))

    # Save model
    model_path = '/models/model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Log to database
    conn = psycopg2.connect(
        host=os.getenv('DB_HOST', 'db'),
        database='mldb',
        user='postgres',
        password=os.getenv('DB_PASSWORD', 'secret')
    )
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS model_runs (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP,
            accuracy FLOAT,
            model_path TEXT
        )
    ''')
    cur.execute(
        'INSERT INTO model_runs (timestamp, accuracy, model_path) VALUES (%s, %s, %s)',
        (datetime.now(), accuracy, model_path)
    )
    conn.commit()
    cur.close()
    conn.close()

    print(f"Model trained! Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    train_model()
```

```dockerfile
# training/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .

CMD ["python", "train.py"]
```

```txt
# training/requirements.txt
scikit-learn==1.3.0
psycopg2-binary==2.9.6
```

### Step 3: API Service

```python
# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import redis
import json
import os

app = Flask(__name__)
CORS(app)

# Load model
with open('/models/model.pkl', 'rb') as f:
    model = pickle.load(f)

# Redis connection
cache = redis.Redis(host=os.getenv('REDIS_HOST', 'redis'), port=6379)

SPECIES = ['setosa', 'versicolor', 'virginica']

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data['features']

    # Check cache
    cache_key = f"pred:{','.join(map(str, features))}"
    cached = cache.get(cache_key)
    if cached:
        return jsonify(json.loads(cached))

    # Predict
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)[0]
    probability = model.predict_proba(features_array)[0]

    result = {
        'prediction': SPECIES[prediction],
        'probabilities': {
            species: float(prob)
            for species, prob in zip(SPECIES, probability)
        }
    }

    # Cache result
    cache.setex(cache_key, 3600, json.dumps(result))

    return jsonify(result)

@app.route('/api/health')
def health():
    return {'status': 'healthy'}

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

```dockerfile
# backend/Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
```

```txt
# backend/requirements.txt
Flask==2.3.0
Flask-CORS==4.0.0
scikit-learn==1.3.0
numpy==1.24.3
redis==4.5.5
```

### Step 4: Frontend

```html
<!-- frontend/index.html -->
<!DOCTYPE html>
<html>
<head>
    <title>Iris Classifier</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; }
        input { margin: 5px; padding: 8px; width: 200px; }
        button { padding: 10px 20px; background: #007bff; color: white; border: none; cursor: pointer; }
        #result { margin-top: 20px; padding: 15px; background: #f0f0f0; }
    </style>
</head>
<body>
    <h1>Iris Flower Classifier</h1>
    <div>
        <input type="number" id="sepal_length" placeholder="Sepal Length" step="0.1">
        <input type="number" id="sepal_width" placeholder="Sepal Width" step="0.1">
        <input type="number" id="petal_length" placeholder="Petal Length" step="0.1">
        <input type="number" id="petal_width" placeholder="Petal Width" step="0.1">
        <button onclick="predict()">Predict</button>
    </div>
    <div id="result"></div>

    <script>
        async function predict() {
            const features = [
                parseFloat(document.getElementById('sepal_length').value),
                parseFloat(document.getElementById('sepal_width').value),
                parseFloat(document.getElementById('petal_length').value),
                parseFloat(document.getElementById('petal_width').value)
            ];

            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({features})
            });

            const result = await response.json();
            document.getElementById('result').innerHTML = `
                <h3>Prediction: ${result.prediction}</h3>
                <p>Probabilities:</p>
                <ul>
                    ${Object.entries(result.probabilities)
                        .map(([k, v]) => `<li>${k}: ${(v*100).toFixed(2)}%</li>`)
                        .join('')}
                </ul>
            `;
        }
    </script>
</body>
</html>
```

```dockerfile
# frontend/Dockerfile
FROM nginx:alpine
COPY index.html /usr/share/nginx/html/
```

### Step 5: Nginx Configuration

```nginx
# nginx/nginx.conf
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server api:5000;
    }

    server {
        listen 80;

        location / {
            root /usr/share/nginx/html;
            index index.html;
        }

        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
```

```dockerfile
# nginx/Dockerfile
FROM nginx:alpine
COPY nginx.conf /etc/nginx/nginx.conf
COPY ../frontend/index.html /usr/share/nginx/html/
```

### Step 6: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  training:
    build: ./training
    volumes:
      - model_data:/models
    environment:
      - DB_HOST=db
      - DB_PASSWORD=secret
    depends_on:
      - db

  api:
    build: ./backend
    volumes:
      - model_data:/models
    environment:
      - REDIS_HOST=redis
    depends_on:
      - training
      - redis
    restart: unless-stopped

  nginx:
    build: ./nginx
    ports:
      - "80:80"
    depends_on:
      - api
    restart: unless-stopped

  db:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: mldb
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:6-alpine
    volumes:
      - redis_data:/data

volumes:
  model_data:
  postgres_data:
  redis_data:
```

### Step 7: Run the Pipeline

```bash
# Build and start all services
docker-compose up -d --build

# Check logs
docker-compose logs -f

# Train model
docker-compose run training python train.py

# Test API directly
curl -X POST http://localhost/api/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

# Open browser to http://localhost
```

### Step 8: Production Optimizations

Add to docker-compose.yml:

```yaml
version: '3.8'

services:
  api:
    build: ./backend
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### Challenge

Extend with:
- Model retraining on schedule
- Metrics dashboard (Grafana)
- Authentication
- Model versioning
- A/B testing

---

## Bonus: Quick Reference

### Project 1 Summary
```bash
docker build -t hello .
docker run hello
```

### Project 2 Summary
```bash
docker build -t flask-app .
docker run -d -p 5000:5000 flask-app
```

### Project 3 Summary
```bash
docker-compose up -d
docker-compose logs -f
docker-compose down
```

### Project 4 Summary
```bash
docker build -t ml-api .
docker run -d -p 8000:8000 ml-api
curl -X POST -H "Content-Type: application/json" \
  -d '{"features": [...]}' http://localhost:8000/predict
```

### Project 5 Summary
```bash
docker volume create data
docker run -v data:/data myapp
```

### Project 6 Summary
```bash
docker-compose up -d --build
docker-compose run training
```

---

## What You've Learned

- ✅ Writing Dockerfiles
- ✅ Building and running containers
- ✅ Port mapping and networking
- ✅ Using Docker Compose
- ✅ Managing volumes
- ✅ Multi-stage builds
- ✅ Health checks
- ✅ Environment variables
- ✅ Multi-container orchestration
- ✅ Production-ready deployments

---

## Next Steps

1. **Review**: Go back through projects and try variations
2. **Troubleshooting**: Check [05_troubleshooting.md](05_troubleshooting.md)
3. **Best Practices**: Read [06_tips_and_best_practices.md](06_tips_and_best_practices.md)
4. **Apply**: Containerize your own projects!

---

**Keep building!** → [05. Troubleshooting](05_troubleshooting.md)

---
