# 05. Troubleshooting Guide

---

## 📋 Contents

1. [Installation Issues](#installation-issues)
2. [Tracking Server Problems](#tracking-server-problems)
3. [UI Access Errors](#ui-access-errors)
4. [Model Loading Failures](#model-loading-failures)
5. [Registry Connection Issues](#registry-connection-issues)
6. [Storage and Artifact Problems](#storage-and-artifact-problems)
7. [Performance Issues](#performance-issues)
8. [Debugging Strategies](#debugging-strategies)

---

## Installation Issues

### Problem: `pip install mlflow` fails

**Symptoms:**
```
ERROR: Could not install packages due to an EnvironmentError
```

**Solutions:**

```bash
# Solution 1: Upgrade pip
pip install --upgrade pip
pip install mlflow

# Solution 2: Use user install
pip install --user mlflow

# Solution 3: Use virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install mlflow

# Solution 4: Install specific version
pip install mlflow==2.8.0
```

### Problem: Import errors after installation

**Symptoms:**
```python
import mlflow
# ModuleNotFoundError: No module named 'mlflow'
```

**Solutions:**

```bash
# Check if mlflow is installed
pip list | grep mlflow

# Check Python environment
which python  # Make sure you're using the right Python

# Reinstall
pip uninstall mlflow
pip install mlflow

# Verify installation
python -c "import mlflow; print(mlflow.__version__)"
```

### Problem: Dependencies conflict

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages
```

**Solutions:**

```bash
# Create fresh environment
python -m venv fresh_env
source fresh_env/bin/activate

# Install minimal MLflow
pip install mlflow

# Install extras as needed
pip install mlflow[extras]
```

---

## Tracking Server Problems

### Problem: Cannot start tracking server

**Symptoms:**
```bash
mlflow server
# Error: Address already in use
```

**Solutions:**

```bash
# Solution 1: Use different port
mlflow server --port 5001

# Solution 2: Kill existing process
# On Linux/Mac
lsof -ti:5000 | xargs kill -9

# On Windows
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Solution 3: Check if port is available
netstat -an | grep 5000
```

### Problem: Server starts but crashes immediately

**Symptoms:**
```bash
mlflow server
# Server exits with error code 1
```

**Solutions:**

```bash
# Check if backend store is accessible
ls -la mlflow.db  # If using SQLite

# Create backend directory if needed
mkdir -p mlruns

# Start with verbose logging
mlflow server --backend-store-uri ./mlruns -v

# Check permissions
chmod 755 mlruns

# Use SQLite backend explicitly
mlflow server --backend-store-uri sqlite:///mlflow.db
```

### Problem: Database connection errors

**Symptoms:**
```
sqlalchemy.exc.OperationalError: (sqlite3.OperationalError) unable to open database file
```

**Solutions:**

```bash
# Ensure database directory exists
mkdir -p /path/to/db/directory

# Use absolute path
mlflow server --backend-store-uri sqlite:////absolute/path/mlflow.db

# Check database permissions
chmod 644 mlflow.db

# For PostgreSQL
mlflow server --backend-store-uri postgresql://user:password@localhost:5432/mlflow

# Test PostgreSQL connection
psql -h localhost -U user -d mlflow -c "SELECT 1;"
```

---

## UI Access Errors

### Problem: UI shows "Unable to load experiments"

**Symptoms:**
- UI loads but shows no experiments
- Console shows connection errors

**Solutions:**

```python
# Check tracking URI
import mlflow
print(mlflow.get_tracking_uri())

# Set correct URI
mlflow.set_tracking_uri("http://localhost:5000")

# Verify experiments exist
from mlflow.tracking import MlflowClient
client = MlflowClient()
experiments = client.search_experiments()
print(experiments)
```

### Problem: UI loads but runs are missing

**Symptoms:**
- Experiments visible but runs don't appear
- Filter issues

**Solutions:**

```python
# Check if runs exist programmatically
import mlflow
runs = mlflow.search_runs(experiment_ids=["0"])
print(f"Found {len(runs)} runs")

# Clear filters in UI
# Click "Clear" button in UI filter section

# Check run status
from mlflow.entities import ViewType
runs = mlflow.search_runs(run_view_type=ViewType.ALL)
print(runs[["run_id", "status"]])
```

### Problem: Artifacts not displaying

**Symptoms:**
- Runs visible but artifacts section is empty
- "Artifact location not accessible" error

**Solutions:**

```bash
# Check artifact location
ls -la mlruns/0/RUN_ID/artifacts/

# Verify artifact root
mlflow server --default-artifact-root ./mlruns

# For S3 artifacts
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
mlflow server --default-artifact-root s3://bucket/path

# Check permissions
chmod -R 755 mlruns
```

---

## Model Loading Failures

### Problem: Cannot load model by run_id

**Symptoms:**
```python
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")
# Error: Run 'RUN_ID' not found
```

**Solutions:**

```python
# Verify run exists
from mlflow.tracking import MlflowClient
client = MlflowClient()
try:
    run = client.get_run("RUN_ID")
    print(f"Run found: {run.info.run_id}")
except Exception as e:
    print(f"Run not found: {e}")

# List all runs to find correct ID
import mlflow
runs = mlflow.search_runs()
print(runs[["run_id", "experiment_id"]])

# Use correct tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
```

### Problem: Model flavor not supported

**Symptoms:**
```python
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")
# Error: No suitable flavor found
```

**Solutions:**

```python
# Load as pyfunc (generic loader)
model = mlflow.pyfunc.load_model("runs:/RUN_ID/model")

# Check what flavors are available
import mlflow
run = mlflow.get_run("RUN_ID")
artifacts = mlflow.tracking.MlflowClient().list_artifacts("RUN_ID", "model")
for artifact in artifacts:
    print(artifact.path)

# Install missing dependencies
pip install scikit-learn  # For sklearn models
pip install tensorflow    # For TensorFlow models
pip install torch         # For PyTorch models
```

### Problem: Model signature errors

**Symptoms:**
```python
predictions = model.predict(X)
# Error: Model input does not match expected signature
```

**Solutions:**

```python
# Check model signature
import mlflow
model_uri = "runs:/RUN_ID/model"
model = mlflow.pyfunc.load_model(model_uri)
print(model.metadata.signature)

# Convert input to correct format
import pandas as pd
if isinstance(X, np.ndarray):
    X_df = pd.DataFrame(X, columns=feature_names)
    predictions = model.predict(X_df)

# Load without signature validation
model = mlflow.pyfunc.load_model(model_uri, suppress_warnings=True)
```

---

## Registry Connection Issues

### Problem: Cannot register model

**Symptoms:**
```python
mlflow.register_model("runs:/RUN_ID/model", "ModelName")
# Error: Model registry not configured
```

**Solutions:**

```bash
# For local registry (file-based)
mlflow server --backend-store-uri sqlite:///mlflow.db

# Set registry URI (if different from tracking)
export MLFLOW_REGISTRY_URI=http://localhost:5000

# In Python
import mlflow
mlflow.set_registry_uri("http://localhost:5000")

# Verify connection
from mlflow.tracking import MlflowClient
client = MlflowClient()
models = client.search_registered_models()
print(f"Found {len(models)} registered models")
```

### Problem: Model version not found

**Symptoms:**
```python
model = mlflow.pyfunc.load_model("models:/ModelName/1")
# Error: Model version not found
```

**Solutions:**

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Check if model exists
try:
    model_info = client.get_registered_model("ModelName")
    print(f"Model exists: {model_info.name}")
except Exception as e:
    print(f"Model not found: {e}")

# List all versions
versions = client.search_model_versions(f"name='ModelName'")
for v in versions:
    print(f"Version {v.version}: {v.current_stage}")

# Load latest version
model = mlflow.pyfunc.load_model("models:/ModelName/latest")

# Load by stage
model = mlflow.pyfunc.load_model("models:/ModelName/Production")
```

### Problem: Stage transition fails

**Symptoms:**
```python
client.transition_model_version_stage("ModelName", 1, "Production")
# Error: Cannot transition model
```

**Solutions:**

```python
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Check current stage
version = client.get_model_version("ModelName", 1)
print(f"Current stage: {version.current_stage}")

# Valid stages: None, Staging, Production, Archived
client.transition_model_version_stage(
    name="ModelName",
    version=1,
    stage="Production",
    archive_existing_versions=True  # Archive old production versions
)

# Check permissions (if using remote server)
# Ensure you have write access to the registry
```

---

## Storage and Artifact Problems

### Problem: S3 artifacts not accessible

**Symptoms:**
```
Error: Unable to access artifact at s3://bucket/path
```

**Solutions:**

```bash
# Set AWS credentials
export AWS_ACCESS_KEY_ID=your_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-east-1

# Install AWS dependencies
pip install boto3

# Test S3 access
aws s3 ls s3://bucket/path

# Use IAM role (on EC2)
mlflow server --default-artifact-root s3://bucket/path

# For MinIO (S3-compatible)
export MLFLOW_S3_ENDPOINT_URL=http://minio:9000
```

### Problem: Artifact too large

**Symptoms:**
```python
mlflow.log_artifact("large_file.pkl")
# Warning: Large artifact detected
```

**Solutions:**

```python
# Option 1: Compress before logging
import gzip
import pickle

with gzip.open("model_compressed.pkl.gz", "wb") as f:
    pickle.dump(model, f)
mlflow.log_artifact("model_compressed.pkl.gz")

# Option 2: Use external storage
# Store large files in S3/GCS and log URL
artifact_url = upload_to_s3(large_file)  # Custom function
mlflow.log_param("artifact_url", artifact_url)

# Option 3: Split into chunks
# Split large artifact and log separately

# Option 4: Increase timeout
mlflow.set_tracking_uri("http://localhost:5000")
import mlflow.tracking
mlflow.tracking._tracking_service.utils._chunk_size = 100 * 1024 * 1024  # 100MB
```

### Problem: Artifacts directory permissions

**Symptoms:**
```
PermissionError: [Errno 13] Permission denied: 'mlruns/...'
```

**Solutions:**

```bash
# Fix permissions
chmod -R 755 mlruns/

# Change ownership
sudo chown -R $USER:$USER mlruns/

# Use different artifact root
mlflow server --default-artifact-root /tmp/mlruns

# Check directory exists
mkdir -p mlruns
```

---

## Performance Issues

### Problem: UI is slow with many runs

**Symptoms:**
- UI takes long time to load
- Filtering is slow

**Solutions:**

```python
# Option 1: Limit runs in search
runs = mlflow.search_runs(
    max_results=100,
    order_by=["start_time DESC"]
)

# Option 2: Use database backend (not filesystem)
# Start server with PostgreSQL
mlflow server --backend-store-uri postgresql://user:password@localhost/mlflow

# Option 3: Clean up old runs
from mlflow.tracking import MlflowClient
client = MlflowClient()

# Delete old runs
experiment_id = "0"
runs = client.search_runs(experiment_id)
for run in runs[-100:]:  # Keep only recent 100
    client.delete_run(run.info.run_id)

# Option 4: Archive old experiments
client.delete_experiment(experiment_id)
```

### Problem: Model serving is slow

**Symptoms:**
```
Prediction requests take too long
```

**Solutions:**

```bash
# Use more workers
mlflow models serve -m models:/ModelName/Production -p 5000 -w 4

# Use gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "mlflow.pyfunc:serve_model(...)"

# Optimize model (batch predictions)
import mlflow
model = mlflow.pyfunc.load_model("models:/ModelName/Production")

# Batch predictions instead of single
predictions = model.predict(X_batch)  # Faster than loop

# Use GPU if available
import torch
model = mlflow.pytorch.load_model("models:/ModelName/Production")
model = model.cuda()
```

### Problem: Logging is slow

**Symptoms:**
```python
# Many log_metric calls taking too long
```

**Solutions:**

```python
# Option 1: Use log_metrics (batch)
# Instead of multiple log_metric calls
# BAD:
mlflow.log_metric("metric1", value1)
mlflow.log_metric("metric2", value2)
mlflow.log_metric("metric3", value3)

# GOOD:
mlflow.log_metrics({
    "metric1": value1,
    "metric2": value2,
    "metric3": value3
})

# Option 2: Use log_batch
from mlflow.entities import Metric, Param
metrics = [
    Metric("loss", 0.5, timestamp=1234, step=i)
    for i in range(100)
]
client.log_batch(run_id, metrics=metrics)

# Option 3: Reduce logging frequency
# Only log every N epochs
if epoch % 10 == 0:
    mlflow.log_metric("loss", loss, step=epoch)
```

---

## Debugging Strategies

### Strategy 1: Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# For MLflow specifically
logger = logging.getLogger("mlflow")
logger.setLevel(logging.DEBUG)

# Now run your code
mlflow.start_run()
# ... you'll see detailed logs
```

### Strategy 2: Inspect MLflow Directories

```bash
# Check experiment structure
tree mlruns/

# Look at specific run
ls -la mlruns/0/RUN_ID/

# Check artifacts
ls -la mlruns/0/RUN_ID/artifacts/

# View metadata
cat mlruns/0/RUN_ID/meta.yaml

# Check params
cat mlruns/0/RUN_ID/params/learning_rate

# Check metrics
cat mlruns/0/RUN_ID/metrics/accuracy
```

### Strategy 3: Use MlflowClient for Debugging

```python
from mlflow.tracking import MlflowClient
import json

client = MlflowClient()

# Get run details
run = client.get_run("RUN_ID")
print(json.dumps(run.to_dictionary(), indent=2))

# List all artifacts
artifacts = client.list_artifacts("RUN_ID")
for artifact in artifacts:
    print(f"{artifact.path} - {artifact.file_size} bytes")

# Get metric history
metric_history = client.get_metric_history("RUN_ID", "loss")
for metric in metric_history:
    print(f"Step {metric.step}: {metric.value}")

# Check experiment
experiment = client.get_experiment("0")
print(f"Experiment: {experiment.name}")
print(f"Artifact location: {experiment.artifact_location}")
```

### Strategy 4: Test Connections

```python
import mlflow

# Test tracking URI
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Test connection
try:
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    print(f"✅ Connected! Found {len(experiments)} experiments")
except Exception as e:
    print(f"❌ Connection failed: {e}")

# Test artifact storage
try:
    with mlflow.start_run():
        mlflow.log_param("test", "value")
        print("✅ Can log parameters")
except Exception as e:
    print(f"❌ Logging failed: {e}")
```

### Strategy 5: Check Environment Variables

```bash
# Print all MLflow-related env vars
env | grep MLFLOW

# Common variables
echo $MLFLOW_TRACKING_URI
echo $MLFLOW_EXPERIMENT_NAME
echo $MLFLOW_ARTIFACT_ROOT
```

---

## Common Error Messages and Solutions

### Error: "No such file or directory: 'mlruns'"

**Solution:**
```bash
mkdir mlruns
# Or set tracking URI
mlflow.set_tracking_uri("file:///absolute/path/to/mlruns")
```

### Error: "Run ID not found"

**Solution:**
```python
# Check you're connected to right tracking server
mlflow.set_tracking_uri("http://localhost:5000")

# List all runs
runs = mlflow.search_runs()
print(runs["run_id"])
```

### Error: "Experiment ID does not exist"

**Solution:**
```python
# Create experiment first
mlflow.create_experiment("experiment_name")

# Or use set_experiment (creates if not exists)
mlflow.set_experiment("experiment_name")
```

### Error: "Cannot log parameters after run has ended"

**Solution:**
```python
# Ensure you're inside with block
with mlflow.start_run():
    mlflow.log_param("param", value)  # ✅ INSIDE
    mlflow.log_metric("metric", value)

# Not here! Run has ended
# mlflow.log_param("param", value)  # ❌ OUTSIDE
```

---

## Quick Diagnostic Checklist

When things go wrong:

- [ ] Is MLflow installed? `pip list | grep mlflow`
- [ ] Is the tracking server running? `netstat -an | grep 5000`
- [ ] Is the tracking URI correct? `mlflow.get_tracking_uri()`
- [ ] Do the directories exist? `ls -la mlruns/`
- [ ] Are permissions correct? `ls -la mlruns/`
- [ ] Are dependencies installed? `pip list`
- [ ] Is the run ID correct? `mlflow.search_runs()`
- [ ] Is the model logged? Check artifacts in UI
- [ ] Are credentials set? (for S3/Azure/GCS)
- [ ] Check logs with verbose mode

---

## Getting Help

If you're still stuck:

1. **Check official docs**: https://mlflow.org/docs/latest/index.html
2. **Search GitHub issues**: https://github.com/mlflow/mlflow/issues
3. **Ask on Stack Overflow**: Tag with `mlflow`
4. **Join Slack**: https://mlflow.org/slack
5. **Check logs carefully** - error messages usually point to the issue!

---
