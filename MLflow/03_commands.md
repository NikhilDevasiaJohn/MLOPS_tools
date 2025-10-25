# 03. MLflow Commands Reference

---

## 📦 Installation & Setup

### Install MLflow

```bash
# Using pip
pip install mlflow

# With extras for all dependencies
pip install mlflow[extras]

# Specific version
pip install mlflow==2.8.0

# With tracking server dependencies
pip install mlflow[extras] psycopg2-binary boto3
```

### Verify Installation

```bash
mlflow --version
# Output: mlflow, version 2.8.0
```

---

## 🎯 MLflow Tracking API (Python)

### Basic Setup

```python
import mlflow

# Set tracking URI (optional, defaults to ./mlruns)
mlflow.set_tracking_uri("http://localhost:5000")

# Set experiment
mlflow.set_experiment("my_experiment")

# Get current experiment
experiment = mlflow.get_experiment_by_name("my_experiment")
print(f"Experiment ID: {experiment.experiment_id}")
```

### Start and Manage Runs

```python
# Start a run
with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    # Your code here

# Start a run with a name
with mlflow.start_run(run_name="baseline_model"):
    # Your code

# Nested runs (for multi-step workflows)
with mlflow.start_run(run_name="parent"):
    mlflow.log_param("algorithm", "ensemble")

    with mlflow.start_run(run_name="model_1", nested=True):
        mlflow.log_metric("accuracy", 0.9)

    with mlflow.start_run(run_name="model_2", nested=True):
        mlflow.log_metric("accuracy", 0.92)

# Get active run info
active_run = mlflow.active_run()
if active_run:
    print(f"Active Run ID: {active_run.info.run_id}")
```

### Logging Parameters

```python
# Log single parameter
mlflow.log_param("learning_rate", 0.01)
mlflow.log_param("n_estimators", 100)

# Log multiple parameters
params = {
    "max_depth": 5,
    "min_samples_split": 2,
    "random_state": 42
}
mlflow.log_params(params)
```

### Logging Metrics

```python
# Log single metric
mlflow.log_metric("accuracy", 0.95)
mlflow.log_metric("f1_score", 0.87)

# Log multiple metrics
metrics = {
    "precision": 0.92,
    "recall": 0.88,
    "auc": 0.94
}
mlflow.log_metrics(metrics)

# Log metric with step (for tracking over epochs)
for epoch in range(100):
    loss = train_epoch()
    mlflow.log_metric("loss", loss, step=epoch)
    mlflow.log_metric("val_loss", val_loss, step=epoch)
```

### Logging Artifacts

```python
# Log single file
mlflow.log_artifact("model.pkl")
mlflow.log_artifact("plot.png")

# Log file to specific artifact path
mlflow.log_artifact("plot.png", artifact_path="plots")

# Log entire directory
mlflow.log_artifacts("output_dir/", artifact_path="outputs")

# Log dictionary as JSON
import json
data = {"config": "values"}
with open("config.json", "w") as f:
    json.dump(data, f)
mlflow.log_artifact("config.json")

# Log text
mlflow.log_text("Model description here", "description.txt")

# Log dictionary directly
mlflow.log_dict({"key": "value"}, "data.json")

# Log figure (matplotlib)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 4, 9])
mlflow.log_figure(fig, "plot.png")
plt.close()
```

### Logging Models

```python
# Scikit-learn
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn

model = RandomForestClassifier()
model.fit(X_train, y_train)

mlflow.sklearn.log_model(model, "model")

# With signature and input example
from mlflow.models.signature import infer_signature

signature = infer_signature(X_train, model.predict(X_train))
input_example = X_train[:5]

mlflow.sklearn.log_model(
    model,
    "model",
    signature=signature,
    input_example=input_example
)

# PyTorch
import mlflow.pytorch
mlflow.pytorch.log_model(pytorch_model, "model")

# TensorFlow/Keras
import mlflow.tensorflow
mlflow.tensorflow.log_model(tf_model, "model")

# XGBoost
import mlflow.xgboost
mlflow.xgboost.log_model(xgb_model, "model")

# Custom model (pyfunc)
import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return model_input * 2

mlflow.pyfunc.log_model("model", python_model=CustomModel())
```

### Loading Models

```python
# Load by run ID
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")
model = mlflow.pyfunc.load_model("runs:/RUN_ID/model")

# Load from model registry
model = mlflow.pyfunc.load_model("models:/ModelName/1")
model = mlflow.pyfunc.load_model("models:/ModelName/Production")

# Load model from local path
model = mlflow.sklearn.load_model("file:///path/to/model")
```

### Tags

```python
# Set single tag
mlflow.set_tag("team", "data-science")
mlflow.set_tag("model_type", "classifier")

# Set multiple tags
tags = {
    "author": "alice",
    "purpose": "production",
    "priority": "high"
}
mlflow.set_tags(tags)
```

### Autologging

```python
# Enable autologging for scikit-learn
mlflow.sklearn.autolog()

# Enable autologging for TensorFlow/Keras
mlflow.tensorflow.autolog()

# Enable autologging for PyTorch
mlflow.pytorch.autolog()

# Enable autologging for XGBoost
mlflow.xgboost.autolog()

# Enable autologging for LightGBM
mlflow.lightgbm.autolog()

# Disable autologging
mlflow.autolog(disable=True)

# Configure autologging
mlflow.sklearn.autolog(
    log_input_examples=True,
    log_model_signatures=True,
    log_models=True,
    max_tuning_runs=5
)
```

---

## 🖥️ MLflow CLI Commands

### Server Management

```bash
# Start UI (uses ./mlruns by default)
mlflow ui

# Start UI on specific host and port
mlflow ui --host 0.0.0.0 --port 5000

# Start UI with backend store
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Start tracking server
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Start server with PostgreSQL backend
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://my-bucket/mlflow \
  --host 0.0.0.0 \
  --port 5000
```

### Running Projects

```bash
# Run MLflow project from current directory
mlflow run .

# Run with parameters
mlflow run . -P alpha=0.5 -P l1_ratio=0.1

# Run from Git repository
mlflow run https://github.com/user/repo -v main

# Run specific entry point
mlflow run . -e train

# Run with specific backend (local, kubernetes, etc.)
mlflow run . --backend kubernetes
```

### Model Serving

```bash
# Serve model from runs
mlflow models serve -m runs:/RUN_ID/model -p 5000

# Serve model from registry
mlflow models serve -m models:/ModelName/Production -p 5000

# Serve with specific host
mlflow models serve -m runs:/RUN_ID/model -h 0.0.0.0 -p 5000

# Serve with workers (for production)
mlflow models serve -m models:/MyModel/1 -p 5000 -w 4
```

### Making Predictions (REST API)

```bash
# Once model is served, make predictions
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "dataframe_split": {
        "columns": ["feature1", "feature2"],
        "data": [[1, 2], [3, 4]]
    }
}'

# For pandas input
curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' -d '{
    "dataframe_records": [
        {"feature1": 1, "feature2": 2},
        {"feature1": 3, "feature2": 4}
    ]
}'
```

### Model Commands

```bash
# Build Docker image for model
mlflow models build-docker -m runs:/RUN_ID/model -n my-model

# Generate Dockerfile
mlflow models generate-dockerfile -m runs:/RUN_ID/model -d ./output

# Predict using CLI
mlflow models predict \
  -m runs:/RUN_ID/model \
  -i input.csv \
  -t csv \
  -o predictions.csv
```

### Experiment Commands

```bash
# List experiments
mlflow experiments list

# Create experiment
mlflow experiments create -n "my_experiment"

# Delete experiment
mlflow experiments delete --experiment-id 1

# Restore experiment
mlflow experiments restore --experiment-id 1

# Rename experiment
mlflow experiments rename --experiment-id 1 --new-name "new_name"
```

### Run Commands

```bash
# List runs in an experiment
mlflow runs list --experiment-id 0

# Describe a run
mlflow runs describe --run-id RUN_ID

# Delete a run
mlflow runs delete --run-id RUN_ID

# Restore a run
mlflow runs restore --run-id RUN_ID
```

---

## 🗄️ Model Registry API

### Register Models

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
result = mlflow.register_model(
    "runs:/RUN_ID/model",
    "ModelName"
)

# Register with description
client.create_registered_model(
    "ModelName",
    tags={"team": "data-science"},
    description="Customer churn prediction model"
)
```

### Manage Model Versions

```python
# Transition model version stage
client.transition_model_version_stage(
    name="ModelName",
    version=1,
    stage="Staging"
)

# Update to production
client.transition_model_version_stage(
    name="ModelName",
    version=2,
    stage="Production",
    archive_existing_versions=True  # Archive old production versions
)

# Update model version description
client.update_model_version(
    name="ModelName",
    version=1,
    description="Baseline model with 85% accuracy"
)

# Set model version tag
client.set_model_version_tag(
    name="ModelName",
    version=1,
    key="validation_status",
    value="approved"
)

# Delete model version tag
client.delete_model_version_tag(
    name="ModelName",
    version=1,
    key="validation_status"
)
```

### Query Models

```python
# Get registered model
model = client.get_registered_model("ModelName")

# Get model version
version = client.get_model_version(name="ModelName", version=1)

# Search registered models
models = client.search_registered_models(
    filter_string="name LIKE 'Customer%'"
)

# Search model versions
versions = client.search_model_versions(
    filter_string="name='ModelName' AND run_id='RUN_ID'"
)

# Get latest versions
latest = client.get_latest_versions("ModelName", stages=["Production"])
```

### Delete Models

```python
# Delete model version
client.delete_model_version(name="ModelName", version=1)

# Delete registered model (all versions)
client.delete_registered_model("ModelName")
```

---

## 🔍 Search and Query API

### Search Runs

```python
from mlflow.entities import ViewType

# Search runs in an experiment
runs = mlflow.search_runs(
    experiment_ids=["0"],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"]
)

# Search across all experiments
all_runs = mlflow.search_runs(
    filter_string="params.model_type = 'random_forest'"
)

# Search with multiple conditions
runs = mlflow.search_runs(
    experiment_ids=["0", "1"],
    filter_string="metrics.accuracy > 0.9 AND params.n_estimators > 100",
    max_results=10,
    order_by=["start_time DESC"]
)

# Include deleted runs
runs = mlflow.search_runs(
    experiment_ids=["0"],
    run_view_type=ViewType.ALL
)
```

### Filter Syntax

```python
# Numeric comparisons
"metrics.accuracy > 0.9"
"params.n_estimators >= 100"
"metrics.loss < 0.1"

# String comparisons
"params.model_type = 'random_forest'"
"tags.team = 'data-science'"

# Logical operators
"metrics.accuracy > 0.9 AND params.max_depth < 10"
"tags.priority = 'high' OR tags.priority = 'critical'"

# Attribute filters
"attribute.status = 'FINISHED'"
"attribute.start_time > 1609459200000"  # Unix timestamp in ms

# Combined
"metrics.accuracy > 0.9 AND params.model_type = 'xgboost' AND tags.env = 'prod'"
```

---

## 🛠️ MlflowClient Advanced Usage

### Working with Experiments

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Create experiment
experiment_id = client.create_experiment(
    "new_experiment",
    artifact_location="s3://bucket/path",
    tags={"team": "ml-team"}
)

# Get experiment
experiment = client.get_experiment(experiment_id)

# Get experiment by name
experiment = client.get_experiment_by_name("my_experiment")

# List experiments
experiments = client.list_experiments()

# Delete experiment
client.delete_experiment(experiment_id)

# Restore experiment
client.restore_experiment(experiment_id)

# Set experiment tag
client.set_experiment_tag(experiment_id, "version", "v2")
```

### Working with Runs

```python
# Create run
run = client.create_run(experiment_id)

# Log param/metric/artifact to existing run
client.log_param(run.info.run_id, "alpha", 0.5)
client.log_metric(run.info.run_id, "rmse", 0.87)
client.log_artifact(run.info.run_id, "model.pkl")

# Log batch (more efficient)
from mlflow.entities import Metric, Param

params = [Param("lr", "0.01"), Param("epochs", "100")]
metrics = [Metric("loss", 0.5, timestamp=1234, step=0)]

client.log_batch(
    run_id=run.info.run_id,
    params=params,
    metrics=metrics
)

# Set tags
client.set_tag(run.info.run_id, "model_type", "classifier")

# Terminate run
client.set_terminated(run.info.run_id, status="FINISHED")

# Delete run
client.delete_run(run.info.run_id)
```

### Download Artifacts

```python
# Download artifacts to local directory
local_path = client.download_artifacts(
    run_id=run.info.run_id,
    path="model",  # artifact path
    dst_path="./downloads"
)

# List artifacts
artifacts = client.list_artifacts(run.info.run_id, path="")
for artifact in artifacts:
    print(f"{artifact.path} - {artifact.file_size} bytes")
```

---

## 🐍 Environment Variables

```bash
# Set tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Set registry URI (if different from tracking)
export MLFLOW_REGISTRY_URI=http://registry-server:5000

# Set experiment name
export MLFLOW_EXPERIMENT_NAME=my_experiment

# Set experiment ID
export MLFLOW_EXPERIMENT_ID=1

# Disable autologging
export MLFLOW_AUTOLOGGING=false

# Set artifact location
export MLFLOW_ARTIFACT_ROOT=s3://my-bucket/mlflow

# In Python
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
```

---

## 📊 Quick Reference Table

### Most Used Commands

| Task | Python API | CLI |
|------|-----------|-----|
| **Start UI** | - | `mlflow ui` |
| **Set experiment** | `mlflow.set_experiment("name")` | - |
| **Start run** | `with mlflow.start_run():` | - |
| **Log param** | `mlflow.log_param("key", val)` | - |
| **Log metric** | `mlflow.log_metric("key", val)` | - |
| **Log model** | `mlflow.sklearn.log_model(m, "model")` | - |
| **Load model** | `mlflow.sklearn.load_model("runs:/ID/model")` | - |
| **Serve model** | - | `mlflow models serve -m ... -p 5000` |
| **Search runs** | `mlflow.search_runs(filter_string="...")` | - |
| **Register model** | `mlflow.register_model("runs:/ID/m", "Name")` | - |

---

## 💡 Common Patterns

### Complete Training Script

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

# Enable autologging
mlflow.sklearn.autolog()

# Set experiment
mlflow.set_experiment("customer_churn")

# Training loop
for n_estimators in [50, 100, 200]:
    with mlflow.start_run(run_name=f"rf_{n_estimators}"):
        # Log additional params
        mlflow.log_param("feature_engineering", "v2")

        # Train
        model = RandomForestClassifier(n_estimators=n_estimators)
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log additional metrics
        mlflow.log_metrics({"accuracy": acc, "f1": f1})

        # Log plots
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.bar(range(len(model.feature_importances_)), model.feature_importances_)
        mlflow.log_figure(fig, "feature_importance.png")
        plt.close()
```

---
