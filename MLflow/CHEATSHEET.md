# MLflow Cheat Sheet

Quick reference for common MLflow operations

---

## 🚀 Installation & Setup

```bash
# Install MLflow
pip install mlflow

# Install with extras
pip install mlflow[extras]

# Check version
mlflow --version
```

---

## 🎯 Basic Tracking

```python
import mlflow

# Set experiment
mlflow.set_experiment("my_experiment")

# Start run
with mlflow.start_run():
    # Log single param/metric
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)

    # Log multiple
    mlflow.log_params({"lr": 0.01, "epochs": 100})
    mlflow.log_metrics({"acc": 0.95, "loss": 0.05})

# Named run
with mlflow.start_run(run_name="baseline"):
    pass
```

---

## 📊 Logging Metrics Over Time

```python
# Log metrics with steps (for training curves)
with mlflow.start_run():
    for epoch in range(100):
        mlflow.log_metric("loss", loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
```

---

## 📦 Logging Artifacts

```python
with mlflow.start_run():
    # Single file
    mlflow.log_artifact("plot.png")

    # File to specific path
    mlflow.log_artifact("plot.png", artifact_path="plots")

    # Directory
    mlflow.log_artifacts("outputs/")

    # Figure
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    mlflow.log_figure(fig, "plot.png")

    # Dictionary as JSON
    mlflow.log_dict({"key": "value"}, "data.json")

    # Text
    mlflow.log_text("Some text", "file.txt")
```

---

## 🤖 Logging Models

```python
import mlflow.sklearn
from mlflow.models.signature import infer_signature

with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)

    # Log model (basic)
    mlflow.sklearn.log_model(model, "model")

    # Log with signature and example
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5]
    )

    # Log and register
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="MyModel"
    )
```

### Other Frameworks

```python
# PyTorch
mlflow.pytorch.log_model(model, "model")

# TensorFlow/Keras
mlflow.tensorflow.log_model(model, "model")

# XGBoost
mlflow.xgboost.log_model(model, "model")

# Generic (pyfunc)
mlflow.pyfunc.log_model("model", python_model=CustomModel())
```

---

## 🔄 Loading Models

```python
# Load by run ID
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")
model = mlflow.pyfunc.load_model("runs:/RUN_ID/model")

# Load from registry
model = mlflow.pyfunc.load_model("models:/ModelName/1")
model = mlflow.pyfunc.load_model("models:/ModelName/Production")
model = mlflow.pyfunc.load_model("models:/ModelName/latest")
```

---

## 🏷️ Tags

```python
with mlflow.start_run():
    # Single tag
    mlflow.set_tag("team", "ml-team")

    # Multiple tags
    mlflow.set_tags({
        "owner": "alice",
        "priority": "high",
        "status": "production"
    })
```

---

## 🪄 Autologging

```python
# Enable for sklearn
mlflow.sklearn.autolog()

# Enable for TensorFlow
mlflow.tensorflow.autolog()

# Enable for PyTorch
mlflow.pytorch.autolog()

# Just train - MLflow logs automatically!
with mlflow.start_run():
    model.fit(X_train, y_train)
```

---

## 🔍 Searching Runs

```python
# Basic search
runs = mlflow.search_runs(experiment_names=["my_experiment"])

# With filter
runs = mlflow.search_runs(
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Multiple conditions
runs = mlflow.search_runs(
    filter_string="metrics.accuracy > 0.9 AND params.lr < 0.1"
)

# Display specific columns
print(runs[["run_id", "params.lr", "metrics.accuracy"]])
```

---

## 🗄️ Model Registry

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Register model
mlflow.register_model("runs:/RUN_ID/model", "ModelName")

# Get model info
model = client.get_registered_model("ModelName")

# List versions
versions = client.search_model_versions("name='ModelName'")

# Transition stage
client.transition_model_version_stage(
    name="ModelName",
    version=1,
    stage="Production"  # None, Staging, Production, Archived
)

# Update description
client.update_model_version(
    name="ModelName",
    version=1,
    description="Best model - 95% accuracy"
)

# Set tags
client.set_model_version_tag(
    name="ModelName",
    version=1,
    key="validated",
    value="true"
)
```

---

## 🖥️ CLI Commands

```bash
# Start UI
mlflow ui
mlflow ui --port 5001

# Start tracking server
mlflow server --host 0.0.0.0 --port 5000

# With database backend
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns

# Serve model
mlflow models serve -m runs:/RUN_ID/model -p 5000
mlflow models serve -m models:/ModelName/Production -p 5000

# List experiments
mlflow experiments list

# Create experiment
mlflow experiments create -n "my_experiment"

# Run project
mlflow run . -P alpha=0.5
```

---

## 🌐 Tracking URI

```python
# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("file:///path/to/mlruns")
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Get current URI
uri = mlflow.get_tracking_uri()

# Using environment variable
import os
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
```

---

## 🔧 MlflowClient Advanced

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Create experiment
exp_id = client.create_experiment("new_exp")

# Get experiment
exp = client.get_experiment(exp_id)
exp = client.get_experiment_by_name("my_exp")

# Get run
run = client.get_run(run_id)

# Download artifacts
local_path = client.download_artifacts(run_id, "model")

# List artifacts
artifacts = client.list_artifacts(run_id)

# Log to existing run
client.log_param(run_id, "param", value)
client.log_metric(run_id, "metric", value)

# Delete run
client.delete_run(run_id)
```

---

## 📝 Common Patterns

### Complete Training Script

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("my_experiment")

with mlflow.start_run(run_name="baseline"):
    # Params
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5
    })

    # Train
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # Metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Model
    mlflow.sklearn.log_model(model, "model")

    # Tags
    mlflow.set_tag("status", "completed")
```

### Hyperparameter Tuning

```python
mlflow.set_experiment("hp_tuning")

for lr in [0.001, 0.01, 0.1]:
    for depth in [3, 5, 10]:
        with mlflow.start_run(run_name=f"lr{lr}_d{depth}"):
            mlflow.log_params({"lr": lr, "depth": depth})
            model = train_model(lr, depth)
            acc = evaluate(model)
            mlflow.log_metric("accuracy", acc)
```

### Parent-Child Runs

```python
with mlflow.start_run(run_name="parent"):
    mlflow.log_param("ensemble_type", "voting")

    with mlflow.start_run(run_name="model1", nested=True):
        mlflow.log_metric("accuracy", 0.92)

    with mlflow.start_run(run_name="model2", nested=True):
        mlflow.log_metric("accuracy", 0.94)

    mlflow.log_metric("ensemble_accuracy", 0.96)
```

---

## 🎯 Quick Filters

```python
# Numeric comparisons
"metrics.accuracy > 0.9"
"params.n_estimators >= 100"

# String comparisons
"params.model_type = 'random_forest'"
"tags.team = 'ml-team'"

# Logical operators
"metrics.accuracy > 0.9 AND params.lr < 0.1"
"tags.status = 'production' OR tags.status = 'staging'"

# Attributes
"attribute.status = 'FINISHED'"
"attribute.start_time > 1609459200000"
```

---

## 🐛 Common Issues

```python
# Issue: Run not found
# Solution: Check tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Issue: Cannot log after run ends
# Solution: Log inside with block
with mlflow.start_run():
    mlflow.log_metric("acc", 0.95)  # ✅

# Issue: Model not loading
# Solution: Use pyfunc for generic loading
model = mlflow.pyfunc.load_model("runs:/ID/model")

# Issue: Slow UI
# Solution: Use database backend
mlflow server --backend-store-uri postgresql://...
```

---

## 🚀 Production Tips

```python
# 1. Always log git commit
import subprocess
commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
mlflow.set_tag("git_commit", commit)

# 2. Log data version
mlflow.log_param("data_version", "v2.1")

# 3. Log training duration
import time
start = time.time()
# train...
mlflow.log_metric("duration_sec", time.time() - start)

# 4. Add descriptions
mlflow.set_tag("description", "Production model - 95% accuracy")

# 5. Use meaningful names
mlflow.set_experiment("production_model_v2")
with mlflow.start_run(run_name="rf_100_d5_20240115"):
    pass
```

---

## 📊 Environment Variables

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
export MLFLOW_EXPERIMENT_NAME=my_experiment
export MLFLOW_EXPERIMENT_ID=1
export MLFLOW_ARTIFACT_ROOT=s3://bucket/path
```

---

## 🔗 Useful Links

- **Docs:** https://mlflow.org/docs/latest/
- **GitHub:** https://github.com/mlflow/mlflow
- **API Reference:** https://mlflow.org/docs/latest/python_api/
- **Slack:** https://mlflow.org/slack

---

## 📌 Remember

- **Set experiment** before starting runs
- **Use meaningful names** for experiments and runs
- **Log everything** - params, metrics, artifacts, model
- **Add tags** for important runs
- **Use autologging** when possible
- **Search and compare** runs regularly

---

**Print this and keep it handy!** 📄
