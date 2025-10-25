# 02. Key Concepts in MLflow

---

## 🎯 Core Hierarchy

```
MLflow
  └── Experiments
        └── Runs
              ├── Parameters
              ├── Metrics
              ├── Artifacts
              ├── Tags
              └── Models
```

---

## 1. Experiments 🧪

### What is an Experiment?
**An experiment is a collection of related runs.** Think of it as a folder that groups similar ML training attempts.

### Example:
```python
import mlflow

# Set experiment (creates if doesn't exist)
mlflow.set_experiment("customer_churn_prediction")

# All runs below will be part of this experiment
with mlflow.start_run():
    # Training code here...
    pass
```

### When to create a new experiment:
- New model architecture (Random Forest vs Neural Network)
- New problem/dataset (Customer Churn vs Product Recommendation)
- New project phase (POC vs Production)

### Naming conventions:
```python
# Good names
"customer_churn_rf"
"image_classifier_resnet50"
"nlp_sentiment_bert"

# Bad names
"experiment1"
"test"
"final"
```

---

## 2. Runs 🏃

### What is a Run?
**A run is a single execution of your ML code.** Every time you train a model, that's one run.

### Anatomy of a Run:

```python
with mlflow.start_run() as run:
    # This creates ONE run
    # Each run has:
    # - Unique run_id
    # - Start time
    # - End time
    # - Status (RUNNING, FINISHED, FAILED)

    print(f"Run ID: {run.info.run_id}")
```

### Run Lifecycle:

```
START ──► RUNNING ──► FINISHED ✅
                  └──► FAILED ❌
```

### Multiple Runs:

```python
mlflow.set_experiment("hyperparameter_tuning")

for learning_rate in [0.001, 0.01, 0.1]:
    with mlflow.start_run():
        # Each iteration = one run
        mlflow.log_param("lr", learning_rate)
        # Train model...
```

This creates 3 separate runs!

---

## 3. Parameters 🎛️

### What are Parameters?
**Inputs to your model** (things you SET before training)

### Examples:
- Hyperparameters: `learning_rate=0.01`, `n_estimators=100`
- Data parameters: `train_size=0.8`, `random_seed=42`
- Model architecture: `num_layers=3`, `dropout=0.5`

### Logging Parameters:

```python
with mlflow.start_run():
    # Single parameter
    mlflow.log_param("learning_rate", 0.01)

    # Multiple parameters
    mlflow.log_params({
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42
    })
```

### Important Rules:
- ⚠️ **Parameters are immutable** (can't change once logged in a run)
- ✅ Log before training starts
- ✅ Use descriptive names
- ❌ Don't log metrics as parameters!

---

## 4. Metrics 📊

### What are Metrics?
**Outputs from your model** (things you MEASURE after training)

### Examples:
- Performance: `accuracy=0.95`, `f1_score=0.87`
- Loss: `train_loss=0.05`, `val_loss=0.08`
- Business: `revenue_impact=10000`, `cost_savings=5000`

### Logging Metrics:

```python
with mlflow.start_run():
    # Train model
    model.fit(X_train, y_train)

    # Single metric
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Multiple metrics
    mlflow.log_metrics({
        "precision": 0.92,
        "recall": 0.88,
        "f1": 0.90
    })
```

### Metrics Over Time (Logging Multiple Values):

```python
with mlflow.start_run():
    for epoch in range(100):
        loss = train_one_epoch()

        # Log same metric multiple times (with step)
        mlflow.log_metric("loss", loss, step=epoch)
```

This creates a curve you can visualize!

---

## 5. Artifacts 📦

### What are Artifacts?
**Files produced during the run** (anything you want to save)

### Examples:
- Models: `model.pkl`, `model.h5`
- Plots: `confusion_matrix.png`, `roc_curve.png`
- Data: `predictions.csv`, `feature_importance.csv`
- Code snapshots
- Configuration files

### Logging Artifacts:

```python
import matplotlib.pyplot as plt
import mlflow

with mlflow.start_run():
    # Train model...

    # 1. Save plot as artifact
    plt.plot(history)
    plt.savefig("training_history.png")
    mlflow.log_artifact("training_history.png")

    # 2. Save entire directory
    mlflow.log_artifacts("outputs/")

    # 3. Log text file
    with open("config.txt", "w") as f:
        f.write("Model configuration...")
    mlflow.log_artifact("config.txt")
```

### Organizing Artifacts:

```python
# Save to specific path in MLflow
mlflow.log_artifact("plot.png", artifact_path="visualizations")
mlflow.log_artifact("model.pkl", artifact_path="models")

# Results in:
# artifacts/
#   └── visualizations/
#       └── plot.png
#   └── models/
#       └── model.pkl
```

---

## 6. Models 🤖

### What is an MLflow Model?
**A standardized format for packaging ML models** with:
- The model itself
- Dependencies (conda.yaml or requirements.txt)
- Signature (input/output schema)
- Example data

### Model Flavors:

MLflow supports many frameworks:

| Flavor | Framework |
|--------|-----------|
| `mlflow.sklearn` | Scikit-learn |
| `mlflow.tensorflow` | TensorFlow/Keras |
| `mlflow.pytorch` | PyTorch |
| `mlflow.xgboost` | XGBoost |
| `mlflow.lightgbm` | LightGBM |
| `mlflow.pyfunc` | Custom Python functions |

### Logging Models:

```python
from sklearn.ensemble import RandomForestClassifier
import mlflow

mlflow.set_experiment("model_logging_demo")

with mlflow.start_run():
    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Log model (automatically saves as artifact)
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name="RandomForestChurn"  # Register to Model Registry
    )
```

### Loading Models:

```python
# Load by run_id
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")

# Load from registry
model = mlflow.pyfunc.load_model("models:/RandomForestChurn/1")

# Make predictions
predictions = model.predict(X_new)
```

---

## 7. Tags 🏷️

### What are Tags?
**Key-value pairs for organizing and filtering runs**

### Use Cases:
- Mark important runs: `important=true`
- Team member: `author=alice`
- Environment: `env=production`
- Purpose: `purpose=hyperparameter_tuning`

### Using Tags:

```python
with mlflow.start_run():
    # Set tags
    mlflow.set_tag("team", "data-science")
    mlflow.set_tag("model_type", "classifier")
    mlflow.set_tag("priority", "high")

    # Set multiple tags
    mlflow.set_tags({
        "author": "alice",
        "purpose": "baseline_model"
    })
```

---

## 8. Model Registry 🗄️

### What is the Model Registry?
**A central repository for managing model versions and lifecycle**

### Model Lifecycle Stages:

```
None ──► Staging ──► Production ──► Archived
```

### Registering a Model:

```python
# Method 1: During logging
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name="CustomerChurnModel"
    )

# Method 2: Register existing model
mlflow.register_model(
    "runs:/RUN_ID/model",
    "CustomerChurnModel"
)
```

### Managing Model Versions:

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Transition to staging
client.transition_model_version_stage(
    name="CustomerChurnModel",
    version=1,
    stage="Staging"
)

# Transition to production
client.transition_model_version_stage(
    name="CustomerChurnModel",
    version=2,
    stage="Production"
)

# Archive old version
client.transition_model_version_stage(
    name="CustomerChurnModel",
    version=1,
    stage="Archived"
)
```

### Loading from Registry:

```python
# Load latest production model
model = mlflow.pyfunc.load_model(
    "models:/CustomerChurnModel/Production"
)

# Load specific version
model = mlflow.pyfunc.load_model(
    "models:/CustomerChurnModel/2"
)
```

---

## 9. Autologging 🪄

### What is Autologging?
**Automatic logging of parameters, metrics, and models** without manual `log_*` calls

### Supported Frameworks:
- Scikit-learn
- TensorFlow/Keras
- PyTorch
- XGBoost
- LightGBM
- And more!

### Using Autologging:

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier

# Enable autologging for sklearn
mlflow.sklearn.autolog()

with mlflow.start_run():
    # Just train - MLflow logs everything automatically!
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    # MLflow automatically logs:
    # - All model parameters
    # - Training score
    # - Model artifact
```

### What Gets Logged:

| Framework | Parameters | Metrics | Model | Extra |
|-----------|-----------|---------|-------|-------|
| sklearn | All hyperparameters | Score | Yes | Training time |
| Keras | Optimizer, epochs | Loss, metrics | Yes | TensorBoard logs |
| PyTorch | - | - | Yes | - |
| XGBoost | All params | Eval metrics | Yes | Feature importance |

---

## 10. Tracking Server 🖥️

### What is a Tracking Server?
**A centralized server where all experiments are stored and shared**

### Modes:

**1. Local (Default)**
```python
# Stores in ./mlruns/
with mlflow.start_run():
    # ...
```

**2. Local Server**
```bash
# Terminal 1: Start server
mlflow server --host 127.0.0.1 --port 5000

# Terminal 2: Point to server
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
```

**3. Remote Server**
```python
import mlflow

# Connect to remote server
mlflow.set_tracking_uri("http://team-server.com:5000")

with mlflow.start_run():
    # Logs to remote server
    mlflow.log_param("lr", 0.01)
```

### Storage:

```
Tracking Server
  ├── Backend Store (metadata)
  │     ├── Local filesystem
  │     ├── SQLite
  │     └── PostgreSQL/MySQL
  └── Artifact Store (files)
        ├── Local filesystem
        ├── S3
        ├── Azure Blob
        └── Google Cloud Storage
```

---

## 📝 Quick Reference Table

| Concept | What | When to Use | Example |
|---------|------|-------------|---------|
| **Experiment** | Group of runs | Different models/problems | `"random_forest_tuning"` |
| **Run** | Single execution | Every training | Auto-generated ID |
| **Parameter** | Input value | Before training | `learning_rate=0.01` |
| **Metric** | Output value | After training | `accuracy=0.95` |
| **Artifact** | File output | Plots, models, data | `model.pkl`, `plot.png` |
| **Tag** | Label | Organization | `team="data-science"` |
| **Model** | Packaged model | Deployment | Saved with dependencies |
| **Registry** | Model versions | Production tracking | Version 1, 2, 3... |

---

## 🎯 Key Takeaways

1. **Experiments** group related runs
2. **Runs** track individual executions
3. **Parameters** = inputs, **Metrics** = outputs
4. **Artifacts** = any files you want to save
5. **Models** are artifacts with special packaging
6. **Registry** manages model lifecycle
7. **Autologging** reduces boilerplate code
8. **Tracking Server** enables team collaboration

---
