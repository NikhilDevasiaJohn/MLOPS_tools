# 06. Tips & Best Practices

---

## 📋 Contents

1. [Golden Rules](#golden-rules)
2. [Experiment Tracking Best Practices](#experiment-tracking-best-practices)
3. [Naming Conventions](#naming-conventions)
4. [Model Management](#model-management)
5. [Production Deployment](#production-deployment)
6. [Team Collaboration](#team-collaboration)
7. [Performance Optimization](#performance-optimization)
8. [Security Considerations](#security-considerations)
9. [Integration with Other Tools](#integration-with-other-tools)

---

## Golden Rules

### 1. Always Use Experiments

**❌ Bad:**
```python
with mlflow.start_run():
    # Uses default experiment
    mlflow.log_metric("accuracy", 0.95)
```

**✅ Good:**
```python
mlflow.set_experiment("customer_churn_v2")
with mlflow.start_run():
    mlflow.log_metric("accuracy", 0.95)
```

**Why:** Organization! Default experiment becomes cluttered quickly.

### 2. Log Everything That Matters

**Minimum to log:**
- All hyperparameters
- All evaluation metrics
- Model artifact
- Code version (Git commit)
- Data version
- Training duration

```python
import mlflow
import time
import subprocess

mlflow.set_experiment("complete_tracking")

with mlflow.start_run():
    start_time = time.time()

    # Log hyperparameters
    mlflow.log_params({
        "learning_rate": 0.01,
        "batch_size": 32,
        "epochs": 100
    })

    # Log data info
    mlflow.log_params({
        "dataset_version": "v2.1",
        "train_size": len(X_train),
        "test_size": len(X_test)
    })

    # Log code version
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"]
    ).decode('ascii').strip()
    mlflow.set_tag("git_commit", git_commit)

    # Train model...

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": 0.95,
        "test_accuracy": 0.92,
        "f1_score": 0.93
    })

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log duration
    duration = time.time() - start_time
    mlflow.log_metric("training_duration_seconds", duration)
```

### 3. Use Meaningful Run Names

**❌ Bad:**
```python
with mlflow.start_run():  # Auto-generated name
    pass
```

**✅ Good:**
```python
with mlflow.start_run(run_name="rf_100trees_depth10_v1"):
    pass
```

### 4. Tag Important Runs

```python
with mlflow.start_run():
    # Train model...

    if accuracy > 0.95:
        mlflow.set_tag("candidate_for_production", "true")
        mlflow.set_tag("reviewed_by", "alice")
        mlflow.set_tag("model_quality", "excellent")
```

### 5. Use Autologging When Possible

```python
import mlflow

# Enable autologging
mlflow.sklearn.autolog()
mlflow.tensorflow.autolog()
mlflow.pytorch.autolog()

# Just train - MLflow handles the rest!
with mlflow.start_run():
    model.fit(X_train, y_train)
```

---

## Experiment Tracking Best Practices

### Organize Experiments by Project Phase

```python
# POC phase
mlflow.set_experiment("customer_churn_poc")

# Development phase
mlflow.set_experiment("customer_churn_dev")

# Production experiments
mlflow.set_experiment("customer_churn_prod")
```

### Create Parent-Child Runs for Complex Workflows

```python
mlflow.set_experiment("ensemble_models")

# Parent run
with mlflow.start_run(run_name="ensemble_v1") as parent_run:
    mlflow.log_param("ensemble_type", "voting")

    # Child run 1: Random Forest
    with mlflow.start_run(run_name="random_forest", nested=True):
        rf_model = train_random_forest()
        mlflow.sklearn.log_model(rf_model, "model")
        mlflow.log_metric("accuracy", 0.92)

    # Child run 2: Gradient Boosting
    with mlflow.start_run(run_name="gradient_boosting", nested=True):
        gb_model = train_gradient_boosting()
        mlflow.sklearn.log_model(gb_model, "model")
        mlflow.log_metric("accuracy", 0.94)

    # Log ensemble results in parent
    ensemble_accuracy = 0.96
    mlflow.log_metric("ensemble_accuracy", ensemble_accuracy)
```

### Track Training Curves

```python
with mlflow.start_run():
    for epoch in range(num_epochs):
        train_loss = train_epoch()
        val_loss = validate()

        # Log both with step
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Early stopping based on metrics
        if should_stop(val_loss):
            mlflow.set_tag("early_stopped", "true")
            mlflow.log_param("stopped_at_epoch", epoch)
            break
```

### Compare Runs Systematically

```python
import mlflow
import pandas as pd

# Search runs with specific criteria
runs = mlflow.search_runs(
    experiment_names=["my_experiment"],
    filter_string="metrics.accuracy > 0.9 AND params.model_type = 'random_forest'",
    order_by=["metrics.accuracy DESC"],
    max_results=10
)

# Analyze top performers
print("Top 10 runs:")
print(runs[["run_id", "params.n_estimators", "metrics.accuracy", "metrics.f1_score"]])

# Statistical comparison
print(f"\nMean accuracy: {runs['metrics.accuracy'].mean():.4f}")
print(f"Std accuracy: {runs['metrics.accuracy'].std():.4f}")
```

---

## Naming Conventions

### Experiment Names

**Pattern:** `{project}_{model_family}_{phase}`

```python
# Good examples
"customer_churn_rf_poc"
"fraud_detection_nn_dev"
"recommendation_xgboost_prod"
"nlp_sentiment_bert_experiments"

# Bad examples
"experiment1"
"test"
"my_model"
"final_final_v2"
```

### Run Names

**Pattern:** `{model_type}_{key_params}_{version}`

```python
# Good examples
with mlflow.start_run(run_name="rf_100trees_depth5_v1"):
with mlflow.start_run(run_name="lstm_128units_dropout02_v3"):
with mlflow.start_run(run_name="bert_large_lr1e5_baseline"):

# Include date for time-series
from datetime import datetime
run_name = f"production_retrain_{datetime.now().strftime('%Y%m%d')}"
with mlflow.start_run(run_name=run_name):
```

### Model Names (Registry)

**Pattern:** `{UseCase}{ModelType}`

```python
# Good examples
"CustomerChurnClassifier"
"FraudDetectionModel"
"ProductRecommender"
"SentimentAnalyzer"

# Bad examples
"Model1"
"best_model"
"final"
```

### Parameter Names

```python
# Use clear, descriptive names
mlflow.log_param("learning_rate", 0.01)  # ✅
mlflow.log_param("lr", 0.01)  # ❌ Too short

mlflow.log_param("num_hidden_layers", 3)  # ✅
mlflow.log_param("n_layers", 3)  # ⚠️ OK but less clear

# Use prefixes for grouped params
mlflow.log_params({
    "model_n_estimators": 100,
    "model_max_depth": 5,
    "data_train_size": 0.8,
    "data_random_seed": 42,
    "optim_learning_rate": 0.01,
    "optim_batch_size": 32
})
```

---

## Model Management

### Always Add Model Signatures

```python
from mlflow.models.signature import infer_signature
import mlflow.sklearn

with mlflow.start_run():
    model.fit(X_train, y_train)

    # Infer signature from training data
    signature = infer_signature(X_train, model.predict(X_train))

    # Log with signature
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        input_example=X_train[:5]
    )
```

### Version Models in Registry

```python
# Register model with meaningful description
result = mlflow.register_model(
    "runs:/RUN_ID/model",
    "CustomerChurnClassifier",
)

# Add version description
from mlflow.tracking import MlflowClient
client = MlflowClient()

client.update_model_version(
    name="CustomerChurnClassifier",
    version=result.version,
    description=f"""
    Random Forest Classifier for customer churn prediction.
    - Accuracy: 95.2%
    - F1 Score: 93.8%
    - Trained on: 2024-01-15
    - Dataset: customer_data_v3.csv
    - Features: 25 demographic and behavioral features
    """
)
```

### Model Lifecycle Management

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Workflow: None → Staging → Production → Archived

# 1. After training, register as None stage
mlflow.register_model("runs:/RUN_ID/model", "MyModel")

# 2. After validation, move to Staging
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Staging"
)

# 3. After A/B testing, promote to Production
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production",
    archive_existing_versions=True  # Archive old production models
)

# 4. When superseded, archive
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Archived"
)
```

### Save Preprocessing Artifacts

```python
import pickle
import mlflow

with mlflow.start_run():
    # Save scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("scaler.pkl")

    # Save feature names
    with open("feature_names.txt", "w") as f:
        f.write("\n".join(feature_names))
    mlflow.log_artifact("feature_names.txt")

    # Save encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    with open("label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    mlflow.log_artifact("label_encoder.pkl")

    # Train and log model
    mlflow.sklearn.log_model(model, "model")
```

---

## Production Deployment

### Use Environment Variables for Configuration

```python
import os
import mlflow

# Configure via environment
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "default")

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
```

### Implement Model Validation Before Deployment

```python
def validate_model(model, X_test, y_test, min_accuracy=0.9):
    """Validate model meets minimum requirements"""
    accuracy = model.score(X_test, y_test)

    if accuracy < min_accuracy:
        raise ValueError(f"Model accuracy {accuracy} below threshold {min_accuracy}")

    return True

with mlflow.start_run():
    model.fit(X_train, y_train)

    # Validate before logging
    if validate_model(model, X_test, y_test, min_accuracy=0.9):
        mlflow.sklearn.log_model(model, "model")
        mlflow.set_tag("validation_passed", "true")
```

### Use Model Aliases for Deployment

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Set alias for current production model
client.set_registered_model_alias(
    name="MyModel",
    alias="production",
    version=3
)

# Load by alias (no hardcoded version numbers!)
model = mlflow.pyfunc.load_model("models:/MyModel@production")
```

### Monitor Model Performance

```python
import mlflow

# In production inference code
def predict_and_log(model, X, run_id=None):
    """Make predictions and log inference metrics"""

    predictions = model.predict(X)

    # Log inference stats
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("inference_batch_size", len(X))
        mlflow.log_metric("prediction_mean", predictions.mean())
        mlflow.log_metric("prediction_std", predictions.std())

    return predictions
```

### Version Deployment Configurations

```python
# deployment_config.yaml
model:
  name: "CustomerChurnClassifier"
  version: 3
  stage: "Production"

serving:
  host: "0.0.0.0"
  port: 5000
  workers: 4

monitoring:
  log_predictions: true
  alert_on_drift: true
```

---

## Team Collaboration

### Use Remote Tracking Server

```bash
# Set up shared tracking server
mlflow server \
  --backend-store-uri postgresql://user:password@db-server:5432/mlflow \
  --default-artifact-root s3://mlflow-artifacts/experiments \
  --host 0.0.0.0 \
  --port 5000
```

```python
# Team members connect to shared server
import mlflow
mlflow.set_tracking_uri("http://mlflow-server:5000")
```

### Use Tags for Team Communication

```python
with mlflow.start_run():
    # Train model...

    # Communication tags
    mlflow.set_tag("owner", "alice@company.com")
    mlflow.set_tag("reviewer", "bob@company.com")
    mlflow.set_tag("status", "ready_for_review")
    mlflow.set_tag("jira_ticket", "ML-123")
    mlflow.set_tag("model_purpose", "production_candidate")
```

### Document Experiments

```python
with mlflow.start_run():
    # Log detailed description
    description = """
    Experiment: Customer Churn Prediction V2

    Changes from V1:
    - Added 5 new behavioral features
    - Increased training data by 20%
    - Switched from Random Forest to XGBoost

    Results:
    - Accuracy improved from 92% to 95%
    - Inference time reduced by 30%

    Next steps:
    - A/B test in production
    - Monitor for data drift
    """

    mlflow.set_tag("description", description)

    # Log configuration
    with open("experiment_config.yaml", "w") as f:
        yaml.dump(config, f)
    mlflow.log_artifact("experiment_config.yaml")
```

### Code Review Integration

```python
import subprocess

def get_git_info():
    """Get current git information"""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode('ascii').strip()

        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        ).decode('ascii').strip()

        return commit, branch
    except:
        return "unknown", "unknown"

with mlflow.start_run():
    commit, branch = get_git_info()

    mlflow.set_tag("git_commit", commit)
    mlflow.set_tag("git_branch", branch)
    mlflow.set_tag("code_review_url", f"https://github.com/org/repo/pull/{pr_number}")
```

---

## Performance Optimization

### Batch Logging

```python
# ❌ Slow: Multiple API calls
for i in range(100):
    mlflow.log_metric(f"metric_{i}", values[i])

# ✅ Fast: Single API call
mlflow.log_metrics({f"metric_{i}": values[i] for i in range(100)})
```

### Use Database Backend for Production

```bash
# SQLite (development only)
mlflow server --backend-store-uri sqlite:///mlflow.db

# PostgreSQL (production)
mlflow server \
  --backend-store-uri postgresql://user:password@localhost:5432/mlflow
```

### Optimize Artifact Storage

```python
# Use cloud storage for artifacts
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://my-bucket/mlflow-artifacts

# Enable artifact proxying for faster downloads
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://my-bucket/mlflow-artifacts \
  --serve-artifacts
```

### Clean Up Old Runs

```python
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta

client = MlflowClient()

# Delete runs older than 90 days
cutoff_date = datetime.now() - timedelta(days=90)
cutoff_timestamp = int(cutoff_date.timestamp() * 1000)

runs = client.search_runs(
    experiment_ids=["0"],
    filter_string=f"attribute.start_time < {cutoff_timestamp}"
)

for run in runs:
    if run.info.lifecycle_stage != "deleted":
        client.delete_run(run.info.run_id)

print(f"Deleted {len(runs)} old runs")
```

---

## Security Considerations

### Secure Tracking Server

```bash
# Use authentication (with nginx/auth proxy)
# Configure HTTPS
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --host 127.0.0.1 \
  --port 5000

# Use nginx for SSL and authentication
```

### Don't Log Sensitive Data

```python
# ❌ Bad: Logging sensitive data
mlflow.log_param("api_key", "secret_key_123")
mlflow.log_param("password", "mypassword")

# ✅ Good: Use environment variables
import os
api_key = os.getenv("API_KEY")  # Don't log this!

# Log only metadata
mlflow.log_param("api_key_name", "production_key")
mlflow.log_param("api_key_rotation_date", "2024-01-15")
```

### Use IAM Roles for Cloud Storage

```bash
# Instead of hardcoding credentials
# Use IAM roles (AWS)
mlflow server \
  --backend-store-uri postgresql://user:password@localhost/mlflow \
  --default-artifact-root s3://my-bucket/mlflow

# MLflow uses IAM role attached to EC2 instance
```

### Implement Access Control

```python
# Use MLflow with authentication backend
# Example: Use Databricks MLflow with built-in access control

# Or implement custom authentication
# Using proxy (nginx, auth0, etc.)
```

---

## Integration with Other Tools

### MLflow + DVC

```python
import mlflow
import dvc.api

# Track data version with DVC
with dvc.api.open('data/dataset.csv', rev='v1.0') as f:
    data = pd.read_csv(f)

with mlflow.start_run():
    # Log DVC data version
    mlflow.log_param("dvc_data_version", "v1.0")
    mlflow.log_param("dvc_commit", dvc.api.get_url('data/dataset.csv'))

    # Train model...
```

### MLflow + Airflow

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow

def train_model(**context):
    mlflow.set_tracking_uri("http://mlflow-server:5000")
    mlflow.set_experiment("airflow_pipeline")

    with mlflow.start_run(run_name=f"daily_train_{context['ds']}"):
        # Training code...
        mlflow.log_metric("accuracy", accuracy)

dag = DAG('ml_training_pipeline', schedule_interval='@daily')

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag
)
```

### MLflow + Docker

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Set MLflow tracking URI
ENV MLFLOW_TRACKING_URI=http://mlflow-server:5000

CMD ["python", "train.py"]
```

### MLflow + Kubernetes

```yaml
# kubernetes-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: ml-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: my-training-image:latest
        env:
        - name: MLFLOW_TRACKING_URI
          value: "http://mlflow-server:5000"
        - name: EXPERIMENT_NAME
          value: "production_training"
```

---

## Summary Checklist

### Before Training
- [ ] Set experiment name
- [ ] Create run with meaningful name
- [ ] Log all hyperparameters
- [ ] Log dataset information
- [ ] Log code version (git commit)

### During Training
- [ ] Log metrics at appropriate intervals
- [ ] Log training curves (loss, accuracy over epochs)
- [ ] Save checkpoints for long training runs

### After Training
- [ ] Log final metrics
- [ ] Log confusion matrix / evaluation plots
- [ ] Log model with signature and input example
- [ ] Add tags for important runs
- [ ] Add description/notes

### Before Deployment
- [ ] Register model in Model Registry
- [ ] Add model version description
- [ ] Transition through stages (Staging → Production)
- [ ] Validate model performance
- [ ] Document deployment configuration

### In Production
- [ ] Monitor inference metrics
- [ ] Log prediction distributions
- [ ] Track model performance over time
- [ ] Set up alerts for drift/degradation
- [ ] Plan for model retraining

---

## Final Tips

1. **Start simple, scale gradually** - Begin with basic tracking, add complexity as needed
2. **Be consistent** - Follow naming conventions across team
3. **Document everything** - Use tags and descriptions liberally
4. **Automate when possible** - Use autologging, scripts, CI/CD
5. **Review regularly** - Clean up old experiments, archive unused models
6. **Share knowledge** - Document patterns, create templates for team
7. **Monitor production** - Track model performance, not just accuracy
8. **Version everything** - Data, code, models, configurations

---
