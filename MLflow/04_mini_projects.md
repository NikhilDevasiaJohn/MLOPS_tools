# 04. Mini Projects - Hands-On Practice

---

## 📋 Project Index

1. **[Basic Experiment Tracking](#project-1-basic-experiment-tracking)** - 15 min
2. **[Model Logging and Loading](#project-2-model-logging-and-loading)** - 20 min
3. **[Hyperparameter Tuning](#project-3-hyperparameter-tuning-with-tracking)** - 25 min
4. **[Model Registry Workflow](#project-4-model-registry-workflow)** - 20 min
5. **[Remote Tracking Server](#project-5-remote-tracking-server)** - 25 min
6. **[Complete MLOps Pipeline](#project-6-complete-mlops-pipeline)** - 60 min

---

## Project 1: Basic Experiment Tracking

**⏱️ Time: 15 minutes**
**🎯 Goal**: Learn to track parameters, metrics, and artifacts

### Step 1: Setup

```bash
mkdir mlflow_project1
cd mlflow_project1

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install mlflow scikit-learn pandas numpy matplotlib
```

### Step 2: Create Training Script

Create `train.py`:

```python
import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt

# Set experiment
mlflow.set_experiment("first_experiment")

# Generate sample data
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run(run_name="baseline_model"):

    # Define hyperparameters
    n_estimators = 100
    max_depth = 5
    random_state = 42

    # Log parameters
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("random_state", random_state)

    # Train model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # Create and log a plot
    fig, ax = plt.subplots()
    feature_importance = model.feature_importances_
    ax.bar(range(len(feature_importance)), feature_importance)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance")
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    # Log a text file
    with open("model_info.txt", "w") as f:
        f.write(f"Model: Random Forest\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
    mlflow.log_artifact("model_info.txt")

    print(f"✅ Logged run with accuracy: {accuracy:.4f}")
```

### Step 3: Run and View Results

```bash
# Run the script
python train.py

# Launch MLflow UI
mlflow ui

# Open browser: http://localhost:5000
```

### Step 4: Explore the UI

1. **Experiments page**: See "first_experiment"
2. **Click on the run**: View all logged data
3. **Artifacts**: Download feature_importance.png and model_info.txt
4. **Metrics**: See accuracy, precision, recall

### ✅ Success Criteria
- [x] Script runs without errors
- [x] MLflow UI shows your experiment
- [x] Artifacts are visible and downloadable
- [x] Metrics are logged correctly

---

## Project 2: Model Logging and Loading

**⏱️ Time: 20 minutes**
**🎯 Goal**: Log models and reload them for predictions

### Step 1: Create Model Training Script

Create `train_model.py`:

```python
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mlflow.models.signature import infer_signature

# Load data
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("model_logging")

with mlflow.start_run(run_name="iris_classifier"):

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Calculate accuracy
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Infer model signature
    signature = infer_signature(X_train, model.predict(X_train))

    # Create input example
    input_example = X_train[:5]

    # Log model with signature and example
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=input_example,
        registered_model_name="IrisClassifier"
    )

    # Save run ID for later use
    run_id = mlflow.active_run().info.run_id
    print(f"✅ Model logged! Run ID: {run_id}")
    print(f"Accuracy: {accuracy:.4f}")

    # Save run_id to file for next script
    with open("run_id.txt", "w") as f:
        f.write(run_id)
```

### Step 2: Create Prediction Script

Create `predict.py`:

```python
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_iris

# Load the run_id
with open("run_id.txt", "r") as f:
    run_id = f.read().strip()

# Method 1: Load by run_id
print("Method 1: Loading by run_id...")
model_uri = f"runs:/{run_id}/model"
model = mlflow.sklearn.load_model(model_uri)

# Make predictions
iris = load_iris()
X_sample = iris.data[:5]

predictions = model.predict(X_sample)
print(f"Predictions: {predictions}")
print(f"Class names: {[iris.target_names[p] for p in predictions]}")

# Method 2: Load from Model Registry
print("\nMethod 2: Loading from Model Registry...")
model_registry = mlflow.pyfunc.load_model("models:/IrisClassifier/latest")
predictions_registry = model_registry.predict(X_sample)
print(f"Predictions from registry: {predictions_registry}")

print("\n✅ Model loaded and predictions made successfully!")
```

### Step 3: Run Both Scripts

```bash
# Train and log model
python train_model.py

# Load model and predict
python predict.py
```

### Step 4: Verify in UI

```bash
mlflow ui
```

Navigate to:
1. **Experiments** → "model_logging" → Click on run
2. **Artifacts** → "model" → See MLmodel file
3. **Models** tab (top) → See "IrisClassifier" registered

### ✅ Success Criteria
- [x] Model is logged with signature
- [x] Model appears in Model Registry
- [x] Predictions work with both loading methods
- [x] Input example is visible in UI

---

## Project 3: Hyperparameter Tuning with Tracking

**⏱️ Time: 25 minutes**
**🎯 Goal**: Track multiple experiments with different hyperparameters

### Step 1: Create Tuning Script

Create `hyperparameter_tuning.py`:

```python
import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from itertools import product

# Load data
wine = load_wine()
X, y = wine.data, wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set experiment
mlflow.set_experiment("hyperparameter_tuning")

# Define hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 10],
    'min_samples_split': [2, 5, 10]
}

# Generate all combinations
param_combinations = list(product(
    param_grid['n_estimators'],
    param_grid['max_depth'],
    param_grid['min_samples_split']
))

print(f"Testing {len(param_combinations)} combinations...")

best_accuracy = 0
best_params = {}
best_run_id = None

# Try each combination
for n_est, max_d, min_samples in param_combinations:

    with mlflow.start_run(run_name=f"rf_n{n_est}_d{max_d}_s{min_samples}"):

        # Log parameters
        params = {
            "n_estimators": n_est,
            "max_depth": max_d,
            "min_samples_split": min_samples
        }
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(
            n_estimators=n_est,
            max_depth=max_d,
            min_samples_split=min_samples,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Log metrics
        mlflow.log_metrics({
            "accuracy": accuracy,
            "f1_score": f1
        })

        # Track best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_run_id = mlflow.active_run().info.run_id

            # Tag best run
            mlflow.set_tag("best_model", "true")

            # Log model
            mlflow.sklearn.log_model(model, "model")

        print(f"n_est={n_est}, max_d={max_d}, min_s={min_samples} → accuracy={accuracy:.4f}")

print(f"\n✅ Best model: accuracy={best_accuracy:.4f}")
print(f"Best params: {best_params}")
print(f"Best run ID: {best_run_id}")
```

### Step 2: Create Analysis Script

Create `analyze_results.py`:

```python
import mlflow
import pandas as pd
import matplotlib.pyplot as plt

# Set experiment
mlflow.set_experiment("hyperparameter_tuning")

# Search all runs
runs = mlflow.search_runs(
    filter_string="",
    order_by=["metrics.accuracy DESC"]
)

# Display top 10
print("Top 10 runs by accuracy:")
print(runs[["run_id", "params.n_estimators", "params.max_depth",
           "params.min_samples_split", "metrics.accuracy"]].head(10))

# Find best run
best_run = runs.iloc[0]
print(f"\n✅ Best Run ID: {best_run['run_id']}")
print(f"Accuracy: {best_run['metrics.accuracy']:.4f}")
print(f"Parameters:")
print(f"  n_estimators: {best_run['params.n_estimators']}")
print(f"  max_depth: {best_run['params.max_depth']}")
print(f"  min_samples_split: {best_run['params.min_samples_split']}")

# Create visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Plot 1: n_estimators vs accuracy
runs_sorted = runs.sort_values('params.n_estimators')
axes[0].scatter(runs_sorted['params.n_estimators'].astype(int),
                runs_sorted['metrics.accuracy'])
axes[0].set_xlabel('n_estimators')
axes[0].set_ylabel('Accuracy')
axes[0].set_title('n_estimators vs Accuracy')

# Plot 2: max_depth vs accuracy
runs_sorted = runs.sort_values('params.max_depth')
axes[1].scatter(runs_sorted['params.max_depth'].astype(int),
                runs_sorted['metrics.accuracy'])
axes[1].set_xlabel('max_depth')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('max_depth vs Accuracy')

# Plot 3: min_samples_split vs accuracy
runs_sorted = runs.sort_values('params.min_samples_split')
axes[2].scatter(runs_sorted['params.min_samples_split'].astype(int),
                runs_sorted['metrics.accuracy'])
axes[2].set_xlabel('min_samples_split')
axes[2].set_ylabel('Accuracy')
axes[2].set_title('min_samples_split vs Accuracy')

plt.tight_layout()
plt.savefig('hyperparameter_analysis.png', dpi=150)
print("\n📊 Saved visualization to hyperparameter_analysis.png")
```

### Step 3: Run Scripts

```bash
# Run hyperparameter tuning (this will take a minute)
python hyperparameter_tuning.py

# Analyze results
python analyze_results.py

# Launch UI
mlflow ui
```

### Step 4: Explore Results

In the UI:
1. Click on "hyperparameter_tuning" experiment
2. Use the "+" button to add columns for parameters
3. Sort by accuracy
4. Compare runs visually
5. Use parallel coordinates plot

### ✅ Success Criteria
- [x] 27 runs are created (3×3×3 combinations)
- [x] Best run is identified and tagged
- [x] Analysis script generates visualization
- [x] Can filter and sort runs in UI

---

## Project 4: Model Registry Workflow

**⏱️ Time: 20 minutes**
**🎯 Goal**: Use Model Registry for version control and lifecycle management

### Step 1: Train Multiple Model Versions

Create `train_versions.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from mlflow.models.signature import infer_signature

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("model_versioning")

# Version 1: Logistic Regression
print("Training Version 1: Logistic Regression...")
with mlflow.start_run(run_name="v1_logistic_regression"):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("model_type", "logistic_regression")

    signature = infer_signature(X_train, model.predict(X_train))

    # Register model
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="CancerDetectionModel"
    )

    print(f"✅ Version 1 accuracy: {accuracy:.4f}")

# Version 2: Random Forest (shallow)
print("\nTraining Version 2: Random Forest (shallow)...")
with mlflow.start_run(run_name="v2_random_forest_shallow"):
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 50)
    mlflow.log_param("max_depth", 5)

    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="CancerDetectionModel"
    )

    print(f"✅ Version 2 accuracy: {accuracy:.4f}")

# Version 3: Random Forest (deep)
print("\nTraining Version 3: Random Forest (deep)...")
with mlflow.start_run(run_name="v3_random_forest_deep"):
    model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 200)
    mlflow.log_param("max_depth", 20)

    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        registered_model_name="CancerDetectionModel"
    )

    print(f"✅ Version 3 accuracy: {accuracy:.4f}")

print("\n✅ All versions trained and registered!")
```

### Step 2: Manage Model Lifecycle

Create `manage_lifecycle.py`:

```python
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get all versions
model_name = "CancerDetectionModel"
versions = client.search_model_versions(f"name='{model_name}'")

print(f"Model '{model_name}' has {len(versions)} versions:\n")

# Display all versions with their metrics
for version in sorted(versions, key=lambda x: int(x.version)):
    run = client.get_run(version.run_id)
    accuracy = run.data.metrics.get("accuracy", "N/A")
    print(f"Version {version.version}:")
    print(f"  Stage: {version.current_stage}")
    print(f"  Accuracy: {accuracy}")
    print(f"  Run ID: {version.run_id}\n")

# Promote version 2 to Staging
print("Promoting Version 2 to Staging...")
client.transition_model_version_stage(
    name=model_name,
    version=2,
    stage="Staging"
)

# Promote version 3 to Production
print("Promoting Version 3 to Production...")
client.transition_model_version_stage(
    name=model_name,
    version=3,
    stage="Production"
)

# Archive version 1
print("Archiving Version 1...")
client.transition_model_version_stage(
    name=model_name,
    version=1,
    stage="Archived"
)

# Add descriptions
client.update_model_version(
    name=model_name,
    version=1,
    description="Baseline logistic regression model"
)

client.update_model_version(
    name=model_name,
    version=2,
    description="Random forest with shallow trees - faster inference"
)

client.update_model_version(
    name=model_name,
    version=3,
    description="Deep random forest - best accuracy, production model"
)

print("\n✅ Model lifecycle managed!")

# Show final state
print("\nFinal model states:")
versions = client.search_model_versions(f"name='{model_name}'")
for version in sorted(versions, key=lambda x: int(x.version)):
    print(f"Version {version.version}: {version.current_stage}")
```

### Step 3: Load Production Model

Create `load_production.py`:

```python
import mlflow
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load production model
model_name = "CancerDetectionModel"
model_uri = f"models:/{model_name}/Production"

print(f"Loading production model: {model_uri}")
model = mlflow.pyfunc.load_model(model_uri)

# Get sample data
data = load_breast_cancer()
X_sample = data.data[:5]

# Make predictions
predictions = model.predict(X_sample)

print("\nPredictions from Production model:")
print(predictions)
print(f"\nPredicted classes: {['Malignant' if p == 0 else 'Benign' for p in predictions]}")

print("\n✅ Successfully loaded and used production model!")
```

### Step 4: Run All Scripts

```bash
python train_versions.py
python manage_lifecycle.py
python load_production.py
```

### ✅ Success Criteria
- [x] 3 model versions are registered
- [x] Versions have different stages (Staging, Production, Archived)
- [x] Can load production model by stage name
- [x] Descriptions are added to versions

---

## Project 5: Remote Tracking Server

**⏱️ Time: 25 minutes**
**🎯 Goal**: Set up and use a centralized tracking server

### Step 1: Start Tracking Server

```bash
# Create directory for server
mkdir mlflow_server
cd mlflow_server

# Start server with SQLite backend
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 \
  --port 5000
```

Keep this terminal running!

### Step 2: Create Client Script (New Terminal)

In a new terminal:

```bash
cd ..
mkdir mlflow_client
cd mlflow_client
```

Create `remote_tracking.py`:

```python
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Connect to remote server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Verify connection
print(f"Tracking URI: {mlflow.get_tracking_uri()}")

# Create experiment
mlflow.set_experiment("remote_experiment")

# Load data
digits = load_digits()
X, y = digits.data, digits.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train and log
print("Training model and logging to remote server...")
with mlflow.start_run(run_name="remote_run"):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

    print(f"✅ Logged to remote server! Accuracy: {accuracy:.4f}")

# List experiments on remote server
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

print(f"\nExperiments on remote server:")
for exp in experiments:
    print(f"  - {exp.name} (ID: {exp.experiment_id})")
```

Run it:

```bash
python remote_tracking.py
```

### Step 3: Access from Different Location

Create `remote_client2.py` in a different directory:

```python
import mlflow

# Connect to same server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Search runs
runs = mlflow.search_runs(experiment_names=["remote_experiment"])

print("Runs from remote server:")
print(runs[["run_id", "params.model_type", "metrics.accuracy"]])

print("\n✅ Successfully accessed remote tracking data!")
```

### Step 4: View in Browser

Open browser and go to: `http://127.0.0.1:5000`

You should see all experiments tracked on the server!

### ✅ Success Criteria
- [x] Tracking server is running
- [x] Can log from remote client
- [x] Can query runs from different location
- [x] UI accessible in browser

---

## Project 6: Complete MLOps Pipeline

**⏱️ Time: 60 minutes**
**🎯 Goal**: Build end-to-end pipeline with data processing, training, evaluation, and deployment

### Project Structure

```bash
mkdir mlops_pipeline
cd mlops_pipeline
mkdir -p src/{data,features,models,evaluation}
touch src/__init__.py
```

### Step 1: Data Loading

Create `src/data/load_data.py`:

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
import mlflow

def load_and_split_data(test_size=0.2, random_state=42):
    """Load and split California housing dataset"""

    # Log data loading
    mlflow.log_param("dataset", "california_housing")
    mlflow.log_param("test_size", test_size)
    mlflow.log_param("random_state", random_state)

    # Load data
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    # Log dataset info
    mlflow.log_param("n_samples", len(df))
    mlflow.log_param("n_features", len(data.feature_names))

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test, data.feature_names

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
```

### Step 2: Feature Engineering

Create `src/features/build_features.py`:

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
import mlflow

def create_features(X_train, X_test, scale=True):
    """Create and transform features"""

    mlflow.log_param("feature_scaling", scale)

    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Log scaler statistics
        mlflow.log_param("scaler_mean", scaler.mean_[0])
        mlflow.log_param("scaler_std", scaler.scale_[0])

        return X_train_scaled, X_test_scaled, scaler

    return X_train, X_test, None

if __name__ == "__main__":
    from src.data.load_data import load_and_split_data
    X_train, X_test, y_train, y_test, _ = load_and_split_data()
    X_train_s, X_test_s, scaler = create_features(X_train, X_test)
    print(f"Scaled train: {X_train_s.shape}")
```

### Step 3: Model Training

Create `src/models/train.py`:

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import mlflow
import mlflow.sklearn
import pickle

def train_model(X_train, y_train, model_type="gbr", **params):
    """Train regression model"""

    mlflow.log_param("model_type", model_type)
    mlflow.log_params(params)

    if model_type == "gbr":
        model = GradientBoostingRegressor(**params)
    elif model_type == "ridge":
        model = Ridge(**params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.fit(X_train, y_train)

    return model

if __name__ == "__main__":
    from src.data.load_data import load_and_split_data
    from src.features.build_features import create_features

    X_train, X_test, y_train, y_test, _ = load_and_split_data()
    X_train, X_test, _ = create_features(X_train, X_test)

    model = train_model(X_train, y_train, model_type="gbr",
                       n_estimators=100, max_depth=3)
    print(f"Model trained: {model}")
```

### Step 4: Evaluation

Create `src/evaluation/evaluate.py`:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import mlflow
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """Evaluate model and log metrics"""

    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Log metrics
    metrics = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    mlflow.log_metrics(metrics)

    # Create residual plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Actual vs Predicted
    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()],
             [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Actual vs Predicted')

    # Residuals
    residuals = y_test - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')

    plt.tight_layout()
    plt.savefig('evaluation_plots.png')
    mlflow.log_artifact('evaluation_plots.png')
    plt.close()

    return metrics

if __name__ == "__main__":
    from src.data.load_data import load_and_split_data
    from src.features.build_features import create_features
    from src.models.train import train_model

    X_train, X_test, y_train, y_test, _ = load_and_split_data()
    X_train, X_test, _ = create_features(X_train, X_test)
    model = train_model(X_train, y_train, model_type="gbr",
                       n_estimators=100, max_depth=3)
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Metrics: {metrics}")
```

### Step 5: Main Pipeline

Create `main.py`:

```python
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.data.load_data import load_and_split_data
from src.features.build_features import create_features
from src.models.train import train_model
from src.evaluation.evaluate import evaluate_model
import pickle

# Configuration
CONFIG = {
    "test_size": 0.2,
    "random_state": 42,
    "scale_features": True,
    "model_type": "gbr",
    "model_params": {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "random_state": 42
    }
}

# Set experiment
mlflow.set_experiment("mlops_pipeline")

# Start MLflow run
with mlflow.start_run(run_name="complete_pipeline"):

    # Log configuration
    mlflow.log_params({"config_" + k: str(v) for k, v in CONFIG.items()})

    # Step 1: Load data
    print("Step 1: Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_and_split_data(
        test_size=CONFIG["test_size"],
        random_state=CONFIG["random_state"]
    )

    # Step 2: Feature engineering
    print("Step 2: Engineering features...")
    X_train, X_test, scaler = create_features(
        X_train, X_test,
        scale=CONFIG["scale_features"]
    )

    # Save scaler
    if scaler:
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        mlflow.log_artifact("scaler.pkl")

    # Step 3: Train model
    print("Step 3: Training model...")
    model = train_model(
        X_train, y_train,
        model_type=CONFIG["model_type"],
        **CONFIG["model_params"]
    )

    # Step 4: Evaluate
    print("Step 4: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test)

    print(f"\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    # Step 5: Log model
    print("\nStep 5: Logging model...")
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=X_train[:5],
        registered_model_name="HousingPriceModel"
    )

    # Add tags
    mlflow.set_tags({
        "project": "mlops_pipeline",
        "team": "data-science",
        "model_family": "ensemble"
    })

    print("\n✅ Pipeline completed successfully!")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

# Launch UI
print("\nLaunch MLflow UI with: mlflow ui")
```

### Step 6: Run Pipeline

```bash
python main.py
```

### Step 7: Model Serving

Create `serve_model.sh`:

```bash
#!/bin/bash

# Get latest production model
mlflow models serve \
  -m "models:/HousingPriceModel/latest" \
  -h 0.0.0.0 \
  -p 5001 \
  --no-conda
```

Make it executable and run:

```bash
chmod +x serve_model.sh
./serve_model.sh
```

### Step 8: Test Deployment

Create `test_deployment.py`:

```python
import requests
import json
import numpy as np

# Sample input (8 features for California housing)
data = {
    "dataframe_split": {
        "columns": ["feature1", "feature2", "feature3", "feature4",
                   "feature5", "feature6", "feature7", "feature8"],
        "data": [[8.3252, 41.0, 6.98, 1.02, 322.0, 2.55, 37.88, -122.23]]
    }
}

# Make prediction request
url = "http://127.0.0.1:5001/invocations"
headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(data), headers=headers)

if response.status_code == 200:
    prediction = response.json()
    print(f"✅ Prediction: {prediction}")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
```

### ✅ Success Criteria
- [x] Complete pipeline runs end-to-end
- [x] All steps logged to MLflow
- [x] Model registered in Model Registry
- [x] Model can be served via REST API
- [x] Predictions work via HTTP requests

---

## 🎓 Summary

You've completed all mini projects! You now know how to:

✅ Track experiments with MLflow
✅ Log and load models
✅ Perform hyperparameter tuning
✅ Use Model Registry
✅ Set up remote tracking servers
✅ Build complete MLOps pipelines

**Next steps:**
- Apply these patterns to your own projects
- Explore advanced features (autologging, custom models)
- Integrate with production systems

---
