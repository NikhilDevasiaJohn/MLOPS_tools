# 01. Introduction to MLflow

---

## 💡 What is MLflow?

**MLflow** is an open-source platform for managing the end-to-end machine learning lifecycle. It helps you:
- Track experiments (parameters, metrics, code versions)
- Package ML code in reusable formats
- Deploy models to various platforms
- Manage and version models in a central repository

### 🔍 Breaking this down:

**MLflow solves 4 main problems:**

1. **Tracking**: How do I keep track of all my experiments?
2. **Reproducibility**: How do I recreate my results?
3. **Deployment**: How do I deploy my model?
4. **Model Management**: How do I manage different model versions?

---

## 🧩 Four Core Components

### 1. MLflow Tracking 📊
**Purpose**: Log and query experiments (parameters, metrics, artifacts)

**What it tracks:**
- Parameters (hyperparameters like learning_rate=0.01)
- Metrics (results like accuracy=0.95)
- Artifacts (files like model.pkl, plots, datasets)
- Code version (Git commit hash)
- Start/end time, duration

**Why it matters:**
- Compare different experiments
- Understand what works and what doesn't
- Share results with team

### 2. MLflow Projects 📦
**Purpose**: Package ML code in reusable, reproducible format

**What it includes:**
- Code structure
- Dependencies (conda.yaml or requirements.txt)
- Entry points (what to run)

**Why it matters:**
- Anyone can run your code
- Reproducible across environments
- Easy to share and reuse

### 3. MLflow Models 🎯
**Purpose**: Package models for deployment in various formats

**Supports:**
- Scikit-learn
- TensorFlow
- PyTorch
- XGBoost
- Custom models
- And many more!

**Why it matters:**
- Deploy to multiple platforms (REST API, cloud, batch)
- Consistent interface across frameworks
- Easy model serving

### 4. MLflow Registry 🗄️
**Purpose**: Central model store with versioning and lifecycle management

**Features:**
- Version control for models
- Stage transitions (Staging → Production)
- Annotations and descriptions
- Access control

**Why it matters:**
- Know which model is in production
- Track model history
- Collaborate on model deployment

---

## 🧠 Why MLflow?

### Common ML Problems:

**Problem 1: Experiment Chaos**
```
You: "Which model performed best last week?"
Your notes: "model_v2_final_FINAL_actually_final.pkl"
```

**MLflow Solution:**
- Automatic tracking of all experiments
- Easy comparison in UI
- Never lose results

**Problem 2: "It Works on My Machine"**
```
Colleague: "Your model doesn't run on my laptop"
You: "Did you install sklearn 0.24.2 and numpy 1.19.5?"
```

**MLflow Solution:**
- Packaged with dependencies
- Reproducible environments
- Works everywhere

**Problem 3: Model Deployment Confusion**
```
Boss: "Which model is running in production?"
You: "Ummm... model_v7... or was it v8?"
```

**MLflow Solution:**
- Model Registry tracks all versions
- Clear staging/production labels
- Audit trail of changes

---

## ⚙️ How MLflow Works

### Simple Workflow:

```python
import mlflow

# 1. Start tracking
mlflow.set_experiment("my_first_experiment")

with mlflow.start_run():
    # 2. Log parameters
    mlflow.log_param("learning_rate", 0.01)

    # 3. Train model
    model = train_model(data, lr=0.01)

    # 4. Log metrics
    mlflow.log_metric("accuracy", 0.95)

    # 5. Save model
    mlflow.sklearn.log_model(model, "model")
```

```bash
# 6. View results
mlflow ui
```

### What Happens Behind the Scenes:

```
Your Code ──► MLflow ──► mlruns/ directory
                            ├── Experiment 1/
                            │   ├── Run 1/
                            │   │   ├── params/
                            │   │   ├── metrics/
                            │   │   └── artifacts/
                            │   └── Run 2/
                            └── Experiment 2/
```

---

## 🎯 Real-World Example

### Without MLflow:
```python
# train.py
from sklearn.ensemble import RandomForestClassifier

# Try learning_rate = 0.01
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print(f"Accuracy: {accuracy}")
# Now you write this down in Excel/notebook 😅
# Then you change the parameter and forget what you tried before...
```

### With MLflow:
```python
# train.py
import mlflow
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment("random_forest_tuning")

with mlflow.start_run():
    n_estimators = 100
    mlflow.log_param("n_estimators", n_estimators)

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    mlflow.sklearn.log_model(model, "model")

# Now everything is tracked automatically! 🎉
# Run `mlflow ui` to see all experiments
```

---

## 📊 Key Benefits

| Benefit | Description |
|---------|-------------|
| **Reproducibility** | Recreate any experiment from history |
| **Comparison** | Compare 100s of runs visually |
| **Collaboration** | Share experiments with team |
| **Organization** | No more "final_model_v2_actual.pkl" |
| **Deployment** | Deploy models easily to any platform |
| **Language Agnostic** | Python, R, Java REST API |

---

## 🔄 MLflow vs Other Tools

| Tool | Purpose | How MLflow Fits |
|------|---------|-----------------|
| **Git** | Code versioning | MLflow tracks code + data + models |
| **DVC** | Data versioning | Use together! DVC for data, MLflow for experiments |
| **TensorBoard** | Visualization | MLflow is broader, supports all frameworks |
| **Weights & Biases** | Experiment tracking | Similar, but MLflow is open-source and self-hosted |

---

## 🎯 Key Takeaways

1. **MLflow = Experiment Tracking + Model Management + Deployment**
2. **4 Components**: Tracking, Projects, Models, Registry
3. **Works with any ML framework**: sklearn, PyTorch, TensorFlow, etc.
4. **Open source** and can be self-hosted
5. **Solves real problems**: No more lost experiments, "works on my machine", deployment confusion

---

## 🚀 What's Next?

Now that you understand WHAT MLflow is and WHY it exists, let's dive into:
- **Key Concepts** (Experiments, Runs, Parameters, Metrics)
- **Hands-on Examples**
- **Real Mini Projects**

---
