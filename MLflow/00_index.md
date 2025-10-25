# MLflow Study Guide - Complete Index

> **A developer-friendly guide to mastering MLflow for ML experiment tracking and model management**

---

## 📚 Table of Contents

### [01. Introduction to MLflow](01_intro.md)
**What you'll learn:**
- What MLflow is and why it exists
- Core components: Tracking, Projects, Models, Registry
- How MLflow fits into ML workflows
- Key problems MLflow solves

**Time:** 10 minutes
**Best for:** Complete beginners

---

### [02. Key Concepts](02_key_concepts.md)
**What you'll learn:**
- Experiments and Runs
- Parameters, Metrics, and Artifacts
- MLflow Tracking Server
- Model Registry and Model Flavors
- MLflow Projects structure
- Autologging capabilities

**Time:** 20 minutes
**Best for:** Understanding MLflow architecture

---

### [03. Commands Reference](03_commands.md)
**What you'll learn:**
- Installation & setup
- Tracking API (`mlflow.log_param`, `mlflow.log_metric`, etc.)
- CLI commands (`mlflow ui`, `mlflow run`)
- Model management (`mlflow models serve`)
- Registry operations
- Environment setup

**Time:** 30 minutes
**Best for:** Command-line reference, quick lookup

---

### [04. Mini Projects](04_mini_projects.md)
**What you'll learn:**
- **Project 1:** Basic experiment tracking (15 min)
- **Project 2:** Model logging and loading (20 min)
- **Project 3:** Hyperparameter tuning with tracking (25 min)
- **Project 4:** Model registry workflow (20 min)
- **Project 5:** Remote tracking server (25 min)
- **Project 6:** Complete MLOps pipeline (60 min)

**Time:** 2.5-3 hours total
**Best for:** Hands-on practice, building muscle memory

---

### [05. Troubleshooting](05_troubleshooting.md)
**What you'll learn:**
- Installation issues
- Tracking server problems
- UI access errors
- Model loading failures
- Registry connection issues
- Storage and artifact problems
- Performance optimization
- Debugging strategies

**Time:** Reference as needed
**Best for:** Fixing errors, understanding what went wrong

---

### [06. Tips & Best Practices](06_tips_and_best_practices.md)
**What you'll learn:**
- Golden rules for experiment tracking
- Project organization
- Model naming conventions
- Registry best practices
- Production deployment strategies
- Team collaboration
- Performance optimization
- Integration with other tools (DVC, Airflow, etc.)

**Time:** 30 minutes
**Best for:** Professional workflows, production use

---

## 🎯 Quick Start Paths

### Path 1: Complete Beginner (4-5 hours)
1. Read [01_intro.md](01_intro.md) - 10 min
2. Read [02_key_concepts.md](02_key_concepts.md) - 20 min
3. Do [04_mini_projects.md](04_mini_projects.md) Project 1 - 15 min
4. Do [04_mini_projects.md](04_mini_projects.md) Project 2 - 20 min
5. Read [03_commands.md](03_commands.md) - 30 min
6. Do [04_mini_projects.md](04_mini_projects.md) Projects 3-4 - 45 min
7. Skim [06_tips_and_best_practices.md](06_tips_and_best_practices.md) - 30 min

### Path 2: Quick Learner (2 hours)
1. Skim [01_intro.md](01_intro.md) + [02_key_concepts.md](02_key_concepts.md) - 15 min
2. Do [04_mini_projects.md](04_mini_projects.md) Projects 1-2 - 35 min
3. Reference [03_commands.md](03_commands.md) as needed - 30 min
4. Read [06_tips_and_best_practices.md](06_tips_and_best_practices.md) - 30 min

### Path 3: Just the Essentials (30 min)
1. Read [01_intro.md](01_intro.md) - 10 min
2. Quick skim [02_key_concepts.md](02_key_concepts.md) - 5 min
3. Bookmark [03_commands.md](03_commands.md) for reference
4. Do [04_mini_projects.md](04_mini_projects.md) Project 1 - 15 min
5. Keep [05_troubleshooting.md](05_troubleshooting.md) handy

---

## 📖 How to Use This Guide

### For Learning
1. **Read sequentially** (01 → 02 → 03 → 04)
2. **Type, don't copy-paste** code
3. **Do all mini projects** for hands-on experience
4. **Experiment with different models** - track everything!

### For Reference
- Use [03_commands.md](03_commands.md) for API/command lookup
- Use [05_troubleshooting.md](05_troubleshooting.md) when stuck
- Use [06_tips_and_best_practices.md](06_tips_and_best_practices.md) for production workflows

### For Quick Recall
- Each file has clear examples
- Tables for quick scanning
- Code-heavy format
- Minimal theory, maximum practice

---

## 🎓 Learning Objectives

After completing this guide, you will be able to:

**Basic Skills:**
- ✅ Track experiments with parameters and metrics
- ✅ Log models and artifacts
- ✅ Launch MLflow UI and explore runs
- ✅ Load and use logged models
- ✅ Compare different experiment runs

**Intermediate Skills:**
- ✅ Set up tracking servers (local & remote)
- ✅ Use Model Registry for versioning
- ✅ Implement hyperparameter tuning workflows
- ✅ Deploy models with MLflow Models
- ✅ Use autologging with popular frameworks

**Advanced Skills:**
- ✅ Design production-ready tracking architecture
- ✅ Implement CI/CD for ML models
- ✅ Manage model lifecycle (staging, production)
- ✅ Integrate with cloud platforms (AWS, Azure, GCP)
- ✅ Optimize tracking performance at scale

---

## 💡 Key Concepts at a Glance

| Concept | Purpose |
|---------|---------|
| **Experiment** | Groups related runs |
| **Run** | Single execution of ML code |
| **Parameter** | Input value (hyperparameter) |
| **Metric** | Output value to track (accuracy, loss) |
| **Artifact** | File output (model, plot, data) |
| **Model** | Packaged ML model with dependencies |
| **Registry** | Central model store with versioning |
| **Tracking Server** | Centralized service for team sharing |

---

## 🔧 Essential Code Snippets

```python
import mlflow

# Start a run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 100)

    # Train model...

    # Log metrics
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_metric("loss", 0.05)

    # Log model
    mlflow.sklearn.log_model(model, "model")

    # Log artifacts
    mlflow.log_artifact("plot.png")
```

```bash
# Launch UI
mlflow ui

# Run project
mlflow run . -P alpha=0.5

# Serve model
mlflow models serve -m runs:/RUN_ID/model -p 5000
```

---

## 🚀 Project Templates

### Simple ML Experiment
```
project/
├── train.py
├── MLproject
├── conda.yaml
├── mlruns/          # Local tracking data
└── models/
```

### Complete MLflow Project
```
project/
├── src/
│   ├── data/
│   │   └── load_data.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── predict.py
│   └── evaluation/
│       └── evaluate.py
├── notebooks/
├── tests/
├── MLproject
├── conda.yaml
├── requirements.txt
└── README.md
```

---

## 📊 Workflow Diagram

```
┌─────────────────────────────────────────────┐
│           Your ML Workflow                   │
├─────────────────────────────────────────────┤
│                                              │
│  1. mlflow.start_run()                       │
│     ↓                                        │
│  2. Train model + log params/metrics         │
│     ↓                                        │
│  3. mlflow.log_model()                       │
│     ↓                                        │
│  4. mlflow ui (view results)                 │
│     ↓                                        │
│  5. Compare runs & select best               │
│     ↓                                        │
│  6. Register model to Registry               │
│     ↓                                        │
│  7. Promote to Production                    │
│     ↓                                        │
│  8. Deploy with mlflow models serve          │
│                                              │
└─────────────────────────────────────────────┘
```

---

## 🎯 Common Use Cases

### Use Case 1: Track Experiment
```python
import mlflow

mlflow.set_experiment("my_experiment")

with mlflow.start_run():
    mlflow.log_param("lr", 0.01)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.sklearn.log_model(model, "model")
```

### Use Case 2: Load and Use Model
```python
import mlflow

# Load model
model = mlflow.sklearn.load_model("runs:/RUN_ID/model")

# Make predictions
predictions = model.predict(X_test)
```

### Use Case 3: Compare Runs
```bash
# Launch UI
mlflow ui

# Navigate to http://localhost:5000
# Compare runs visually
```

### Use Case 4: Register and Deploy
```python
# Register model
result = mlflow.register_model(
    "runs:/RUN_ID/model",
    "MyModel"
)

# Transition to production
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="MyModel",
    version=1,
    stage="Production"
)
```

---

## 🆘 Quick Help

**Stuck?**
1. Check [05_troubleshooting.md](05_troubleshooting.md)
2. Run `mlflow --help` for CLI help
3. Check `mlruns/` directory for tracking files
4. Look at UI for detailed run information

**Need help?**
- MLflow Docs: https://mlflow.org/docs/latest/index.html
- MLflow GitHub: https://github.com/mlflow/mlflow
- Community Slack: https://mlflow.org/slack

---

## 📝 Practice Checklist

Mark your progress:

- [ ] Completed [01_intro.md](01_intro.md)
- [ ] Completed [02_key_concepts.md](02_key_concepts.md)
- [ ] Completed [03_commands.md](03_commands.md)
- [ ] Completed Mini Project 1
- [ ] Completed Mini Project 2
- [ ] Completed Mini Project 3
- [ ] Completed Mini Project 4
- [ ] Completed Mini Project 5
- [ ] Completed Mini Project 6
- [ ] Read [06_tips_and_best_practices.md](06_tips_and_best_practices.md)
- [ ] Tracked own ML experiments
- [ ] Used Model Registry
- [ ] Deployed a model with MLflow
- [ ] Set up remote tracking server

---

## 🏆 Mastery Goals

**Beginner Level:**
- Can track basic experiments
- Understands parameters vs metrics
- Can launch and use UI

**Intermediate Level:**
- Can log and load models
- Can use Model Registry
- Can set up tracking servers

**Advanced Level:**
- Can design production architectures
- Can integrate with CI/CD pipelines
- Can manage model lifecycle at scale

---

## 🔖 Bookmark These

**Most Important:**
1. [03_commands.md](03_commands.md) - API & CLI reference
2. [05_troubleshooting.md](05_troubleshooting.md) - Error fixes

**For Deep Understanding:**
1. [02_key_concepts.md](02_key_concepts.md) - Architecture
2. [06_tips_and_best_practices.md](06_tips_and_best_practices.md) - Professional workflows

**For Practice:**
1. [04_mini_projects.md](04_mini_projects.md) - All projects

---

## 📈 Next Steps After This Guide

1. **Apply to real projects**
   - Track all your experiments
   - Use Model Registry
   - Deploy models to production

2. **Explore advanced features**
   - MLflow on Databricks
   - Custom model flavors
   - Plugin development

3. **Integrate with ecosystem**
   - Use with DVC for data versioning
   - Integrate with Airflow for orchestration
   - Connect to Kubernetes for deployment

4. **Join community**
   - Share your learnings
   - Contribute to MLflow
   - Help others

---

## ✨ Final Tips

1. **Track everything** - Storage is cheap, insights are valuable
2. **Name experiments clearly** - Your future self will thank you
3. **Use autologging** - Less code, more productivity
4. **Compare runs often** - Find patterns in what works
5. **Document decisions** - Use tags and notes liberally

---

**Happy Tracking! 🚀**

*Remember: MLflow makes ML experiments reproducible, comparable, and deployable!*

---
