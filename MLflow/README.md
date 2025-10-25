# MLflow Complete Study Guide

**A comprehensive, hands-on guide to mastering MLflow for ML experiment tracking and model management**

---

## 🚀 Quick Start

```bash
# Install MLflow
pip install mlflow

# Your first MLflow script
python -c "
import mlflow
with mlflow.start_run():
    mlflow.log_param('learning_rate', 0.01)
    mlflow.log_metric('accuracy', 0.95)
print('✅ First run logged!')
"

# View in UI
mlflow ui
# Open: http://localhost:5000
```

---

## 📚 What's Inside

This guide contains **7 comprehensive documents** covering everything from basics to production deployment:

### Core Documents

| File | What You'll Learn | Time | Best For |
|------|------------------|------|----------|
| [00_index.md](00_index.md) | Navigation and learning paths | 5 min | Start here! |
| [01_intro.md](01_intro.md) | What MLflow is and why it exists | 10 min | Complete beginners |
| [02_key_concepts.md](02_key_concepts.md) | Experiments, Runs, Models, Registry | 20 min | Understanding architecture |
| [03_commands.md](03_commands.md) | Complete API & CLI reference | 30 min | Command lookup |
| [04_mini_projects.md](04_mini_projects.md) | 6 hands-on projects | 3 hours | Practical experience |
| [05_troubleshooting.md](05_troubleshooting.md) | Common errors and fixes | Reference | When stuck |
| [06_tips_and_best_practices.md](06_tips_and_best_practices.md) | Production patterns | 30 min | Professional workflows |

### Quick References

- **[CHEATSHEET.md](CHEATSHEET.md)** - One-page quick reference for common commands

---

## 🎯 Learning Paths

Choose your path based on your time and experience:

### 🟢 Complete Beginner (4-5 hours)
Perfect if you're new to MLflow and experiment tracking

1. **Read** [01_intro.md](01_intro.md) - Understand what MLflow is (10 min)
2. **Read** [02_key_concepts.md](02_key_concepts.md) - Learn the architecture (20 min)
3. **Do** [04_mini_projects.md](04_mini_projects.md) Projects 1-2 - Get hands-on (35 min)
4. **Read** [03_commands.md](03_commands.md) - Learn the commands (30 min)
5. **Do** [04_mini_projects.md](04_mini_projects.md) Projects 3-4 - Build on basics (45 min)
6. **Read** [06_tips_and_best_practices.md](06_tips_and_best_practices.md) - Learn best practices (30 min)

### 🟡 Quick Learner (2 hours)
You know ML, just need to learn MLflow

1. **Skim** [01_intro.md](01_intro.md) + [02_key_concepts.md](02_key_concepts.md) (15 min)
2. **Do** [04_mini_projects.md](04_mini_projects.md) Projects 1-2 (35 min)
3. **Read** [03_commands.md](03_commands.md) as reference (20 min)
4. **Do** [04_mini_projects.md](04_mini_projects.md) Project 3 (25 min)
5. **Read** [06_tips_and_best_practices.md](06_tips_and_best_practices.md) (30 min)

### 🔴 Just the Essentials (30 min)
Quick overview before diving into your project

1. **Read** [01_intro.md](01_intro.md) (10 min)
2. **Do** [04_mini_projects.md](04_mini_projects.md) Project 1 (15 min)
3. **Bookmark** [03_commands.md](03_commands.md) and [CHEATSHEET.md](CHEATSHEET.md)
4. **Keep** [05_troubleshooting.md](05_troubleshooting.md) handy

---

## 💡 What Makes This Guide Special

### ✅ Hands-On First
- 6 complete mini-projects you can run immediately
- Real code, not pseudo-code
- Copy-paste friendly examples

### ✅ Production-Ready
- Best practices from real-world deployments
- Team collaboration patterns
- Integration with other tools (DVC, Airflow, Docker, K8s)

### ✅ Problem-Solution Format
- Common issues with solutions
- Debugging strategies
- Performance optimization tips

### ✅ Complete Coverage
- Tracking API
- Model Registry
- Remote servers
- Deployment patterns

---

## 🎓 Learning Objectives

After completing this guide, you will:

**Basic Skills:**
- ✅ Track experiments with parameters and metrics
- ✅ Log and load models
- ✅ Use MLflow UI effectively
- ✅ Compare experiment runs

**Intermediate Skills:**
- ✅ Set up tracking servers
- ✅ Use Model Registry for versioning
- ✅ Implement hyperparameter tuning workflows
- ✅ Deploy models with MLflow

**Advanced Skills:**
- ✅ Design production tracking architectures
- ✅ Integrate with CI/CD pipelines
- ✅ Manage model lifecycle at scale
- ✅ Optimize performance for team use

---

## 🛠️ Prerequisites

### Required
- Python 3.7+
- Basic ML knowledge (training models, metrics)
- Familiarity with scikit-learn or similar

### Optional (but helpful)
- Git basics
- Command line experience
- Docker knowledge (for deployment)

---

## 📖 How to Use This Guide

### For Learning
1. **Follow the order** - Documents build on each other
2. **Type, don't copy-paste** - Builds muscle memory
3. **Do all mini projects** - Hands-on practice is key
4. **Experiment freely** - Break things and fix them!

### For Reference
- Use [03_commands.md](03_commands.md) for command lookup
- Use [CHEATSHEET.md](CHEATSHEET.md) for quick reference
- Use [05_troubleshooting.md](05_troubleshooting.md) when stuck
- Use [06_tips_and_best_practices.md](06_tips_and_best_practices.md) for production

### For Teams
- Share [00_index.md](00_index.md) as entry point
- Use [06_tips_and_best_practices.md](06_tips_and_best_practices.md) for team standards
- Reference [03_commands.md](03_commands.md) for consistency

---

## 🎯 Quick Examples

### Track Your First Experiment

```python
import mlflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Load data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2
)

# Set experiment
mlflow.set_experiment("iris_classification")

# Track experiment
with mlflow.start_run(run_name="random_forest_baseline"):
    # Log parameters
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 5)

    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X_train, y_train)

    # Log metrics
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # Log model
    mlflow.sklearn.log_model(model, "model")

print(f"✅ Logged run with accuracy: {accuracy:.4f}")
```

### Compare Multiple Runs

```python
import mlflow
import pandas as pd

# Search all runs
runs = mlflow.search_runs(
    experiment_names=["iris_classification"],
    order_by=["metrics.accuracy DESC"]
)

# Display top 5
print("Top 5 runs:")
print(runs[["run_id", "params.n_estimators", "metrics.accuracy"]].head())
```

### Register and Deploy Model

```python
import mlflow

# Register best model
mlflow.register_model(
    "runs:/RUN_ID/model",
    "IrisClassifier"
)

# Promote to production
from mlflow.tracking import MlflowClient
client = MlflowClient()
client.transition_model_version_stage(
    name="IrisClassifier",
    version=1,
    stage="Production"
)

# Load production model
model = mlflow.pyfunc.load_model("models:/IrisClassifier/Production")
predictions = model.predict(new_data)
```

---

## 🆘 Getting Help

### Within This Guide
1. Check [05_troubleshooting.md](05_troubleshooting.md) for common issues
2. Search [03_commands.md](03_commands.md) for specific commands
3. Review [06_tips_and_best_practices.md](06_tips_and_best_practices.md) for patterns

### External Resources
- **Official Docs:** https://mlflow.org/docs/latest/index.html
- **GitHub:** https://github.com/mlflow/mlflow
- **Stack Overflow:** Tag `mlflow`
- **Community Slack:** https://mlflow.org/slack

---

## 🔄 Companion Guides

This MLflow guide is part of a complete MLOps toolkit:

- **[DVC Guide](../DVC/)** - Data versioning and pipeline management
- **MLflow Guide** (you are here) - Experiment tracking and model management

Use together for complete MLOps coverage:
- **DVC** for data/pipeline versioning
- **MLflow** for experiment tracking and model deployment

---

## 📝 Practice Checklist

Track your progress:

- [ ] ✅ Installed MLflow
- [ ] ✅ Completed [01_intro.md](01_intro.md)
- [ ] ✅ Completed [02_key_concepts.md](02_key_concepts.md)
- [ ] ✅ Completed [03_commands.md](03_commands.md)
- [ ] ✅ Completed Mini Project 1
- [ ] ✅ Completed Mini Project 2
- [ ] ✅ Completed Mini Project 3
- [ ] ✅ Completed Mini Project 4
- [ ] ✅ Completed Mini Project 5
- [ ] ✅ Completed Mini Project 6
- [ ] ✅ Read [06_tips_and_best_practices.md](06_tips_and_best_practices.md)
- [ ] ✅ Used MLflow in own project
- [ ] ✅ Set up tracking server
- [ ] ✅ Used Model Registry
- [ ] ✅ Deployed a model

---

## 🏆 Next Steps

After completing this guide:

1. **Apply to Real Projects**
   - Track all your ML experiments
   - Use Model Registry for versioning
   - Deploy models to production

2. **Explore Advanced Features**
   - MLflow Projects for reproducibility
   - Custom model flavors
   - Multi-step workflows
   - Hyperparameter tuning at scale

3. **Integrate with Your Stack**
   - MLflow + DVC (data versioning)
   - MLflow + Airflow (orchestration)
   - MLflow + Docker (containerization)
   - MLflow + Kubernetes (deployment)

4. **Share Knowledge**
   - Train your team
   - Create team standards
   - Contribute to MLflow community

---

## ✨ Final Words

**MLflow makes ML experiments reproducible, comparable, and deployable.**

This guide gives you everything you need to:
- Track experiments effectively
- Manage models professionally
- Deploy with confidence
- Collaborate with teams

**Now go build something awesome!** 🚀

---

## 📄 License & Attribution

This guide is created for educational purposes.

- **MLflow** is an open-source project by Databricks
- **Official MLflow docs:** https://mlflow.org
- **MLflow GitHub:** https://github.com/mlflow/mlflow

---

## 🤝 Contributing

Found an issue or want to improve this guide?

- Report issues
- Suggest improvements
- Share your experience

**Happy Learning!** 🎉

---

*Last Updated: 2025*
*MLflow Version: 2.8+*
