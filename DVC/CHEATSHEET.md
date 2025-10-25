# DVC Cheatsheet - Quick Reference

> One-page reference for the most common DVC commands and workflows

---

## 🚀 Setup & Installation

```bash
# Install DVC
pip install dvc
pip install dvc[s3]      # With S3 support
pip install dvc[gdrive]  # With Google Drive
pip install dvc[all]     # All remotes

# Initialize
git init                 # Git first
dvc init                 # Then DVC
git add .dvc .dvcignore
git commit -m "Init DVC"
```

---

## 📦 File Tracking

```bash
# Track file/directory
dvc add data.csv
dvc add data/

# Commit to Git
git add data.csv.dvc .gitignore
git commit -m "Track data"

# Restore from cache
dvc checkout data.csv
dvc checkout             # All files
```

---

## ☁️ Remote Storage

```bash
# Add remote
dvc remote add -d myremote /tmp/dvc-storage          # Local
dvc remote add -d s3 s3://bucket/path                # S3
dvc remote add -d gdrive gdrive://folder-id          # Google Drive

# Commit config
git add .dvc/config
git commit -m "Add remote"

# Upload/download
dvc push                 # Upload to remote
dvc pull                 # Download from remote
dvc fetch                # Download to cache only

# List remotes
dvc remote list
```

---

## 🔄 Pipeline

### Create Pipeline

```bash
# Method 1: Using commands
dvc stage add -n prepare \
  -d raw_data.csv \
  -o processed_data.csv \
  python prepare.py

dvc stage add -n train \
  -d processed_data.csv \
  -p train.learning_rate,train.epochs \
  -o model.pkl \
  -M metrics.json \
  python train.py

# Method 2: Edit dvc.yaml directly
vim dvc.yaml
```

### Run Pipeline

```bash
dvc repro                # Run pipeline
dvc repro -f             # Force re-run all
dvc repro train          # Run up to 'train' stage

# View pipeline
dvc dag                  # Text diagram
dvc dag --md             # Markdown format
```

---

## 📊 Experiments

### Parameters

```bash
# params.yaml
train:
  learning_rate: 0.01
  epochs: 100

# Show params
dvc params diff
dvc params diff main     # Compare with branch
```

### Metrics

```bash
# Show metrics
dvc metrics show
dvc metrics show -a      # All branches

# Compare
dvc metrics diff
dvc metrics diff main
dvc metrics diff HEAD~1
```

---

## 🔍 Status & Information

```bash
# Check status
dvc status               # Pipeline status
dvc status -c            # Cloud status
dvc diff                 # Show changes

# Pipeline info
dvc dag                  # Show graph
dvc stage list           # List stages
```

---

## 🛠️ Common Workflows

### Workflow 1: Track New Data

```bash
dvc add data.csv
git add data.csv.dvc .gitignore
git commit -m "Add data"
dvc push
git push
```

### Workflow 2: Clone & Get Data

```bash
git clone <repo-url>
cd project
dvc pull
```

### Workflow 3: Run Experiment

```bash
# Edit params
vim params.yaml

# Run pipeline
dvc repro

# Check results
dvc metrics show

# Commit
git add params.yaml dvc.lock metrics.json
git commit -m "Exp: increased learning rate"
dvc push
git push
```

### Workflow 4: Compare Experiments

```bash
# Create branch
git checkout -b experiment-1

# Make changes
vim params.yaml
dvc repro

# Commit
git add -A
git commit -m "Experiment 1"

# Compare
dvc metrics diff main
dvc params diff main
```

---

## 🔧 Flags Reference

### dvc add
```bash
-d, --desc           # Description
--file              # Custom .dvc filename
```

### dvc stage add
```bash
-n, --name          # Stage name
-d, --deps          # Dependency
-o, --outs          # Output (cached)
-O, --outs-no-cache # Output (not cached)
-p, --params        # Parameter
-m, --metrics       # Metric (cached)
-M, --metrics-no-cache  # Metric (not cached)
--plots-no-cache    # Plot file
```

### dvc repro
```bash
-f, --force         # Force re-run
-s, --single-item   # Run single stage
-p, --pipeline      # Reproduce entire pipeline
```

### dvc push/pull
```bash
-r, --remote        # Specific remote
-j, --jobs          # Parallel jobs
-v, --verbose       # Verbose output
```

---

## 📁 File Structure

```
project/
├── .dvc/
│   ├── cache/          # Local cache
│   ├── config          # DVC config
│   └── .gitignore
├── .git/
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── src/
│   ├── prepare.py
│   ├── train.py
│   └── evaluate.py
├── data.csv.dvc        # Pointer file
├── dvc.yaml            # Pipeline
├── dvc.lock            # Lock file
├── params.yaml         # Parameters
├── metrics.json        # Metrics
└── .gitignore
```

---

## 📝 dvc.yaml Template

```yaml
stages:
  prepare:
    cmd: python src/prepare.py
    deps:
      - data/raw.csv
      - src/prepare.py
    params:
      - prepare.split
    outs:
      - data/processed.csv

  train:
    cmd: python src/train.py
    deps:
      - data/processed.csv
      - src/train.py
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: python src/evaluate.py
    deps:
      - models/model.pkl
      - src/evaluate.py
    metrics:
      - metrics/eval.json:
          cache: false
```

---

## 📝 params.yaml Template

```yaml
prepare:
  split: 0.8
  random_state: 42

train:
  learning_rate: 0.01
  epochs: 100
  batch_size: 32

evaluate:
  threshold: 0.5
```

---

## 🎯 Python Code Templates

### Load Parameters
```python
import yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)

lr = params['train']['learning_rate']
epochs = params['train']['epochs']
```

### Save Metrics
```python
import json

metrics = {
    'accuracy': 0.95,
    'loss': 0.12
}

with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

### Full Training Script
```python
import pandas as pd
import pickle
import yaml
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load params
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load data
train = pd.read_csv('data/train.csv')
X_train = train.drop('target', axis=1)
y_train = train['target']

# Train
model = RandomForestClassifier(
    n_estimators=params['train']['n_estimators'],
    random_state=42
)
model.fit(X_train, y_train)

# Save model
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate
y_pred = model.predict(X_train)
metrics = {'accuracy': accuracy_score(y_train, y_pred)}

# Save metrics
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

## 🚨 Quick Troubleshooting

| Error | Quick Fix |
|-------|-----------|
| `command not found: dvc` | `pip install dvc` |
| `not initialized` | `dvc init` |
| `.git not found` | `git init` first |
| `file already exists` | `git rm --cached file` then `dvc add` |
| `failed to push` | Check `dvc remote list` |
| `authentication failed` | Re-run `dvc push` (OAuth) or check credentials |
| `file not in cache` | `dvc pull` |
| `circular dependency` | Fix deps in `dvc.yaml` |

---

## 🔑 Key Concepts

| What | Where | Git? | Purpose |
|------|-------|------|---------|
| Large files | `.dvc/cache/` | ❌ | Local storage |
| Pointer files | `*.dvc` | ✅ | Track large files |
| Pipeline | `dvc.yaml` | ✅ | Define workflow |
| Lock | `dvc.lock` | ✅ | Reproducibility |
| Parameters | `params.yaml` | ✅ | Hyperparameters |
| Metrics | `metrics.json` | ✅ | Results |
| Remote | Cloud/storage | ❌ | Shared storage |

---

## ⚡ Performance Tips

```bash
# Use hardlinks (faster, saves space)
dvc config cache.type hardlink

# Parallel jobs
dvc pull -j 8
dvc push -j 8

# Shared cache
dvc config cache.dir /shared/cache --global

# Clean old cache
dvc gc -w                # Workspace only
dvc gc -c                # Not in Git history
```

---

## 🔒 Security

```bash
# Never commit secrets!
# Use environment variables in params.yaml
database:
  password: ${DB_PASSWORD}

# Secure remote credentials
dvc remote modify myremote access_key_id ${AWS_ACCESS_KEY_ID}

# .gitignore
.env
*.key
credentials.json
```

---

## 📋 Daily Checklist

### Starting Work
```bash
git pull
dvc pull
```

### Ending Work
```bash
dvc status
git status
dvc push
git add -A
git commit -m "..."
git push
```

---

## 🎓 Remember

1. **DVC works WITH Git, not instead of it**
2. **Small files → Git, Large files → DVC**
3. **Always: `dvc push && git push`**
4. **`dvc repro` > running scripts manually**
5. **`.dvc` files go in Git, actual data doesn't**

---

## 🆘 Help

```bash
dvc --help
dvc add --help
dvc repro --help

# Version
dvc version

# Doctor (check setup)
dvc doctor
```

**Resources:**
- Docs: https://dvc.org/doc
- Chat: https://dvc.org/chat
- GitHub: https://github.com/iterative/dvc

---

## 💡 One-Liners

```bash
# Initialize project
git init && dvc init && git add .dvc .dvcignore && git commit -m "Init"

# Track and push data
dvc add data.csv && git add data.csv.dvc .gitignore && git commit -m "Add data" && dvc push && git push

# Quick experiment
git checkout -b exp && vim params.yaml && dvc repro && dvc metrics diff main

# Compare all experiments
for branch in $(git branch | cut -c 3-); do echo "=== $branch ==="; git checkout $branch; dvc metrics show; done
```

---

**Print this page and keep it handy!** 📄

---
