# 06. DVC Tips & Best Practices

---

## 🎯 Golden Rules

### Rule 1: Git + DVC = Best Friends
```bash
# Always do both
git add .
git commit -m "Update"
dvc push
git push

# Make it a habit
alias gpush='dvc push && git push'
```

**Why:** DVC pointers go in Git, actual data goes to DVC remote. Both need to be synced!

---

### Rule 2: Never Commit Large Files to Git
```bash
# ❌ WRONG
git add large_dataset.csv
git commit -m "Add data"  # Git repo will be huge!

# ✅ CORRECT
dvc add large_dataset.csv
git add large_dataset.csv.dvc .gitignore
git commit -m "Track data with DVC"
```

**Rule of thumb:** Files > 1MB → use DVC, not Git

---

### Rule 3: Commit `.dvc` Files, Not Actual Data
```bash
# What goes in Git:
git add data.csv.dvc        # ✅ Pointer file
git add dvc.yaml            # ✅ Pipeline definition
git add dvc.lock            # ✅ Pipeline state
git add params.yaml         # ✅ Parameters
git add metrics.json        # ✅ Metrics

# What stays out of Git:
# data.csv                  # ❌ Large file (DVC handles it)
# model.pkl                 # ❌ Large file (DVC handles it)
# .dvc/cache/               # ❌ Local cache (auto-ignored)
```

---

### Rule 4: Always Use `dvc repro`, Not Manual Scripts
```bash
# ❌ WRONG - Not reproducible
python prepare.py
python train.py
python evaluate.py

# ✅ CORRECT - Reproducible pipeline
dvc repro

# Why?
# - Uses cache (faster)
# - Tracks dependencies
# - Ensures reproducibility
# - Updates dvc.lock automatically
```

---

### Rule 5: Pin Your Random Seeds
```python
# ✅ Always set seeds for reproducibility
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

set_seed(42)
```

**In params.yaml:**
```yaml
train:
  random_state: 42
  seed: 42
```

---

## 📁 Project Structure Best Practices

### Recommended Structure
```
ml-project/
├── .dvc/                   # DVC configuration
├── .git/                   # Git repository
├── data/
│   ├── raw/               # Original, immutable data
│   ├── processed/         # Cleaned, transformed data
│   └── features/          # Feature-engineered data
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks (exploratory)
├── src/                   # Source code
│   ├── data/
│   │   ├── download.py
│   │   └── prepare.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   └── visualization/
│       └── visualize.py
├── metrics/               # Metrics files
├── plots/                 # Plot data
├── params.yaml            # Hyperparameters
├── dvc.yaml               # Pipeline definition
├── dvc.lock               # Pipeline lock file
├── requirements.txt       # Python dependencies
├── .gitignore
├── .dvcignore
└── README.md
```

### What to Track Where

| Type | Tool | Why |
|------|------|-----|
| Code (`.py`) | Git | Small, changes often |
| Config (`.yaml`) | Git | Small, need version history |
| Data (`data/`) | DVC | Large, changes rarely |
| Models (`.pkl`) | DVC | Large, binary |
| Metrics (`.json`) | Git | Small, compare experiments |
| Notebooks (`.ipynb`) | Git | Code + docs |

---

## 🔄 Pipeline Best Practices

### 1. Break Into Small Stages
```yaml
# ✅ GOOD - Small, focused stages
stages:
  download:
    cmd: python src/data/download.py
    outs:
      - data/raw/data.csv

  clean:
    cmd: python src/data/clean.py
    deps:
      - data/raw/data.csv
    outs:
      - data/processed/clean.csv

  features:
    cmd: python src/features/build.py
    deps:
      - data/processed/clean.csv
    outs:
      - data/features/features.csv

  train:
    cmd: python src/models/train.py
    deps:
      - data/features/features.csv
    outs:
      - models/model.pkl
```

**Why:**
- Better caching (only re-run what changed)
- Easier debugging
- Clearer pipeline

---

### 2. Explicitly List All Dependencies
```yaml
# ❌ BAD - Missing dependencies
stages:
  train:
    cmd: python train.py
    outs:
      - model.pkl

# ✅ GOOD - All dependencies listed
stages:
  train:
    cmd: python src/models/train.py
    deps:
      - src/models/train.py        # Code dependency
      - data/features/features.csv # Data dependency
      - src/models/utils.py        # Helper code
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - models/model.pkl
```

**Why:** DVC needs to know when to re-run stages

---

### 3. Use Parameters File
```yaml
# params.yaml
data:
  download:
    url: https://example.com/data.csv

prepare:
  test_size: 0.2
  random_state: 42

train:
  model: random_forest
  n_estimators: 100
  max_depth: 10
  learning_rate: 0.01

evaluate:
  metrics:
    - accuracy
    - precision
    - recall
```

**In code:**
```python
import yaml

with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Use params
test_size = params['prepare']['test_size']
n_estimators = params['train']['n_estimators']
```

**Benefits:**
- Single source of truth
- Easy to compare experiments
- Track config changes in Git

---

### 4. Track Metrics Properly
```yaml
stages:
  evaluate:
    cmd: python src/models/evaluate.py
    deps:
      - models/model.pkl
      - data/test.csv
    metrics:
      - metrics/metrics.json:
          cache: false  # ← Important! Never cache metrics
```

**In code:**
```python
import json

metrics = {
    'accuracy': 0.95,
    'precision': 0.93,
    'recall': 0.94,
    'f1_score': 0.935
}

with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)
```

---

## 🔬 Experiment Tracking

### Strategy 1: Use Git Branches
```bash
# Main branch = production/best model
git checkout main

# Create experiment branch
git checkout -b exp-larger-model

# Modify params
vim params.yaml

# Run experiment
dvc repro

# Commit results
git add params.yaml dvc.lock metrics/
git commit -m "Exp: Larger model (500 trees)"

# Compare with main
dvc metrics diff main
dvc params diff main

# If better, merge to main
git checkout main
git merge exp-larger-model
```

**Benefits:**
- Clear history of experiments
- Easy to compare
- Can return to any experiment

---

### Strategy 2: Use Git Tags
```bash
# Tag important milestones
git tag -a v1.0-baseline -m "Baseline model"
git tag -a v1.1-tuned -m "Hyperparameter tuned"
git tag -a v2.0-best -m "Best model - 95% accuracy"

# Compare any two versions
dvc metrics diff v1.0-baseline v2.0-best

# Return to specific version
git checkout v1.0-baseline
dvc checkout
```

---

### Strategy 3: Systematic Naming
```bash
# Branch naming convention
exp-YYYYMMDD-description
exp-20240115-more-features
exp-20240116-deeper-network
exp-20240117-regularization

# Commit message template
git commit -m "Exp: [CHANGE] - [RESULT]

Details:
- Parameter: learning_rate=0.01
- Metric: accuracy=0.95 (+0.03)
- Runtime: 120s"
```

---

## 💾 Data Management

### 1. Organize Data by Mutability
```
data/
├── raw/              # ← Never modified (read-only)
│   └── original.csv
├── interim/          # ← Temporary processing
│   └── step1.csv
├── processed/        # ← Final cleaned data
│   └── clean.csv
└── external/         # ← External data sources
    └── reference.csv
```

**Track appropriately:**
```bash
# Raw data - track at folder level
dvc add data/raw

# Processed data - track individual files
dvc add data/processed/clean.csv
```

---

### 2. Version Your Data
```bash
# Method 1: Git tags
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -m "Data v1.0"
git tag -a data-v1.0 -m "Initial dataset"

# Later, when data changes
dvc add data/dataset.csv
git add data/dataset.csv.dvc
git commit -m "Data v2.0 - added 1000 samples"
git tag -a data-v2.0 -m "Extended dataset"

# Switch between versions
git checkout data-v1.0
dvc checkout
```

---

### 3. Document Your Data
```bash
# Create data README
cat > data/README.md << 'EOF'
# Dataset Documentation

## Raw Data
- Source: https://example.com
- Download date: 2024-01-15
- Size: 10,000 rows
- Features: 50 columns
- Target: binary classification

## Processed Data
- Cleaning steps:
  1. Removed duplicates
  2. Handled missing values
  3. Scaled features
- Final size: 9,500 rows
- Split: 80% train, 20% test
EOF

git add data/README.md
git commit -m "Document data"
```

---

## ⚡ Performance Optimization

### 1. Use Hardlinks (Faster, Less Space)
```bash
# Configure cache type
dvc config cache.type hardlink,symlink

# Fallback order: hardlink → symlink → copy
# Hardlink is fastest and saves space
```

### 2. Parallel Jobs
```bash
# Configure parallel downloads/uploads
dvc config cache.jobs 4

# Use when running
dvc pull --jobs 8
dvc push --jobs 8
```

### 3. Shared Cache Across Projects
```bash
# Set global cache directory
dvc config cache.dir /shared/dvc-cache --global

# All projects use same cache
# Saves disk space
# Faster when reusing data
```

### 4. Shallow Clone
```bash
# Clone repo without DVC data
git clone <repo-url>

# Pull only what you need
dvc pull data/subset/
```

---

## 🔒 Security Best Practices

### 1. Never Commit Secrets
```bash
# ❌ WRONG
# params.yaml:
# database:
#   password: "secretpassword123"

# ✅ CORRECT
# Use environment variables
# params.yaml:
# database:
#   password: ${DB_PASSWORD}

# Set in environment
export DB_PASSWORD="secretpassword123"

# Or use .env file (add to .gitignore!)
echo "DB_PASSWORD=secret" > .env
echo ".env" >> .gitignore
```

### 2. Secure Remote Credentials
```bash
# Don't put credentials in .dvc/config
# ❌ WRONG
dvc remote modify myremote access_key_id AKIAXXXXX

# ✅ CORRECT - Use environment variables
dvc remote modify myremote access_key_id ${AWS_ACCESS_KEY_ID}

# Or use credential helpers
dvc remote modify s3remote credentialpath ~/.aws/credentials
```

### 3. .gitignore Important Files
```bash
# .gitignore
.env
*.key
credentials.json
secrets/
.aws/
```

---

## 📊 Monitoring & Logging

### 1. Log Pipeline Runs
```bash
# Create logging wrapper
cat > run_pipeline.sh << 'EOF'
#!/bin/bash
echo "Pipeline started at $(date)" >> pipeline.log
dvc repro -v 2>&1 | tee -a pipeline.log
echo "Pipeline completed at $(date)" >> pipeline.log
EOF

chmod +x run_pipeline.sh
./run_pipeline.sh
```

### 2. Track Experiment Metadata
```python
# src/models/train.py
import json
import time
from datetime import datetime

metadata = {
    'timestamp': datetime.now().isoformat(),
    'duration_seconds': training_time,
    'python_version': sys.version,
    'package_versions': {
        'sklearn': sklearn.__version__,
        'pandas': pd.__version__
    }
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```

---

## 🎓 Team Collaboration

### 1. Team Workflow
```bash
# Setup (once per teammate)
git clone <repo-url>
cd project
dvc pull

# Daily workflow
git pull
dvc pull           # Get latest data
# ... work ...
dvc repro          # Run experiments
git add -A
git commit -m "Experiment description"
dvc push
git push
```

### 2. Code Review Checklist
```markdown
## DVC PR Checklist
- [ ] dvc.lock updated?
- [ ] metrics.json included?
- [ ] params.yaml changes documented?
- [ ] All .dvc files committed?
- [ ] README updated if needed?
- [ ] dvc push completed?
```

### 3. Onboarding New Team Members
```bash
# Create onboarding script
cat > setup.sh << 'EOF'
#!/bin/bash
echo "Setting up project..."

# Install dependencies
pip install -r requirements.txt

# Pull data
dvc pull

# Verify setup
dvc status
dvc dag

echo "Setup complete! Run 'dvc repro' to test."
EOF
```

---

## 🚀 CI/CD Integration

### GitHub Actions Example
```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on: [push]

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Setup DVC
        run: |
          pip install dvc[s3]
          dvc remote modify storage --local access_key_id ${{ secrets.AWS_KEY }}
          dvc remote modify storage --local secret_access_key ${{ secrets.AWS_SECRET }}

      - name: Pull data
        run: dvc pull

      - name: Run pipeline
        run: dvc repro

      - name: Check metrics
        run: dvc metrics show
```

---

## 📝 Documentation Tips

### 1. Self-Documenting Pipeline
```yaml
# dvc.yaml
stages:
  prepare_data:  # Clear stage names
    desc: "Clean and split data into train/test"  # Add descriptions!
    cmd: python src/data/prepare.py
    deps:
      - data/raw/data.csv
    outs:
      - data/processed/train.csv
      - data/processed/test.csv
```

### 2. README Template
```markdown
# Project Name

## Setup
\`\`\`bash
git clone <repo>
pip install -r requirements.txt
dvc pull
\`\`\`

## Run Pipeline
\`\`\`bash
dvc repro
\`\`\`

## Experiment
1. Edit `params.yaml`
2. Run `dvc repro`
3. Check results: `dvc metrics show`

## Project Structure
- `data/` - Datasets (DVC tracked)
- `src/` - Source code
- `models/` - Trained models (DVC tracked)
- `params.yaml` - Hyperparameters
```

---

## 🎯 Quick Tips Summary

| Tip | Command/Action |
|-----|----------------|
| Always commit both | `dvc push && git push` |
| Check before commit | `dvc status && git status` |
| Clean cache | `dvc gc -w` |
| Compare experiments | `dvc metrics diff <branch>` |
| Force re-run | `dvc repro -f` |
| Show pipeline | `dvc dag` |
| Validate setup | `dvc doctor` |
| Parallel jobs | `dvc pull -j 4` |
| Use hardlinks | `dvc config cache.type hardlink` |

---

## 💡 Mental Models

### Think of DVC as...

**1. Git for data**
- Commits → `dvc add`
- Push → `dvc push`
- Pull → `dvc pull`
- Branches → Same Git branches!

**2. Make for ML**
- Makefile → `dvc.yaml`
- Dependencies → `deps:`
- Targets → `outs:`
- Make → `dvc repro`

**3. Package manager for data**
- package.json → `dvc.yaml`
- package-lock.json → `dvc.lock`
- node_modules/ → `.dvc/cache/`
- npm install → `dvc pull`

---

## 🏆 Pro Tips

**1. Alias common commands:**
```bash
# Add to ~/.bashrc or ~/.zshrc
alias dr='dvc repro'
alias ds='dvc status'
alias dm='dvc metrics show'
alias dp='dvc push && git push'
alias gpull='git pull && dvc pull'
```

**2. Pre-commit hooks:**
```bash
# .git/hooks/pre-commit
#!/bin/bash
dvc status --cloud
if [ $? -ne 0 ]; then
    echo "DVC files not pushed to remote!"
    echo "Run: dvc push"
    exit 1
fi
```

**3. Quick experiment template:**
```bash
# Create experiment script
cat > experiment.sh << 'EOF'
#!/bin/bash
BRANCH="exp-$(date +%Y%m%d-%H%M%S)"
git checkout -b $BRANCH
# ... make changes ...
dvc repro
git add -A
git commit -m "Experiment: $1"
dvc metrics diff main
EOF
chmod +x experiment.sh

# Run: ./experiment.sh "description"
```

---

## 🎓 Learning Path

**Week 1:** Basic tracking
- `dvc init`, `dvc add`, `dvc push/pull`

**Week 2:** Pipelines
- `dvc.yaml`, `dvc repro`, `dvc dag`

**Week 3:** Experiments
- Parameters, metrics, branches

**Week 4:** Advanced
- CI/CD, shared cache, optimization

---

## ✅ Final Checklist

Before each commit:
```bash
- [ ] dvc status (check tracked files)
- [ ] git status (check Git files)
- [ ] dvc metrics show (verify results)
- [ ] dvc dag (pipeline makes sense?)
- [ ] dvc push (upload data)
- [ ] git push (upload code)
```

Before each pull request:
```bash
- [ ] dvc.lock updated
- [ ] All experiments documented
- [ ] README updated
- [ ] Tests pass
- [ ] Metrics improved
```

---
