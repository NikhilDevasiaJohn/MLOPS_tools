# 03. DVC Commands Reference

---

## 🚀 Installation & Setup

```bash
# Install DVC
pip install dvc

# Install with specific remote support
pip install dvc[s3]        # AWS S3
pip install dvc[gdrive]    # Google Drive
pip install dvc[azure]     # Azure
pip install dvc[all]       # All remotes

# Initialize DVC in your project
dvc init

# What it does:
# ✅ Creates .dvc/ directory
# ✅ Creates .dvc/.gitignore
# ✅ Creates .dvc/config
```

---

## 📦 Tracking Files

### `dvc add` - Track a file or directory

```bash
# Track a single file
dvc add data.csv
# Creates: data.csv.dvc
# Adds to .gitignore: data.csv

# Track a directory
dvc add data/
# Creates: data.dvc

# What happens:
# 1. File moved to .dvc/cache/
# 2. Creates .dvc pointer file
# 3. Original file replaced with link/copy
# 4. File added to .gitignore

# Commit the .dvc file
git add data.csv.dvc .gitignore
git commit -m "Add dataset"
```

**Example workflow:**
```bash
# Download/create a dataset
wget https://example.com/data.csv

# Track it with DVC
dvc add data.csv

# Commit to Git
git add data.csv.dvc .gitignore
git commit -m "Track data.csv"
```

---

### `dvc checkout` - Restore files from cache

```bash
# Restore all DVC-tracked files
dvc checkout

# Restore specific file
dvc checkout data.csv.dvc

# Use case: After git checkout, restore data
git checkout main
dvc checkout  # Restore data for main branch
```

---

## ☁️ Remote Storage

### `dvc remote` - Configure remote storage

```bash
# Add local remote
dvc remote add -d myremote /tmp/dvc-storage

# Add S3 remote
dvc remote add -d s3remote s3://mybucket/path

# Add Google Drive remote
dvc remote add -d gdrive gdrive://folder-id

# List remotes
dvc remote list

# Remove remote
dvc remote remove myremote

# Set default remote
dvc remote default s3remote

# Modify remote URL
dvc remote modify myremote url /new/path
```

**Common remote configurations:**

```bash
# Local directory (good for testing)
dvc remote add -d local /mnt/external-drive/dvc-storage

# AWS S3
dvc remote add -d s3 s3://my-bucket/dvc-store
dvc remote modify s3 region us-west-2

# Google Drive
dvc remote add -d gdrive gdrive://1a2b3c4d5e6f7g8h9i

# SSH/SFTP
dvc remote add -d ssh-storage ssh://user@example.com/path

# Azure Blob
dvc remote add -d azure azure://mycontainer/path
```

---

### `dvc push` - Upload to remote storage

```bash
# Push all tracked files
dvc push

# Push specific file
dvc push data.csv.dvc

# Push to specific remote
dvc push -r myremote

# What it does:
# ✅ Uploads files from .dvc/cache/ to remote
# ✅ Only uploads new/changed files
# ✅ Uses MD5 to avoid duplicates
```

**Example workflow:**
```bash
# Track new data
dvc add model.pkl

# Commit pointer
git add model.pkl.dvc
git commit -m "Add trained model"

# Upload to remote
dvc push

# Push to Git
git push
```

---

### `dvc pull` - Download from remote storage

```bash
# Pull all tracked files
dvc pull

# Pull specific file
dvc pull data.csv.dvc

# Pull from specific remote
dvc pull -r myremote

# What it does:
# ✅ Downloads files from remote to .dvc/cache/
# ✅ Links/copies files to workspace
# ✅ Only downloads missing files
```

**Example workflow:**
```bash
# Clone repository
git clone https://github.com/user/project.git
cd project

# Download data
dvc pull

# Now you have code (from Git) + data (from DVC)
```

---

### `dvc fetch` - Download to cache only

```bash
# Fetch without checking out
dvc fetch

# Difference from pull:
# fetch: Downloads to cache only
# pull:  Downloads + restores to workspace

# Use case: Pre-download for later use
dvc fetch           # Download everything
dvc checkout data/  # Use only needed files
```

---

## 🔄 Pipeline Commands

### `dvc stage add` - Add a pipeline stage

```bash
# Add stage manually
dvc stage add -n prepare \
  -d raw_data.csv \
  -o processed_data.csv \
  python prepare.py

# Add training stage
dvc stage add -n train \
  -d processed_data.csv \
  -d train.py \
  -p train.learning_rate,train.epochs \
  -o model.pkl \
  -M metrics.json \
  python train.py

# Flags:
# -n  : Stage name
# -d  : Dependency
# -o  : Output (cached)
# -O  : Output (not cached)
# -p  : Parameter
# -m  : Metric (cached)
# -M  : Metric (not cached)
```

**This creates/updates `dvc.yaml`:**
```yaml
stages:
  prepare:
    cmd: python prepare.py
    deps:
      - raw_data.csv
    outs:
      - processed_data.csv

  train:
    cmd: python train.py
    deps:
      - processed_data.csv
      - train.py
    params:
      - train.learning_rate
      - train.epochs
    outs:
      - model.pkl
    metrics:
      - metrics.json:
          cache: false
```

---

### `dvc repro` - Reproduce pipeline

```bash
# Run entire pipeline
dvc repro

# Run up to specific stage
dvc repro train

# Force re-run (ignore cache)
dvc repro -f

# What it does:
# ✅ Checks dependencies (files, params)
# ✅ Runs only changed stages (smart caching!)
# ✅ Updates dvc.lock
# ✅ Generates outputs
```

**Example:**
```bash
# Edit params.yaml
vim params.yaml  # Change learning_rate

# Reproduce pipeline
dvc repro
# Output: Only 'train' stage runs (prepare is cached!)

# Commit changes
git add dvc.lock params.yaml metrics.json
git commit -m "Tune learning rate"
```

---

### `dvc dag` - Show pipeline graph

```bash
# Show pipeline as text
dvc dag

# Output:
#+------------+
#| raw_data   |
#+------------+
#      *
#      *
#      *
#+------------+
#| prepare    |
#+------------+
#      *
#      *
#      *
#+------------+
#| train      |
#+------------+

# Show as Mermaid diagram
dvc dag --mermaid

# Save as image (requires graphviz)
dvc dag --dot | dot -Tpng -o pipeline.png
```

---

## 📊 Experiments & Metrics

### `dvc params` - Work with parameters

```bash
# Show all parameters
dvc params diff

# Compare with another branch
dvc params diff experiment-branch

# Show specific params
dvc params diff --targets train.learning_rate
```

---

### `dvc metrics` - Work with metrics

```bash
# Show current metrics
dvc metrics show

# Compare metrics across branches
dvc metrics diff

# Compare with specific revision
dvc metrics diff HEAD~1

# Show specific metric
dvc metrics show metrics.json

# Output:
# metrics.json:
#   accuracy: 0.95
#   loss: 0.12
```

**Example comparison:**
```bash
# Current branch
dvc metrics show
# accuracy: 0.95

# Compare with main
dvc metrics diff main
# metrics.json:
#   accuracy:
#     main: 0.92
#     workspace: 0.95  # +3% improvement!
```

---

### `dvc plots` - Visualize metrics

```bash
# Show plots
dvc plots show

# Compare plots across experiments
dvc plots diff

# Show specific plot
dvc plots show training_history.csv

# Output HTML comparison
dvc plots diff --open
```

---

## 🔍 Status & Information

### `dvc status` - Check pipeline status

```bash
# Check pipeline status
dvc status

# Check remote status
dvc status -r myremote

# Outputs:
# - Changed dependencies
# - Changed outputs
# - Stages to re-run
```

---

### `dvc diff` - Show file changes

```bash
# Show what changed
dvc diff

# Compare with specific commit
dvc diff HEAD~1

# Output:
# Added:
#   - new_data.csv
# Modified:
#   - model.pkl
# Deleted:
#   - old_data.csv
```

---

### `dvc list` - List files in repository

```bash
# List DVC-tracked files in remote repo
dvc list https://github.com/user/repo data/

# List specific path
dvc list --dvc-only https://github.com/user/repo
```

---

### `dvc get` - Download specific file

```bash
# Download file from DVC repo
dvc get https://github.com/user/repo data/file.csv

# Download to specific path
dvc get https://github.com/user/repo data/file.csv -o myfile.csv

# Use case: Get data without cloning entire repo
```

---

### `dvc import` - Import and track file from another repo

```bash
# Import file and create .dvc tracking
dvc import https://github.com/user/repo data/file.csv

# What it does:
# ✅ Downloads file
# ✅ Creates file.csv.dvc
# ✅ Links to source repo (can update later!)
```

---

## 🛠️ Utility Commands

### `dvc gc` - Garbage collect old cache

```bash
# Remove unused cache files
dvc gc

# Remove cache not used by current workspace
dvc gc -w

# Remove cache not in any branch/tag
dvc gc -a

# Dry run (show what would be deleted)
dvc gc -w --dry

# Use case: Clean up old experiments to save space
```

---

### `dvc move` - Rename tracked file

```bash
# Rename DVC-tracked file
dvc move old_data.csv new_data.csv

# What it does:
# ✅ Renames file
# ✅ Updates .dvc file
# ✅ Updates .gitignore
```

---

### `dvc unprotect` - Make file writable

```bash
# Make DVC file editable
dvc unprotect data.csv

# Use case: DVC files are read-only by default
# Unprotect to modify in place
```

---

### `dvc config` - Configure DVC

```bash
# Set config value
dvc config core.remote myremote

# Show config
dvc config --list

# Unset config
dvc config --unset core.autostage
```

---

## 📋 Command Cheatsheet

| Task | Command |
|------|---------|
| **Setup** | |
| Initialize DVC | `dvc init` |
| **Tracking** | |
| Track file | `dvc add data.csv` |
| Restore files | `dvc checkout` |
| **Remote** | |
| Add remote | `dvc remote add -d name url` |
| Upload files | `dvc push` |
| Download files | `dvc pull` |
| **Pipeline** | |
| Add stage | `dvc stage add -n name ...` |
| Run pipeline | `dvc repro` |
| Show graph | `dvc dag` |
| **Experiments** | |
| Show metrics | `dvc metrics show` |
| Compare metrics | `dvc metrics diff` |
| Show params | `dvc params diff` |
| **Info** | |
| Check status | `dvc status` |
| Show changes | `dvc diff` |
| **Cleanup** | |
| Remove old cache | `dvc gc` |

---

## 🎯 Common Workflows

### Workflow 1: Track new dataset
```bash
dvc add data.csv
git add data.csv.dvc .gitignore
git commit -m "Add dataset"
dvc push
git push
```

### Workflow 2: Get data from repo
```bash
git clone repo-url
dvc pull
```

### Workflow 3: Run experiment
```bash
# Edit params
vim params.yaml

# Run pipeline
dvc repro

# Check results
dvc metrics show

# Commit
git add dvc.lock params.yaml metrics.json
git commit -m "Experiment: higher learning rate"
```

### Workflow 4: Compare experiments
```bash
# Create experiment branch
git checkout -b experiment-1
# ... make changes ...
dvc repro
git add -A
git commit -m "Experiment 1"

# Compare
dvc metrics diff main
dvc params diff main
```

---

## 💡 Pro Tips

1. **Always commit `.dvc` files to Git**
   ```bash
   git add data.csv.dvc
   git commit -m "Track data"
   ```

2. **Push to both Git and DVC**
   ```bash
   dvc push && git push
   ```

3. **Use `dvc repro` instead of running scripts manually**
   - It's faster (uses cache)
   - It's reproducible
   - It tracks everything

4. **Check status before committing**
   ```bash
   dvc status
   git status
   ```

5. **Use meaningful commit messages**
   ```bash
   git commit -m "Add data preprocessing pipeline"
   git commit -m "Tune model: learning_rate=0.01"
   ```

---
