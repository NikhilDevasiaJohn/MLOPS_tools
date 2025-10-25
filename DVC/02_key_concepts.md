# 02. DVC Key Concepts

---

## 📁 Core DVC Files

### 1. `.dvc` Files (Pointer Files)

**What:** Lightweight metadata files that track large files/folders
**Extension:** `.dvc` (e.g., `data.csv.dvc`, `model.pkl.dvc`)
**Stored in:** Git repository
**Points to:** Actual large files in cache/remote storage

**Example `data.csv.dvc`:**
```yaml
outs:
- md5: a3b2c1d4e5f6...
  size: 1048576
  path: data.csv
```

**Key info stored:**
- `md5`: Hash of the file (fingerprint)
- `size`: File size in bytes
- `path`: Original file name

**Remember:** The `.dvc` file is small (few lines), actual file is large!

---

### 2. `dvc.yaml` (Pipeline Definition)

**What:** Defines your ML pipeline stages (data processing, training, evaluation)
**Purpose:** Automate and reproduce workflows
**Stored in:** Git repository

**Example:**
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

**Key sections:**
- `cmd`: Shell command to run
- `deps`: Dependencies (inputs) - triggers re-run if changed
- `outs`: Outputs - files created by this stage
- `params`: Hyperparameters from `params.yaml`
- `metrics`: Metric files to track

---

### 3. `dvc.lock` (Lockfile)

**What:** Auto-generated file that captures exact state of pipeline execution
**Purpose:** Ensures reproducibility by locking dependencies
**Stored in:** Git repository
**Never edit manually!**

**Example `dvc.lock`:**
```yaml
schema: '2.0'
stages:
  train:
    cmd: python train.py
    deps:
    - path: processed_data.csv
      md5: e3b0c44298fc1c14
      size: 524288
    - path: train.py
      md5: d41d8cd98f00b204
      size: 2048
    params:
      params.yaml:
        train.learning_rate: 0.001
        train.epochs: 10
    outs:
    - path: model.pkl
      md5: 7d8e4f9a2b1c3d5e
      size: 1048576
```

**What it locks:**
- Exact file versions (MD5 hashes)
- Exact parameter values
- Command that was run

---

### 4. `.dvc/` Directory (Local Cache)

**What:** Hidden folder where DVC stores actual file contents
**Location:** `.dvc/cache/`
**Structure:** Content-addressable storage (files stored by MD5 hash)

**Directory structure:**
```
.dvc/
├── cache/                    # Cached files
│   └── files/
│       └── md5/
│           ├── a3/
│           │   └── b2c1d4... # Actual file content
│           └── e3/
│               └── b0c442... # Another file
├── config                    # DVC configuration
├── .gitignore               # Auto-generated
└── tmp/                     # Temporary files
```

**Remember:**
- Files in cache are stored by hash, not original names
- `.dvc/cache` should be in `.gitignore`
- Cache is local - use `dvc push` to backup to remote

---

### 5. Remote Storage

**What:** Cloud or local storage for your large files
**Purpose:** Share data across team, backup, restore
**Common options:**
- Local directory
- AWS S3
- Google Drive
- Azure Blob
- SSH server

**Configuration example:**
```bash
# Add remote storage
dvc remote add -d myremote /tmp/dvc-storage
dvc remote add -d s3remote s3://mybucket/dvcstore
dvc remote add -d gdrive gdrive://folder-id
```

**In `.dvc/config`:**
```ini
[core]
    remote = myremote
['remote "myremote"']
    url = /tmp/dvc-storage
```

---

## 🔄 Key Concepts Summary

### Pipeline Stages

**What:** Individual steps in your ML workflow
**Why:** Break complex workflows into manageable, cacheable steps

**Example workflow:**
```
raw_data → [prepare] → processed_data → [train] → model → [evaluate] → metrics
```

Each stage:
- Has dependencies (inputs)
- Produces outputs
- Runs only when dependencies change (smart caching!)

---

### Dependencies (`deps`)

**What:** Files that a stage needs as input
**Behavior:** If any dependency changes, stage re-runs

**Example:**
```yaml
stages:
  train:
    cmd: python train.py
    deps:
      - train.py           # Code dependency
      - data.csv          # Data dependency
      - config.json       # Config dependency
```

---

### Outputs (`outs`)

**What:** Files created by a stage
**Types:**
- Regular outputs (cached)
- Metrics (not cached, tracked)
- Plots (for visualization)

**Example:**
```yaml
outs:
  - model.pkl              # Regular output (cached)
  - predictions.csv        # Regular output
metrics:
  - metrics.json:          # Metric file (not cached)
      cache: false
plots:
  - confusion_matrix.csv   # Plot data
```

---

### Parameters (`params`)

**What:** Hyperparameters and config values
**Stored in:** `params.yaml` file
**Purpose:** Track experiment configurations

**`params.yaml` example:**
```yaml
train:
  learning_rate: 0.001
  epochs: 100
  batch_size: 32

model:
  hidden_layers: [128, 64, 32]
  dropout: 0.3
```

**Reference in `dvc.yaml`:**
```yaml
stages:
  train:
    params:
      - train.learning_rate
      - train.epochs
```

---

### Metrics

**What:** Numerical results from experiments (accuracy, loss, etc.)
**Format:** JSON or YAML
**Not cached:** Always committed to Git for comparison

**Example `metrics.json`:**
```json
{
  "accuracy": 0.95,
  "loss": 0.12,
  "f1_score": 0.93
}
```

**Track with:**
```bash
dvc metrics show
dvc metrics diff
```

---

## 📊 Visual Relationships

```
┌─────────────────────────────────────────────────────┐
│                    Your Project                      │
├─────────────────────────────────────────────────────┤
│                                                      │
│  Git Tracks:                DVC Tracks:             │
│  ├── code (*.py)            ├── data.csv.dvc        │
│  ├── dvc.yaml               ├── model.pkl.dvc       │
│  ├── dvc.lock               │                       │
│  ├── params.yaml            │                       │
│  └── metrics.json           │                       │
│                             │                       │
│                             ▼                       │
│                      .dvc/cache/                    │
│                      (local storage)                │
│                             │                       │
│                             ▼                       │
│                      Remote Storage                 │
│                   (S3, GDrive, etc.)                │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Quick Reference Table

| Concept | What It Is | Where It Lives | Git Tracked? |
|---------|-----------|----------------|--------------|
| `.dvc` file | Pointer to large file | Project root | ✅ Yes |
| `dvc.yaml` | Pipeline definition | Project root | ✅ Yes |
| `dvc.lock` | Pipeline state/lock | Project root | ✅ Yes |
| `params.yaml` | Hyperparameters | Project root | ✅ Yes |
| `metrics.json` | Experiment results | Project root | ✅ Yes |
| `.dvc/cache` | Local file storage | `.dvc/` folder | ❌ No |
| Remote | Cloud/shared storage | External | ❌ No |
| Actual data | Large files | Cache/Remote | ❌ No |

---

## 💡 Mental Model

**Think of DVC as a smart layer on top of Git:**

1. **Git** = Tracks what changed (code, configs, pointers)
2. **DVC** = Manages where large files are stored and how to reproduce pipelines
3. **`.dvc` files** = Like Git commits for individual files
4. **`dvc.yaml`** = Like a Makefile for ML pipelines
5. **`dvc.lock`** = Like package-lock.json for exact reproducibility

---

## 🔑 Key Takeaways

1. **`.dvc` files** are pointers (metadata), not the actual data
2. **`dvc.yaml`** defines your pipeline, **`dvc.lock`** locks the exact state
3. **Cache** is local, **remote** is for sharing/backup
4. **Parameters** and **metrics** help track experiments
5. **Stages** break work into reusable, cacheable steps

---
