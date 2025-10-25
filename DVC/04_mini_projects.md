# 04. DVC Mini Projects - Hands-On Practice

---

## 🎓 Project 1: Simple Data Tracking

**Goal:** Learn basic DVC file tracking
**Time:** 10 minutes
**Skills:** `dvc add`, `dvc push`, `dvc pull`

### Step 1: Setup

```bash
# Create project
mkdir dvc-project-1
cd dvc-project-1

# Initialize Git and DVC
git init
dvc init

# Commit DVC initialization
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Step 2: Create sample data

```bash
# Create a CSV file
cat > data.csv << EOF
name,age,salary
Alice,25,50000
Bob,30,60000
Charlie,35,70000
David,40,80000
Eve,45,90000
EOF

# Check file size
ls -lh data.csv
```

### Step 3: Track with DVC

```bash
# Add file to DVC tracking
dvc add data.csv

# What was created?
ls -la
# - data.csv.dvc (pointer file)
# - .gitignore (contains data.csv)

# Check .gitignore
cat .gitignore
# /data.csv

# Commit to Git
git add data.csv.dvc .gitignore
git commit -m "Track data.csv with DVC"
```

### Step 4: Setup remote storage

```bash
# Create local remote (for practice)
mkdir -p /tmp/dvc-storage

# Add remote
dvc remote add -d local /tmp/dvc-storage

# Commit config
git add .dvc/config
git commit -m "Configure remote storage"
```

### Step 5: Push data

```bash
# Upload to remote
dvc push

# Verify upload
ls /tmp/dvc-storage/files/md5/
```

### Step 6: Simulate team member pulling data

```bash
# Delete local file (simulate fresh clone)
rm data.csv

# Pull from remote
dvc pull

# Verify file is back
cat data.csv
```

**✅ What you learned:**
- Track files with `dvc add`
- Setup remote storage
- Share data with `dvc push/pull`
- `.dvc` files are pointers to actual data

---

## 🎓 Project 2: Simple ML Pipeline

**Goal:** Build a complete ML pipeline with DVC
**Time:** 30 minutes
**Skills:** `dvc.yaml`, `dvc repro`, parameters, metrics

### Project Structure

```
dvc-project-2/
├── data/
│   └── raw_data.csv
├── src/
│   ├── prepare.py
│   ├── train.py
│   └── evaluate.py
├── params.yaml
└── dvc.yaml
```

### Step 1: Setup

```bash
# Create project
mkdir -p dvc-project-2/src
cd dvc-project-2

# Initialize
git init
dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"
```

### Step 2: Create dataset

```bash
# Create data directory
mkdir -p data

# Create sample dataset
cat > data/raw_data.csv << EOF
feature1,feature2,label
1.2,3.4,0
2.3,4.5,1
3.4,5.6,0
4.5,6.7,1
5.6,7.8,0
6.7,8.9,1
7.8,9.0,0
8.9,10.1,1
9.0,11.2,0
10.1,12.3,1
1.5,3.8,0
2.8,4.9,1
3.9,5.2,0
4.2,6.1,1
5.3,7.4,0
EOF

# Track raw data
dvc add data/raw_data.csv
git add data/raw_data.csv.dvc data/.gitignore
git commit -m "Add raw dataset"
```

### Step 3: Create parameters file

```bash
cat > params.yaml << EOF
prepare:
  train_split: 0.8
  random_seed: 42

train:
  n_estimators: 100
  max_depth: 5
  random_state: 42

evaluate:
  threshold: 0.5
EOF

git add params.yaml
git commit -m "Add parameters"
```

### Step 4: Create preprocessing script

```python
cat > src/prepare.py << 'EOF'
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load params
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load data
df = pd.read_csv('data/raw_data.csv')

# Split data
train, test = train_test_split(
    df,
    train_size=params['prepare']['train_split'],
    random_state=params['prepare']['random_seed']
)

# Save
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)

print(f"Train size: {len(train)}")
print(f"Test size: {len(test)}")
EOF
```

### Step 5: Create training script

```python
cat > src/train.py << 'EOF'
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

# Load params
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load data
train = pd.read_csv('data/train.csv')
X_train = train[['feature1', 'feature2']]
y_train = train['label']

# Train model
model = RandomForestClassifier(
    n_estimators=params['train']['n_estimators'],
    max_depth=params['train']['max_depth'],
    random_state=params['train']['random_state']
)
model.fit(X_train, y_train)

# Save model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained successfully")
EOF
```

### Step 6: Create evaluation script

```python
cat > src/evaluate.py << 'EOF'
import pandas as pd
import pickle
import json
import yaml
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load params
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load test data
test = pd.read_csv('data/test.csv')
X_test = test[['feature1', 'feature2']]
y_test = test['label']

# Predict
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred)),
    'recall': float(recall_score(y_test, y_pred))
}

# Save metrics
with open('metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("Evaluation complete:")
print(json.dumps(metrics, indent=2))
EOF
```

### Step 7: Commit code

```bash
git add src/
git commit -m "Add pipeline scripts"
```

### Step 8: Create DVC pipeline

```bash
# Add prepare stage
dvc stage add -n prepare \
  -d data/raw_data.csv \
  -d src/prepare.py \
  -p prepare.train_split,prepare.random_seed \
  -o data/train.csv \
  -o data/test.csv \
  python src/prepare.py

# Add train stage
dvc stage add -n train \
  -d data/train.csv \
  -d src/train.py \
  -p train.n_estimators,train.max_depth,train.random_state \
  -o model.pkl \
  python src/train.py

# Add evaluate stage
dvc stage add -n evaluate \
  -d model.pkl \
  -d data/test.csv \
  -d src/evaluate.py \
  -M metrics.json \
  python src/evaluate.py

# Check dvc.yaml
cat dvc.yaml
```

### Step 9: Run pipeline

```bash
# Install dependencies (if needed)
pip install pandas scikit-learn pyyaml

# Run entire pipeline
dvc repro

# What happens:
# 1. prepare: Splits data
# 2. train: Trains model
# 3. evaluate: Evaluates and saves metrics

# Check metrics
dvc metrics show

# Check outputs
ls -la
# - data/train.csv
# - data/test.csv
# - model.pkl
# - metrics.json
# - dvc.lock (created)
```

### Step 10: Commit results

```bash
git add dvc.yaml dvc.lock .gitignore metrics.json
git commit -m "Add ML pipeline and baseline results"
```

### Step 11: Run experiment (change parameters)

```bash
# Edit parameters
sed -i 's/n_estimators: 100/n_estimators: 200/' params.yaml

# Run pipeline (only train and evaluate will run!)
dvc repro

# Compare metrics
dvc metrics diff HEAD

# Commit experiment
git add params.yaml dvc.lock metrics.json
git commit -m "Experiment: increase n_estimators to 200"
```

### Step 12: Visualize pipeline

```bash
# Show pipeline graph
dvc dag

# Expected output:
# +----------------+
# | data/raw_data  |
# +----------------+
#         *
#         *
#         *
# +----------------+
# | prepare        |
# +----------------+
#         *
#         *
#         *
# +----------------+
# | train          |
# +----------------+
#         *
#         *
#         *
# +----------------+
# | evaluate       |
# +----------------+
```

**✅ What you learned:**
- Create multi-stage pipeline with `dvc.yaml`
- Use parameters from `params.yaml`
- Track metrics with `metrics.json`
- Run pipeline with `dvc repro` (smart caching!)
- Compare experiments with `dvc metrics diff`

---

## 🎓 Project 3: Multiple Experiments with Branches

**Goal:** Use Git branches for experiment tracking
**Time:** 20 minutes
**Skills:** Git integration, experiment comparison

### Step 1: Baseline (continuing from Project 2)

```bash
# Ensure you're on main branch
git checkout -b main 2>/dev/null || git checkout main

# Run baseline
dvc repro
dvc metrics show
```

### Step 2: Experiment 1 - Tune tree depth

```bash
# Create experiment branch
git checkout -b exp-depth-10

# Edit params
cat > params.yaml << EOF
prepare:
  train_split: 0.8
  random_seed: 42

train:
  n_estimators: 100
  max_depth: 10
  random_state: 42

evaluate:
  threshold: 0.5
EOF

# Run pipeline
dvc repro

# Commit
git add params.yaml dvc.lock metrics.json
git commit -m "Experiment: max_depth=10"

# Check metrics
dvc metrics show
```

### Step 3: Experiment 2 - More trees

```bash
# Create another experiment
git checkout main
git checkout -b exp-trees-300

# Edit params
cat > params.yaml << EOF
prepare:
  train_split: 0.8
  random_seed: 42

train:
  n_estimators: 300
  max_depth: 5
  random_state: 42

evaluate:
  threshold: 0.5
EOF

# Run pipeline
dvc repro

# Commit
git add params.yaml dvc.lock metrics.json
git commit -m "Experiment: n_estimators=300"
```

### Step 4: Compare all experiments

```bash
# Compare with main
dvc metrics diff main

# Compare parameters
dvc params diff main

# Switch between experiments
git checkout main
dvc checkout
dvc metrics show

git checkout exp-depth-10
dvc checkout
dvc metrics show

git checkout exp-trees-300
dvc checkout
dvc metrics show
```

### Step 5: Merge best experiment

```bash
# Assume exp-trees-300 was best
git checkout main
git merge exp-trees-300 -m "Merge best experiment"

# Now main has the best model
dvc repro  # Should be cached, nothing runs
```

**✅ What you learned:**
- Use Git branches for experiments
- Compare experiments across branches
- Keep track of multiple configurations
- Merge winning experiments to main

---

## 🎓 Project 4: Data Versioning

**Goal:** Version your dataset and switch between versions
**Time:** 15 minutes
**Skills:** Data versioning, time travel

### Step 1: Setup

```bash
mkdir dvc-project-4
cd dvc-project-4
git init
dvc init
git add .dvc .dvcignore
git commit -m "Init"
```

### Step 2: Version 1 of dataset

```bash
# Create small dataset
cat > data.csv << EOF
id,value
1,10
2,20
3,30
EOF

# Track it
dvc add data.csv
git add data.csv.dvc .gitignore
git commit -m "Data v1: 3 rows"

# Tag it
git tag -a v1-data -m "Dataset version 1"
```

### Step 3: Version 2 of dataset (add more rows)

```bash
# Update dataset
cat > data.csv << EOF
id,value
1,10
2,20
3,30
4,40
5,50
EOF

# Track changes
dvc add data.csv
git add data.csv.dvc
git commit -m "Data v2: 5 rows"

git tag -a v2-data -m "Dataset version 2"
```

### Step 4: Version 3 of dataset (clean data)

```bash
# Clean dataset
cat > data.csv << EOF
id,value,category
1,10,A
2,20,B
3,30,A
4,40,B
5,50,A
6,60,B
EOF

# Track changes
dvc add data.csv
git add data.csv.dvc
git commit -m "Data v3: added category column"

git tag -a v3-data -m "Dataset version 3"
```

### Step 5: Time travel!

```bash
# Check current version
cat data.csv  # 6 rows, 3 columns

# Go back to v1
git checkout v1-data
dvc checkout
cat data.csv  # 3 rows, 2 columns

# Go to v2
git checkout v2-data
dvc checkout
cat data.csv  # 5 rows, 2 columns

# Back to latest
git checkout main
dvc checkout
cat data.csv  # 6 rows, 3 columns
```

### Step 6: View history

```bash
# See all data versions
git log --oneline --decorate

# See what changed in data.csv.dvc
git log -p data.csv.dvc
```

**✅ What you learned:**
- Version datasets like code
- Use Git tags for dataset versions
- Travel back to any dataset version
- Track dataset evolution

---

## 🎓 Project 5: Remote Storage with Google Drive

**Goal:** Use Google Drive as DVC remote
**Time:** 15 minutes
**Skills:** Cloud storage integration

### Step 1: Setup

```bash
# Install DVC with Google Drive support
pip install dvc[gdrive]

mkdir dvc-project-5
cd dvc-project-5
git init
dvc init
```

### Step 2: Create and track data

```bash
# Create large-ish file
dd if=/dev/urandom of=large_file.bin bs=1M count=10

# Track it
dvc add large_file.bin
git add large_file.bin.dvc .gitignore
git commit -m "Add large file"
```

### Step 3: Setup Google Drive remote

```bash
# Add Google Drive remote
# You'll need a Google Drive folder ID
# Get it from: https://drive.google.com/drive/folders/YOUR_FOLDER_ID
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID

# Commit config
git add .dvc/config
git commit -m "Add Google Drive remote"
```

### Step 4: Push to Google Drive

```bash
# First push will open browser for authentication
dvc push

# Follow OAuth flow in browser
# File will be uploaded to your Google Drive folder
```

### Step 5: Simulate teammate

```bash
# Delete local data
rm large_file.bin
rm -rf .dvc/cache

# Pull from Google Drive
dvc pull

# File is back!
ls -lh large_file.bin
```

**✅ What you learned:**
- Configure cloud remote (Google Drive)
- Authenticate with OAuth
- Share large files via cloud
- Team collaboration with DVC

---

## 🎓 Project 6: Complete Real-World Project

**Goal:** Build a complete ML project with best practices
**Time:** 45 minutes

### Project: Iris Classification Pipeline

```
iris-project/
├── data/
│   └── raw/
├── src/
│   ├── prepare.py
│   ├── train.py
│   └── evaluate.py
├── models/
├── metrics/
├── plots/
├── params.yaml
├── dvc.yaml
└── .dvc/
```

### Step 1: Setup project

```bash
mkdir -p iris-project/{data/raw,src,models,metrics,plots}
cd iris-project

git init
dvc init
git add .dvc .dvcignore
git commit -m "Initialize project"

# Setup remote (local for practice)
mkdir -p /tmp/iris-storage
dvc remote add -d storage /tmp/iris-storage
git add .dvc/config
git commit -m "Configure storage"
```

### Step 2: Download data

```python
cat > src/download.py << 'EOF'
from sklearn.datasets import load_iris
import pandas as pd

# Load iris dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = iris.target

# Save
df.to_csv('data/raw/iris.csv', index=False)
print(f"Downloaded {len(df)} samples")
EOF

python src/download.py

# Track raw data
dvc add data/raw/iris.csv
git add data/raw/iris.csv.dvc data/raw/.gitignore src/download.py
git commit -m "Add iris dataset"
```

### Step 3: Create pipeline

**params.yaml:**
```yaml
cat > params.yaml << 'EOF'
prepare:
  test_size: 0.2
  random_state: 42

train:
  model_type: random_forest
  n_estimators: 100
  max_depth: 5
  random_state: 42

featurize:
  use_sepal: true
  use_petal: true
EOF

git add params.yaml
git commit -m "Add parameters"
```

**src/prepare.py:**
```python
cat > src/prepare.py << 'EOF'
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

# Load params
with open('params.yaml') as f:
    params = yaml.safe_load(f)

# Load data
df = pd.read_csv('data/raw/iris.csv')

# Split
train, test = train_test_split(
    df,
    test_size=params['prepare']['test_size'],
    random_state=params['prepare']['random_state'],
    stratify=df['species']
)

# Save
train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)

print(f"Train: {len(train)}, Test: {len(test)}")
EOF
```

**src/train.py:**
```python
cat > src/train.py << 'EOF'
import pandas as pd
import pickle
import yaml
from sklearn.ensemble import RandomForestClassifier

with open('params.yaml') as f:
    params = yaml.safe_load(f)

train = pd.read_csv('data/train.csv')
X = train.drop('species', axis=1)
y = train['species']

model = RandomForestClassifier(
    n_estimators=params['train']['n_estimators'],
    max_depth=params['train']['max_depth'],
    random_state=params['train']['random_state']
)
model.fit(X, y)

with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained")
EOF
```

**src/evaluate.py:**
```python
cat > src/evaluate.py << 'EOF'
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix

with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

test = pd.read_csv('data/test.csv')
X_test = test.drop('species', axis=1)
y_test = test['species']

y_pred = model.predict(X_test)

metrics = {
    'accuracy': accuracy_score(y_test, y_pred),
    'report': classification_report(y_test, y_pred, output_dict=True)
}

with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# Save confusion matrix
cm = confusion_matrix(y_test, y_pred)
pd.DataFrame(cm).to_csv('plots/confusion_matrix.csv')

print(f"Accuracy: {metrics['accuracy']:.3f}")
EOF

git add src/
git commit -m "Add pipeline scripts"
```

### Step 4: Create DVC pipeline

```bash
dvc stage add -n prepare \
  -d data/raw/iris.csv \
  -d src/prepare.py \
  -p prepare \
  -o data/train.csv \
  -o data/test.csv \
  python src/prepare.py

dvc stage add -n train \
  -d data/train.csv \
  -d src/train.py \
  -p train \
  -o models/model.pkl \
  python src/train.py

dvc stage add -n evaluate \
  -d models/model.pkl \
  -d data/test.csv \
  -d src/evaluate.py \
  -M metrics/metrics.json \
  --plots-no-cache plots/confusion_matrix.csv \
  python src/evaluate.py

git add dvc.yaml
git commit -m "Define pipeline"
```

### Step 5: Run and track

```bash
# Install dependencies
pip install pandas scikit-learn pyyaml

# Run pipeline
dvc repro

# View results
dvc metrics show
cat metrics/metrics.json

# Commit
git add dvc.lock metrics/ plots/ .gitignore
git commit -m "Baseline model results"

# Push data and models
dvc push
git push
```

### Step 6: Run experiments

```bash
# Experiment 1: More trees
git checkout -b exp-trees-200
sed -i 's/n_estimators: 100/n_estimators: 200/' params.yaml
dvc repro
git add params.yaml dvc.lock metrics/
git commit -m "Exp: 200 trees"

# Experiment 2: Deeper trees
git checkout main
git checkout -b exp-depth-10
sed -i 's/max_depth: 5/max_depth: 10/' params.yaml
dvc repro
git add params.yaml dvc.lock metrics/
git commit -m "Exp: depth 10"

# Compare
git checkout main
dvc metrics diff exp-trees-200
dvc metrics diff exp-depth-10
```

**✅ What you learned:**
- Build complete ML project structure
- Organize code, data, models, metrics
- Run systematic experiments
- Track everything with Git + DVC
- Real-world best practices

---

## 🎯 Summary Table

| Project | Focus | Key Skills | Time |
|---------|-------|------------|------|
| 1 | Data Tracking | `dvc add`, push/pull | 10 min |
| 2 | ML Pipeline | `dvc.yaml`, repro, metrics | 30 min |
| 3 | Experiments | Git branches, comparison | 20 min |
| 4 | Versioning | Time travel, tags | 15 min |
| 5 | Cloud Storage | Google Drive remote | 15 min |
| 6 | Real Project | Best practices | 45 min |

---

## 💡 Practice Tips

1. **Start small:** Complete projects 1-3 before moving to complex ones
2. **Type commands:** Don't copy-paste, build muscle memory
3. **Break things:** Delete files, try to recover them
4. **Experiment:** Change parameters, see what happens
5. **Commit often:** Practice the Git + DVC workflow
6. **Read outputs:** Understand what each command does

---

## 📚 Next Steps

After completing these projects:
1. Use your own datasets
2. Build your own ML pipelines
3. Integrate with CI/CD
4. Try DVC Studio for visualization
5. Explore advanced features (plots, DVCLive)

---
