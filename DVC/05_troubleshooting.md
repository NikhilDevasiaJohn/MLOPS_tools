# 05. DVC Troubleshooting & Common Errors

---

## 🚨 Installation Issues

### Error: `dvc: command not found`

**Problem:** DVC not installed or not in PATH

**Solution:**
```bash
# Install DVC
pip install dvc

# Verify installation
dvc version

# If still not found, check PATH
which dvc

# Alternative: Install with specific remote
pip install dvc[s3]      # For S3
pip install dvc[gdrive]  # For Google Drive
pip install dvc[all]     # For all remotes
```

---

### Error: `No module named 'dvc'`

**Problem:** DVC installed in wrong Python environment

**Solution:**
```bash
# Check Python version
python --version
pip --version

# Install in current environment
python -m pip install dvc

# If using virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install dvc
```

---

## 🚨 Initialization Errors

### Error: `DVC is not initialized in this directory`

**Problem:** Running DVC commands before `dvc init`

**Solution:**
```bash
# Initialize DVC
dvc init

# Commit initialization
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# If .git doesn't exist either
git init
dvc init
```

---

### Error: `.git directory not found`

**Problem:** DVC requires Git repository

**Solution:**
```bash
# Initialize Git first
git init

# Then initialize DVC
dvc init
```

**Remember:** DVC works ON TOP of Git, not standalone!

---

## 🚨 File Tracking Issues

### Error: `output 'file.csv' already exists`

**Problem:** Trying to add file that's already tracked by Git

**Solution:**
```bash
# Option 1: Remove from Git first
git rm --cached file.csv
git commit -m "Remove from Git"
dvc add file.csv

# Option 2: Check .gitignore
cat .gitignore
# If file is there, it's probably already tracked by DVC
ls file.csv.dvc

# Option 3: Force add
dvc add file.csv --force
```

---

### Error: `failed to find 'data.csv.dvc'`

**Problem:** `.dvc` file missing or deleted

**Solution:**
```bash
# Option 1: Recover from Git
git checkout data.csv.dvc

# Option 2: Re-track file
dvc add data.csv

# Option 3: Check Git history
git log --all --full-history -- "*.dvc"
```

---

### Error: `file 'data.csv' is modified`

**Problem:** Tracked file was modified outside DVC

**Solution:**
```bash
# Check what changed
dvc status

# Option 1: Keep changes and update tracking
dvc add data.csv
git add data.csv.dvc
git commit -m "Update data"

# Option 2: Discard changes
dvc checkout data.csv

# Option 3: Compare changes
dvc diff
```

---

## 🚨 Remote Storage Errors

### Error: `failed to push data to the remote`

**Problem:** Remote not configured or inaccessible

**Solution:**
```bash
# Check remote configuration
dvc remote list

# If no remote:
dvc remote add -d myremote /tmp/dvc-storage
git add .dvc/config
git commit -m "Add remote"

# Test remote connection
dvc remote list
dvc push -v  # Verbose mode for details

# Check permissions
ls -la /tmp/dvc-storage  # For local remote
```

---

### Error: `authentication failed`

**Problem:** Credentials missing or incorrect (S3, GDrive, etc.)

**Solutions by remote type:**

**AWS S3:**
```bash
# Check AWS credentials
aws configure list

# Set credentials
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret

# Or configure remote with credentials
dvc remote modify myremote access_key_id YOUR_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET

# Or use AWS profile
dvc remote modify myremote profile myprofile
```

**Google Drive:**
```bash
# Re-authenticate
dvc push  # Will open browser for OAuth

# Check credentials
cat .dvc/tmp/gdrive-user-credentials.json

# Clear and re-auth
rm .dvc/tmp/gdrive-user-credentials.json
dvc push
```

**SSH:**
```bash
# Check SSH key
ssh -T user@host

# Add SSH key
ssh-add ~/.ssh/id_rsa

# Configure remote with key
dvc remote modify myremote password ''
dvc remote modify myremote ask_password false
```

---

### Error: `unable to find file in cache`

**Problem:** File not in local cache or remote

**Solution:**
```bash
# Check cache
dvc cache dir
ls .dvc/cache/files/md5/

# Pull from remote
dvc pull

# If file is truly missing, re-create
dvc status  # See what's missing
# Re-run pipeline or re-add file
dvc repro

# Check remote
dvc status -r myremote
```

---

## 🚨 Pipeline Errors

### Error: `stage 'train' does not exist`

**Problem:** Stage not defined in `dvc.yaml`

**Solution:**
```bash
# Check dvc.yaml
cat dvc.yaml

# Add missing stage
dvc stage add -n train \
  -d data.csv \
  -o model.pkl \
  python train.py

# Or edit dvc.yaml manually
vim dvc.yaml
```

---

### Error: `circular dependency detected`

**Problem:** Stage A depends on B, B depends on A

**Solution:**
```bash
# Check dependencies
dvc dag

# Fix circular dependency in dvc.yaml
# Example of circular dependency:
# stage1:
#   deps: [output2.csv]
#   outs: [output1.csv]
# stage2:
#   deps: [output1.csv]  # ← Depends on stage1
#   outs: [output2.csv]  # ← Stage1 depends on this!

# Fix by removing circular reference
vim dvc.yaml
```

---

### Error: `failed to run command`

**Problem:** Command in `dvc.yaml` failed

**Solution:**
```bash
# Check command manually
cat dvc.yaml  # Find the cmd
python train.py  # Run manually to see error

# Common issues:
# 1. Missing dependencies
pip install -r requirements.txt

# 2. Missing files
dvc pull  # Get data from remote

# 3. Wrong Python version
python --version

# 4. Syntax error in script
python -m py_compile train.py

# Run with verbose mode
dvc repro -v
```

---

### Error: `output 'file.csv' already tracked by SCM`

**Problem:** Output file tracked by Git, not DVC

**Solution:**
```bash
# Remove from Git
git rm --cached file.csv

# Add to .gitignore (should be automatic)
echo "file.csv" >> .gitignore

# Re-run pipeline
dvc repro

# Commit changes
git add .gitignore dvc.yaml dvc.lock
git commit -m "Fix tracked outputs"
```

---

### Error: `failed to load params from 'params.yaml'`

**Problem:** Parameters file missing or malformed

**Solution:**
```bash
# Check if file exists
ls params.yaml

# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('params.yaml'))"

# Common YAML errors:
# ❌ Bad indentation
# ❌ Missing colons
# ❌ Tabs instead of spaces

# Example fix:
cat > params.yaml << EOF
train:
  learning_rate: 0.01
  epochs: 100
EOF

# Test again
dvc repro
```

---

### Error: `stage 'name' changed`

**Problem:** Stage definition changed, need to re-run

**Solution:**
```bash
# This is actually expected behavior!
# DVC detected changes and wants to re-run

# Option 1: Re-run pipeline
dvc repro

# Option 2: Force re-run all stages
dvc repro -f

# Option 3: Update lock without running
dvc commit

# Check what changed
dvc status
```

---

## 🚨 Merge Conflicts

### Error: Merge conflict in `.dvc` file

**Problem:** Two branches modified same `.dvc` file

**Solution:**
```bash
# Check conflict
git status
cat data.csv.dvc

# It will look like:
# <<<<<<< HEAD
# outs:
# - md5: abc123
# =======
# outs:
# - md5: def456
# >>>>>>> branch

# Option 1: Accept one version
git checkout --ours data.csv.dvc    # Keep current branch
git checkout --theirs data.csv.dvc  # Keep incoming branch

# Option 2: Choose which data you want
dvc checkout  # Get data from current .dvc file

# Mark resolved
git add data.csv.dvc
git commit -m "Resolve conflict"
```

---

### Error: Merge conflict in `dvc.lock`

**Problem:** Two branches ran pipeline differently

**Solution:**
```bash
# Usually it's safe to regenerate dvc.lock
# Choose one version
git checkout --ours dvc.lock
# Or
git checkout --theirs dvc.lock

# Re-run pipeline to regenerate
dvc repro

# Commit new lock file
git add dvc.lock
git commit -m "Regenerate lock file"
```

---

## 🚨 Cache & Storage Issues

### Error: `no space left on device`

**Problem:** Disk full from DVC cache

**Solution:**
```bash
# Check cache size
du -sh .dvc/cache

# Option 1: Clean unused cache
dvc gc -w  # Remove files not used in workspace
dvc gc -c  # Remove files not in Git history

# Option 2: Clear all cache (dangerous!)
rm -rf .dvc/cache
dvc pull  # Re-download what you need

# Option 3: Move cache location
dvc cache dir /path/to/larger/disk
```

---

### Error: `file already exists in cache`

**Problem:** Cache corruption or duplicate

**Solution:**
```bash
# Usually harmless, but if causing issues:

# Option 1: Remove and re-add
dvc remove data.csv.dvc
dvc add data.csv

# Option 2: Clear specific cache entry
# Find MD5 from .dvc file
cat data.csv.dvc
# Remove from cache
rm .dvc/cache/files/md5/XX/yyy...

# Re-add
dvc add data.csv
```

---

## 🚨 Performance Issues

### Error: `dvc pull` is very slow

**Problem:** Large files or slow connection

**Solution:**
```bash
# Option 1: Pull only what you need
dvc pull data.csv.dvc  # Specific file
dvc pull -r myremote   # Specific remote

# Option 2: Use parallel downloads
dvc config cache.jobs 4
dvc pull

# Option 3: Use partial checkout
dvc fetch  # Just download to cache
dvc checkout data/  # Checkout only needed folder

# Option 4: Check network
# For S3, try different region
dvc remote modify myremote region us-east-1
```

---

### Error: `dvc repro` runs slowly

**Problem:** Not using cache effectively

**Solution:**
```bash
# Check what's being re-run
dvc status

# Tips for faster pipelines:
# 1. Break into smaller stages
# 2. Use parameters properly
# 3. Check dependencies

# Enable persistent cache
dvc config cache.type hardlink

# Use shared cache across projects
dvc cache dir /shared/cache
```

---

## 🚨 Reproducibility Issues

### Error: Different results on different machines

**Problem:** Non-deterministic code or missing dependencies

**Checklist:**
```bash
# 1. Check random seeds
# In your code:
# np.random.seed(42)
# random.seed(42)
# tf.random.set_seed(42)

# 2. Pin dependencies
pip freeze > requirements.txt

# 3. Check parameter files
git log -p params.yaml

# 4. Verify data versions
dvc diff

# 5. Check lock file
git diff dvc.lock

# 6. Same Python version?
python --version
```

---

## 🛠️ Debugging Tips

### General Debugging Strategy

```bash
# 1. Use verbose mode
dvc repro -v
dvc pull -v
dvc push -v

# 2. Check status
dvc status
git status

# 3. View pipeline
dvc dag

# 4. Check configuration
dvc config --list
cat .dvc/config

# 5. Validate files
cat dvc.yaml
cat dvc.lock
cat params.yaml

# 6. Manual testing
# Run commands from dvc.yaml manually
python train.py

# 7. Check logs
# DVC doesn't have logs, but you can redirect
dvc repro > log.txt 2>&1
```

---

### Reset Everything (Nuclear Option)

```bash
# ⚠️ WARNING: This deletes everything!

# Backup first
cp -r .dvc .dvc.backup
git branch backup-$(date +%s)

# Remove DVC
rm -rf .dvc
rm *.dvc
rm dvc.yaml dvc.lock

# Re-initialize
dvc init
git add .dvc .dvcignore
git commit -m "Reinitialize DVC"

# Re-add files
dvc add data.csv
# Recreate pipeline...
```

---

## 📋 Quick Troubleshooting Checklist

| Symptom | Quick Check | Fix |
|---------|------------|-----|
| File not found | `ls file` | `dvc pull` or `dvc checkout` |
| Command failed | Run manually | Fix script or install deps |
| Auth error | Check credentials | Re-authenticate |
| Slow performance | `dvc status` | Use cache, smaller stages |
| Merge conflict | `git status` | Choose version, re-run |
| Cache full | `du -sh .dvc/cache` | `dvc gc` |
| Wrong results | Check params/seed | Pin versions, set seeds |

---

## 🆘 Getting Help

```bash
# Built-in help
dvc --help
dvc add --help
dvc repro --help

# Check version
dvc version

# Validate installation
dvc doctor

# Community resources
# - GitHub: https://github.com/iterative/dvc
# - Discord: https://dvc.org/chat
# - Forum: https://discuss.dvc.org
# - Docs: https://dvc.org/doc
```

---

## 💡 Prevention Tips

**1. Commit often:**
```bash
# After each step
dvc add data.csv
git add data.csv.dvc
git commit -m "Track data"
```

**2. Test pipeline incrementally:**
```bash
# Don't write entire pipeline at once
dvc stage add -n step1 ...
dvc repro
# Test it works
dvc stage add -n step2 ...
dvc repro
```

**3. Use version control:**
```bash
# Tag important versions
git tag -a v1.0 -m "First working model"

# Branch for experiments
git checkout -b experiment-1
```

**4. Backup remotes:**
```bash
# Use multiple remotes
dvc remote add backup /backup/location
dvc push -r backup
```

**5. Document your workflow:**
```bash
# Create README.md
echo "## Setup" > README.md
echo "dvc pull" >> README.md
echo "dvc repro" >> README.md
```

---

## 🎯 Common Error Patterns

### Pattern 1: "Works on my machine"
- **Cause:** Missing `dvc.lock`, different data versions
- **Fix:** Commit `dvc.lock`, use `dvc pull`

### Pattern 2: "Pipeline won't re-run"
- **Cause:** Outputs cached, no changes detected
- **Fix:** `dvc repro -f` or change params/deps

### Pattern 3: "Data disappeared"
- **Cause:** Switched branches without `dvc checkout`
- **Fix:** `dvc checkout` after `git checkout`

### Pattern 4: "Can't push/pull"
- **Cause:** Remote misconfigured or no access
- **Fix:** Check `dvc remote list`, test credentials

### Pattern 5: "Git repo too large"
- **Cause:** Large files committed to Git, not DVC
- **Fix:** `git rm --cached`, use `dvc add` instead

---
