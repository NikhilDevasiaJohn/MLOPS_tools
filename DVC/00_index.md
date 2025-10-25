# DVC Study Guide - Complete Index

> **A developer-friendly guide to mastering DVC for AI/ML projects**

---

## 📚 Table of Contents

### [01. Introduction to DVC](01_intro.md)
**What you'll learn:**
- What DVC is and why it exists
- How DVC works with Git
- The library analogy for understanding DVC
- Key problems DVC solves

**Time:** 10 minutes
**Best for:** Complete beginners

---

### [02. Key Concepts](02_key_concepts.md)
**What you'll learn:**
- `.dvc` files (pointer files)
- `dvc.yaml` (pipeline definition)
- `dvc.lock` (pipeline lockfile)
- `.dvc/` directory and cache
- Remote storage
- Stages, dependencies, outputs, parameters, metrics

**Time:** 20 minutes
**Best for:** Understanding DVC architecture

---

### [03. Commands Reference](03_commands.md)
**What you'll learn:**
- Installation & setup
- File tracking (`dvc add`, `dvc checkout`)
- Remote storage (`dvc remote`, `dvc push/pull`)
- Pipeline commands (`dvc stage add`, `dvc repro`, `dvc dag`)
- Experiments (`dvc metrics`, `dvc params`)
- Utilities (`dvc gc`, `dvc diff`)

**Time:** 30 minutes
**Best for:** Command-line reference, quick lookup

---

### [04. Mini Projects](04_mini_projects.md)
**What you'll learn:**
- **Project 1:** Simple data tracking (10 min)
- **Project 2:** ML pipeline with DVC (30 min)
- **Project 3:** Multiple experiments with branches (20 min)
- **Project 4:** Data versioning (15 min)
- **Project 5:** Google Drive remote (15 min)
- **Project 6:** Complete real-world project (45 min)

**Time:** 2-3 hours total
**Best for:** Hands-on practice, building muscle memory

---

### [05. Troubleshooting](05_troubleshooting.md)
**What you'll learn:**
- Installation issues
- File tracking errors
- Remote storage problems
- Pipeline errors
- Merge conflicts
- Cache & storage issues
- Performance problems
- Debugging strategies

**Time:** Reference as needed
**Best for:** Fixing errors, understanding what went wrong

---

### [06. Tips & Best Practices](06_tips_and_best_practices.md)
**What you'll learn:**
- Golden rules
- Project structure
- Pipeline best practices
- Experiment tracking strategies
- Data management
- Performance optimization
- Security
- Team collaboration
- CI/CD integration

**Time:** 30 minutes
**Best for:** Professional workflows, production use

---

## 🎯 Quick Start Paths

### Path 1: Complete Beginner (4-5 hours)
1. Read [01_intro.md](01_intro.md) - 10 min
2. Read [02_key_concepts.md](02_key_concepts.md) - 20 min
3. Do [04_mini_projects.md](04_mini_projects.md) Project 1 - 10 min
4. Do [04_mini_projects.md](04_mini_projects.md) Project 2 - 30 min
5. Read [03_commands.md](03_commands.md) - 30 min
6. Do [04_mini_projects.md](04_mini_projects.md) Projects 3-4 - 35 min
7. Skim [06_tips_and_best_practices.md](06_tips_and_best_practices.md) - 20 min

### Path 2: Quick Learner (2 hours)
1. Skim [01_intro.md](01_intro.md) + [02_key_concepts.md](02_key_concepts.md) - 15 min
2. Do [04_mini_projects.md](04_mini_projects.md) Projects 1-2 - 40 min
3. Reference [03_commands.md](03_commands.md) as needed - 30 min
4. Read [06_tips_and_best_practices.md](06_tips_and_best_practices.md) - 30 min

### Path 3: Just the Essentials (30 min)
1. Read [01_intro.md](01_intro.md) - 10 min
2. Quick skim [02_key_concepts.md](02_key_concepts.md) - 5 min
3. Bookmark [03_commands.md](03_commands.md) for reference
4. Do [04_mini_projects.md](04_mini_projects.md) Project 1 - 10 min
5. Keep [05_troubleshooting.md](05_troubleshooting.md) handy

---

## 📖 How to Use This Guide

### For Learning
1. **Read sequentially** (01 → 02 → 03 → 04)
2. **Type, don't copy-paste** commands
3. **Do all mini projects** for hands-on experience
4. **Experiment and break things** - then fix them!

### For Reference
- Use [03_commands.md](03_commands.md) for command lookup
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
- ✅ Initialize DVC in a project
- ✅ Track large files with DVC
- ✅ Configure remote storage
- ✅ Push/pull data to/from remote
- ✅ Share data with teammates

**Intermediate Skills:**
- ✅ Create ML pipelines with `dvc.yaml`
- ✅ Use parameters and metrics
- ✅ Run reproducible experiments
- ✅ Track and compare experiments
- ✅ Version datasets

**Advanced Skills:**
- ✅ Design efficient pipeline architectures
- ✅ Optimize cache performance
- ✅ Implement team workflows
- ✅ Integrate with CI/CD
- ✅ Debug complex issues

---

## 💡 Key Concepts at a Glance

| Concept | File | Purpose |
|---------|------|---------|
| Pointer | `.dvc` | Links to large files |
| Pipeline | `dvc.yaml` | Defines ML workflow |
| Lock | `dvc.lock` | Ensures reproducibility |
| Cache | `.dvc/cache/` | Local file storage |
| Remote | Cloud/storage | Shared file storage |
| Params | `params.yaml` | Hyperparameters |
| Metrics | `metrics.json` | Experiment results |

---

## 🔧 Essential Commands

```bash
# Setup
dvc init

# Track data
dvc add data.csv
git add data.csv.dvc .gitignore
git commit -m "Track data"

# Remote storage
dvc remote add -d storage /tmp/dvc-storage
dvc push

# Pipeline
dvc stage add -n train -d data.csv -o model.pkl python train.py
dvc repro

# Experiments
dvc metrics show
dvc metrics diff
```

---

## 🚀 Project Templates

### Simple ML Project
```
project/
├── data/
│   └── raw/data.csv
├── src/
│   ├── train.py
│   └── evaluate.py
├── models/
├── metrics/
├── params.yaml
└── dvc.yaml
```

### Complete ML Project
```
project/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── src/
│   ├── data/
│   ├── features/
│   └── models/
├── models/
├── notebooks/
├── metrics/
├── plots/
├── params.yaml
├── dvc.yaml
└── README.md
```

---

## 📊 Workflow Diagram

```
┌─────────────────────────────────────────────┐
│           Your ML Project                    │
├─────────────────────────────────────────────┤
│                                              │
│  1. Edit code/params                         │
│     ↓                                        │
│  2. dvc repro                                │
│     ↓                                        │
│  3. dvc metrics show                         │
│     ↓                                        │
│  4. git add dvc.lock params.yaml metrics.json│
│     ↓                                        │
│  5. git commit -m "Experiment: ..."          │
│     ↓                                        │
│  6. dvc push && git push                     │
│                                              │
└─────────────────────────────────────────────┘
```

---

## 🎯 Common Use Cases

### Use Case 1: Track Large Dataset
```bash
dvc add data/large_dataset.csv
git add data/large_dataset.csv.dvc .gitignore
git commit -m "Add dataset"
dvc push
```

### Use Case 2: Run ML Pipeline
```bash
dvc repro
dvc metrics show
```

### Use Case 3: Compare Experiments
```bash
git checkout experiment-branch
dvc checkout
dvc metrics diff main
```

### Use Case 4: Clone Project & Get Data
```bash
git clone repo-url
cd project
dvc pull
```

---

## 🆘 Quick Help

**Stuck?**
1. Check [05_troubleshooting.md](05_troubleshooting.md)
2. Run `dvc status` and `git status`
3. Try `dvc repro -v` for verbose output
4. Read error messages carefully

**Need help?**
- DVC Docs: https://dvc.org/doc
- DVC Discord: https://dvc.org/chat
- GitHub Issues: https://github.com/iterative/dvc/issues

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
- [ ] Built own ML project with DVC
- [ ] Used DVC in team setting
- [ ] Integrated DVC with CI/CD

---

## 🏆 Mastery Goals

**Beginner Level:**
- Can track files and use remotes
- Understands basic concepts
- Can follow existing pipelines

**Intermediate Level:**
- Can create pipelines from scratch
- Can run and track experiments
- Can debug common issues

**Advanced Level:**
- Can design efficient architectures
- Can optimize team workflows
- Can integrate with production systems

---

## 🔖 Bookmark These

**Most Important:**
1. [03_commands.md](03_commands.md) - Command reference
2. [05_troubleshooting.md](05_troubleshooting.md) - Error fixes

**For Deep Understanding:**
1. [02_key_concepts.md](02_key_concepts.md) - Architecture
2. [06_tips_and_best_practices.md](06_tips_and_best_practices.md) - Professional workflows

**For Practice:**
1. [04_mini_projects.md](04_mini_projects.md) - All projects

---

## 📈 Next Steps After This Guide

1. **Apply to real project**
   - Use your own dataset
   - Build your own pipeline
   - Track your experiments

2. **Explore advanced features**
   - DVC Studio (web UI)
   - DVCLive (experiment tracking)
   - CML (CI/CD for ML)

3. **Join community**
   - Share your learnings
   - Help others
   - Contribute to DVC

---

## ✨ Final Tips

1. **Practice daily** - Even 15 minutes helps
2. **Build real projects** - Apply what you learn
3. **Make mistakes** - That's how you learn
4. **Ask questions** - Community is helpful
5. **Share knowledge** - Teaching reinforces learning

---

**Happy Learning! 🚀**

*Remember: DVC is just Git for data. If you know Git, you already understand 80% of DVC!*

---
