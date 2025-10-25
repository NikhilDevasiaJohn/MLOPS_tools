# 01. Introduction to DVC (Data Version Control)

## 💡 What is DVC?
**DVC (Data Version Control)** is an open-source tool that helps you manage:
- Datasets
- Machine learning models
- Experiments
- Pipelines

It works **on top of Git** — so your code and metadata are version-controlled in Git, while large files are stored elsewhere (like Google Drive, S3, or local storage).

**Breaking this down:**
1. **Git stores**: Your Python code, configuration files, and small tracking files (metadata)
   - These are lightweight and Git can handle them easily

2. **Elsewhere stores**: Your actual large files like:
   - Big datasets (100MB, 1GB, 10GB+)
   - Trained model files
   - Images, videos, etc.
   - These are too big for Git

**Why separate?**
- Git gets slow with large files
- So DVC creates a small "pointer file" that Git tracks
- The actual large file goes to cloud storage (Google Drive, AWS S3) or your local folder

**Simple analogy:**
Think of it like a library system:
- **Git** = The library catalog (small cards with book info)
- **DVC remote storage** = The actual shelves where books are stored

---

## 🧠 Why DVC?
When you build ML projects, you deal with:
- Large datasets (can’t commit to Git)
- Model files that change over time
- Experiment results that need tracking
- Pipelines that need reproducibility

DVC helps by:
- Tracking large files with lightweight `.dvc` metadata files
- Linking data, code, and results together
- Enabling you to reproduce results anytime

---

## ⚙️ How DVC Works
- **Git** tracks code & metadata (small files)
- **DVC** tracks large data & models
- **Remote storage** stores the actual content

### Simplified Workflow:
