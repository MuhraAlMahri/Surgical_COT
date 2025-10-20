# GitHub Setup Instructions

Complete guide for pushing your Surgical_COT repository to GitHub.

## 🎯 Quick Start

**The easiest way:**

```bash
cd /l/users/muhra.almahri/Surgical_COT
./push_to_github.sh
```

This automated script will handle everything for you! Just follow the prompts.

---

## 📋 Manual Steps (if you prefer)

### Step 1: Create GitHub Repository

1. Go to [https://github.com/new](https://github.com/new)
2. Fill in:
   - **Repository name**: `Surgical_COT`
   - **Description**: `Stage-wise training strategies for medical Visual Question Answering`
   - **Visibility**: Public (recommended for research) or Private
   - **❌ DO NOT** check "Initialize with README" (we already have one)
   - **❌ DO NOT** add .gitignore or license (we already have them)
3. Click "Create repository"

### Step 2: Authenticate with GitHub

You need to set up authentication. Choose **ONE** method:

#### Option A: Personal Access Token (Recommended) ⭐

1. Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
2. Click "Generate new token" → "Generate new token (classic)"
3. Give it a name: `Surgical_COT_Upload`
4. Select scopes: Check **`repo`** (full control of private repositories)
5. Click "Generate token"
6. **⚠️ COPY THE TOKEN NOW** - you won't see it again!
7. Save it somewhere secure (password manager)

When pushing, use:
- Username: `MuhraAlMahri`
- Password: `<your-personal-access-token>`

#### Option B: SSH Key

```bash
# Generate SSH key (if you don't have one)
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub:
# 1. Go to: https://github.com/settings/keys
# 2. Click "New SSH key"
# 3. Paste your public key
# 4. Save
```

### Step 3: Initialize Git and Push

```bash
cd /l/users/muhra.almahri/Surgical_COT

# Initialize git repository
git init

# Configure git (if not already done)
git config user.name "Muhra Al Mahri"
git config user.email "your.email@example.com"

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Complete Surgical COT experiments"

# Add remote repository
git remote add origin https://github.com/MuhraAlMahri/Surgical_COT.git

# Rename branch to main
git branch -M main

# Push to GitHub
git push -u origin main
```

When prompted for credentials:
- **Username**: `MuhraAlMahri`
- **Password**: Your personal access token (from Step 2)

---

## ✅ After Successful Push

### 1. Configure Repository Settings

Go to [https://github.com/MuhraAlMahri/Surgical_COT](https://github.com/MuhraAlMahri/Surgical_COT)

**Add Topics** (at the top of the page):
```
medical-ai
vision-language-models
vqa
visual-question-answering
curriculum-learning
medical-imaging
surgical-ai
deep-learning
pytorch
qwen
lora
fine-tuning
```

**Settings to Enable**:
- ✅ Issues (for bug reports and discussions)
- ✅ Discussions (optional, for community)
- ✅ Wiki (optional)

### 2. Add Repository Description

Click "⚙️ Settings" → "General" → Edit description:
```
Stage-wise training strategies for medical VQA: Comparing CXRTrek Sequential (77.59%) vs Curriculum Learning (64.24%) using Qwen2-VL-2B-Instruct
```

### 3. Create Release (Optional but Recommended)

1. Go to "Releases" → "Create a new release"
2. Tag version: `v1.0.0`
3. Release title: `v1.0.0 - Initial Release`
4. Description:
```markdown
## 🎉 Initial Release

Complete implementation of 4 medical VQA training strategies.

### 🏆 Results
- **CXRTrek Sequential**: 77.59% (WINNER) ✅
- **Qwen Ordering**: 67.12%
- **Curriculum Learning**: 64.24%
- **Random Baseline**: 64.24%

### ✨ Features
- Complete training and evaluation scripts
- Verified results with job logs
- Comprehensive documentation
- Production-ready CXRTrek Sequential model
- MIT License

### 📊 Highlights
- 4,114 test samples evaluated
- Three clinical reasoning stages
- Qwen2-VL-2B-Instruct base model
- LoRA fine-tuning implementation
- SLURM job templates included
```

5. Click "Publish release"

### 4. Pin Repository (Optional)

1. Go to your profile: [https://github.com/MuhraAlMahri](https://github.com/MuhraAlMahri)
2. Click "Customize your pins"
3. Select `Surgical_COT`
4. Save

---

## 🚫 Troubleshooting

### Issue 1: Authentication Failed

**Error**: `remote: Invalid username or password`

**Solution**:
- If using HTTPS, ensure you're using a **Personal Access Token**, not your GitHub password
- Generate a new token at: https://github.com/settings/tokens
- Use token as password when prompted

### Issue 2: Repository Already Exists

**Error**: `remote: Repository not found` or `repository already exists`

**Solution**:
```bash
# Remove the remote
git remote remove origin

# Add it again
git remote add origin https://github.com/MuhraAlMahri/Surgical_COT.git

# Try pushing again
git push -u origin main
```

### Issue 3: Large Files Rejected

**Error**: `remote: error: File is too large`

**Solution**:
```bash
# Check file sizes
find . -type f -size +100M

# Large files are already in .gitignore
# If you want to include model checkpoints, use Git LFS:
git lfs install
git lfs track "*.bin"
git lfs track "*.safetensors"
git add .gitattributes
git commit -m "Add Git LFS tracking"
git push
```

### Issue 4: Permission Denied (SSH)

**Error**: `Permission denied (publickey)`

**Solution**:
```bash
# Test SSH connection
ssh -T git@github.com

# If it fails, generate and add SSH key:
ssh-keygen -t ed25519 -C "your_email@example.com"
cat ~/.ssh/id_ed25519.pub
# Copy and add to: https://github.com/settings/keys

# Or use HTTPS instead:
git remote set-url origin https://github.com/MuhraAlMahri/Surgical_COT.git
```

### Issue 5: Need to Create Repository First

**Error**: `Repository not found`

**Solution**:
1. The repository doesn't exist on GitHub yet
2. Go to [https://github.com/new](https://github.com/new)
3. Create repository named `Surgical_COT`
4. Don't initialize with README
5. Then run push command again

---

## 📊 What Gets Uploaded

### ✅ Files Included (via .gitignore):
- All Python scripts (`.py`)
- All documentation (`.md`)
- Configuration files (`requirements.txt`, `LICENSE`, etc.)
- SLURM job scripts (`.slurm`)
- Small JSON files (<10MB)
- Evaluation results (JSON format)

### ❌ Files Excluded (via .gitignore):
- Model checkpoints (`*.bin`, `*.safetensors`) - Too large
- Large image datasets - Use external hosting
- Training logs - Too many files
- Temporary files - Not needed
- Cache files - Generated at runtime

**Note**: Model checkpoints are excluded because they're 4-5GB each. Instead:
- Document how to train models
- Provide checkpoint URLs (if hosted elsewhere)
- Users can reproduce by training

---

## 🎓 Best Practices After Upload

### 1. Add README Badges

Edit `README.md` and add at the top:

```markdown
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/MuhraAlMahri/Surgical_COT?style=social)](https://github.com/MuhraAlMahri/Surgical_COT/stargazers)
```

### 2. Create GitHub Pages (Optional)

```bash
# Create gh-pages branch
git checkout --orphan gh-pages
git rm -rf .
echo "# Surgical COT Documentation" > index.md
echo "Visit main repository: https://github.com/MuhraAlMahri/Surgical_COT" >> index.md
git add index.md
git commit -m "Initial gh-pages"
git push origin gh-pages
git checkout main
```

Enable in Settings → Pages → Source: `gh-pages` branch

### 3. Set Up Branch Protection (Optional)

Settings → Branches → Add rule for `main`:
- ✅ Require pull request reviews
- ✅ Require status checks to pass
- ✅ Include administrators

---

## 📧 Need Help?

- **GitHub Docs**: https://docs.github.com/en/get-started
- **Git Tutorial**: https://git-scm.com/docs/gittutorial
- **GitHub Support**: https://support.github.com

---

## 🎉 Success Checklist

After successful upload, you should see:

- ✅ Repository visible at: https://github.com/MuhraAlMahri/Surgical_COT
- ✅ README displays correctly on main page
- ✅ All documentation files present
- ✅ LICENSE file shows MIT
- ✅ Topics added (medical-ai, vqa, etc.)
- ✅ Repository description set
- ✅ Issues/Discussions enabled
- ✅ Green commit history

**Congratulations! Your research is now open source! 🎊**

---

**Last Updated**: October 20, 2025

