# 🚀 GitHub Upload Guide

## Quick Steps to Upload Your Project to GitHub

### Step 1: Create a New Repository on GitHub

1. Go to https://github.com
2. Click the **"+"** icon (top right) → **"New repository"**
3. Fill in the details:
   - **Repository name**: `tb-detection` (or your preferred name)
   - **Description**: "TB Detection using Vision Transformers and Lung Segmentation"
   - **Visibility**: Choose Public or Private
   - **DON'T initialize with README** (we already have one)
4. Click **"Create repository"**

### Step 2: Initialize Git in Your Local Project

Open PowerShell in your project directory (`C:\Users\calso\Downloads\buc`):

```powershell
# Navigate to project directory
cd C:\Users\calso\Downloads\buc

# Initialize git repository
git init

# Add all files (respects .gitignore)
git add .

# Check what will be committed
git status
```

**You should see:**
- ✅ going.ipynb
- ✅ README.md
- ✅ SETUP.md
- ✅ .gitignore
- ✅ datasets/tbx11k-simplified/data.csv
- ✅ .gitkeep files

**You should NOT see (excluded by .gitignore):**
- ❌ results/visualizations/
- ❌ results/reports/
- ❌ results/mask_cache/
- ❌ models/*.keras
- ❌ venv/
- ❌ __pycache__/

### Step 3: Make Your First Commit

```powershell
# Create first commit
git commit -m "Initial commit: TB detection pipeline with ViT and lung segmentation"
```

### Step 4: Connect to GitHub and Push

Replace `yourusername` with your actual GitHub username:

```powershell
# Add remote repository
git remote add origin https://github.com/yourusername/tb-detection.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

**If prompted for credentials:**
- Use your GitHub username
- For password, use a **Personal Access Token** (not your actual password)
- Get token from: GitHub → Settings → Developer settings → Personal access tokens

### Step 5: Verify Upload

1. Go to your GitHub repository URL
2. You should see:
   - ✅ README.md displayed on homepage
   - ✅ going.ipynb in file list
   - ✅ Directory structure preserved
   - ✅ .gitignore working (no results/ or venv/)

## 📦 What Gets Uploaded vs Excluded

### ✅ UPLOADED (Included in Git)
```
✅ going.ipynb                 # Main pipeline notebook
✅ README.md                   # Project documentation
✅ SETUP.md                    # Setup instructions
✅ GITHUB_UPLOAD.md            # This guide
✅ .gitignore                  # Git configuration
✅ datasets/tbx11k-simplified/data.csv   # Dataset metadata
✅ datasets/tbx11k-simplified/.gitkeep   # Preserve directory
✅ models/.gitkeep             # Preserve directory
✅ results/.gitkeep            # Preserve directory
```

### ❌ EXCLUDED (Not uploaded - too large or generated)
```
❌ results/                    # All generated outputs
   ├── visualizations/         # ~10-50 MB (users generate)
   ├── reports/                # ~1-5 MB (users generate)
   ├── models/                 # ~500 MB (users generate)
   └── mask_cache/             # ~500 MB-2 GB (users generate)

❌ models/best_model (1).keras # ~100-500 MB (users download)
❌ datasets/*/images/          # ~2-5 GB (users download)
❌ venv/                       # Virtual environment (users create)
❌ __pycache__/                # Python cache (auto-generated)
❌ .ipynb_checkpoints/         # Jupyter checkpoints
```

## 🔄 Future Updates

When you make changes, use this workflow:

```powershell
# Check what changed
git status

# Add changes
git add .

# Commit with descriptive message
git commit -m "Add feature X" 
# or
git commit -m "Fix bug in phase Y"
# or  
git commit -m "Update documentation"

# Push to GitHub
git push
```

## 🌟 Make Your Repository Stand Out

### Add Repository Details on GitHub:

1. Go to your repo → **Settings**
2. Add:
   - **Description**: "TB Detection using Vision Transformers and Lung Segmentation"
   - **Topics/Tags**: `deep-learning`, `medical-imaging`, `tuberculosis`, `vision-transformer`, `pytorch`, `tensorflow`, `computer-vision`, `healthcare-ai`
   - **Website**: (if you have a project page)

### Add a LICENSE (Optional but Recommended)

```powershell
# Create LICENSE file (MIT License example)
git add LICENSE
git commit -m "Add MIT license"
git push
```

Common licenses:
- **MIT**: Very permissive, allows commercial use
- **GPL-3.0**: Open source, derivative works must be open source
- **Apache-2.0**: Permissive, includes patent grant

### Pin Important Files

On GitHub, you can "pin" your repository to your profile to showcase it!

## 📊 Repository Size Estimate

**Your Git repo will be:** ~2-10 MB (very small!)

**Breakdown:**
- going.ipynb: ~500 KB - 2 MB
- data.csv: ~100-500 KB
- Documentation: ~50 KB
- .gitkeep files: ~1 KB total

**Users download separately:** ~3-7 GB (dataset + models + generated outputs)

## 🔐 Important: Credentials Security

**NEVER commit:**
- ❌ API keys
- ❌ Passwords
- ❌ AWS credentials
- ❌ Personal data
- ❌ .env files with secrets

**Already included in .gitignore:**
```
.env
.env.local
secrets.yml
credentials.json
```

## 🐛 Troubleshooting

### "Permission denied (publickey)"
Use HTTPS instead of SSH:
```powershell
git remote set-url origin https://github.com/yourusername/tb-detection.git
```

### "Large files detected"
If Git complains about large files:
```powershell
# Check what's trying to be committed
git status

# If something shouldn't be there, add to .gitignore
echo "large-file.keras" >> .gitignore
git rm --cached large-file.keras
git commit -m "Remove large file"
```

### "Repository already exists"
```powershell
# Remove old origin and add new one
git remote remove origin
git remote add origin https://github.com/yourusername/new-repo-name.git
git push -u origin main
```

## ✅ Checklist Before Pushing

- [ ] Created .gitignore file
- [ ] Verified large files are excluded (`git status`)
- [ ] Committed with meaningful message
- [ ] No credentials/secrets in code
- [ ] README.md is complete
- [ ] Created GitHub repository
- [ ] Pushed successfully
- [ ] Verified files appear on GitHub
- [ ] Added repository description and topics

## 📧 Next Steps After Upload

1. **Add a badge** to README (optional):
   ```markdown
   [![GitHub Stars](https://img.shields.io/github/stars/yourusername/tb-detection)](https://github.com/yourusername/tb-detection)
   ```

2. **Share your work**:
   - LinkedIn
   - Twitter
   - Research communities
   - Academic conferences

3. **Enable GitHub Pages** (optional):
   - Create a project website from your README

4. **Set up Releases**:
   - Tag versions: `v1.0.0`, `v1.1.0`, etc.
   - Create releases for major updates

---

**🎉 Congratulations! Your project is now on GitHub!** 🚀

**Repository URL:** `https://github.com/yourusername/tb-detection`

Share it with the world! ⭐
