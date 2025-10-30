# 🚀 Quick Start: Documentation CI

## First Time Setup

Run this once to test locally:

```bash
cd /home/elle/Dropbox/Work/Codes/pygeon/docs
make strict
```

If you see errors, fix them before committing!

## Daily Workflow

### ✅ Recommended Flow

```bash
# 1. Make changes to your code
vim src/pygeon/geometry/domain.py

# 2. Test documentation locally
cd docs && make strict

# 3. Fix any warnings
# (edit files as needed)

# 4. Commit and push
git add .
git commit -m "Add new geometry features"
git push

# 5. Check GitHub Actions
# Go to: https://github.com/compgeo-mox/pygeon/actions
# Your build will run automatically!
```

### 🔧 If CI Fails

1. **Open the PR/commit on GitHub**
2. **Look at the "Documentation" check** - Click "Details"
3. **Read the summary** - Shows first 20 warnings/errors
4. **Fix locally**:
   ```bash
   cd docs
   make strict  # Reproduce the error
   # Fix the issue
   make strict  # Verify it's fixed
   ```
5. **Push the fix**
6. **CI will re-run automatically**

## Testing Commands

```bash
# Regular build (allows warnings)
make html

# Strict build (like CI, fails on warnings)
make strict

# Detailed strict build with report
./build_strict.sh

# Just regenerate API docs
make apidoc

# Clean everything
make clean
```

## What Gets Checked

- ✅ All Python modules can be imported
- ✅ No syntax errors in docstrings
- ✅ All `:class:`, `:func:`, `:ref:` links work
- ✅ No undefined labels
- ✅ No broken cross-references
- ✅ Proper RST formatting

## Pro Tips

### Tip 1: Test Before Push
```bash
# Always run this before pushing
make strict
```
If it passes, CI will pass!

### Tip 2: Check Specific Files
```bash
# Only check one module's docs
sphinx-build -W -n -b html . _build/html 2>&1 | grep "geometry"
```

### Tip 3: Download Built Docs from CI
1. Go to Actions tab
2. Click on your workflow run
3. Scroll down to "Artifacts"
4. Download "documentation" to see the built HTML

### Tip 4: Watch for Common Issues
```python
# ❌ Bad - undefined reference
"""See :class:`Foo`"""

# ✅ Good - full path
"""See :class:`pygeon.geometry.Foo`"""
```

## Files You Created

- `.github/workflows/docs.yml` - GitHub Actions workflow
- `docs/build_strict.sh` - Strict local testing script  
- `docs/CI_REFERENCE.md` - Detailed CI documentation
- `docs/QUALITY_ASSURANCE.md` - QA overview
- Updated `docs/Makefile` - Added `make strict` target
- Updated `docs/README.md` - Added CI section

## Need Help?

1. **Check the docs**: `docs/CI_REFERENCE.md`
2. **Run locally**: `make strict` to reproduce
3. **Check logs**: Full output in GitHub Actions
4. **Review examples**: See existing docstrings in the codebase

---

**Remember**: Green builds = happy maintainers = merged PRs! 🎉
