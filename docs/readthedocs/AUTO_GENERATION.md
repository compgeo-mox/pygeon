# Automatic Documentation Updates - Options

## Current Setup ✓

Your documentation is now configured to **automatically update** when you add new files! Here's how:

### 1. ReadTheDocs Auto-Generation

**File**: `.readthedocs.yaml`

```yaml
pre_build:
  - sphinx-apidoc -f -o docs/api src/pygeon --separate --no-toc
```

**What it does**:
- Every time you push to GitHub, ReadTheDocs automatically:
  1. Detects new Python files in `src/pygeon/`
  2. Generates `.rst` documentation files for them
  3. Builds the HTML documentation
  
**Result**: Your docs are ALWAYS up-to-date with your code! 🎉

### 2. Local Auto-Generation

**File**: `docs/Makefile`

The Makefile now runs `sphinx-apidoc` before every build:

```bash
make html    # Auto-generates API docs, then builds HTML
make apidoc  # Only regenerate API docs without building
```

### 3. Manual Script (Optional)

**File**: `docs/generate_api_docs.sh`

Run manually when you want:
```bash
cd docs
./generate_api_docs.sh
```

## How It Works

### When you add a new file:

**Example**: You create `src/pygeon/discretizations/fem/new_method.py`

1. **Locally**: Run `make html`
   - `sphinx-apidoc` detects the new file
   - Creates `docs/api/pygeon.discretizations.fem.new_method.rst`
   - Builds documentation with your new module

2. **On ReadTheDocs**: Push to GitHub
   - ReadTheDocs runs `sphinx-apidoc` automatically
   - Generates the `.rst` file
   - Publishes updated docs

### You don't need to:
- ❌ Manually create `.rst` files for new modules
- ❌ Update `index.rst` or module lists
- ❌ Commit generated API documentation files

### You only need to:
- ✅ Write good docstrings in your Python code
- ✅ Push your code to GitHub
- ✅ Documentation updates automatically!

## Git Workflow

### Option A: Commit Generated Files (Current)
**Pros**: Easy to review what changed
**Cons**: Git history includes auto-generated files

```bash
git add docs/api/*.rst
git commit -m "Add new module"
git push
```

### Option B: Ignore Generated Files (Recommended)
**Pros**: Cleaner git history, no merge conflicts in generated files
**Cons**: Can't preview docs without building locally

To switch to this approach:

1. Add to `docs/.gitignore`:
   ```
   api/pygeon*.rst
   ```

2. Remove tracked files:
   ```bash
   git rm --cached docs/api/pygeon*.rst
   git commit -m "Stop tracking auto-generated API docs"
   ```

3. Done! Files are generated on-demand during builds

## GitHub Actions (Optional Advanced Setup)

You could also add a GitHub Action to auto-generate docs on every PR:

**File**: `.github/workflows/docs.yml`

```yaml
name: Documentation
on: [push, pull_request]
jobs:
  build-docs:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.12'
      - run: pip install -r docs/requirements.txt
      - run: cd docs && make html
      - uses: actions/upload-artifact@v3
        with:
          name: documentation
          path: docs/_build/html
```

This would:
- Build docs on every push
- Catch documentation errors in CI
- Generate preview artifacts

## Recommendation

**Best practice**: Use Option B (ignore generated files) with the current ReadTheDocs setup:

1. ✅ Auto-generation is already configured
2. ✅ Add `api/pygeon*.rst` to `.gitignore`
3. ✅ Only commit hand-written documentation
4. ✅ Let ReadTheDocs handle the API docs

**Result**: Clean git history + always up-to-date documentation!
