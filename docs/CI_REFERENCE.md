# Documentation CI/CD Reference

## GitHub Actions Workflow

**File**: `.github/workflows/docs.yml`

### What it does

The workflow runs on:
- Every push to `main` or `develop` branches
- Every pull request targeting `main` or `develop`

### Build Process

1. **Checkout code** - Gets your repository
2. **Set up Python 3.12** - Matches ReadTheDocs environment
3. **Install dependencies** - Installs Sphinx and your package
4. **Generate API docs** - Runs `sphinx-apidoc` to create module documentation
5. **Build with strict checks** - Uses these flags:
   - `-W`: Warnings become errors (build fails)
   - `-n`: Nitpicky mode (detects broken links)
   - `--keep-going`: Shows all issues, not just first
6. **Generate report** - Creates a summary in the PR
7. **Upload artifacts** - HTML available for download (7 days)

### Interpreting Results

**✅ Green check** - Documentation is perfect, no warnings
**❌ Red X** - Build failed, check the logs

Common issues:
- **Broken references** - `:class:`, `:func:` pointing to non-existent items
- **Undefined labels** - `:ref:` pointing to missing labels
- **Import errors** - Module can't be imported
- **RST syntax errors** - Malformed reStructuredText

### Local Testing (Before Push)

Test exactly what CI will test:

```bash
# Option 1: Use the strict build script
cd docs
./build_strict.sh

# Option 2: Use Makefile target
cd docs
make strict

# Option 3: Manual command
cd docs
sphinx-build -W -n --keep-going -b html . _build/html
```

All three methods use the same flags as CI.

## Quick Fixes for Common Warnings

### Missing References

```python
# ❌ Bad - module doesn't exist
"""See :class:`pygeon.nonexistent.Foo`"""

# ✅ Good - correct path
"""See :class:`pygeon.geometry.Domain`"""
```

### Broken Links

```rst
.. ❌ Bad - undefined label
See :ref:`installation` for details

.. ✅ Good - define the label
.. _installation:

Installation
============
```

### Docstring Formatting

```python
# ❌ Bad - incorrect indentation
"""
Args:
  param: Description
"""

# ✅ Good - proper formatting
"""
Args:
    param: Description
"""
```

## PR Workflow

1. **Make changes** - Update code and docstrings
2. **Test locally** - Run `make strict` to catch issues early
3. **Create PR** - Push your branch
4. **Check CI** - Wait for GitHub Actions to run
5. **Fix issues** - If red X, check logs and fix
6. **Merge** - Once green, you're good to go!

## Customizing the Workflow

### Make it less strict

Remove the `-W` flag to allow warnings:

```yaml
# In .github/workflows/docs.yml
sphinx-build -n --keep-going -b html . _build/html
```

### Check only specific warnings

Add `-W` with `--keep-going` to see all warnings but only fail on errors:

```yaml
sphinx-build --keep-going -b html . _build/html
```

### Add coverage check

Ensure all modules are documented:

```bash
# Add to workflow
sphinx-build -b coverage . _build/coverage
cat _build/coverage/python.txt
```

## Debugging Failed Builds

1. **Read the summary** - GitHub adds a summary to each PR
2. **Check the full log** - Click "Details" next to the failed check
3. **Download artifacts** - Download the HTML to see what was built
4. **Test locally** - Run `make strict` to reproduce

## Resources

- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [ReadTheDocs Build Process](https://docs.readthedocs.io/en/stable/builds.html)
- [GitHub Actions Docs](https://docs.github.com/en/actions)
