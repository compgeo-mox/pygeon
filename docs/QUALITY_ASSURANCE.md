# Documentation Quality Assurance Setup

## ✅ What's Configured

Your PyGeoN repository now has comprehensive documentation quality checks:

### 1. GitHub Actions CI/CD (`.github/workflows/docs.yml`)

**Triggers**: Every push and PR to `main`/`develop`

**Checks**:
- ✅ Documentation builds successfully
- ✅ Zero warnings (warnings treated as errors)
- ✅ No broken references or links
- ✅ All modules can be imported

**Output**:
- Detailed summary report in PR
- Built HTML artifacts (downloadable for 7 days)
- Clear pass/fail status

### 2. Local Testing Tools

#### Quick Test
```bash
make strict
```
Same checks as CI, fast feedback

#### Detailed Test
```bash
./build_strict.sh
```
Detailed report with warning counts and log file

#### Regular Build
```bash
make html
```
Normal build (allows warnings for development)

### 3. Documentation

- **README.md** - Quick start guide
- **CI_REFERENCE.md** - Complete CI documentation
- **AUTO_GENERATION.md** - API docs generation details

## 🔄 Your Workflow

### Development Cycle

1. **Write code** with good docstrings
2. **Test locally**: `make strict` (catches issues early)
3. **Create PR** 
4. **CI runs automatically** - checks documentation quality
5. **Fix any issues** if CI fails (see PR summary)
6. **Merge** when green ✅

### Before Committing

```bash
cd docs
make strict
```

If it passes locally, it will pass CI!

## 🐛 Common Issues and Fixes

### "WARNING: undefined label"

**Problem**: Reference to non-existent label
```rst
See :ref:`nonexistent-label`
```

**Fix**: Define the label
```rst
.. _installation-guide:

Installation Guide
==================
```

### "WARNING: py:class reference target not found"

**Problem**: Link to class/function doesn't exist
```python
"""See :class:`pygeon.wrong.Path`"""
```

**Fix**: Use correct import path
```python
"""See :class:`pygeon.geometry.Domain`"""
```

### "WARNING: Unexpected indentation"

**Problem**: Incorrect docstring formatting

**Fix**: Follow proper RST syntax (see CI_REFERENCE.md)

## 📊 Monitoring

### Check Documentation Health

1. **GitHub Actions tab** - See all recent builds
2. **PR checks** - Each PR shows documentation status
3. **ReadTheDocs** - Production builds after merge

### Key Metrics

- ✅ **Green builds** - Documentation is consistent
- ⏱️ **Build time** - Should be < 5 minutes
- 📦 **Artifact size** - Built HTML for review

## 🎯 Benefits

### For Contributors
- **Catch errors early** - Before review, not after
- **Clear feedback** - Exactly what's wrong and where
- **Consistent quality** - Same checks for everyone

### For Maintainers
- **Automated reviews** - CI checks documentation quality
- **Always up-to-date** - Auto-generated from code
- **Professional docs** - Zero warnings, clean builds

### For Users
- **Reliable documentation** - Always reflects code
- **No broken links** - All references verified
- **Complete coverage** - All modules documented

## 🚀 Next Steps

1. **Test the setup**: Run `make strict` locally
2. **Create a test PR**: Trigger the CI workflow
3. **Review the output**: Check the PR summary
4. **Fix any warnings**: Clean up existing issues
5. **Maintain quality**: Keep builds green!

## 📚 Additional Resources

- [`README.md`](README.md) - Building documentation
- [`CI_REFERENCE.md`](CI_REFERENCE.md) - Detailed CI docs
- [`AUTO_GENERATION.md`](AUTO_GENERATION.md) - Auto-generation setup
- [Sphinx Documentation](https://www.sphinx-doc.org/)
- [ReadTheDocs Docs](https://docs.readthedocs.io/)

---

**Questions?** Check the reference docs or open an issue!
