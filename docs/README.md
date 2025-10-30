# PyGeoN Documentation

This directory contains the documentation source files for PyGeoN.

## Building the Documentation Locally

### Prerequisites

Install the documentation requirements:

```bash
pip install -r requirements.txt
```

### Build HTML Documentation

From this directory, run:

```bash
make html
```

The generated HTML documentation will be in `_build/html/`. Open `_build/html/index.html` in your browser.

**Note**: The `make html` command automatically:

- Collects generated reports via `readthedocs/hooks/pre_build_reports.sh` into `docs/reports/`
- Runs `sphinx-apidoc` to generate API documentation from your source code

So your API and reports sections are always up-to-date.

### Manually Regenerate API Documentation

If you want to regenerate only the API documentation without building:

```bash
make apidoc
```

### Build PDF Documentation

```bash
make latexpdf
```

### Clean Build Files

```bash
make clean
```

This removes the `_build/` directory but keeps the generated API documentation.

## Generated Reports

Place any Markdown reports you generate into `docs/reports_src/` (or provide a generator `docs/reports_src/generate_reports.sh`).

On both ReadTheDocs and local builds, `readthedocs/hooks/pre_build_reports.sh` will copy `docs/reports_src/*.md` into `docs/reports/`, which is included in the documentation under the "Reports" section.

Locally you can run just the collection step with:

```bash
cd docs
make reports
```

## ReadTheDocs

The documentation is automatically built and hosted on ReadTheDocs when changes are pushed to the repository. ReadTheDocs automatically:

1. Installs your package
2. Runs `sphinx-apidoc` to generate API docs from source code
3. Builds the HTML documentation

**You don't need to commit generated API `.rst` files** - they are auto-generated during each build!

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `api/` - API reference documentation (auto-generated, can be git-ignored)
- `_static/` - Static files (CSS, images, etc.)
- `requirements.txt` - Python packages needed to build the docs
- `Makefile` - Build automation (includes auto-generation of API docs)

## Contributing

When adding new modules or features:

1. **Add docstrings** to all public functions and classes (Google or NumPy style)
2. **Include type hints** in function signatures
3. **Test locally**: Run `make html` to verify documentation builds correctly
4. **No manual API files needed**: The API documentation is auto-generated from docstrings

### What to Commit

✅ **DO commit**:
- Hand-written documentation (`.rst` files in docs root)
- `index.rst`, `installation.rst`, `tutorials.rst`, etc.
- Configuration files (`conf.py`, `Makefile`, `.readthedocs.yaml`)

❌ **DON'T commit** (auto-generated):
- `api/*.rst` files (except manually curated overview pages if desired)
- `_build/` directory
- `_autosummary/` directory

## Style Guide

- Use Google or NumPy style docstrings
- Include type hints in function signatures
- Provide examples in docstrings where helpful
- Keep line length to 88 characters (ruff standard)

## Continuous Integration

The documentation is automatically tested on every PR and push via GitHub Actions:

**Workflow**: `.github/workflows/docs.yml`

**What it does**:
- Builds documentation with warnings treated as errors (`-W` flag)
- Detects broken links and references (`-n` flag)
- Shows all warnings, not just the first (`--keep-going`)
- Generates a detailed report in the PR summary
- Uploads built HTML as an artifact (available for 7 days)

**If CI fails**: Check the workflow logs in the "Actions" tab for specific warnings/errors to fix. The summary will show the first 20 issues found.

**Test locally before pushing**:
```bash
# Test with the same strict settings as CI
make strict

# Or use the detailed script
./build_strict.sh
```

For more details, see [`CI_REFERENCE.md`](readthedocs/CI_REFERENCE.md).

## Automatic API Documentation

The API documentation is generated automatically using `sphinx-apidoc`:

```bash
sphinx-apidoc -f -o docs/api src/pygeon --separate --no-toc
```

Options used:
- `-f`: Overwrite existing files
- `-o docs/api`: Output to docs/api directory
- `--separate`: Create separate pages for each module
- `--no-toc`: Don't create a table of contents file (we use our own)

This ensures the documentation always matches your current source code!
