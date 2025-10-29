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

### Build PDF Documentation

```bash
make latexpdf
```

### Clean Build Files

```bash
make clean
```

## ReadTheDocs

The documentation is automatically built and hosted on ReadTheDocs when changes are pushed to the repository.

## Documentation Structure

- `conf.py` - Sphinx configuration
- `index.rst` - Main documentation page
- `api/` - API reference documentation
- `_static/` - Static files (CSS, images, etc.)
- `requirements.txt` - Python packages needed to build the docs

## Contributing

When adding new modules or features:

1. Add docstrings to all public functions and classes
2. Update or create relevant `.rst` files in the `api/` directory
3. Add entries to the table of contents in `index.rst` if needed
4. Test the documentation locally before committing

## Style Guide

- Use Google or NumPy style docstrings
- Include type hints in function signatures
- Provide examples in docstrings where helpful
- Keep line length to 88 characters (ruff standard)
