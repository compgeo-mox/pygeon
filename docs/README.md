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