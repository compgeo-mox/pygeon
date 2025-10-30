#!/bin/bash
# Generate API documentation for PyGeoN
# This script is run automatically by ReadTheDocs and can be run locally

set -e

echo "==================================="
echo "Auto-generating API documentation"
echo "==================================="

cd "$(dirname "$0")"

# Remove old generated files (optional)
# rm -f api/pygeon*.rst

# Generate API documentation
sphinx-apidoc -f -o api ../src/pygeon --separate --no-toc

echo ""
echo "✓ API documentation generated successfully!"
echo "  Location: docs/api/"
echo ""
echo "To build HTML docs, run: make html"
