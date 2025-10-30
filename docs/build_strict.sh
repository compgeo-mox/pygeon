#!/bin/bash
# Strict documentation build - matches GitHub Actions CI checks
# Use this to test locally before pushing

set -e

cd "$(dirname "$0")"

echo "=========================================="
echo "Strict Documentation Build"
echo "=========================================="
echo ""

# Clean previous build
echo "1. Cleaning previous build..."
make clean

# Generate API docs
echo ""
echo "2. Generating API documentation..."
make apidoc

# Build with strict settings
echo ""
echo "3. Building documentation with strict checks..."
echo "   (warnings = errors, broken links detected)"
echo ""

# Same flags as GitHub Actions:
# -W: Turn warnings into errors
# -n: Nitpicky mode (warn about all missing references)
# --keep-going: Don't stop at first error, show all issues
sphinx-build -W -n --keep-going -b html . _build/html 2>&1 | tee build.log

echo ""
echo "=========================================="
echo "Build Summary"
echo "=========================================="

warnings=$(grep -c "WARNING" build.log || echo "0")
errors=$(grep -c "ERROR" build.log || echo "0")

echo "Warnings: $warnings"
echo "Errors: $errors"

if [ "$warnings" -gt 0 ] || [ "$errors" -gt 0 ]; then
    echo ""
    echo "⚠️  Issues found (first 20):"
    echo "-------------------------------------------"
    grep -E "(WARNING|ERROR)" build.log | head -20
    echo "-------------------------------------------"
    echo ""
    echo "❌ Build would FAIL on GitHub Actions"
    exit 1
else
    echo ""
    echo "✅ All checks passed!"
    echo "Documentation ready at: _build/html/index.html"
fi
