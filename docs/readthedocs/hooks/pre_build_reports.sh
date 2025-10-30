#!/usr/bin/env bash
# Collect or generate documentation reports before Sphinx build.
# This script is safe to run on ReadTheDocs and locally.
#
# Behavior:
# - If docs/reports_src/ exists, copy all *.md into docs/reports/
# - Ensures docs/reports/ exists
# - You can extend this script to run your own generators (nbconvert, scripts, etc.)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DOCS_DIR="$REPO_ROOT/docs"
REPORTS_DIR="$DOCS_DIR/reports"
REPORTS_SRC_DIR="$DOCS_DIR/reports_src"

mkdir -p "$REPORTS_DIR"

# Optional user hook: run a generator if present
# Example: python docs/reports_src/generate_reports.py
if [[ -x "$REPORTS_SRC_DIR/generate_reports.sh" ]]; then
  echo "Running user report generator: $REPORTS_SRC_DIR/generate_reports.sh"
  (cd "$REPORTS_SRC_DIR" && ./generate_reports.sh)
fi

# Copy markdown reports from reports_src to reports
if [[ -d "$REPORTS_SRC_DIR" ]]; then
  echo "Collecting markdown reports from $REPORTS_SRC_DIR -> $REPORTS_DIR"
  # Copy only .md files, non-recursive by default
  find "$REPORTS_SRC_DIR" -maxdepth 1 -type f -name "*.md" -print0 | xargs -0 -I{} cp -f {} "$REPORTS_DIR" || true
fi

echo "Reports pre-build complete. Contents of $REPORTS_DIR:"
ls -1 "$REPORTS_DIR" || true
