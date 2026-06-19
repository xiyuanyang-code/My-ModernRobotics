#!/bin/bash
# Plot training results from logs
# Usage: bash scripts/plot_results.sh [--log_dir outputs]

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== Plotting Results ==="
$PYTHON src/plot_results.py "$@"
