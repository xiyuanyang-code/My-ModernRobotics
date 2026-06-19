#!/bin/bash
# Run all RL ablation experiments
# Usage: bash scripts/run_ablations.sh [--ablation algo_compare] [--max_steps 100000]

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== Running RL Ablations ==="
$PYTHON src/run_ablations.py "$@"
