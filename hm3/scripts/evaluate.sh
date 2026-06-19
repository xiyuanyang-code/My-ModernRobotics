#!/bin/bash
# Evaluate a trained checkpoint
# Usage: bash scripts/evaluate.sh --checkpoint outputs/ppo/best.pt --algo ppo [--record]

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== Evaluating Checkpoint ==="
$PYTHON src/evaluate.py "$@"
