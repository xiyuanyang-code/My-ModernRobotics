#!/bin/bash
# Quick smoke test: train PPO and SAC for a small number of steps
# Usage: bash scripts/quick_test.sh

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== Quick Smoke Test ==="
echo ""

echo "[1/2] Testing PPO (5k steps)..."
$PYTHON src/train_ppo.py --max_steps 5000 --eval_freq 2000 --output_dir outputs --seed 42
echo ""

echo "[2/2] Testing SAC (5k steps)..."
$PYTHON src/train_sac.py --max_steps 5000 --eval_freq 2000 --output_dir outputs --seed 42
echo ""

echo "=== Smoke test complete! ==="
echo "Check outputs/quick_test/ for logs and plots."
