#!/usr/bin/env bash
# Run all BC ablation experiments.
#
# Usage:
#   bash scripts/run_bc_ablations.sh                      # all ablations
#   bash scripts/run_bc_ablations.sh --ablation chunk_length   # just K=1 vs K=8

set -euo pipefail
cd "$(dirname "$0")/.."

python src/run_bc_ablations.py \
    --max_epochs 100 \
    --seed 42 \
    --device auto \
    "$@"
