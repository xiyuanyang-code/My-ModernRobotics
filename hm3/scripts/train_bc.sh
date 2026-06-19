#!/usr/bin/env bash
# Train MSE Behavior Cloning on PushT with action chunking.
#
# Usage:
#   bash scripts/train_bc.sh                # default K=8
#   bash scripts/train_bc.sh --chunk_length 1   # K=1
#   bash scripts/train_bc.sh --chunk_length 4   # K=4
#
# All extra args are forwarded to src/train_bc.py.

set -euo pipefail
cd "$(dirname "$0")/.."

python src/train_bc.py \
    --chunk_length 8 \
    --hidden_dim 256 \
    --num_layers 3 \
    --lr 1e-3 \
    --batch_size 256 \
    --max_epochs 200 \
    --eval_freq 10 \
    --eval_episodes 10 \
    --ema_decay 0.999 \
    --grad_clip 1.0 \
    --obs_norm 1 \
    --action_norm 1 \
    --seed 42 \
    --device auto \
    "$@"
