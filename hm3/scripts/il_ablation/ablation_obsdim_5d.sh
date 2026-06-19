#!/usr/bin/env bash
# Ablation 2: Observation Dimension — 5-dim (full state)
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="outputs/ablations/bc_obs_dim"
EPOCHS=${1:-200}
DEVICE=${2:-auto}
export CUDA_VISIBLE_DEVICES=4

echo ">>> Training 5-dim (full state) ..."
python src/train_bc.py \
    --chunk_length 8 \
    --hidden_dim 256 \
    --num_layers 2 \
    --activation tanh \
    --max_epochs "$EPOCHS" \
    --eval_freq 10 \
    --eval_episodes 10 \
    --batch_size 256 \
    --lr 1e-3 \
    --delta_max 100 \
    --use_full_state 1 \
    --obs_norm 1 \
    --action_norm 1 \
    --ema_decay 0.999 \
    --seed 42 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR/5d_full_state" \
    --data_dir data/lerobot/pusht
