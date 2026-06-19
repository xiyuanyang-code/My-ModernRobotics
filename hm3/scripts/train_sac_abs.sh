#!/bin/bash
# Train SAC on PushT (aligned with PPO config)
# Usage: bash scripts/train_sac.sh

export CUDA_VISIBLE_DEVICES=2
set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== SAC Training ==="
$PYTHON src/train_sac.py \
  --max_steps 1000000 \
  --lr_actor 3e-4 \
  --lr_critic 3e-4 \
  --hidden_dim 1024 \
  --num_layers 4 \
  --batch_size 2048 \
  --buffer_size 1000000 \
  --tau 0.005 \
  --obs_norm 1 \
  --reward_shaping 0 \
  --action_space abs \
  --eval_freq 2000 \
  --seed 42 \
  --output_dir outputs \
  "$@"
