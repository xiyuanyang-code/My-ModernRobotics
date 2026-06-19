#!/bin/bash
# Train PPO on PushT (aggressive config for strong GPU)
# Usage: bash scripts/train_ppo.sh [--override args...]

export CUDA_VISIBLE_DEVICES=0

set -e
cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== PPO Training (aggressive) ==="
$PYTHON src/train_ppo.py \
  --max_steps 1000000 \
  --lr 3e-4 \
  --lr_schedule none \
  --hidden_dim 1024 \
  --num_layers 4 \
  --num_envs 64 \
  --rollout_steps 512 \
  --batch_size 2048 \
  --epochs_per_update 4 \
  --reward_shaping 0 \
  --curriculum 0 \
  --action_space abs \
  --eval_freq 2000 \
  --seed 42 \
  --output_dir outputs \
  "$@"
