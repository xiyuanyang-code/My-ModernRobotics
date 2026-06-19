#!/bin/bash
# Ablation 2: Action space — SAC + delta (baseline)
# Same as 01_algo_sac.sh, kept here for easy comparison within group

export CUDA_VISIBLE_DEVICES=2
set -e
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== [Ablation] action_space: SAC + delta ==="
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
  --reward_shaping 1 \
  --action_space delta \
  --delta_max 100.0 \
  --eval_freq 2000 \
  --seed 42 \
  --output_dir outputs/ablation/02_action/sac_delta \
  "$@"
