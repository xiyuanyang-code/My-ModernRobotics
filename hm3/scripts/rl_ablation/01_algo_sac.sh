#!/bin/bash
# Ablation 1: Algorithm comparison — SAC
# Default: delta + reward_shaping=on + obs_norm=on

export CUDA_VISIBLE_DEVICES=1
set -e
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== [Ablation] algo_compare: SAC ==="
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
  --output_dir outputs/ablation/01_algo/sac \
  "$@"
