#!/bin/bash
# Ablation 1: Algorithm comparison — PPO (baseline)
# Default: delta + reward_shaping=on + obs_norm=on

export CUDA_VISIBLE_DEVICES=1
set -e
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== [Ablation] algo_compare: PPO ==="
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
  --reward_shaping 1 \
  --obs_norm 1 \
  --curriculum 0 \
  --action_space delta \
  --delta_max 100.0 \
  --eval_freq 2000 \
  --seed 42 \
  --output_dir outputs/ablation/01_algo/ppo \
  "$@"
