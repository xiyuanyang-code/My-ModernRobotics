#!/bin/bash
# Ablation 3: Reward shaping — PPO + shaping OFF
# Compare against 01_algo_ppo.sh (shaping ON)

export CUDA_VISIBLE_DEVICES=2
set -e
cd "$(dirname "$0")/../.."

PYTHON=${PYTHON:-".venv/bin/python"}

echo "=== [Ablation] reward_shaping: PPO + OFF ==="
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
  --obs_norm 1 \
  --curriculum 0 \
  --action_space delta \
  --delta_max 100.0 \
  --eval_freq 2000 \
  --seed 42 \
  --output_dir outputs/ablation/03_reward/ppo_shaping_off \
  "$@"
