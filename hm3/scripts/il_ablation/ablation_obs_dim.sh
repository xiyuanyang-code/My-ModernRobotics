#!/usr/bin/env bash
# ============================================================================
# Ablation 2: Observation Dimension Comparison
# 2-dim (agent_xy only) vs 5-dim (full state from video reconstruction)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="outputs/ablations/bc_obs_dim"
EPOCHS=${1:-200}
DEVICE=${2:-auto}

echo "============================================================"
echo "  Ablation 2: Observation Dimension (2d vs 5d)"
echo "  Epochs: $EPOCHS  Device: $DEVICE"
echo "============================================================"

# 2-dim: agent_xy only
echo ""
echo ">>> Training 2-dim (agent_xy only) ..."
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
    --use_full_state 0 \
    --obs_norm 1 \
    --action_norm 1 \
    --ema_decay 0.999 \
    --seed 42 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR/2d_agent_only" \
    --data_dir data/lerobot/pusht \
    2>&1 | tail -3

# 5-dim: full state (agent + block from video)
echo ""
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
    --data_dir data/lerobot/pusht \
    2>&1 | tail -3

echo ""
echo ">>> Collecting results..."
python scripts/il_ablation/collect_results.py "$OUTPUT_DIR" \
    --title "Ablation: Obs Dimension (2d vs 5d)"

echo ""
echo "Done! Results in $OUTPUT_DIR/"
