#!/usr/bin/env bash
# ============================================================================
# Ablation 3: EMA (Exponential Moving Average) Comparison
# EMA off (decay=0) vs EMA on (decay=0.999)
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="outputs/ablations/bc_ema"
EPOCHS=${1:-200}
DEVICE=${2:-auto}

echo "============================================================"
echo "  Ablation 3: EMA (off vs decay=0.999)"
echo "  Epochs: $EPOCHS  Device: $DEVICE"
echo "============================================================"

# Without EMA
echo ""
echo ">>> Training without EMA (decay=0) ..."
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
    --ema_decay 0.0 \
    --seed 42 \
    --device "$DEVICE" \
    --output_dir "$OUTPUT_DIR/ema_off" \
    --data_dir data/lerobot/pusht \
    2>&1 | tail -3

# With EMA (decay=0.999)
echo ""
echo ">>> Training with EMA (decay=0.999) ..."
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
    --output_dir "$OUTPUT_DIR/ema_0999" \
    --data_dir data/lerobot/pusht \
    2>&1 | tail -3

echo ""
echo ">>> Collecting results..."
python scripts/il_ablation/collect_results.py "$OUTPUT_DIR" \
    --title "Ablation: EMA (off vs 0.999)"

echo ""
echo "Done! Results in $OUTPUT_DIR/"
