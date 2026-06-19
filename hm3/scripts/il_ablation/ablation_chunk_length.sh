#!/usr/bin/env bash
# ============================================================================
# Ablation 1: Chunk Length Comparison
# K=1 vs K=4 vs K=8 vs K=16
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="outputs/ablations/bc_chunk_length"
EPOCHS=${1:-200}
DEVICE=${2:-auto}
export CUDA_VISIBLE_DEVICES=3

echo "============================================================"
echo "  Ablation 1: Chunk Length (K=1, K=4, K=8, K=16)"
echo "  Epochs: $EPOCHS  Device: $DEVICE"
echo "============================================================"

for K in 1 4 8 16; do
    echo ""
    echo ">>> Training K=$K ..."
    python src/train_bc.py \
        --chunk_length "$K" \
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
        --output_dir "$OUTPUT_DIR/K${K}" \
        --data_dir data/lerobot/pusht \
        2>&1 | tail -3
done

echo ""
echo ">>> Collecting results..."
python scripts/il_ablation/collect_results.py "$OUTPUT_DIR" \
    --title "Ablation: Chunk Length (K=1 vs K=4 vs K=8 vs K=16)"

echo ""
echo "Done! Results in $OUTPUT_DIR/"
