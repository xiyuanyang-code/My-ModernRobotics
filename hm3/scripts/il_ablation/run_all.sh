#!/usr/bin/env bash
# ============================================================================
# Run all 3 BC ablation experiments.
#
# Usage:
#   bash scripts/il_ablation/run_all.sh              # default 100 epochs, auto device
#   bash scripts/il_ablation/run_all.sh 200 cuda:0    # 200 epochs, GPU
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")/../.."

EPOCHS=${1:-100}
DEVICE=${2:-auto}

echo "############################################################"
echo "  Running all BC ablations"
echo "  Epochs: $EPOCHS  Device: $DEVICE"
echo "############################################################"

echo ""
echo "=== Ablation 1/3: Chunk Length ==="
bash scripts/il_ablation/ablation_chunk_length.sh "$EPOCHS" "$DEVICE"

echo ""
echo "=== Ablation 2/3: Observation Dimension ==="
bash scripts/il_ablation/ablation_obs_dim.sh "$EPOCHS" "$DEVICE"

echo ""
echo "=== Ablation 3/3: EMA ==="
bash scripts/il_ablation/ablation_ema.sh "$EPOCHS" "$DEVICE"

echo ""
echo "############################################################"
echo "  All ablations complete!"
echo "  Results in: outputs/ablations/bc_*/"
echo "############################################################"
