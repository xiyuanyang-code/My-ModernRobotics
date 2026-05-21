#!/bin/bash

# iLQR Ablation: 10 R_SCALE values × 1 environment, with timing
# Records runtime per r_scale to demonstrate speed optimization.

set -e

# ---- Hyperparameters ----
SEED=2025
SIM_TIME=4
EVAL_EPISODES=1
DEMO_SEED=2025
ENV_NAMES=(
    "TwoLinkArm-v0"
    "TwoLinkArm-limited-torque-v0"
)

# ---- R_SCALE values (10 points, log-spaced) ----
R_SCALES=(0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

echo "=========================================="
echo "  iLQR R_SCALE Ablation with Timing"
echo "=========================================="
echo "  Environments: ${ENV_NAMES[*]}"
echo "  Sim time:     ${SIM_TIME}s"
echo "  R_SCALES:     ${R_SCALES[*]}"
echo "=========================================="

mkdir -p results
TIMING_FILE="results/ablation_timing.csv"
echo "env_name,r_scale,elapsed_sec" > "$TIMING_FILE"

for ENV_NAME in "${ENV_NAMES[@]}"; do
    echo ""
    echo "=========================================="
    echo "  Environment: $ENV_NAME"
    echo "=========================================="

    for R_SCALE in "${R_SCALES[@]}"; do
        echo ""
        echo "------------------------------------------"
        echo "r_scale = $R_SCALE"
        echo "------------------------------------------"

        START_TIME=$(date +%s.%N)

        python dualarm_entry.py \
            --env-name "$ENV_NAME" \
            --eval-episodes "$EVAL_EPISODES" \
            --seed "$SEED" \
            --demo-seed "$DEMO_SEED" \
            --sim-time "$SIM_TIME" \
            --r-scale "$R_SCALE"

        END_TIME=$(date +%s.%N)
        ELAPSED=$(python3 -c "print(round($END_TIME - $START_TIME, 2))")

        echo "$ENV_NAME,$R_SCALE,$ELAPSED" >> "$TIMING_FILE"
        echo "r_scale=$R_SCALE | elapsed=${ELAPSED}s"
    done
done

echo ""
echo "=========================================="
echo "  Ablation complete! Timing saved to $TIMING_FILE"
echo "=========================================="
cat "$TIMING_FILE"
