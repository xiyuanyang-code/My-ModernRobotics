#!/bin/bash

# iLQR Two-Link Arm Experiment Runner
# All parameters hardcoded below. Edit directly to change.

set -e

# ---- Hyperparameters ----
SEED=2025
SIM_TIME=4
EVAL_EPISODES=1
DEMO_SEED=2025
R_SCALE=0.001

# ---- Environments ----
ENVIRONMENTS=(
    "TwoLinkArm-v0"
    "TwoLinkArm-limited-torque-v0"
    "TwoLinkArm-v1"
    "TwoLinkArm-limited-torque-v1"
)

echo "=========================================="
echo "  iLQR Two-Link Arm Experiments"
echo "=========================================="
echo "  Seed:           $SEED"
echo "  Sim time:       ${SIM_TIME}s"
echo "  Eval episodes:  $EVAL_EPISODES"
echo "  Environments:   ${ENVIRONMENTS[*]}"
echo "=========================================="

for ENV_NAME in "${ENVIRONMENTS[@]}"; do
    echo ""
    echo "------------------------------------------"
    echo "Testing: $ENV_NAME"
    echo "------------------------------------------"

    python dualarm_entry.py \
        --env-name "$ENV_NAME" \
        --eval-episodes "$EVAL_EPISODES" \
        --seed "$SEED" \
        --demo-seed "$DEMO_SEED" \
        --sim-time "$SIM_TIME" \
        --r-scale $R_SCALE

    echo "Completed: $ENV_NAME"
done

echo ""
echo "=========================================="
echo "  All done! Results: $RESULT_DIR/"
echo "=========================================="
