#!/bin/bash
# Experiment: Single-camera calibration with different configurations
# Goal: Find the best calibration setup to minimize intrinsic error

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

python3 scripts/experiment_single_camera.py
