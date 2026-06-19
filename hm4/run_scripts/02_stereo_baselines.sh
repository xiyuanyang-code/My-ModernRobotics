#!/bin/bash
# Experiment: Stereo calibration with different configurations
# Goal: Improve stereo intrinsic accuracy and baseline estimation

set -e
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

python3 scripts/experiment_stereo.py
