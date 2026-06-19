"""Plot results from training logs."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.plotting import plot_reward_curve, plot_comparison, plot_loss_curve


def main():
    parser = argparse.ArgumentParser(description="Plot training results")
    parser.add_argument("--log_dir", type=str, default="outputs", help="Directory containing training logs")
    parser.add_argument("--smooth", type=float, default=0.9, help="Smoothing weight for EMA")
    args = parser.parse_args()

    log_dir = os.path.join(PROJECT_ROOT, args.log_dir)
    if not os.path.exists(log_dir):
        print(f"Directory not found: {log_dir}")
        return

    # Find all CSV logs
    logs = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f == "train_log.csv":
                logs.append(os.path.join(root, f))

    if not logs:
        print(f"No train_log.csv files found in {log_dir}")
        return

    print(f"Found {len(logs)} log files:")
    for log in logs:
        print(f"  {log}")

    # Generate individual plots
    for log_path in logs:
        rel = os.path.relpath(log_path, PROJECT_ROOT)
        out_path = log_path.replace("train_log.csv", "reward_curve.png")
        plot_reward_curve(log_path, out_path, args.smooth)

    # If multiple logs, generate comparison
    if len(logs) > 1:
        labels = [os.path.relpath(os.path.dirname(l), log_dir) for l in logs]
        comparison_path = os.path.join(log_dir, "comparison.png")
        plot_comparison(logs, labels, comparison_path, args.smooth)

    print("\nDone!")


if __name__ == "__main__":
    main()
