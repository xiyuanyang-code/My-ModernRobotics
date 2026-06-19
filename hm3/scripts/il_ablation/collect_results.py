#!/usr/bin/env python3
"""Collect BC ablation results and generate comparison plots + summary table.

Usage:
    python scripts/il_ablation/collect_results.py <ablation_dir>

Reads all train_log.csv files under <ablation_dir>, generates comparison plots,
and prints a summary table.
"""

import argparse
import csv
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))


def load_csv(path):
    """Load CSV log into dict of lists."""
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if not v or v.strip() == "":
                    continue
                if k not in data:
                    data[k] = []
                try:
                    data[k].append(float(v))
                except ValueError:
                    pass
    return data


def find_best_reward(data):
    """Find best eval_reward and ema_eval_reward from log data."""
    best_eval = max(data.get("eval_reward", [-float("inf")]))
    best_ema = max(data.get("ema_eval_reward", [-float("inf")]))
    final_train = data.get("train_loss", [float("nan")])[-1]
    final_val = data.get("val_loss", [float("nan")])[-1]
    return {
        "best_eval_reward": best_eval if best_eval > -float("inf") else None,
        "best_ema_reward": best_ema if best_ema > -float("inf") else None,
        "final_train_loss": final_train,
        "final_val_loss": final_val,
    }


def collect(ablation_dir):
    """Collect results from all runs in an ablation directory."""
    # Find all run directories (contain train_log.csv)
    log_files = glob.glob(os.path.join(ablation_dir, "**/train_log.csv"), recursive=True)

    results = {}
    for log_path in sorted(log_files):
        run_dir = os.path.dirname(log_path)
        run_name = os.path.relpath(run_dir, ablation_dir)
        # Use the immediate subdirectory name as the variant name
        variant = run_name.split(os.sep)[0]

        data = load_csv(log_path)
        if not data:
            continue

        info = find_best_reward(data)
        info["run_dir"] = run_dir
        info["log_path"] = log_path
        results[variant] = info

    return results


def print_table(results, title="Ablation Results"):
    """Print a formatted summary table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  {'Variant':<25} {'Best Eval':>10} {'Best EMA':>10} {'Train Loss':>12} {'Val Loss':>12}")
    print(f"  {'-'*69}")

    for variant, info in sorted(results.items()):
        best_eval = f"{info['best_eval_reward']:.4f}" if info["best_eval_reward"] is not None else "N/A"
        best_ema = f"{info['best_ema_reward']:.4f}" if info["best_ema_reward"] is not None else "N/A"
        train_l = f"{info['final_train_loss']:.6f}"
        val_l = f"{info['final_val_loss']:.6f}"
        print(f"  {variant:<25} {best_eval:>10} {best_ema:>10} {train_l:>12} {val_l:>12}")

    print(f"  {'-'*69}")


def generate_plot(results, out_path, title="Ablation Comparison"):
    """Generate comparison plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray"]

    for i, (variant, info) in enumerate(sorted(results.items())):
        data = load_csv(info["log_path"])
        color = colors[i % len(colors)]

        # Left: loss curves
        epochs = data.get("epoch", [])
        train_loss = data.get("train_loss", [])
        val_loss = data.get("val_loss", [])
        if epochs and train_loss:
            axes[0].plot(epochs, train_loss, color=color, alpha=0.5, linestyle="-")
        if epochs and val_loss:
            axes[0].plot(epochs, val_loss, color=color, label=variant, linewidth=2)

        # Right: eval reward
        eval_epochs = []
        eval_rewards = []
        for j, e in enumerate(data.get("epoch", [])):
            if j < len(data.get("eval_reward", [])):
                eval_epochs.append(e)
                eval_rewards.append(data["eval_reward"][j])
        if eval_epochs:
            axes[1].plot(eval_epochs, eval_rewards, color=color, label=variant, linewidth=2)

        # EMA reward
        ema_rewards = data.get("ema_eval_reward", [])
        if eval_epochs and ema_rewards:
            axes[1].plot(eval_epochs[:len(ema_rewards)], ema_rewards,
                        color=color, linestyle="--", alpha=0.7)

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].set_title("Training / Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale("log")

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Eval Reward")
    axes[1].set_title("Evaluation Reward (solid=raw, dashed=EMA)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved plot: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ablation_dir", help="Directory containing ablation run outputs")
    parser.add_argument("--title", default=None, help="Plot title")
    args = parser.parse_args()

    title = args.title or os.path.basename(args.ablation_dir)
    results = collect(args.ablation_dir)

    if not results:
        print("No results found!")
        return

    print_table(results, title)
    generate_plot(results, os.path.join(args.ablation_dir, "comparison.png"), title)


if __name__ == "__main__":
    main()
