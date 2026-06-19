"""Plotting utilities for training curves."""

import os
import csv
import numpy as np


def smooth(values, weight=0.9):
    """Exponential moving average smoothing."""
    smoothed = []
    last = values[0]
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def load_csv(path):
    """Load a CSV log file into a dict of lists (numeric columns only)."""
    data = {}
    with open(path, "r") as f:
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
                    pass  # skip non-numeric columns
    return data


def _align_xy(data, x_key, y_key):
    """Extract aligned (x, y) pairs where both keys have numeric values."""
    x_all = data.get(x_key, [])
    y_all = data.get(y_key, [])
    if not x_all or not y_all:
        return [], []
    # Re-read raw CSV to keep row alignment
    xs, ys = [], []
    with open(data.get("_path", ""), "r") as f:
        reader = csv.DictReader(f)
        x_vals, y_vals = [], []
        for row in reader:
            try:
                x_vals.append(float(row.get(x_key, "")))
            except (ValueError, KeyError):
                x_vals.append(None)
            try:
                y_vals.append(float(row.get(y_key, "")))
            except (ValueError, KeyError):
                y_vals.append(None)
        for x, y in zip(x_vals, y_vals):
            if x is not None and y is not None:
                xs.append(x)
                ys.append(y)
    return xs, ys


def plot_reward_curve(log_path, out_path, smooth_weight=0.9):
    """Plot evaluation reward vs training steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_csv(log_path)
    data["_path"] = log_path
    x_key = "step" if data.get("step") else "epoch"
    steps, rewards = _align_xy(data, x_key, "eval_reward")

    if not steps or not rewards:
        print(f"Warning: no eval_reward data found in {log_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, rewards, alpha=0.3, color="blue", label="Raw")
    ax.plot(steps, smooth(rewards, smooth_weight), color="blue", label="Smoothed")
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Evaluation Reward")
    ax.set_title("Training Reward Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved reward plot to {out_path}")


def plot_comparison(log_paths, labels, out_path, smooth_weight=0.9):
    """Plot multiple reward curves on the same axes."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["blue", "red", "green", "orange", "purple", "brown"]

    for i, (log_path, label) in enumerate(zip(log_paths, labels)):
        data = load_csv(log_path)
        data["_path"] = log_path
        steps, rewards = _align_xy(data, "step", "eval_reward")
        if steps and rewards:
            color = colors[i % len(colors)]
            ax.plot(steps, smooth(rewards, smooth_weight), color=color, label=label)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Evaluation Reward")
    ax.set_title("Reward Curve Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison plot to {out_path}")


def plot_loss_curve(log_path, out_path, loss_key="loss", smooth_weight=0.9):
    """Plot training loss vs steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_csv(log_path)
    steps = data.get("step", data.get("epoch", data.get("total_steps", [])))
    losses = data.get(loss_key, [])

    if not steps or not losses:
        print(f"Warning: no '{loss_key}' data found in {log_path}")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(steps, losses, alpha=0.3, color="red", label="Raw")
    ax.plot(steps, smooth(losses, smooth_weight), color="red", label="Smoothed")
    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Loss")
    ax.set_title(f"Training {loss_key} Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved loss plot to {out_path}")
