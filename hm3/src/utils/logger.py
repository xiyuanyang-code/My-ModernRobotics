"""Logging utilities — CSV, console, and TensorBoard."""

import csv
import os
from datetime import datetime


class CSVLogger:
    """Append rows to a CSV file with dynamic columns."""

    ALL_FIELDS = [
        "step", "total_steps", "epoch",
        "pg_loss", "vf_loss", "entropy", "total_loss",
        "clip_fraction", "explained_variance", "grad_norm", "lr",
        "q_loss", "q1", "q2", "policy_loss", "alpha", "alpha_loss",
        "q_grad_norm", "actor_grad_norm",
        "train_loss", "val_loss",
        "eval_reward", "eval_std", "ema_eval_reward", "ema_eval_std",
        "rollout_reward_mean", "rollout_reward_max", "rollout_reward_min", "rollout_reward_std",
        "action_mean", "action_std", "value_mean",
        "num_episodes", "ep_reward_mean", "ep_reward_max", "ep_reward_min", "ep_len_mean",
        "fps",
    ]

    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.file = open(path, "a", newline="")
        self.writer = csv.DictWriter(
            self.file, fieldnames=self.ALL_FIELDS, extrasaction="ignore"
        )
        if os.path.getsize(path) == 0:
            self.writer.writeheader()

    def log(self, row: dict):
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        self.file.close()


class TensorboardLogger:
    """Thin wrapper around torch.utils.tensorboard.SummaryWriter.

    Organizes scalars into groups: train/, rollout/, ep/, eval/, perf/.
    """

    def __init__(self, log_dir):
        from torch.utils.tensorboard import SummaryWriter
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def log_scalars(self, main_tag: str, tag_scalar_dict: dict, step: int):
        """Log multiple scalars under one group, e.g. 'train/'."""
        for tag, value in tag_scalar_dict.items():
            if value is not None:
                self.writer.add_scalar(f"{main_tag}/{tag}", value, step)

    def log_scalar(self, tag: str, value, step: int):
        self.writer.add_scalar(tag, value, step)

    def close(self):
        self.writer.close()


class ConsoleLogger:
    """Prints training info to stdout."""

    def __init__(self, prefix=""):
        self.prefix = prefix

    def log(self, step, **kwargs):
        parts = [f"Step {step:>8d}"]
        for k, v in kwargs.items():
            if isinstance(v, float):
                parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(f"[{self.prefix}] " + " | ".join(parts))
