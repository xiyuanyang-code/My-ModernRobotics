"""Run behavior cloning ablation experiments.

Compares:
  1. Chunk length: K=1 vs K=8 (required)
  2. Obs normalization on/off
  3. Action normalization on/off
  4. EMA vs raw policy

Generates comparison plots and a summary table.
"""

import argparse
import os
import sys
import json
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


ABLATIONS = {
    # name: {description, overrides}
    "chunk_length": {
        "description": "Compare action chunk lengths K=1 vs K=8",
        "runs": {
            "K1": {"chunk_length": 1},
            "K8": {"chunk_length": 8},
        },
    },
    "obs_norm": {
        "description": "Compare with/without observation normalization",
        "runs": {
            "obs_norm_on": {"obs_norm": 1},
            "obs_norm_off": {"obs_norm": 0},
        },
    },
    "action_norm": {
        "description": "Compare with/without action normalization",
        "runs": {
            "action_norm_on": {"action_norm": 1},
            "action_norm_off": {"action_norm": 0},
        },
    },
    "ema": {
        "description": "Compare EMA vs raw policy (K=8)",
        "runs": {
            "ema_on": {"ema_decay": 0.999, "chunk_length": 8},
            "ema_off": {"ema_decay": 0.0, "chunk_length": 8},
        },
    },
    "network_size": {
        "description": "Compare different MLP sizes (K=8)",
        "runs": {
            "small": {"hidden_dim": 64, "num_layers": 2, "chunk_length": 8},
            "medium": {"hidden_dim": 256, "num_layers": 3, "chunk_length": 8},
            "large": {"hidden_dim": 512, "num_layers": 4, "chunk_length": 8},
        },
    },
}


def run_single(args, overrides, run_name, output_dir):
    """Launch a single BC training run as a subprocess."""
    cmd = [
        sys.executable, os.path.join(PROJECT_ROOT, "src", "train_bc.py"),
        "--output_dir", output_dir,
        "--seed", str(args.seed),
        "--max_epochs", str(args.max_epochs),
        "--device", args.device,
        "--data_dir", args.data_dir,
    ]

    for key, val in overrides.items():
        cmd.extend([f"--{key}", str(val)])

    log_path = os.path.join(output_dir, f"{run_name}_stdout.log")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running: {run_name}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Log: {log_path}")

    with open(log_path, "w") as log_f:
        result = subprocess.run(cmd, stdout=log_f, stderr=subprocess.STDOUT, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"  ⚠ Run {run_name} failed (exit code {result.returncode})")
    else:
        print(f"  ✓ Run {run_name} completed")

    return result.returncode == 0


def generate_comparison_plot(ablation_name, run_dirs, output_path):
    """Generate a comparison plot for an ablation group."""
    from src.utils.plotting import plot_comparison

    log_paths = []
    labels = []
    for name, base_dir in run_dirs.items():
        run_dir = find_run_dir(base_dir)
        log_path = os.path.join(run_dir, "train_log.csv")
        if os.path.exists(log_path):
            log_paths.append(log_path)
            labels.append(name)

    if log_paths:
        plot_comparison(log_paths, labels, output_path)
        print(f"  Saved comparison plot: {output_path}")


def find_run_dir(base_dir):
    """Find the actual run directory (train_bc.py creates a timestamped subdirectory)."""
    if os.path.exists(os.path.join(base_dir, "train_log.csv")):
        return base_dir
    # Look for subdirectories containing train_log.csv
    for entry in sorted(os.listdir(base_dir), reverse=True):
        sub = os.path.join(base_dir, entry)
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "train_log.csv")):
            return sub
    return base_dir


def collect_results(ablation_name, run_dirs):
    """Collect final eval rewards from each run."""
    results = {}
    for name, base_dir in run_dirs.items():
        run_dir = find_run_dir(base_dir)
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            results[name] = {
                "train_pairs": cfg.get("train_pairs", "N/A"),
                "val_pairs": cfg.get("val_pairs", "N/A"),
            }

        # Find best reward from train_log.csv
        import csv
        log_path = os.path.join(run_dir, "train_log.csv")
        if os.path.exists(log_path):
            best_reward = -float("inf")
            best_ema_reward = -float("inf")
            with open(log_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("eval_reward"):
                        try:
                            r = float(row["eval_reward"])
                            if r > best_reward:
                                best_reward = r
                        except ValueError:
                            pass
                    if row.get("ema_eval_reward"):
                        try:
                            r = float(row["ema_eval_reward"])
                            if r > best_ema_reward:
                                best_ema_reward = r
                        except ValueError:
                            pass
            results.setdefault(name, {})
            results[name]["best_eval_reward"] = round(best_reward, 4) if best_reward > -float("inf") else "N/A"
            results[name]["best_ema_reward"] = round(best_ema_reward, 4) if best_ema_reward > -float("inf") else "N/A"

    return results


def main():
    parser = argparse.ArgumentParser(description="Run BC ablation experiments")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=list(ABLATIONS.keys()) + ["all"],
                        help="Which ablation to run")
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=str, default="outputs/ablations/bc")
    parser.add_argument("--data_dir", type=str, default="data/lerobot/pusht")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.ablation == "all":
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = [args.ablation]

    all_results = {}

    for ablation_name in ablation_names:
        ablation = ABLATIONS[ablation_name]
        print(f"\n{'#'*60}")
        print(f"Ablation: {ablation_name} — {ablation['description']}")
        print(f"{'#'*60}")

        ablation_dir = os.path.join(args.output_dir, f"{ablation_name}_{timestamp}")
        run_dirs = {}

        # Run each variant
        for run_name, overrides in ablation["runs"].items():
            run_output = os.path.join(ablation_dir, run_name)
            success = run_single(args, overrides, run_name, run_output)
            run_dirs[run_name] = run_output

        # Generate comparison plot
        plot_path = os.path.join(ablation_dir, f"{ablation_name}_comparison.png")
        generate_comparison_plot(ablation_name, run_dirs, plot_path)

        # Collect results
        results = collect_results(ablation_name, run_dirs)
        all_results[ablation_name] = results

        # Print summary table
        print(f"\n  Summary for {ablation_name}:")
        print(f"  {'Variant':<20} {'Best Reward':>12} {'EMA Reward':>12}")
        print(f"  {'-'*44}")
        for name, res in results.items():
            r = res.get("best_eval_reward", "N/A")
            e = res.get("best_ema_reward", "N/A")
            print(f"  {name:<20} {str(r):>12} {str(e):>12}")

    # Save overall results
    results_path = os.path.join(args.output_dir, f"ablation_results_{timestamp}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll results saved to {results_path}")


if __name__ == "__main__":
    main()
