"""Run RL ablation experiments for Problem 1.

Runs multiple training configurations and generates comparison plots.
"""

import argparse
import os
import sys
import json
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

ABLATIONS = {
    # Ablation 1: PPO vs SAC (on-policy vs off-policy)
    "algo_compare": {
        "description": "PPO vs SAC (on-policy vs off-policy)",
        "runs": [
            {"name": "PPO", "script": "src/train_ppo.py", "args": ["--output_dir", "outputs/ablations/algo_compare/ppo"]},
            {"name": "SAC", "script": "src/train_sac.py", "args": ["--output_dir", "outputs/ablations/algo_compare/sac"]},
        ],
    },
    # Ablation 2: Observation normalization
    "obs_norm": {
        "description": "Observation normalization ON vs OFF (PPO)",
        "runs": [
            {"name": "PPO_obs_norm_on", "script": "src/train_ppo.py", "args": ["--obs_norm", "1", "--output_dir", "outputs/ablations/obs_norm/ppo_on"]},
            {"name": "PPO_obs_norm_off", "script": "src/train_ppo.py", "args": ["--obs_norm", "0", "--output_dir", "outputs/ablations/obs_norm/ppo_off"]},
        ],
    },
    # Ablation 3: Hidden dimension
    "hidden_dim": {
        "description": "Hidden dimension 64 vs 256 (PPO)",
        "runs": [
            {"name": "PPO_h64", "script": "src/train_ppo.py", "args": ["--hidden_dim", "64", "--output_dir", "outputs/ablations/hidden_dim/ppo_64"]},
            {"name": "PPO_h256", "script": "src/train_ppo.py", "args": ["--hidden_dim", "256", "--output_dir", "outputs/ablations/hidden_dim/ppo_256"]},
        ],
    },
    # Ablation 4: Learning rate
    "lr": {
        "description": "Learning rate 1e-4 vs 3e-4 vs 1e-3 (PPO)",
        "runs": [
            {"name": "PPO_lr1e4", "script": "src/train_ppo.py", "args": ["--lr", "0.0001", "--output_dir", "outputs/ablations/lr/ppo_1e4"]},
            {"name": "PPO_lr3e4", "script": "src/train_ppo.py", "args": ["--lr", "0.0003", "--output_dir", "outputs/ablations/lr/ppo_3e4"]},
            {"name": "PPO_lr1e3", "script": "src/train_ppo.py", "args": ["--lr", "0.001", "--output_dir", "outputs/ablations/lr/ppo_1e3"]},
        ],
    },
}


def run_single(run_config, max_steps, seed, device, python_bin):
    """Launch a single training run."""
    cmd = [
        python_bin, run_config["script"],
        "--max_steps", str(max_steps),
        "--seed", str(seed),
        "--device", device,
    ] + run_config["args"]

    print(f"\n{'='*60}")
    print(f"Running: {run_config['name']}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    subprocess.run(cmd, cwd=PROJECT_ROOT)


def generate_comparison_plot(ablation_name, ablation_config, output_dir):
    """Generate comparison plot for an ablation."""
    from src.utils.plotting import plot_comparison

    log_paths = []
    labels = []
    for run in ablation_config["runs"]:
        # Find output dir from args
        for i, arg in enumerate(run["args"]):
            if arg == "--output_dir":
                log_path = os.path.join(PROJECT_ROOT, run["args"][i + 1], "train_log.csv")
                if os.path.exists(log_path):
                    log_paths.append(log_path)
                    labels.append(run["name"])
                break

    if log_paths:
        out_path = os.path.join(output_dir, f"{ablation_name}_comparison.png")
        plot_comparison(log_paths, labels, out_path)


def main():
    parser = argparse.ArgumentParser(description="Run RL ablation experiments")
    parser.add_argument("--ablation", type=str, default="all", help="Which ablation to run, or 'all'")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Max training steps per run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter")
    parser.add_argument("--plot_only", action="store_true", help="Only generate plots, don't train")
    args = parser.parse_args()

    os.makedirs(os.path.join(PROJECT_ROOT, "outputs", "ablations"), exist_ok=True)

    if args.ablation == "all":
        ablations_to_run = ABLATIONS
    else:
        if args.ablation not in ABLATIONS:
            print(f"Unknown ablation: {args.ablation}")
            print(f"Available: {list(ABLATIONS.keys())}")
            return
        ablations_to_run = {args.ablation: ABLATIONS[args.ablation]}

    for name, config in ablations_to_run.items():
        print(f"\n{'#'*60}")
        print(f"Ablation: {config['description']}")
        print(f"{'#'*60}")

        if not args.plot_only:
            for run in config["runs"]:
                run_single(run, args.max_steps, args.seed, args.device, args.python)

        # Generate comparison plot
        generate_comparison_plot(name, config, os.path.join(PROJECT_ROOT, "outputs", "ablations"))

    print("\nAll ablations complete!")


if __name__ == "__main__":
    main()
