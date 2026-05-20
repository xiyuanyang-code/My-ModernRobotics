"""
PID Parameter Sensitivity Analysis Script.

Varies each PID parameter independently while keeping others fixed,
and evaluates performance metrics across multiple seeds.
"""

import json
import os
from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np

from cartpole_entry import PIDParams, evaluate_params, run_episode


@dataclass
class SensitivityConfig:
    """Configuration for sensitivity analysis of a single parameter."""

    param_name: str
    values: list[float]
    base_params: PIDParams


# Base parameters (current best)
BASE_PARAMS = PIDParams(
    theta_kp=46,
    theta_kd=11,
    theta_ki=0.0,
    x_kp=3.2,
    x_kd=5,
    x_ki=0.3,
    force_limit=10.0,
    integral_limit=5.0,
    hysteresis=1.5,
)

# Parameter ranges to sweep (50+ data points for smooth curves)
SENSITIVITY_CONFIGS = [
    SensitivityConfig("theta_kp", np.linspace(10, 90, 50).tolist(), BASE_PARAMS),
    SensitivityConfig("theta_kd", np.linspace(2, 30, 50).tolist(), BASE_PARAMS),
    SensitivityConfig("theta_ki", np.linspace(0.0, 8.0, 50).tolist(), BASE_PARAMS),
    SensitivityConfig("x_kp", np.linspace(0.5, 8.0, 50).tolist(), BASE_PARAMS),
    SensitivityConfig("x_kd", np.linspace(1, 12, 50).tolist(), BASE_PARAMS),
    SensitivityConfig("x_ki", np.linspace(0.0, 2.0, 50).tolist(), BASE_PARAMS),
    SensitivityConfig("hysteresis", np.linspace(0.2, 5.0, 50).tolist(), BASE_PARAMS),
]

# Representative values for detailed trajectory comparison (3 values per param)
REPRESENTATIVE_VALUES = {
    "theta_kp": [20, 46, 70],      # low, base, high
    "theta_kd": [5, 11, 20],       # low, base, high
    "theta_ki": [0.0, 1.0, 5.0],   # none, moderate, high
    "x_kp": [1.0, 3.2, 6.0],      # low, base, high
    "x_kd": [2, 5, 10],           # low, base, high
    "x_ki": [0.0, 0.3, 1.0],      # none, base, high
    "hysteresis": [0.5, 1.5, 3.0], # low, base, high
}


def run_sensitivity_analysis(
    configs: list[SensitivityConfig],
    seeds: list[int] = None,
    output_dir: str = "sensitivity_results",
):
    """
    Run sensitivity analysis for each parameter configuration.

    Args:
        configs: List of SensitivityConfig objects.
        seeds: Random seeds for evaluation.
        output_dir: Directory to save results.
    """
    if seeds is None:
        seeds = [7 + i for i in range(5)]

    os.makedirs(output_dir, exist_ok=True)
    all_results = {}

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Analyzing parameter: {config.param_name}")
        print(f"Values: {config.values}")
        print(f"{'='*60}")

        param_results = []

        for value in config.values:
            # Create params with current value
            params_dict = asdict(config.base_params)
            params_dict[config.param_name] = value
            params = PIDParams(**params_dict)

            # Evaluate
            metrics = evaluate_params(params, seeds)

            result = {
                "value": value,
                "metrics": metrics,
            }
            param_results.append(result)

            print(f"  {config.param_name}={value:.2f}: "
                  f"reward={metrics['reward']:.1f}, "
                  f"steps={metrics['steps']:.0f}, "
                  f"stable_ratio={metrics['stable_ratio']:.3f}, "
                  f"switch_rate={metrics['switch_rate']:.3f}")

        all_results[config.param_name] = param_results

    # Save results
    results_path = os.path.join(output_dir, "sensitivity_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    return all_results


def plot_sensitivity_results(all_results: dict, output_dir: str = "sensitivity_results"):
    """Generate sensitivity analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    metrics_to_plot = [
        ("reward", "Total Reward"),
        ("stable_ratio", "Stable Ratio"),
        ("switch_rate", "Switch Rate"),
        ("mean_abs_theta", r"Mean $|\theta|$ (rad)"),
    ]

    for param_name, results in all_results.items():
        values = [r["value"] for r in results]

        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle(f"Sensitivity Analysis: {param_name}", fontsize=14)

        for idx, (metric_key, metric_label) in enumerate(metrics_to_plot):
            ax = axes[idx // 2][idx % 2]
            metric_values = [r["metrics"][metric_key] for r in results]

            ax.plot(values, metric_values, "o-", color="tab:blue", linewidth=2, markersize=8)
            ax.set_xlabel(param_name)
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.3)

            # Highlight best value
            if metric_key in ["reward", "stable_ratio"]:
                best_idx = np.argmax(metric_values)
            else:
                best_idx = np.argmin(metric_values)
            ax.scatter(values[best_idx], metric_values[best_idx], color="red", s=100, zorder=5)

        fig.tight_layout()
        plot_path = os.path.join(output_dir, f"sensitivity_{param_name}.pdf")
        fig.savefig(plot_path, dpi=150, format="pdf")
        plt.close(fig)
        print(f"Plot saved to {plot_path}")


def plot_trajectory_comparison(
    param_name: str,
    representative_values: list[float],
    base_params: PIDParams,
    seed: int = 42,
    output_dir: str = "sensitivity_results",
):
    """
    Generate side-by-side trajectory comparison for representative parameter values.

    Args:
        param_name: Name of the parameter to vary.
        representative_values: List of 3 values (low, base, high) to compare.
        base_params: Base PID parameters.
        seed: Random seed for reproducibility.
        output_dir: Directory to save plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Trajectory Comparison: {param_name}", fontsize=16)

    labels = ["Low", "Base", "High"]

    for col, (value, label) in enumerate(zip(representative_values, labels)):
        # Create params with current value
        params_dict = asdict(base_params)
        params_dict[param_name] = value
        params = PIDParams(**params_dict)

        # Run episode
        result, state_history, force_history, action_history, _, _, _ = run_episode(
            params=params, seed=seed, record=False
        )
        states = np.asarray(state_history)
        forces = np.asarray(force_history)
        actions = np.asarray(action_history)
        time_s = np.arange(len(states)) * 0.02

        # Plot pole angle
        ax = axes[0][col]
        ax.plot(time_s, states[:, 2], color="tab:green", linewidth=0.5, alpha=0.3)
        ax.plot(time_s, _smooth(states[:, 2]), color="tab:green", linewidth=2.0)
        ax.set_title(f"{label}: {param_name}={value}\nθ (rad)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Pole Angle")

        # Plot cart position
        ax = axes[1][col]
        ax.plot(time_s, states[:, 0], color="tab:blue", linewidth=0.5, alpha=0.3)
        ax.plot(time_s, _smooth(states[:, 0]), color="tab:blue", linewidth=2.0)
        ax.set_title(f"x (m)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Cart Position")

        # Plot control force
        ax = axes[2][col]
        ax.plot(time_s[:-1], forces, color="tab:purple", linewidth=0.5, alpha=0.3)
        ax.plot(time_s[:-1], _smooth(forces), color="tab:purple", linewidth=2.0)
        ax.set_title(f"Force (N)\nReward={result.total_reward:.0f}")
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.3)
        if col == 0:
            ax.set_ylabel("Control Force")

    fig.tight_layout()
    plot_path = os.path.join(output_dir, f"trajectory_comparison_{param_name}.pdf")
    fig.savefig(plot_path, dpi=150, format="pdf")
    plt.close(fig)
    print(f"Trajectory comparison saved to {plot_path}")


def _smooth(signal, window=15):
    """Simple moving average for plotting."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    smoothed = np.convolve(signal, kernel, mode="same")
    smoothed[:window//2] = signal[:window//2]
    smoothed[-(window//2):] = signal[-(window//2):]
    return smoothed


def main():
    """Run sensitivity analysis and generate plots."""
    print("Starting PID Parameter Sensitivity Analysis")
    print(f"Base parameters: {asdict(BASE_PARAMS)}")

    # Run analysis
    all_results = run_sensitivity_analysis(SENSITIVITY_CONFIGS)

    # Generate sensitivity trend plots
    plot_sensitivity_results(all_results)

    # Generate trajectory comparison plots for each parameter
    print("\nGenerating trajectory comparison plots...")
    for param_name, values in REPRESENTATIVE_VALUES.items():
        plot_trajectory_comparison(param_name, values, BASE_PARAMS)

    print("\nSensitivity analysis complete!")


if __name__ == "__main__":
    main()
