import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from robotics_control.cartpole import (
    make_cartpole_env_from_config,
    reset_cartpole_with_random_state,
)
from robotics_control.pid import CartPolePIDController

USE_RANDOM_INIT = True
RANDOM_X_RANGE = (-0.4, 0.4)
RANDOM_X_DOT_RANGE = (-0.15, 0.15)
RANDOM_THETA_RANGE = (-0.5236, 0.5236)  # +-30 deg
RANDOM_THETA_DOT_RANGE = (-0.3, 0.3)
DEFAULT_DT = 0.02
DEFAULT_MAX_STEPS = 1000


def _safe_reset(env, seed=None):
    obs, info = env.reset(seed=seed)
    return obs


def _safe_step(env, action):
    obs, reward, terminated, truncated, info = env.step(action)
    done = bool(terminated or truncated)
    return obs, reward, done, info


@dataclass
class PIDParams:
    """
    Parameters for the CartPole PID controller.

    Gains for angle loop (theta) and cart centering loop (x).
    Tuning order:
      1. Set theta_kp / theta_kd first (primary stabilization).
      2. Add small x_kp / x_kd for centering.
      3. Add tiny integral terms only if persistent bias remains.
      4. Increase hysteresis if action switches too frequently.
    """

    theta_kp: float = 50.0
    theta_kd: float = 20.0
    theta_ki: float = 0.0
    x_kp: float = 2.0
    x_kd: float = 3.0
    x_ki: float = 0.0
    force_limit: float = 10.0
    integral_limit: float = 5.0
    hysteresis: float = 0.5


@dataclass
class EpisodeResult:
    total_reward: float
    steps: int
    mean_abs_theta: float
    mean_abs_x: float
    stable_ratio: float
    switch_rate: float
    score: float
    final_obs: list[float]


def _smooth_signal(signal, window=15):
    """Apply moving average smoothing to a signal."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    smoothed = np.convolve(signal, kernel, mode="same")
    # Fix edges
    smoothed[: window // 2] = signal[: window // 2]
    smoothed[-(window // 2) :] = signal[-(window // 2) :]
    return smoothed


def _compute_energy(states, m_cart=1.0, m_pole=0.1, l=0.5, g=9.81):
    """Compute kinetic + potential energy (simplified)."""
    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]
    # Simplified energy: KE from cart + pole angular, PE from pole height
    ke = 0.5 * m_cart * x_dot**2 + 0.5 * m_pole * (l * theta_dot) ** 2
    pe = m_pole * g * l * (1 - np.cos(theta))
    return ke + pe


def _compute_switch_rate(actions, window=10):
    """Compute action switch rate in a sliding window."""
    switches = np.abs(np.diff(actions)) > 0.5
    rate = np.convolve(switches, np.ones(window) / window, mode="same")
    return rate


def _plot_cart_state(time_s, states, save_path):
    """Plot cart position and velocity."""
    x = states[:, 0]
    x_dot = states[:, 1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, x, color="tab:blue", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, x_dot, color="tab:orange", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(x), color="tab:blue", linewidth=2.5, label="x (m)")
    ax.plot(time_s, _smooth_signal(x_dot), color="tab:orange", linewidth=2.5, label=r"$\dot{x}$ (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cart State")
    ax.set_title("Cart State")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_pole_state(time_s, states, save_path):
    """Plot pole angle and angular velocity."""
    theta = states[:, 2]
    theta_dot = states[:, 3]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, theta, color="tab:green", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, theta_dot, color="tab:red", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(theta), color="tab:green", linewidth=2.5, label=r"$\theta$ (rad)")
    ax.plot(time_s, _smooth_signal(theta_dot), color="tab:red", linewidth=2.5, label=r"$\dot{\theta}$ (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pole State")
    ax.set_title("Pole State")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_control(time_s, forces, actions, save_path):
    """Plot PID force and discrete action."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s[:-1], forces, color="tab:purple", linewidth=0.5, alpha=0.3)
    ax.plot(time_s[:-1], _smooth_signal(forces), color="tab:purple", linewidth=2.5, label="PID Force")
    ax.step(time_s[:-1], actions, where="post", color="tab:brown", linewidth=1.0, alpha=0.7, label="Action")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Control")
    ax.set_title("Control Signal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_basic_states(time_s, state_history, force_history, action_history, save_dir, prefix):
    """Generate basic state/control plots as PDF."""
    states = np.asarray(state_history, dtype=float)
    forces = np.asarray(force_history, dtype=float)
    actions = np.asarray(action_history, dtype=float)

    _plot_cart_state(time_s, states, os.path.join(save_dir, f"{prefix}_cart_state.pdf"))
    _plot_pole_state(time_s, states, os.path.join(save_dir, f"{prefix}_pole_state.pdf"))
    _plot_control(time_s, forces, actions, os.path.join(save_dir, f"{prefix}_control.pdf"))


def _plot_phase_portrait(states, save_path):
    """Plot theta vs theta_dot phase portrait."""
    theta = states[:, 2]
    theta_dot = states[:, 3]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(theta, theta_dot, color="tab:blue", linewidth=0.8, alpha=0.7)
    ax.scatter(theta[0], theta_dot[0], color="green", s=80, zorder=5, label="Start")
    ax.scatter(theta[-1], theta_dot[-1], color="red", s=80, zorder=5, label="End")
    ax.set_xlabel(r"$\theta$ (rad)")
    ax.set_ylabel(r"$\dot{\theta}$ (rad/s)")
    ax.set_title("Phase Portrait")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_energy(time_s, states, save_path):
    """Plot system energy over time."""
    energy = _compute_energy(states)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, energy, color="tab:orange", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(energy), color="tab:orange", linewidth=2.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("System Energy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_error_magnitude(time_s, states, save_path):
    """Plot |theta| and |x| over time."""
    x = states[:, 0]
    theta = states[:, 2]
    abs_theta = np.abs(theta)
    abs_x = np.abs(x)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, abs_theta, color="tab:green", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, abs_x, color="tab:blue", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(abs_theta), color="tab:green", linewidth=2.5, label=r"$|\theta|$")
    ax.plot(time_s, _smooth_signal(abs_x), color="tab:blue", linewidth=2.5, label=r"$|x|$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error Magnitude")
    ax.set_title("Error Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_switch_rate(time_s, actions, save_path, window=10):
    """Plot action switch rate."""
    rate = _compute_switch_rate(actions, window)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s[: len(rate)], rate, color="tab:red", linewidth=0.5, alpha=0.3)
    ax.plot(time_s[: len(rate)], _smooth_signal(rate), color="tab:red", linewidth=2.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Switch Rate")
    ax.set_title(f"Action Switch Rate (window={window})")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_integral_terms(time_s, integral_history, save_path):
    """Plot integral terms over time."""
    integrals = np.asarray(integral_history, dtype=float)

    fig, ax = plt.subplots(figsize=(6, 6))
    if integrals.ndim == 2:
        ax.plot(time_s[: len(integrals)], integrals[:, 0], color="tab:purple", linewidth=0.5, alpha=0.3)
        ax.plot(time_s[: len(integrals)], integrals[:, 1], color="tab:brown", linewidth=0.5, alpha=0.3)
        ax.plot(time_s[: len(integrals)], _smooth_signal(integrals[:, 0]), color="tab:purple", linewidth=2.5, label=r"$\int \theta$")
        ax.plot(time_s[: len(integrals)], _smooth_signal(integrals[:, 1]), color="tab:brown", linewidth=2.5, label=r"$\int x$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Integral Value")
    ax.set_title("Integral Terms")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_cumulative_reward(rewards, save_path):
    """Plot cumulative reward over time."""
    cumulative = np.cumsum(rewards)
    steps = np.arange(len(cumulative))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(steps, cumulative, color="tab:cyan", linewidth=0.5, alpha=0.3)
    ax.plot(steps, _smooth_signal(cumulative), color="tab:cyan", linewidth=2.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Reward")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def _plot_all_metrics(
    time_s, state_history, force_history, action_history,
    integral_history, reward_history, save_dir, prefix
):
    """Generate all 6 metric plots as PDF."""
    states = np.asarray(state_history, dtype=float)
    actions = np.asarray(action_history, dtype=float)
    rewards = np.asarray(reward_history, dtype=float)

    os.makedirs(save_dir, exist_ok=True)

    _plot_phase_portrait(states, os.path.join(save_dir, f"{prefix}_phase_portrait.pdf"))
    _plot_energy(time_s, states, os.path.join(save_dir, f"{prefix}_energy.pdf"))
    _plot_error_magnitude(time_s, states, os.path.join(save_dir, f"{prefix}_error_magnitude.pdf"))
    _plot_switch_rate(time_s, actions, os.path.join(save_dir, f"{prefix}_switch_rate.pdf"))
    _plot_integral_terms(time_s, integral_history, os.path.join(save_dir, f"{prefix}_integral_terms.pdf"))
    _plot_cumulative_reward(rewards, os.path.join(save_dir, f"{prefix}_cumulative_reward.pdf"))


def _make_env():
    return make_cartpole_env_from_config(
        config_path="CartPole-v0_config.yaml", env_id="CartPoleUprightStabilize-v0"
    )


def _make_controller(params: PIDParams) -> CartPolePIDController:
    """
    Create a CartPolePIDController from the given PIDParams.
    """
    kwargs = asdict(params)
    return CartPolePIDController(**kwargs)


def run_episode(params, seed, record=False, max_steps=DEFAULT_MAX_STEPS):
    env = _make_env()
    controller = _make_controller(params=params)
    controller.reset()

    _safe_reset(env)
    if USE_RANDOM_INIT:
        obs = reset_cartpole_with_random_state(
            env,
            seed=seed,
            x_range=RANDOM_X_RANGE,
            x_dot_range=RANDOM_X_DOT_RANGE,
            theta_range=RANDOM_THETA_RANGE,
            theta_dot_range=RANDOM_THETA_DOT_RANGE,
        )
    else:
        obs = _safe_reset(env)

    render_mode = "rgb_array"
    state_history = [np.asarray(obs, dtype=float).copy()]
    force_history = []
    action_history = []
    integral_history = []
    reward_history = []
    video_frames = []
    total_reward = 0.0
    stable_count = 0

    if record:
        first_frame = env.render()
        if first_frame is not None:
            video_frames.append(first_frame)

    for _ in range(max_steps):
        action = controller.compute(obs, dt=DEFAULT_DT)
        obs, reward, done, info = _safe_step(env, action)
        total_reward += float(reward)
        state_history.append(np.asarray(obs, dtype=float).copy())
        action_history.append(int(action))
        force_history.append(controller._last_force)
        integral_history.append([controller._theta_integral, controller._x_integral])
        reward_history.append(float(reward))
        stable_count += int(bool(info.get("is_stable_upright", False)))

        if record:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame)

        if done:
            break

    states = np.asarray(state_history, dtype=float)
    theta = states[:, 2]
    x = states[:, 0]
    discrete_actions = np.asarray(action_history, dtype=float)
    if discrete_actions.size > 1:
        switch_rate = float(np.mean(np.abs(np.diff(discrete_actions)) > 1e-8))
    else:
        switch_rate = 0.0

    score = (
        total_reward
        + 0.15 * len(action_history)
        - 30.0 * float(np.mean(np.abs(theta)))
        - 5.0 * float(np.mean(np.abs(x)))
        - 6.0 * switch_rate
    )

    result = EpisodeResult(
        total_reward=float(total_reward),
        steps=len(action_history),
        mean_abs_theta=float(np.mean(np.abs(theta))) if theta.size else 0.0,
        mean_abs_x=float(np.mean(np.abs(x))) if x.size else 0.0,
        stable_ratio=float(stable_count / max(len(action_history), 1)),
        switch_rate=switch_rate,
        score=float(score),
        final_obs=np.asarray(states[-1], dtype=float).tolist(),
    )
    return result, state_history, force_history, action_history, integral_history, reward_history, video_frames


def evaluate_params(params, seeds):
    results = []
    for seed in seeds:
        episode_result, _, _, _, _, _, _ = run_episode(params=params, seed=seed, record=False)
        results.append(episode_result)

    mean_score = float(np.mean([r.score for r in results]))
    mean_reward = float(np.mean([r.total_reward for r in results]))
    mean_steps = float(np.mean([r.steps for r in results]))
    mean_theta = float(np.mean([r.mean_abs_theta for r in results]))
    mean_x = float(np.mean([r.mean_abs_x for r in results]))
    mean_stable = float(np.mean([r.stable_ratio for r in results]))
    mean_switch = float(np.mean([r.switch_rate for r in results]))
    return {
        "score": mean_score,
        "reward": mean_reward,
        "steps": mean_steps,
        "mean_abs_theta": mean_theta,
        "mean_abs_x": mean_x,
        "stable_ratio": mean_stable,
        "switch_rate": mean_switch,
        "episodes": [asdict(r) for r in results],
    }


def _build_run_dir(params: PIDParams) -> str:
    """Build a timestamped run directory name encoding key PID gains."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = (
        f"thkp{params.theta_kp:.0f}"
        f"_thkd{params.theta_kd:.0f}"
        f"_xkp{params.x_kp:.0f}"
        f"_xkd{params.x_kd:.0f}"
    )
    return os.path.join("runs", f"{ts}_{tag}")


def save_demo_artifacts(params, metrics, output_prefix, demo_seed):
    run_dir = _build_run_dir(params)
    os.makedirs(run_dir, exist_ok=True)

    result, state_history, force_history, action_history, integral_history, reward_history, video_frames = run_episode(
        params=params, seed=demo_seed, record=True
    )
    time_s = np.arange(len(state_history), dtype=float) * DEFAULT_DT
    video_path = os.path.join(run_dir, f"{output_prefix}.mp4")
    summary_path = os.path.join(run_dir, f"{output_prefix}_summary.json")

    if video_frames:
        imageio.mimsave(video_path, video_frames, fps=int(round(1.0 / DEFAULT_DT)))

    # Generate all metric plots
    _plot_all_metrics(
        time_s=time_s,
        state_history=state_history,
        force_history=force_history,
        action_history=action_history,
        integral_history=integral_history,
        reward_history=reward_history,
        save_dir=run_dir,
        prefix=output_prefix,
    )

    # Generate basic state/control plots
    _plot_basic_states(time_s, state_history, force_history, action_history, run_dir, output_prefix)
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "demo_seed": demo_seed,
                "params": asdict(params),
                "aggregate_metrics": metrics,
                "demo_result": asdict(result),
            },
            fh,
            indent=2,
        )
    return video_path, run_dir, summary_path, result

def _get_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config_data = json.load(file)
    return config_data


def _run_mode(args):
    eval_seeds = [args.seed + i for i in range(args.eval_episodes)]
    config_path = args.config_path
    config = _get_config(config_path=config_path)

    params = PIDParams(
        **config
    )

    metrics = evaluate_params(params, eval_seeds)

    video_path, run_dir, summary_path, demo_result = save_demo_artifacts(
        params=params,
        metrics=metrics,
        output_prefix="cartpole_pid_discrete",
        demo_seed=args.demo_seed,
    )

    print("")
    print("=== CartPole PID RESULT ===")
    print("Best params:")
    print(json.dumps(asdict(params), indent=2))
    print("Aggregate metrics:")
    print(json.dumps(metrics, indent=2))
    print("Demo episode:")
    print(json.dumps(asdict(demo_result), indent=2))
    print(f"Video saved to {video_path}")
    print(f"Plots saved to {run_dir}")
    print(f"Summary saved to {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tune and demonstrate PID control on the custom upright CartPole."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=5,
        help="How many randomized episodes to average per candidate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Base seed for the evaluation episodes.",
    )
    parser.add_argument(
        "--demo-seed",
        type=int,
        default=123,
        help="Seed used for the saved demonstration rollout.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="params_config/carpole/base.json",
        help="Seed used for the saved demonstration rollout.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _run_mode(args)


if __name__ == "__main__":
    main()
