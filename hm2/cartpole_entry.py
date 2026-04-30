import argparse
import json
import os
from dataclasses import asdict, dataclass

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


def _safe_reset(env):
    out = env.reset()
    if isinstance(out, tuple):
        return out[0]
    return out


def _safe_step(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        done = bool(terminated or truncated)
        return obs, reward, done, info
    obs, reward, done, info = out
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


def _plot_rollout(
    time_s, state_history, force_history, action_history, save_path, title
):
    states = np.asarray(state_history, dtype=float)
    forces = np.asarray(force_history, dtype=float)
    actions = np.asarray(action_history, dtype=float)
    if states.ndim != 2 or states.shape[1] < 4:
        raise ValueError(
            f"Expected state history with shape [N,4+], got {states.shape}."
        )

    x = states[:, 0]
    x_dot = states[:, 1]
    theta = states[:, 2]
    theta_dot = states[:, 3]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(time_s, x, label="cart position x (m)", color="tab:blue")
    axes[0].plot(time_s, x_dot, label="cart velocity x_dot (m/s)", color="tab:orange")
    axes[0].set_ylabel("Cart State")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(time_s, theta, label="pole theta (rad)", color="tab:green")
    axes[1].plot(time_s, theta_dot, label="pole omega (rad/s)", color="tab:red")
    axes[1].set_ylabel("Pole State")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(time_s[:-1], forces, label="PID force command", color="tab:purple")
    axes[2].step(
        time_s[:-1],
        actions,
        where="post",
        label="applied action",
        color="tab:brown",
        alpha=0.7,
    )
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Control")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


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
    video_frames = []
    total_reward = 0.0
    stable_count = 0

    if record:
        first_frame = env.render(mode=render_mode)
        if first_frame is not None:
            video_frames.append(first_frame)

    for _ in range(max_steps):
        action = controller.compute(obs, dt=DEFAULT_DT)
        obs, reward, done, info = _safe_step(env, action)
        total_reward += float(reward)
        state_history.append(np.asarray(obs, dtype=float).copy())
        action_history.append(int(action))
        force_history.append(controller._last_force)
        stable_count += int(bool(info.get("is_stable_upright", False)))

        if record:
            frame = env.render(mode=render_mode)
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
    return result, state_history, force_history, action_history, video_frames


def evaluate_params(params, seeds):
    results = []
    for seed in seeds:
        episode_result, _, _, _, _ = run_episode(params=params, seed=seed, record=False)
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


def save_demo_artifacts(params, metrics, output_prefix, demo_seed):
    os.makedirs("videos", exist_ok=True)
    result, state_history, force_history, action_history, video_frames = run_episode(
        params=params, seed=demo_seed, record=True
    )
    time_s = np.arange(len(state_history), dtype=float) * DEFAULT_DT
    video_path = os.path.join("videos", f"{output_prefix}.mp4")
    curve_path = os.path.join("videos", f"{output_prefix}_states.png")
    summary_path = os.path.join("videos", f"{output_prefix}_summary.json")

    if video_frames:
        imageio.mimsave(video_path, video_frames, fps=int(round(1.0 / DEFAULT_DT)))
    _plot_rollout(
        time_s=time_s,
        state_history=state_history,
        force_history=force_history,
        action_history=action_history,
        save_path=curve_path,
        title="CartPole PID rollout",
    )
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
    return video_path, curve_path, summary_path, result


def _run_mode(args):
    eval_seeds = [args.seed + i for i in range(args.eval_episodes)]

    params = PIDParams(
        theta_kp=50.0,
        theta_kd=20.0,
        theta_ki=0.0,
        x_kp=2.0,
        x_kd=3.0,
        x_ki=0.0,
        force_limit=10.0,
        integral_limit=5.0,
        hysteresis=0.5,
    )

    metrics = evaluate_params(params, eval_seeds)

    video_path, curve_path, summary_path, demo_result = save_demo_artifacts(
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
    print(f"State curves saved to {curve_path}")
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
    return parser.parse_args()


def main():
    args = parse_args()
    _run_mode(args)


if __name__ == "__main__":
    main()
