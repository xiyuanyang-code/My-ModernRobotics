import argparse
import json
import os
from dataclasses import asdict, dataclass

import gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

import robotics_control  # noqa: F401
from robotics_control.ilqr import calc_ilqr_input, clip_action

DEFAULT_ENV_NAME = "TwoLinkArm-limited-torque-random-goal-v0"
DEFAULT_SIM_TIME = 10.0
DEFAULT_RECORD_FPS = 30


@dataclass
class ILQRParams:
    mock_param: float


@dataclass
class EpisodeMetrics:
    total_reward: float
    steps: int
    success: bool
    final_goal_error_norm: float
    mean_goal_error_norm: float
    mean_control_norm: float
    max_control_norm: float
    settled_ratio: float
    score: float


def _sync_goal(sim_env, env):
    sim_env.unwrapped.goal_q = env.unwrapped.goal_q.copy()
    sim_env.unwrapped.goal_dq = env.unwrapped.goal_dq.copy()


def _reset_env_pair(env_name, seed):
    env = gym.make(env_name, noise_free=True)
    sim_env = gym.make(env_name, noise_free=True)
    if seed is not None:
        env.seed(seed)
        sim_env.seed(seed + 10_000)
    env.reset()
    sim_env.reset()
    _sync_goal(sim_env, env)
    return env, sim_env


def _run_episode(
    params,
    env_name,
    seed,
    sim_time=DEFAULT_SIM_TIME,
    record=False,
    video_path=None,
    plot_path=None,
):
    env, sim_env = _reset_env_pair(env_name, seed)
    sim_fps = int(round(1.0 / env.dt))
    record_stride = max(1, int(round(sim_fps / DEFAULT_RECORD_FPS)))

    video_frames = []
    if record:
        frame = env.render(mode="rgb_array")
        if frame is not None:
            video_frames.append(frame)

    u_cmd_filt = None
    total_reward = 0.0
    settled_steps = 0
    state_history = [np.asarray(env.state, dtype=float).copy()]
    goal_history = [np.asarray(env.goal, dtype=float).copy()]
    control_history = []
    plan_cost_history = []

    for i in range(int(sim_time * sim_fps)):
        # TODO: generate actions here using ilqr
        control_action = np.zeros(2)

        obs, reward, done, _ = env.step(control_action.copy())
        total_reward += float(reward)

        state = np.asarray(obs, dtype=float).copy()
        goal = np.asarray(env.goal, dtype=float).copy()
        err = state - goal
        err_norm = float(np.linalg.norm(err))
        settled_steps += int(err_norm < 0.05)
        state_history.append(state)
        goal_history.append(goal)
        control_history.append(np.asarray(control_action, dtype=float).copy())

        if record and i % record_stride == 0:
            frame = env.render(mode="rgb_array")
            if frame is not None:
                video_frames.append(frame)

        if done:
            break

    states = np.asarray(state_history, dtype=float)
    goals = np.asarray(goal_history, dtype=float)
    controls = (
        np.asarray(control_history, dtype=float)
        if control_history
        else np.zeros((0, env.action_space.shape[0]))
    )
    goal_errors = states - goals
    error_norms = np.linalg.norm(goal_errors, axis=1)
    control_norms = (
        np.linalg.norm(controls, axis=1) if controls.size else np.zeros(0, dtype=float)
    )
    success = bool(
        np.allclose(states[-1][:2], goals[-1][:2], atol=0.03)
        and np.allclose(states[-1][2:], 0.0, atol=0.05)
    )

    metrics = EpisodeMetrics(
        total_reward=float(total_reward),
        steps=int(len(control_history)),
        success=success,
        final_goal_error_norm=float(error_norms[-1]),
        mean_goal_error_norm=float(np.mean(error_norms)),
        mean_control_norm=float(np.mean(control_norms)) if control_norms.size else 0.0,
        max_control_norm=float(np.max(control_norms)) if control_norms.size else 0.0,
        settled_ratio=float(settled_steps / max(len(control_history), 1)),
        score=float(
            total_reward
            - 120.0 * float(error_norms[-1])
            - 18.0 * float(np.mean(error_norms))
            - 0.8 * (float(np.mean(control_norms)) if control_norms.size else 0.0)
            + 200.0 * float(success)
            + 50.0 * float(settled_steps / max(len(control_history), 1))
        ),
    )

    if record and plot_path is not None:
        time_s = np.arange(states.shape[0], dtype=float) * env.dt
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
        axes[0].plot(time_s, states[:, 0], label="q0", color="tab:blue")
        axes[0].plot(time_s, states[:, 1], label="q1", color="tab:orange")
        axes[0].plot(
            time_s, goals[:, 0], "--", label="goal q0", color="tab:blue", alpha=0.5
        )
        axes[0].plot(
            time_s, goals[:, 1], "--", label="goal q1", color="tab:orange", alpha=0.5
        )
        axes[0].set_ylabel("Joint Position")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc="best")

        axes[1].plot(time_s, states[:, 2], label="dq0", color="tab:green")
        axes[1].plot(time_s, states[:, 3], label="dq1", color="tab:red")
        axes[1].set_ylabel("Joint Velocity")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="best")

        if controls.size:
            control_time = time_s[:-1]
            axes[2].plot(control_time, controls[:, 0], label="u0", color="tab:purple")
            axes[2].plot(control_time, controls[:, 1], label="u1", color="tab:brown")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Torque")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="best")
        fig.suptitle("Two-Link Arm iLQR Rollout")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    if record and video_path is not None and video_frames:
        imageio.mimsave(video_path, video_frames, fps=DEFAULT_RECORD_FPS)

    return metrics, states, goals, controls, plan_cost_history


def evaluate_params(params, env_name, seeds, sim_time=DEFAULT_SIM_TIME):
    episodes = []
    for seed in seeds:
        metrics, _, _, _, _ = _run_episode(
            params=params,
            env_name=env_name,
            seed=seed,
            sim_time=sim_time,
            record=False,
        )
        episodes.append(metrics)

    return {
        "score": float(np.mean([m.score for m in episodes])),
        "reward": float(np.mean([m.total_reward for m in episodes])),
        "success_rate": float(np.mean([float(m.success) for m in episodes])),
        "final_goal_error_norm": float(
            np.mean([m.final_goal_error_norm for m in episodes])
        ),
        "mean_goal_error_norm": float(
            np.mean([m.mean_goal_error_norm for m in episodes])
        ),
        "mean_control_norm": float(np.mean([m.mean_control_norm for m in episodes])),
        "settled_ratio": float(np.mean([m.settled_ratio for m in episodes])),
        "episodes": [asdict(m) for m in episodes],
    }


def save_demo(params, metrics, env_name, demo_seed, sim_time, prefix):
    os.makedirs("videos", exist_ok=True)
    video_path = os.path.join("videos", f"{prefix}.mp4")
    plot_path = os.path.join("videos", f"{prefix}_states.png")
    summary_path = os.path.join("videos", f"{prefix}_summary.json")

    demo_metrics, states, goals, controls, plan_cost_history = _run_episode(
        params=params,
        env_name=env_name,
        seed=demo_seed,
        sim_time=sim_time,
        record=True,
        video_path=video_path,
        plot_path=plot_path,
    )

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "env_name": env_name,
                "demo_seed": demo_seed,
                "params": asdict(params),
                "aggregate_metrics": metrics,
                "demo_metrics": asdict(demo_metrics),
                "final_state": states[-1].tolist(),
                "goal": goals[-1].tolist(),
                "mean_plan_control_norm": float(np.mean(plan_cost_history))
                if plan_cost_history
                else 0.0,
                "max_control_norm": float(np.max(np.linalg.norm(controls, axis=1)))
                if controls.size
                else 0.0,
            },
            fh,
            indent=2,
        )

    return video_path, plot_path, summary_path, demo_metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Demonstrate a strong group of iLQR parameters for the planar two-link arm."
        )
    )
    parser.add_argument(
        "--env-name", default=DEFAULT_ENV_NAME, help="Gym environment id."
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=4,
        help="Episodes averaged per parameter group.",
    )
    parser.add_argument(
        "--seed", type=int, default=5, help="Base random seed for search."
    )
    parser.add_argument(
        "--demo-seed", type=int, default=123, help="Seed for the saved demo rollout."
    )
    parser.add_argument(
        "--sim-time",
        type=float,
        default=DEFAULT_SIM_TIME,
        help="Simulation time in seconds.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    eval_seeds = [args.seed + i for i in range(args.eval_episodes)]

    best_params = ILQRParams(mock_param=0.0)
    best_metrics = evaluate_params(
        best_params, args.env_name, eval_seeds, sim_time=args.sim_time
    )
    history = []

    video_path, plot_path, summary_path, demo_metrics = save_demo(
        params=best_params,
        metrics=best_metrics,
        env_name=args.env_name,
        demo_seed=args.demo_seed,
        sim_time=args.sim_time,
        prefix="two_link_arm_ilqr",
    )

    print("")
    print("=== TWO-LINK ARM iLQR RESULT ===")
    print("Best parameter group:")
    print(json.dumps(asdict(best_params), indent=2))
    print("Aggregate metrics:")
    print(json.dumps(best_metrics, indent=2))
    if history:
        print(f"Search generations completed: {len(history)}")
    print("Demo metrics:")
    print(json.dumps(asdict(demo_metrics), indent=2))
    print(f"Video saved to {video_path}")
    print(f"State plot saved to {plot_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
