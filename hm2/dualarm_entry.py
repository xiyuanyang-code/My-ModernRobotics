import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime

import gymnasium as gym
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import robotics_control  # noqa: F401
from robotics_control.ilqr import calc_ilqr_input, clip_action
from robotics_control.arm_renderer import create_arm_video
from plot_results import (
    plot_joint_position,
    plot_joint_velocity,
    plot_control_torque,
    plot_error_norm,
    plot_control_effort,
    plot_energy,
    plot_reward,
)

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


def _reset_env_pair(env_name, seed, render_mode=None):
    kwargs = {"noise_free": True}
    if render_mode:
        kwargs["render_mode"] = render_mode
    env = gym.make(env_name, **kwargs)
    sim_env = gym.make(env_name, noise_free=True)
    env.reset(seed=seed)
    sim_env.reset(seed=seed + 10_000 if seed is not None else None)
    _sync_goal(sim_env, env)
    return env, sim_env


def _run_episode(
    params,
    env_name,
    seed,
    sim_time=DEFAULT_SIM_TIME,
    r_scale=0.01,
    record=False,
    video_path=None,
    plot_path=None,
):
    render_mode = "rgb_array" if record else None
    env, sim_env = _reset_env_pair(env_name, seed, render_mode=render_mode)
    sim_fps = int(round(1.0 / env.unwrapped.dt))
    record_stride = max(1, int(round(sim_fps / DEFAULT_RECORD_FPS)))

    video_frames = []
    if record:
        frame = env.render()
        if frame is not None:
            video_frames.append(frame)

    u_cmd_filt = None
    total_reward = 0.0
    settled_steps = 0
    state_history = [np.asarray(env.unwrapped.state, dtype=float).copy()]
    goal_history = [np.asarray(env.unwrapped.goal, dtype=float).copy()]
    control_history = []
    reward_history = []
    plan_cost_history = []

    # iLQR parameters
    tN = 100
    ilqr_max_iter = 100000
    ilqr_tol = 1e-6
    ilqr_reg = 1e-2

    # Buffer for planned control sequence
    U_plan = None
    plan_idx = 0
    replan_interval = 50  # Replan every N steps
    total_steps = int(sim_time * sim_fps)

    pbar = tqdm(total=total_steps, desc=f"", unit="step")
    for i in range(total_steps):
        # Replan using iLQR periodically or when plan is exhausted
        if U_plan is None or plan_idx >= len(U_plan) or i % replan_interval == 0:
            _sync_goal(sim_env, env)
            # Warm-start: shift previous plan forward
            u_warm = None
            if U_plan is not None and plan_idx < len(U_plan):
                u_warm = np.zeros((tN, env.action_space.shape[0]))
                remaining = U_plan[plan_idx:]
                u_warm[:len(remaining)] = remaining
            U_plan = calc_ilqr_input(
                env=env,
                sim_env=sim_env,
                tN=tN,
                max_iter=ilqr_max_iter,
                tol=ilqr_tol,
                reg=ilqr_reg,
                r_scale=r_scale,
                velocity_q_scale=1.0,
                u_init=u_warm,
            )
            plan_idx = 0

        control_action = U_plan[plan_idx].copy()
        plan_idx += 1

        obs, reward, terminated, truncated, _ = env.step(control_action.copy())
        done = terminated or truncated
        total_reward += float(reward)

        state = np.asarray(obs, dtype=float).copy()
        goal = np.asarray(env.unwrapped.goal, dtype=float).copy()
        err = state - goal
        err_norm = float(np.linalg.norm(err))
        settled_steps += int(err_norm < 0.05)
        state_history.append(state)
        goal_history.append(goal)
        control_history.append(np.asarray(control_action, dtype=float).copy())
        reward_history.append(float(reward))

        if record and i % record_stride == 0:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame)

        pbar.update(1)
        pbar.set_postfix(err=f"{err_norm:.4f}")

        if done:
            pbar.write(f"[Step {i+1}/{total_steps}] Goal reached!")
            break

    pbar.close()

    print(f"\n=== Episode Summary ===")
    print(f"Steps: {len(control_history)}")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Final error: {err_norm:.4f}")

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
        time_s = np.arange(states.shape[0], dtype=float) * env.unwrapped.dt
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

    return metrics, states, goals, controls, plan_cost_history, reward_history


def evaluate_params(params, env_name, seeds, sim_time=DEFAULT_SIM_TIME, r_scale=0.01):
    episodes = []
    for seed in seeds:
        metrics, _, _, _, _, _ = _run_episode(
            params=params,
            env_name=env_name,
            seed=seed,
            sim_time=sim_time,
            r_scale=r_scale,
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


def save_demo(params, metrics, env_name, demo_seed, sim_time, prefix, result_dir, r_scale=0.01):
    os.makedirs(result_dir, exist_ok=True)
    video_path = os.path.join(result_dir, f"{prefix}.mp4")
    plot_path = os.path.join(result_dir, f"{prefix}_states.png")
    summary_path = os.path.join(result_dir, f"{prefix}_summary.json")

    demo_metrics, states, goals, controls, plan_cost_history, reward_history = _run_episode(
        params=params,
        env_name=env_name,
        seed=demo_seed,
        sim_time=sim_time,
        r_scale=r_scale,
        record=True,
        video_path=video_path,
        plot_path=plot_path,
    )

    # Save trajectory data for post-processing plots
    np.savez(
        os.path.join(result_dir, f"{prefix}_trajectory.npz"),
        states=states,
        goals=goals,
        controls=controls,
        rewards=np.asarray(reward_history, dtype=float),
    )

    # Generate PDF plots
    dt = 0.001  # TwoLinkArm default
    n_steps = states.shape[0]
    time_s = np.arange(n_steps, dtype=float) * dt
    control_time = time_s[:-1] if controls.shape[0] == n_steps - 1 else time_s

    plot_joint_position(time_s, states, goals,
                        os.path.join(result_dir, "plot_joint_position.pdf"))
    plot_joint_velocity(time_s, states,
                        os.path.join(result_dir, "plot_joint_velocity.pdf"))
    plot_control_torque(control_time, controls,
                        os.path.join(result_dir, "plot_control_torque.pdf"))
    plot_error_norm(time_s, states, goals,
                    os.path.join(result_dir, "plot_error_norm.pdf"))
    plot_control_effort(control_time, controls, result_dir, prefix="plot")
    plot_energy(time_s, states,
                os.path.join(result_dir, "plot_energy.pdf"))
    if reward_history:
        reward_time = control_time[:len(reward_history)]
        plot_reward(reward_time, np.asarray(reward_history),
                    os.path.join(result_dir, "plot_reward.pdf"))

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
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Directory to save all results. Auto-generated as results/<env_name>_<timestamp> if not set.",
    )
    parser.add_argument(
        "--r-scale",
        type=float,
        default=0.01,
        help="Control cost multiplier. Smaller = more aggressive torques = faster reaching.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve result directory
    if args.result_dir:
        result_dir = args.result_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = os.path.join("results", f"{timestamp}_{args.env_name}")
    os.makedirs(result_dir, exist_ok=True)

    eval_seeds = [args.seed + i for i in range(args.eval_episodes)]

    best_params = ILQRParams(mock_param=0.0)
    best_metrics = evaluate_params(
        best_params, args.env_name, eval_seeds, sim_time=args.sim_time, r_scale=args.r_scale
    )
    history = []

    # Save hyperparameters
    hyperparams = {
        "env_name": args.env_name,
        "sim_time": args.sim_time,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "demo_seed": args.demo_seed,
        "r_scale": args.r_scale,
        "tN": 100,
        "ilqr_max_iter": 100000,
        "ilqr_tol": 1e-6,
        "ilqr_reg": 1e-2,
        "replan_interval": 50,
        "Q_terminal_diag": 1e4,
    }
    with open(os.path.join(result_dir, "hyperparameters.json"), "w", encoding="utf-8") as fh:
        json.dump(hyperparams, fh, indent=2)

    video_path, plot_path, summary_path, demo_metrics = save_demo(
        params=best_params,
        metrics=best_metrics,
        env_name=args.env_name,
        demo_seed=args.demo_seed,
        sim_time=args.sim_time,
        prefix="two_link_arm_ilqr",
        result_dir=result_dir,
        r_scale=args.r_scale,
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
    print(f"Results saved to {result_dir}")
    print(f"  Video: {video_path}")
    print(f"  State plot: {plot_path}")
    print(f"  Summary: {summary_path}")
    print(f"  Hyperparameters: {os.path.join(result_dir, 'hyperparameters.json')}")


if __name__ == "__main__":
    main()