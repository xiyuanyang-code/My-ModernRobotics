"""Comprehensive plotting script for iLQR two-link arm results.

Usage:
    python plot_results.py <result_dir>

Reads trajectory.npz and generates PDF plots in the same directory.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _smooth_signal(signal, window=15):
    """Moving average with edge preservation."""
    if len(signal) < window:
        return signal
    kernel = np.ones(window) / window
    smoothed = np.convolve(signal, kernel, mode="same")
    smoothed[: window // 2] = signal[: window // 2]
    smoothed[-(window // 2) :] = signal[-(window // 2) :]
    return smoothed


def _end_effector_pos(q, l1=0.5, l2=0.75):
    """Compute (x, y) of the end effector from joint angles."""
    x1 = l1 * np.cos(q[:, 0])
    y1 = l1 * np.sin(q[:, 0])
    x2 = x1 + l2 * np.cos(q[:, 0] + q[:, 1])
    y2 = y1 + l2 * np.sin(q[:, 0] + q[:, 1])
    return x1, y1, x2, y2


def _kinetic_energy(states, l1=0.5, l2=0.75, m1=0.33, m2=0.55):
    """Approximate kinetic energy from joint velocities."""
    dq0 = states[:, 2]
    dq1 = states[:, 3]
    # Simplified: KE ~ 0.5 * I_eff * dq^2
    I1 = (1 / 3 * m1 + m2) * l1 ** 2
    I2 = 1 / 3 * m2 * l2 ** 2
    ke = 0.5 * I1 * dq0 ** 2 + 0.5 * I2 * (dq0 + dq1) ** 2
    return ke


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_joint_position(time_s, states, goals, save_path):
    """Joint position tracking with goals."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, states[:, 0], color="tab:blue", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, states[:, 1], color="tab:orange", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(states[:, 0]), color="tab:blue", linewidth=2.5,
            label=r"$q_0$")
    ax.plot(time_s, _smooth_signal(states[:, 1]), color="tab:orange", linewidth=2.5,
            label=r"$q_1$")
    ax.plot(time_s, goals[:, 0], "--", color="tab:blue", linewidth=1.2, alpha=0.5,
            label=r"$q_0^*$")
    ax.plot(time_s, goals[:, 1], "--", color="tab:orange", linewidth=1.2, alpha=0.5,
            label=r"$q_1^*$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Position (rad)")
    ax.set_title("Joint Position Tracking")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_joint_velocity(time_s, states, save_path):
    """Joint velocities over time."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, states[:, 2], color="tab:green", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, states[:, 3], color="tab:red", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(states[:, 2]), color="tab:green", linewidth=2.5,
            label=r"$\dot{q}_0$")
    ax.plot(time_s, _smooth_signal(states[:, 3]), color="tab:red", linewidth=2.5,
            label=r"$\dot{q}_1$")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Velocity (rad/s)")
    ax.set_title("Joint Velocity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_control_torque(time_s, controls, save_path):
    """Control torques over time."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, controls[:, 0], color="tab:purple", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, controls[:, 1], color="tab:brown", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(controls[:, 0]), color="tab:purple", linewidth=2.5,
            label=r"$\tau_0$")
    ax.plot(time_s, _smooth_signal(controls[:, 1]), color="tab:brown", linewidth=2.5,
            label=r"$\tau_1$")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque (N m)")
    ax.set_title("Control Torque")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_error_norm(time_s, states, goals, save_path):
    """Position and velocity error norms over time."""
    pos_err = np.linalg.norm(states[:, :2] - goals[:, :2], axis=1)
    vel_err = np.linalg.norm(states[:, 2:] - goals[:, 2:], axis=1)
    total_err = np.linalg.norm(states - goals, axis=1)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, pos_err, color="tab:blue", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, vel_err, color="tab:red", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(pos_err), color="tab:blue", linewidth=2.5,
            label=r"$\|q - q^*\|$")
    ax.plot(time_s, _smooth_signal(vel_err), color="tab:red", linewidth=2.5,
            label=r"$\|\dot{q} - \dot{q}^*\|$")
    ax.plot(time_s, _smooth_signal(total_err), color="tab:green", linewidth=2.0,
            alpha=0.7, label=r"$\|x - x^*\|$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Error Norm")
    ax.set_title("Tracking Error")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_control_effort(time_s, controls, save_dir, prefix="plot"):
    """Control effort ||u|| over time — two separate square PDFs."""
    u_norm = np.linalg.norm(controls, axis=1)
    u_cumsum = np.cumsum(u_norm)

    # Instantaneous
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, u_norm, color="tab:purple", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(u_norm), color="tab:purple", linewidth=2.5,
            label=r"$\|u\|$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque Norm")
    ax.set_title("Instantaneous Control Effort")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}_effort_instant.pdf"), dpi=150, format="pdf")
    plt.close(fig)

    # Cumulative
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, u_cumsum, color="tab:orange", linewidth=2.0,
            label=r"$\sum \|u\|$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Cumulative Torque")
    ax.set_title("Cumulative Control Effort")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f"{prefix}_effort_cumulative.pdf"), dpi=150, format="pdf")
    plt.close(fig)


def plot_end_effector(states, goals, save_path, l1=0.5, l2=0.75):
    """End-effector trajectory in Cartesian space."""
    _, _, x_ee, y_ee = _end_effector_pos(states, l1, l2)
    _, _, gx_ee, gy_ee = _end_effector_pos(goals, l1, l2)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x_ee, y_ee, color="tab:blue", linewidth=1.5, alpha=0.7, label="Actual")
    ax.plot(x_ee[0], y_ee[0], "go", markersize=10, label="Start", zorder=5)
    ax.plot(x_ee[-1], y_ee[-1], "rs", markersize=10, label="End", zorder=5)
    ax.plot(gx_ee[-1], gy_ee[-1], "b*", markersize=15, label="Goal", zorder=5)
    # Draw goal region
    circle = plt.Circle((gx_ee[-1], gy_ee[-1]), 0.05, color="blue",
                         alpha=0.1, linestyle="--", fill=False, linewidth=1.5)
    ax.add_patch(circle)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("End-Effector Trajectory")
    ax.set_aspect("equal")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_phase_portrait(states, save_path):
    """Phase portrait: q vs dq for each joint."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.plot(states[:, 0], states[:, 2], color="tab:blue", linewidth=1.0, alpha=0.7)
    ax1.plot(states[0, 0], states[0, 2], "go", markersize=10, label="Start", zorder=5)
    ax1.plot(states[-1, 0], states[-1, 2], "rs", markersize=10, label="End", zorder=5)
    ax1.set_xlabel(r"$q_0$ (rad)")
    ax1.set_ylabel(r"$\dot{q}_0$ (rad/s)")
    ax1.set_title("Joint 0 Phase Portrait")
    ax1.legend(loc="best")
    ax1.grid(True, alpha=0.3)

    ax2.plot(states[:, 1], states[:, 3], color="tab:orange", linewidth=1.0, alpha=0.7)
    ax2.plot(states[0, 1], states[0, 3], "go", markersize=10, label="Start", zorder=5)
    ax2.plot(states[-1, 1], states[-1, 3], "rs", markersize=10, label="End", zorder=5)
    ax2.set_xlabel(r"$q_1$ (rad)")
    ax2.set_ylabel(r"$\dot{q}_1$ (rad/s)")
    ax2.set_title("Joint 1 Phase Portrait")
    ax2.legend(loc="best")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_energy(time_s, states, save_path):
    """Kinetic energy over time."""
    ke = _kinetic_energy(states)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, ke, color="tab:orange", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(ke), color="tab:orange", linewidth=2.5,
            label="Kinetic Energy")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (J)")
    ax.set_title("System Energy")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_reward(time_s, rewards, save_path):
    """Per-step reward over time."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(time_s, rewards, color="tab:cyan", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(rewards), color="tab:cyan", linewidth=2.5,
            label="Reward")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Reward")
    ax.set_title("Step Reward")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


def plot_summary_dashboard(time_s, states, goals, controls, save_path):
    """All-in-one dashboard with 6 subplots."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    # (0,0) Joint Position
    ax = axes[0, 0]
    ax.plot(time_s, states[:, 0], color="tab:blue", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, states[:, 1], color="tab:orange", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(states[:, 0]), color="tab:blue", linewidth=2.5,
            label=r"$q_0$")
    ax.plot(time_s, _smooth_signal(states[:, 1]), color="tab:orange", linewidth=2.5,
            label=r"$q_1$")
    ax.plot(time_s, goals[:, 0], "--", color="tab:blue", alpha=0.5, linewidth=1.2)
    ax.plot(time_s, goals[:, 1], "--", color="tab:orange", alpha=0.5, linewidth=1.2)
    ax.set_ylabel("Position (rad)")
    ax.set_title("Joint Position")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (0,1) Joint Velocity
    ax = axes[0, 1]
    ax.plot(time_s, states[:, 2], color="tab:green", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, states[:, 3], color="tab:red", linewidth=0.5, alpha=0.3)
    ax.plot(time_s, _smooth_signal(states[:, 2]), color="tab:green", linewidth=2.5,
            label=r"$\dot{q}_0$")
    ax.plot(time_s, _smooth_signal(states[:, 3]), color="tab:red", linewidth=2.5,
            label=r"$\dot{q}_1$")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_title("Joint Velocity")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,0) Control Torque
    ax = axes[1, 0]
    ctrl_t = time_s[:-1] if controls.shape[0] == time_s.shape[0] - 1 else time_s
    ax.plot(ctrl_t, controls[:, 0], color="tab:purple", linewidth=0.5, alpha=0.3)
    ax.plot(ctrl_t, controls[:, 1], color="tab:brown", linewidth=0.5, alpha=0.3)
    ax.plot(ctrl_t, _smooth_signal(controls[:, 0]), color="tab:purple", linewidth=2.5,
            label=r"$\tau_0$")
    ax.plot(ctrl_t, _smooth_signal(controls[:, 1]), color="tab:brown", linewidth=2.5,
            label=r"$\tau_1$")
    ax.set_ylabel("Torque (N m)")
    ax.set_title("Control Torque")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (1,1) Error Norm (log scale)
    ax = axes[1, 1]
    pos_err = np.linalg.norm(states[:, :2] - goals[:, :2], axis=1)
    vel_err = np.linalg.norm(states[:, 2:] - goals[:, 2:], axis=1)
    ax.plot(time_s, _smooth_signal(pos_err), color="tab:blue", linewidth=2.5,
            label=r"$\|q - q^*\|$")
    ax.plot(time_s, _smooth_signal(vel_err), color="tab:red", linewidth=2.5,
            label=r"$\|\dot{q} - \dot{q}^*\|$")
    ax.set_ylabel("Error Norm")
    ax.set_title("Tracking Error")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,0) End-Effector Trajectory
    ax = axes[2, 0]
    _, _, x_ee, y_ee = _end_effector_pos(states)
    _, _, gx_ee, gy_ee = _end_effector_pos(goals)
    ax.plot(x_ee, y_ee, color="tab:blue", linewidth=1.5, alpha=0.7, label="Actual")
    ax.plot(x_ee[0], y_ee[0], "go", markersize=8, label="Start", zorder=5)
    ax.plot(x_ee[-1], y_ee[-1], "rs", markersize=8, label="End", zorder=5)
    ax.plot(gx_ee[-1], gy_ee[-1], "b*", markersize=12, label="Goal", zorder=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("End-Effector Path")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    # (2,1) Control Effort
    ax = axes[2, 1]
    u_norm = np.linalg.norm(controls, axis=1)
    ax.plot(ctrl_t, u_norm, color="tab:purple", linewidth=0.5, alpha=0.3)
    ax.plot(ctrl_t, _smooth_signal(u_norm), color="tab:purple", linewidth=2.5,
            label=r"$\|u\|$")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Torque Norm")
    ax.set_title("Control Effort")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle("iLQR Two-Link Arm — Summary Dashboard", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, format="pdf")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate plots from iLQR results.")
    parser.add_argument("result_dir", type=str, help="Directory containing trajectory.npz")
    args = parser.parse_args()

    result_dir = args.result_dir
    traj_path = os.path.join(result_dir, "two_link_arm_ilqr_trajectory.npz")

    if not os.path.exists(traj_path):
        print(f"Error: {traj_path} not found.")
        sys.exit(1)

    data = np.load(traj_path)
    states = data["states"]
    goals = data["goals"]
    controls = data["controls"]
    rewards = data["rewards"] if "rewards" in data else None
    dt = 0.001  # TwoLinkArm default

    n_steps = states.shape[0]
    time_s = np.arange(n_steps, dtype=float) * dt
    control_time = time_s[:-1] if controls.shape[0] == n_steps - 1 else time_s

    print(f"Loaded {n_steps} states, {controls.shape[0]} controls from {traj_path}")
    print(f"Generating plots in {result_dir}/ ...")

    # Generate all plots
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
    if rewards is not None:
        reward_time = time_s[:-1] if rewards.shape[0] == n_steps - 1 else time_s[:rewards.shape[0]]
        plot_reward(reward_time, rewards,
                    os.path.join(result_dir, "plot_reward.pdf"))

    print("Done! Generated plots:")
    for f in sorted(os.listdir(result_dir)):
        if f.endswith(".pdf"):
            print(f"  {f}")


if __name__ == "__main__":
    main()
