"""Matplotlib-based renderer for TwoLinkArm environment."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D


def render_arm_frame(q, goal_q, l1=0.5, l2=0.75, figsize=(4, 4)):
    """Render a single frame of the two-link arm using matplotlib.

    Parameters
    ----------
    q : array-like, shape (2,)
        Current joint angles [q0, q1].
    goal_q : array-like, shape (2,)
        Goal joint angles.
    l1, l2 : float
        Link lengths.
    figsize : tuple
        Figure size.

    Returns
    -------
    frame : np.ndarray, shape (size, size, 3), dtype uint8
        RGB array of the rendered frame.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    max_len = l1 + l2
    bound = max_len * 1.5
    ax.set_xlim(-bound, bound)
    ax.set_ylim(-bound, bound)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Goal arm (transparent)
    gx1 = l1 * np.cos(goal_q[0])
    gy1 = l1 * np.sin(goal_q[0])
    gx2 = gx1 + l2 * np.cos(goal_q[0] + goal_q[1])
    gy2 = gy1 + l2 * np.sin(goal_q[0] + goal_q[1])
    ax.plot([0, gx1], [0, gy1], 'b-', linewidth=4, alpha=0.25)
    ax.plot([gx1, gx2], [gy1, gy2], 'b-', linewidth=3, alpha=0.25)

    # Current arm
    x1 = l1 * np.cos(q[0])
    y1 = l1 * np.sin(q[0])
    x2 = x1 + l2 * np.cos(q[0] + q[1])
    y2 = y1 + l2 * np.sin(q[0] + q[1])
    ax.plot([0, x1], [0, y1], 'r-', linewidth=4)
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3)

    # Joints
    ax.plot(0, 0, 'ko', markersize=8)
    ax.plot(x1, y1, 'ko', markersize=6)
    ax.plot(x2, y2, 'go', markersize=8)

    # Goal end effector
    ax.plot(gx2, gy2, 'b*', markersize=10, alpha=0.5)

    # Convert to RGB array
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    frame = np.frombuffer(renderer.tostring_rgb(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)

    return frame


def create_arm_video(states, goals, video_path, fps=30, l1=0.5, l2=0.75):
    """Create a video from arm trajectory using matplotlib.

    Parameters
    ----------
    states : np.ndarray, shape (T, 4)
        State trajectory [q0, q1, dq0, dq1].
    goals : np.ndarray, shape (T, 4)
        Goal trajectory.
    video_path : str
        Path to save video.
    fps : int
        Frames per second.
    l1, l2 : float
        Link lengths.
    """
    import imageio.v2 as imageio

    frames = []
    stride = max(1, int(round(1000 / fps / (1000 * states[0].shape[0] if False else 1))))

    for i in range(0, len(states), max(1, len(states) // 200)):  # ~200 frames max
        q = states[i][:2]
        goal_q = goals[i][:2]
        frame = render_arm_frame(q, goal_q, l1=l1, l2=l2)
        frames.append(frame)

    if frames:
        imageio.mimsave(video_path, frames, fps=fps)
        return True
    return False
