"""LQR, iLQR and MPC.

Costs and goals follow ``arm_env.TwoLinkArmEnv``: stage cost penalizes
``x - goal`` with ``env.Q`` and ``u`` with ``env.R`` (same structure as the
environment reward, up to the conventional ``0.5`` factor used for smooth
derivatives in iLQR).

Dynamics rollouts use ``env.step(u, dt)[0]`` after setting ``env.state = x``,
so Jacobians are taken w.r.t. the **discrete** one-step map
``x_{k+1} = f(x_k, u_k)``.

Torques are projected with ``clip_action`` to ``env.action_space`` everywhere
(open-loop, line search, and dynamics) so planning matches
``LimitedTorqueTwoLinkArmEnv``. Optional ``r_scale`` / ``velocity_q_scale``
inflate ``R`` and the velocity block of ``Q`` inside the planner only, to reduce
chatter near the goal.

By default ``calc_ilqr_input`` tracks a **minimum-jerk** joint-space reference
from the current state to ``env.goal`` over the horizon (``use_smooth_reference``).
Optional ``v_max`` / ``a_max`` lengthen that reference so analytic peak joint
speed and acceleration respect bounds. Pass ``x_ref`` explicitly or set
``use_smooth_reference=False`` to track the fixed goal only.
"""

from __future__ import annotations

import numpy as np


def _goal_vector(env) -> np.ndarray:
    """Stacked goal ``[q_goal, dq_goal]`` (matches ``arm_env.goal``)."""
    return np.asarray(env.goal, dtype=float).ravel()


def _action_bounds(env):
    """Return ``(low, high)`` arrays for torque clipping (``LimitedTorque`` etc.)."""
    core = getattr(env, "unwrapped", env)
    space = core.action_space
    low = np.asarray(space.low, dtype=float).ravel()
    high = np.asarray(space.high, dtype=float).ravel()
    return low, high


def clip_action(env, u):
    """Clip ``u`` to ``env.action_space`` (finite limits or ±inf)."""
    u = np.asarray(u, dtype=float).ravel()
    low, high = _action_bounds(env)
    return np.clip(u, low, high)


def simulate_dynamics_next(env, x, u, dt=None):
    """One discrete step of the arm (same map as ``env.step`` on state ``x``).

    Parameters
    ----------
    env: gym.core.Env
        Must expose ``state``, ``step``, and ``dt`` like ``TwoLinkArmEnv``.
    x: np.ndarray
        State ``[q, dq]`` before the step.
    u: np.ndarray
        Torque command.
    dt: float, optional
        Integration step; defaults to ``env.dt``.

    Returns
    -------
    next_x: np.ndarray
        State after applying ``u`` for one step.
    """
    return np.asarray(next_x, dtype=float).ravel()


def approximate_A(env, x, u, delta=1e-5, dt=1e-5):
    """Columns are :math:`\\partial \\dot{x} / \\partial x_j` at ``(x, u)``."""
    return A


def approximate_B(env, x, u, delta=1e-5, dt=1e-5):
    """Columns are :math:`\\partial \\dot{x} / \\partial u_j` at ``(x, u)``."""
    return B


def cost_inter(cost_env, x, u, Q=None, R=None, x_ref=None):
    """Intermediate cost: tracking + control penalty.

    ``l(x,u) = 0.5 * (x-x_ref)^T Q (x-x_ref) + 0.5 * u^T R u``

    If ``x_ref`` is ``None``, uses ``env.goal`` as the setpoint (original
    behaviour).
    """
    return cost, l_x, l_xx, l_u, l_uu, l_ux


def cost_final(cost_env, x, Q=None, x_ref=None):
    """Terminal cost ``0.5 * (x-x_ref)^T Q (x-x_ref)``.

    If ``x_ref`` is ``None``, uses ``env.goal``.
    """
    return cost, l_x, l_xx


def simulate(cost_env, sim_env, x0, U, Q=None, R=None, X_ref=None, x_terminal=None):
    """Roll out open-loop ``U`` from ``x0`` on ``sim_env``; return cost and trajectory.

    Parameters
    ----------
    cost_env: gym.core.Env
        Used for ``Q``, ``R``, and setpoints in cost.
    sim_env: gym.core.Env
        Used for dynamics (should be noise-free / separate instance).
    X_ref: np.ndarray, optional
        If given, shape ``(tN+1, n)``; stage ``k`` uses ``X_ref[k]``.
    x_terminal: np.ndarray, optional
        Terminal setpoint; if ``None`` and ``X_ref`` is set, uses ``X_ref[-1]``;
        if both ``None``, uses ``env.goal``.
    x0: np.ndarray
    U: np.ndarray, shape ``(tN, action_dim)``

    Returns
    -------
    total_cost: float
    X: np.ndarray, shape ``(tN+1, state_dim)``
    """
    return 0, np.zeros(2)


def calc_ilqr_input(
    env,
    sim_env,
    tN=50,
    max_iter=100,
    tol=1e-2,
    reg=1e-2,
    line_search_alphas=None,
    r_scale=6.0,
    velocity_q_scale=18.0,
    u_init=None,
    x_ref=None,
    use_smooth_reference: bool = True,
    v_max=None,
    a_max=None,
):
    """Iterative LQR: returns a torque sequence ``U`` of length ``tN``.

    Uses ``env.state`` as the initial ``x0``, and ``env`` for cost matrices /
    goal. All dynamics Jacobians and rollouts use ``sim_env`` so the real
    ``env`` is not mutated during planning (copy ``sim_env`` first).

    Parameters
    ----------
    env, sim_env: gym.core.Env
        ``TwoLinkArmEnv`` (or compatible API).
    tN: int
        Horizon (number of controls).
    max_iter: int
        Maximum iLQR outer iterations.
    tol: float
        Stop when cost improvement falls below this threshold.
    reg: float
        Tikhonov term on ``Q_uu`` for stable inversion (larger → smoother gains).
    line_search_alphas: iterable of float, optional
        Backtracking coefficients applied to feedforward ``k``.
    r_scale: float
        Multiplier on ``env.R`` in the planner (``>1`` discourages large torques).
    velocity_q_scale: float
        Multiplier on the velocity block of ``Q`` (``>1`` damps endpoint oscillation).
    u_init: np.ndarray, optional
        Warm-start control sequence, shape ``(tN, m)`` (will be clipped).
    x_ref: np.ndarray, optional
        Reference state trajectory, shape ``(tN+1, state_dim)``. Overrides
        ``use_smooth_reference`` when given.
    use_smooth_reference: bool
        If True (default) and ``x_ref`` is not given, build a minimum-jerk
        joint-space reference from ``x0`` to ``env.goal`` over the horizon.
    v_max: float or array, optional
        Per-joint max ``|dq|`` (rad/s) for the built-in reference. ``None`` =
        unconstrained.
    a_max: float or array, optional
        Per-joint max ``|ddot{q}|`` (rad/s²) for the built-in reference.

    Returns
    -------
    U: np.ndarray, shape ``(tN, action_dim)``
    """
    U = np.zeros(2)

    return U
