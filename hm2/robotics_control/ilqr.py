"""LQR, iLQR and MPC.

Optimized implementation that uses standalone numpy dynamics functions
instead of gym environment calls for the inner iLQR loop.  The gym env
is still used at the top level for reset/step/render.

Homework cost function:
  - Intermediate: l(x,u) = ||u||^2  (no state tracking)
  - Terminal:     l_f(x) = 10^4 * ||x - x*||^2
"""

from __future__ import annotations

import sys
import time
import numpy as np


# ---------------------------------------------------------------------------
# Standalone vectorized dynamics (replicates arm_env._step without gym)
# ---------------------------------------------------------------------------

def _dynamics_step(x, u, dt, K1, K2, K3, K4):
    """One Euler step of the 2-link arm dynamics.

    Works on single states (shape (4,)) or batches (shape (..., 4)).
    """
    q1 = x[..., 1]
    C2 = np.cos(q1)
    S2 = np.sin(q1)

    M11 = K1 + K2 * C2
    M12 = K3 + K4 * C2
    M21 = M12
    M22 = K3

    dq0 = x[..., 2]
    dq1 = x[..., 3]

    H1 = -K2 * S2 * dq0 * dq1 - 0.5 * K2 * S2 * dq1 ** 2
    H2 = 0.5 * K2 * S2 * dq0 ** 2

    denom = M12 ** 2 - M11 * M22
    ddq1 = (H2 * M11 - H1 * M21 - M11 * u[..., 1] + M21 * u[..., 0]) / denom
    ddq0 = (-H2 + u[..., 1] - M22 * ddq1) / M21

    dq_new0 = dq0 + ddq0 * dt
    dq_new1 = dq1 + ddq1 * dt
    q_new0 = x[..., 0] + dq_new0 * dt
    q_new1 = q1 + dq_new1 * dt

    return np.stack([q_new0, q_new1, dq_new0, dq_new1], axis=-1)


def _dynamics_rollout(x0, U, dt, K1, K2, K3, K4):
    """Forward rollout from x0 using control sequence U.

    Returns X of shape (tN+1, 4).
    """
    tN = U.shape[0]
    n = x0.shape[-1]
    X = np.empty((tN + 1, n))
    X[0] = x0
    for k in range(tN):
        X[k + 1] = _dynamics_step(X[k], U[k], dt, K1, K2, K3, K4)
    return X


def _dynamics_rollout_feedback(x0, K_gains, k_ff, X_nom, U_nom, dt,
                                K1, K2, K3, K4, alpha):
    """Line-search rollout with feedback gains.

    U_new[k] = U_nom[k] + alpha * k_ff[k] + K_gains[k] @ (X_new[k] - X_nom[k])

    Returns (X_new, U_new).
    """
    tN = U_nom.shape[0]
    n = x0.shape[-1]
    X_new = np.empty((tN + 1, n))
    U_new = np.empty_like(U_nom)
    X_new[0] = x0
    for k in range(tN):
        dx = X_new[k] - X_nom[k]
        U_new[k] = U_nom[k] + alpha * k_ff[k] + K_gains[k] @ dx
        X_new[k + 1] = _dynamics_step(X_new[k], U_new[k], dt, K1, K2, K3, K4)
    return X_new, U_new


# ---------------------------------------------------------------------------
# Vectorized Jacobian computation via central finite differences
# ---------------------------------------------------------------------------

def _approximate_AB_batch(X, U, dt, K1, K2, K3, K4, delta=1e-5):
    """Compute A_k = df/dx and B_k = df/du for all timesteps at once.

    Returns A shape (tN, 4, 4) and B shape (tN, 4, 2).
    """
    tN = U.shape[0]
    n = X.shape[1]
    m = U.shape[1]
    X_km1 = X[:-1]  # (tN, 4)

    A = np.empty((tN, n, n))
    for j in range(n):
        e = np.zeros(n)
        e[j] = delta
        x_p = _dynamics_step(X_km1 + e, U, dt, K1, K2, K3, K4)
        x_m = _dynamics_step(X_km1 - e, U, dt, K1, K2, K3, K4)
        A[:, :, j] = (x_p - x_m) / (2.0 * delta)

    B = np.empty((tN, n, m))
    for j in range(m):
        e = np.zeros(m)
        e[j] = delta
        x_p = _dynamics_step(X_km1, U + e, dt, K1, K2, K3, K4)
        x_m = _dynamics_step(X_km1, U - e, dt, K1, K2, K3, K4)
        B[:, :, j] = (x_p - x_m) / (2.0 * delta)

    return A, B


# ---------------------------------------------------------------------------
# Cost functions
# ---------------------------------------------------------------------------

def _cost_intermediate_batch(U, R):
    """Intermediate cost for all timesteps: l(x,u) = u^T R u.

    Returns costs (tN,), l_u (tN, m), l_uu (tN, m, m).
    l_x, l_xx, l_ux are zero (no state dependence).
    """
    # u^T R u for each timestep
    Ru = U @ R  # (tN, m)
    costs = 0.5 * np.sum(U * Ru, axis=1)  # (tN,)
    return costs, Ru, np.tile(R, (U.shape[0], 1, 1))


def _cost_terminal(x_last, x_goal, Q_terminal):
    """Terminal cost: 0.5 * (x - x*)^T Q_f (x - x*).

    Returns scalar cost, gradient (n,), Hessian (n, n).
    """
    dx = x_last - x_goal
    cost = 0.5 * dx @ Q_terminal @ dx
    l_f_x = Q_terminal @ dx
    return cost, l_f_x, Q_terminal


def _compute_trajectory_cost(X, U, R, Q_terminal, x_goal):
    """Total trajectory cost: sum of intermediate + terminal."""
    inter_costs, _, _ = _cost_intermediate_batch(U, R)
    term_cost, _, _ = _cost_terminal(X[-1], x_goal, Q_terminal)
    return float(np.sum(inter_costs) + term_cost)


# ---------------------------------------------------------------------------
# Public helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------

def _goal_vector(env) -> np.ndarray:
    """Stacked goal ``[q_goal, dq_goal]`` (matches ``arm_env.goal``)."""
    core = getattr(env, "unwrapped", env)
    return np.asarray(core.goal, dtype=float).ravel()


def clip_action(env, u):
    """Clip ``u`` to ``env.action_space`` (finite limits or +-inf)."""
    u = np.asarray(u, dtype=float)
    core = getattr(env, "unwrapped", env)
    space = core.action_space
    low = np.asarray(space.low, dtype=float)
    high = np.asarray(space.high, dtype=float)
    if u.ndim == 2:
        return np.clip(u, low[None, :], high[None, :])
    return np.clip(u.ravel(), low.ravel(), high.ravel())


# ---------------------------------------------------------------------------
# Main iLQR solver
# ---------------------------------------------------------------------------

def calc_ilqr_input(
    env,
    sim_env,
    tN=50,
    max_iter=100,
    tol=1e-2,
    reg=1e-2,
    line_search_alphas=None,
    r_scale=0.01,
    velocity_q_scale=1.0,
    u_init=None,
    x_ref=None,
    use_smooth_reference: bool = True,
    v_max=None,
    a_max=None,
):
    """Iterative LQR: returns a torque sequence U of length tN.

    Optimized version: uses standalone numpy dynamics instead of gym env
    calls for the inner loop.  The gym env is only used to extract
    parameters (K1-K4, dt, Q, R, goal) and for action clipping.

    Parameters
    ----------
    env, sim_env
        TwoLinkArmEnv (or compatible API).  ``sim_env`` is kept for
        backward compatibility but is no longer used internally.
    tN: int
        Horizon (number of controls).
    max_iter: int
        Maximum iLQR outer iterations.
    tol: float
        Stop when cost improvement falls below this threshold.
    reg: float
        Tikhonov term on Q_uu for stable inversion.
    line_search_alphas: iterable of float, optional
        Backtracking coefficients applied to feedforward k.
    r_scale: float
        Multiplier on R in the planner (unused in homework config).
    velocity_q_scale: float
        Multiplier on velocity block of Q (unused in homework config).
    u_init: np.ndarray, optional
        Warm-start control sequence, shape (tN, m).
    x_ref: np.ndarray, optional
        Reference state trajectory, shape (tN+1, state_dim).
    use_smooth_reference: bool
        If True and x_ref not given, build minimum-jerk reference.
    v_max, a_max: unused, kept for API compatibility.

    Returns
    -------
    U: np.ndarray, shape (tN, action_dim)
    """
    core = getattr(env, "unwrapped", env)
    n = env.observation_space.shape[0]  # state dim = 4
    m = env.action_space.shape[0]       # action dim = 2

    # Extract dynamics parameters from env
    K1, K2, K3, K4 = core.K1, core.K2, core.K3, core.K4
    dt = core.dt

    x0 = np.asarray(core.state, dtype=float).ravel()
    x_goal = _goal_vector(core)

    # ------------------------------------------------------------------
    # Cost matrices (homework specification)
    # ------------------------------------------------------------------
    # Intermediate cost: ||u||^2 only  (no state tracking)
    R_planner = np.eye(m) * r_scale
    # Terminal cost: 10^4 * ||x - x*||^2
    Q_terminal = np.eye(n) * 1e4

    # ------------------------------------------------------------------
    # Build reference trajectory
    # ------------------------------------------------------------------
    if x_ref is not None:
        X_ref = np.asarray(x_ref, dtype=float)
    elif use_smooth_reference:
        t_arr = np.linspace(0, 1, tN + 1)
        dx = x_goal[:core.DOF] - x0[:core.DOF]
        s = 10 * t_arr**3 - 15 * t_arr**4 + 6 * t_arr**5
        ds = 30 * t_arr**2 - 60 * t_arr**3 + 30 * t_arr**4
        X_ref = np.zeros((tN + 1, n))
        for i in range(core.DOF):
            X_ref[:, i] = x0[i] + dx[i] * s
            X_ref[:, core.DOF + i] = dx[i] * ds / (tN * core.dt)
    else:
        X_ref = np.tile(x_goal, (tN + 1, 1))

    # ------------------------------------------------------------------
    # Initialize control sequence
    # ------------------------------------------------------------------
    if u_init is not None:
        U = clip_action(env, np.asarray(u_init, dtype=float))
    else:
        U = np.zeros((tN, m))

    if line_search_alphas is None:
        line_search_alphas = 0.5 ** np.arange(10)

    current_reg = reg
    print_interval = max(1, max_iter // 20)

    # ------------------------------------------------------------------
    # Main iLQR loop
    # ------------------------------------------------------------------
    for iteration in range(max_iter):
        t_start = time.time()

        # Forward rollout (vectorized, no gym calls)
        X = _dynamics_rollout(x0, U, dt, K1, K2, K3, K4)
        J = _compute_trajectory_cost(X, U, R_planner, Q_terminal, x_goal)

        # Linearize dynamics (vectorized)
        A_arr, B_arr = _approximate_AB_batch(X, U, dt, K1, K2, K3, K4)

        # Cost gradients (homework: ||u||^2 only, no state cost)
        l_u_arr = U @ R_planner           # (tN, m)
        l_uu_arr = np.tile(R_planner, (tN, 1, 1))  # (tN, m, m)
        # l_x, l_xx, l_ux are all zero -- no state cost at intermediate steps

        # Terminal cost gradients
        _, l_f_x, l_f_xx = _cost_terminal(X[tN], x_goal, Q_terminal)

        # Backward pass
        V_x = l_f_x
        V_xx = l_f_xx

        K_list = []
        k_list = []
        expected_decrease = 0.0

        backward_success = True
        for k in range(tN - 1, -1, -1):
            A_k = A_arr[k]
            B_k = B_arr[k]

            Q_x = A_k.T @ V_x                        # l_x = 0
            Q_u = l_u_arr[k] + B_k.T @ V_x
            Q_xx = A_k.T @ V_xx @ A_k                 # l_xx = 0
            Q_uu = l_uu_arr[k] + B_k.T @ V_xx @ B_k
            Q_ux = B_k.T @ V_xx @ A_k                 # l_ux = 0

            Q_uu_reg = Q_uu + current_reg * np.eye(m)

            try:
                L = np.linalg.cholesky(Q_uu_reg)
                K = -np.linalg.solve(L.T, np.linalg.solve(L, Q_ux))
                k_ff = -np.linalg.solve(L.T, np.linalg.solve(L, Q_u))
            except np.linalg.LinAlgError:
                backward_success = False
                break

            K_list.insert(0, K)
            k_list.insert(0, k_ff)

            expected_decrease += float(k_ff @ Q_u)

            V_x = Q_x + K.T @ Q_uu @ k_ff + K.T @ Q_u + Q_ux.T @ k_ff
            V_xx = Q_xx + K.T @ Q_uu @ K + K.T @ Q_ux + Q_ux.T @ K
            V_xx = 0.5 * (V_xx + V_xx.T)

        if not backward_success:
            current_reg = max(current_reg * 2.0, 1e-8)
            continue

        # Line search
        best_J = J
        best_U = U.copy()
        improved = False

        for alpha in line_search_alphas:
            X_new, U_new = _dynamics_rollout_feedback(
                x0, K_list, k_list, X, U, dt, K1, K2, K3, K4, alpha
            )
            J_new = _compute_trajectory_cost(
                X_new, U_new, R_planner, Q_terminal, x_goal
            )

            if J_new < best_J:
                best_J = J_new
                best_U = U_new.copy()
                improved = True
                break

        if improved:
            U = best_U
            current_reg = max(current_reg * 0.5, 1e-10)

            if abs(J - best_J) < tol:
                # print(f"  [iLQR] Converged at iter {iteration}: "
                    #   f"dJ={abs(J - best_J):.6f} < tol={tol}")
                break
        else:
            current_reg = min(current_reg * 2.0, 1e4)
            if current_reg >= 1e4:
                # print(f"  [iLQR] Regularization saturated at iter {iteration}")
                break

        # Progress logging
        if (iteration + 1) % print_interval == 0 or iteration == 0:
            elapsed = time.time() - t_start
            dJ = abs(J - best_J) if improved else float('inf')
            x_err = np.linalg.norm(X[tN][:core.DOF] - x_goal[:core.DOF])
            # print(f"  [iLQR] iter {iteration+1}/{max_iter} | "
            #       f"J={J:.2f} | dJ={dJ:.6f} | reg={current_reg:.2e} | "
            #       f"x_err={x_err:.4f} | {elapsed:.2f}s")
            sys.stdout.flush()

    return U
