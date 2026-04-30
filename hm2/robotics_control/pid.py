"""
PID controller scaffold for CartPole homework.

File description
----------------
- `PIDController` is the minimal interface used by the runner script.
- Students are expected to implement a concrete controller in this file for
  `UprightStabilizeCartPoleEnv` from `robotics_control/cartpole.py`.

How to build a good PID for UprightStabilizeCartPoleEnv
--------------------------------------------------------
1. Parse observation as `[x, x_dot, theta, theta_dot]`.
2. Use `theta` (upright angle error around zero) as the primary signal.
3. Add cart centering terms (`x`, `x_dot`) as secondary stabilization.
4. Include integral anti-windup (clamp integral term).
5. Limit commanded force and map to discrete action:
   - force >= 0 -> action 1 (push right)
   - force < 0 -> action 0 (push left)
6. Optionally add hysteresis/deadband near zero force to reduce action
   chattering under the environment's discrete action space.

Recommended tuning order (practical)
------------------------------------
1. Start with PD only on angle: tune `kp_theta`, then `kd_theta`.
2. Add small `kp_x`, `kd_x` to keep cart near center.
3. Add tiny integral terms only if steady-state drift remains.
4. Validate over many random resets, not one trajectory.

You can also use other population-based or heristic search algorithms to tune the parameters. If you do so, please document your changes clearly in your submission.
"""

from __future__ import annotations

from abc import ABC

import numpy as np


class PIDController(ABC):
    """Generic PID controller interface."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def reset(self) -> None:
        """Reset integral/derivative memory."""
        return None

    def compute(self, observation: np.ndarray, dt: float) -> int:
        """Return control action for current observation."""
        return 0


class CartPolePIDController(PIDController):
    """
    Dual-loop PID controller for UprightStabilizeCartPoleEnv.

    Primary loop: PD on pole angle theta (error = theta - 0).
    Secondary loop: PD on cart position x (error = x - 0).
    Optional integral terms with anti-windup clamping.
    Hysteresis deadband reduces action chattering near zero force.
    """

    def __init__(
        self,
        theta_kp: float = 50.0,
        theta_kd: float = 20.0,
        theta_ki: float = 0.0,
        x_kp: float = 2.0,
        x_kd: float = 3.0,
        x_ki: float = 0.0,
        force_limit: float = 10.0,
        integral_limit: float = 5.0,
        hysteresis: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.theta_kp = theta_kp
        self.theta_kd = theta_kd
        self.theta_ki = theta_ki
        self.x_kp = x_kp
        self.x_kd = x_kd
        self.x_ki = x_ki
        self.force_limit = force_limit
        self.integral_limit = integral_limit
        self.hysteresis = hysteresis

        self._theta_integral = 0.0
        self._x_integral = 0.0
        self._prev_theta = 0.0
        self._prev_x = 0.0
        self._initialized = False
        self._last_force = 0.0

    def reset(self) -> None:
        """Clear all integrator and derivative memory."""
        self._theta_integral = 0.0
        self._x_integral = 0.0
        self._prev_theta = 0.0
        self._prev_x = 0.0
        self._initialized = False
        self._last_force = 0.0

    def compute(self, observation: np.ndarray, dt: float) -> int:
        """
        Compute discrete action from observation.

        Args:
            observation: [x, x_dot, theta, theta_dot].
            dt: Timestep in seconds.

        Returns:
            0 for push left, 1 for push right.
        """
        x, x_dot, theta, theta_dot = observation[:4]

        if not self._initialized:
            self._prev_theta = theta
            self._prev_x = x
            self._initialized = True

        # Angle PD + I
        theta_error = theta
        theta_deriv = (theta - self._prev_theta) / dt if dt > 0 else 0.0
        self._theta_integral += theta_error * dt
        self._theta_integral = np.clip(
            self._theta_integral, -self.integral_limit, self.integral_limit
        )
        f_angle = (
            self.theta_kp * theta_error
            + self.theta_kd * theta_dot
            + self.theta_ki * self._theta_integral
        )

        # Cart PD + I
        x_error = x
        x_deriv = (x - self._prev_x) / dt if dt > 0 else 0.0
        self._x_integral += x_error * dt
        self._x_integral = np.clip(
            self._x_integral, -self.integral_limit, self.integral_limit
        )
        f_cart = (
            self.x_kp * x_error
            + self.x_kd * x_dot
            + self.x_ki * self._x_integral
        )

        # Total force: angle stabilization takes priority; cart centering is secondary
        force = f_angle + f_cart

        # Hysteresis deadband to reduce chattering
        if abs(force) < self.hysteresis:
            force = 0.0

        # Clip to force limit
        force = np.clip(force, -self.force_limit, self.force_limit)

        self._prev_theta = theta
        self._prev_x = x
        self._last_force = float(force)

        # Map to discrete action: right (1) if force >= 0, left (0) otherwise
        return 1 if force >= 0 else 0
