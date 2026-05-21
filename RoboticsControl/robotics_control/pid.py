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
