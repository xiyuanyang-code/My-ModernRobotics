"""CartPole helpers: env factory from expert config and PID interface."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Tuple

import gym
import numpy as np


@dataclass(frozen=True)
class CartPoleEnvSpec:
    """Minimal CartPole spec inferred from expert model config."""

    env_id: str
    obs_dim: int
    action_dim: int


def _infer_dims_from_keras_yaml(config_path: str | Path) -> Tuple[int, int]:
    """Return ``(obs_dim, action_dim)`` from a Keras YAML model config file."""
    path = Path(config_path)
    content = path.read_text(encoding="utf-8")

    # Legacy expert config stores input shape as:
    # batch_input_shape: !!python/tuple [null, 4]
    obs_match = re.search(
        r"batch_input_shape:\s*!!python/tuple\s*\[\s*(?:null|None)\s*,\s*(\d+)\s*\]",
        content,
    )
    if obs_match is None:
        raise ValueError(f"Cannot infer observation dim from config: {config_path}")
    obs_dim = int(obs_match.group(1))

    unit_matches = re.findall(r"^\s*units:\s*(\d+)\s*$", content, flags=re.MULTILINE)
    if not unit_matches:
        raise ValueError(f"Cannot infer action dim from config: {config_path}")
    action_dim = int(unit_matches[-1])

    return obs_dim, action_dim


def build_cartpole_env_spec(
    config_path: str | Path, env_id: str = "CartPole-v0"
) -> CartPoleEnvSpec:
    """Build an env spec by reading expert model YAML."""
    obs_dim, action_dim = _infer_dims_from_keras_yaml(config_path)
    return CartPoleEnvSpec(env_id=env_id, obs_dim=obs_dim, action_dim=action_dim)


def make_cartpole_env_from_config(
    config_path: str | Path, env_id: str = "CartPole-v0", **gym_kwargs
):
    """Create and validate CartPole env using model config dimensions."""
    spec = build_cartpole_env_spec(config_path=config_path, env_id=env_id)
    env = gym.make(spec.env_id, **gym_kwargs)
    if int(env.observation_space.shape[0]) != spec.obs_dim:
        raise ValueError(
            f"Observation dim mismatch: env={env.observation_space.shape[0]} vs config={spec.obs_dim}"
        )
    if int(env.action_space.n) != spec.action_dim:
        raise ValueError(
            f"Action dim mismatch: env={env.action_space.n} vs config={spec.action_dim}"
        )
    return env


def sample_random_cartpole_state_params(
    seed: int | None = None,
    x_range: Tuple[float, float] = (-0.05, 0.05),
    x_dot_range: Tuple[float, float] = (-0.05, 0.05),
    theta_range: Tuple[float, float] = (-0.05, 0.05),
    theta_dot_range: Tuple[float, float] = (-0.05, 0.05),
):
    """Sample random initial state parameters for CartPole."""
    rng = np.random.default_rng(seed)
    return {
        "x": float(rng.uniform(*x_range)),
        "x_dot": float(rng.uniform(*x_dot_range)),
        "theta": float(rng.uniform(*theta_range)),
        "theta_dot": float(rng.uniform(*theta_dot_range)),
    }


def reset_cartpole_with_random_state(
    env,
    seed: int | None = None,
    x_range: Tuple[float, float] = (-0.05, 0.05),
    x_dot_range: Tuple[float, float] = (-0.05, 0.05),
    theta_range: Tuple[float, float] = (-0.05, 0.05),
    theta_dot_range: Tuple[float, float] = (-0.05, 0.05),
) -> np.ndarray:
    """Reset env and overwrite state with random sampled values."""
    try:
        env.reset(seed=seed)
    except TypeError:
        env.reset()
    params = sample_random_cartpole_state_params(
        seed=seed,
        x_range=x_range,
        x_dot_range=x_dot_range,
        theta_range=theta_range,
        theta_dot_range=theta_dot_range,
    )
    state = np.array(
        [params["x"], params["x_dot"], params["theta"], params["theta_dot"]],
        dtype=float,
    )
    env.unwrapped.state = state.copy()
    return state


class SwingUpStabilizeCartPoleEnv(gym.Env):
    """CartPole swing-up environment with low-motion stabilization objective."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        max_episode_steps: int = 1000,
        x_threshold: float = 2.4,
        force_mag: float = 10.0,
        tau: float = 0.02,
        down_theta_noise: float = 0.08,
        init_x_range: Tuple[float, float] = (-0.05, 0.05),
        init_x_dot_range: Tuple[float, float] = (-0.05, 0.05),
        init_theta_dot_range: Tuple[float, float] = (-0.05, 0.05),
    ) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = float(force_mag)
        self.tau = float(tau)

        self.max_episode_steps = int(max_episode_steps)
        self.x_threshold = float(x_threshold)
        self.theta_limit = 4.0 * math.pi
        self.down_theta_noise = float(down_theta_noise)
        self.init_x_range = init_x_range
        self.init_x_dot_range = init_x_dot_range
        self.init_theta_dot_range = init_theta_dot_range

        high = np.array(
            [
                self.x_threshold * 2.0,
                np.finfo(np.float32).max,
                self.theta_limit,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(-high, high, dtype=np.float32)

        self.state = np.zeros(4, dtype=np.float64)
        self.steps = 0
        self.np_random, _ = gym.utils.seeding.np_random(None)

    @staticmethod
    def _angle_normalize(theta: float) -> float:
        return ((theta + math.pi) % (2.0 * math.pi)) - math.pi

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self.seed(seed)
        x = self.np_random.uniform(*self.init_x_range)
        x_dot = self.np_random.uniform(*self.init_x_dot_range)
        theta = math.pi + self.np_random.uniform(
            -self.down_theta_noise, self.down_theta_noise
        )
        theta_dot = self.np_random.uniform(*self.init_theta_dot_range)
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps = 0
        return self.state.astype(np.float32)

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if int(action) == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * costheta * costheta / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps += 1

        theta_err = self._angle_normalize(theta)
        # Reward = upright progress + low-motion stabilization.
        upright_reward = math.cos(theta_err)
        stabilization_penalty = (
            0.5 * theta_err * theta_err
            + 0.2 * theta_dot * theta_dot
            + 0.05 * x * x
            + 0.1 * x_dot * x_dot
        )
        reward = upright_reward - stabilization_penalty
        if (
            abs(theta_err) < math.radians(10.0)
            and abs(theta_dot) < 0.4
            and abs(x_dot) < 0.15
            and abs(x) < 0.5
        ):
            reward += 2.0

        done = bool(abs(x) > self.x_threshold or self.steps >= self.max_episode_steps)
        info = {
            "theta_error": float(theta_err),
            "x": float(x),
            "x_dot": float(x_dot),
            "theta_dot": float(theta_dot),
            "is_stable_upright": bool(
                abs(theta_err) < math.radians(10.0) and abs(theta_dot) < 0.4
            ),
        }
        return self.state.astype(np.float32), float(reward), done, info

    def render(self, mode="human"):
        # Reuse Gym CartPole renderer by syncing temporary env state.
        from gym.envs.classic_control.cartpole import CartPoleEnv

        if not hasattr(self, "_render_env"):
            self._render_env = CartPoleEnv()
        self._render_env.state = self.state.copy()
        return self._render_env.render(mode=mode)

    def close(self):
        if hasattr(self, "_render_env"):
            self._render_env.close()
            del self._render_env


class UprightStabilizeCartPoleEnv(SwingUpStabilizeCartPoleEnv):
    """
    CartPole stabilization task with near-upright random initialization.

    Task implementation notes (for PID homework)
    --------------------------------------------
    Goal:
    - Keep pole near upright (`theta ~= 0`) while minimizing cart motion and
      surviving as many steps as possible.

    Observation semantics:
    - `state[0] = x`          (cart position)
    - `state[1] = x_dot`      (cart velocity)
    - `state[2] = theta`      (angle, upright around 0 rad)
    - `state[3] = theta_dot`  (angular velocity)

    Discrete action semantics:
    - `0`: push cart left with fixed magnitude (`-force_mag`)
    - `1`: push cart right with fixed magnitude (`+force_mag`)

    Suggested controller design steps:
    1. Start with angle PD: `u_theta = kp_theta*theta + kd_theta*theta_dot`.
    2. Add cart-centering PD:
       `u_x = kp_x*x + kd_x*x_dot`.
    3. Combine and clamp:
       `u = clip(u_theta + u_x, -force_limit, force_limit)`.
    4. Convert `u` to discrete action by sign (optionally hysteresis near 0).
    5. Add integral terms only if persistent bias/drift exists.
    6. Tune for robustness over randomized resets (not just one seed).

    Practical tuning order:
    - Increase `kp_theta` until response is fast but not unstable.
    - Increase `kd_theta` to damp oscillation.
    - Add small `kp_x`, `kd_x` for center recovery.
    - Keep integral gains small and clamp integral memory.
    """

    def __init__(
        self,
        init_theta_max_deg: float = 30.0,
        init_x_range: Tuple[float, float] = (-0.1, 0.1),
        init_x_dot_range: Tuple[float, float] = (-0.15, 0.15),
        init_theta_dot_range: Tuple[float, float] = (-0.3, 0.3),
        **kwargs,
    ) -> None:
        super().__init__(
            init_x_range=init_x_range,
            init_x_dot_range=init_x_dot_range,
            init_theta_dot_range=init_theta_dot_range,
            **kwargs,
        )
        self.init_theta_max_rad = math.radians(float(init_theta_max_deg))
        self.down_theta_noise = 0.0  # Disable inherited down-hanging reset behavior.

    def reset(self, **kwargs):
        seed = kwargs.pop("seed", None)
        if seed is not None:
            self.seed(seed)
        x = self.np_random.uniform(*self.init_x_range)
        x_dot = self.np_random.uniform(*self.init_x_dot_range)
        theta = self.np_random.uniform(
            -self.init_theta_max_rad, self.init_theta_max_rad
        )
        theta_dot = self.np_random.uniform(*self.init_theta_dot_range)
        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float64)
        self.steps = 0
        return self.state.astype(np.float32)
