"""Environment creation and wrappers for PushT."""

import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure project root is on path so gym_pusht can be imported
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gym_pusht  # noqa: F401 - registers gym_pusht/PushT-v0

# Headless rendering setup (must be set before any pygame import)
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

if not os.environ.get("XDG_RUNTIME_DIR"):
    runtime_dir = Path(f"/tmp/xdg-runtime-{os.getuid()}")
    runtime_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.chmod(0o700)
    os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)


# Known observation bounds for PushT state obs:
# [agent_x, agent_y, block_x, block_y, block_angle]
# positions in [0, 512], angle in [0, 2*pi]
OBS_HIGH = np.array([512.0, 512.0, 512.0, 512.0, 2 * np.pi], dtype=np.float32)


class CurriculumWrapper(gym.Wrapper):
    """Start with block near goal, gradually expand to full random range.

    progress=0 → block starts within 100px of goal (easy)
    progress=1 → block starts anywhere in [100, 400] (hard, default)
    """

    GOAL_POS = np.array([256.0, 256.0])
    EASY_RADIUS = 100.0   # block starts within this of goal
    FULL_RANGE = (100.0, 400.0)

    def __init__(self, env):
        super().__init__(env)
        self.progress = 0.0  # 0=easy, 1=full random

    def set_progress(self, progress):
        self.progress = np.clip(progress, 0.0, 1.0)

    def reset(self, **kwargs):
        # First reset to get a valid agent position
        obs, info = self.env.reset(**kwargs)

        if self.progress < 1.0:
            # Sample block pos near goal, expanding toward full range as progress increases
            angle = np.random.uniform(0, 2 * np.pi)
            max_radius = self.EASY_RADIUS * (1 - self.progress) + 200 * self.progress
            radius = np.random.uniform(0, max_radius)
            block_pos = self.GOAL_POS + radius * np.array([np.cos(angle), np.sin(angle)])
            block_pos = np.clip(block_pos, 50, 460)
            block_angle = np.random.uniform(-np.pi, np.pi)

            # Re-reset with specific block position
            try:
                obs, info = self.env.reset(options={
                    "reset_to_state": [obs[0], obs[1],
                                       block_pos[0], block_pos[1], block_angle]
                })
            except Exception:
                pass  # fallback to default reset

        return obs, info


class AbsActionWrapper(gym.ActionWrapper):
    """Convert [-1, 1] actions to absolute positions [0, 512].

    Network outputs [x, y] in [-1, 1] (from tanh).
    We map to [0, 512]: target = (action + 1) / 2 * 512.
    """

    def action(self, action):
        target = (action + 1.0) / 2.0 * 512.0
        return np.clip(target, 0, 512).astype(np.float32)


class DeltaActionWrapper(gym.ActionWrapper):
    """Convert delta actions to absolute positions.

    Network outputs [dx, dy] in [-1, 1] (from tanh).
    We scale to [-delta_max, delta_max] and add to current agent position.
    """

    def __init__(self, env, delta_max=100.0):
        super().__init__(env)
        self.delta_max = delta_max
        self._agent_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # obs = [agent_x, agent_y, block_x, block_y, block_angle]
        self._agent_pos = obs[:2].copy()
        return obs, info

    def action(self, action):
        # action in [-1, 1] from tanh → scale to delta
        delta = action * self.delta_max
        # Add to current agent position
        target = self._agent_pos + delta
        # Clip to arena bounds
        target = np.clip(target, 0, 512)
        return target.astype(np.float32)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(self.action(action))
        self._agent_pos = obs[:2].copy()
        return obs, reward, terminated, truncated, info


class RewardShapingWrapper(gym.RewardWrapper):
    """Touch bonus + coverage delta reward.

    reward = touch_bonus + delta * scale
      touch_bonus: +1 when coverage changes (agent moved the block)
      delta: coverage_after - coverage_before (push quality)
    """

    def __init__(self, env, delta_scale=10.0):
        super().__init__(env)
        self.delta_scale = delta_scale
        self._prev_coverage = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._prev_coverage = 0.0
        return obs, info

    def step(self, action):
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        # env_reward = clip(coverage / 0.95, 0, 1)
        coverage = min(env_reward * 0.95, 1.0)

        delta = coverage - self._prev_coverage
        self._prev_coverage = coverage

        if coverage >= 0.95:
            reward = 20.0
        elif abs(delta) > 1e-6:
            reward = 1.0 + delta * self.delta_scale
        else:
            reward = 0.0

        return obs, reward, terminated, truncated, info


class FixedObsNormWrapper(gym.ObservationWrapper):
    """Fixed normalization: divide by known max values → obs in ~[0, 1]."""

    def __init__(self, env):
        super().__init__(env)
        self.obs_high = OBS_HIGH.copy()

    def observation(self, obs):
        return (obs / self.obs_high).astype(np.float32)


class AgentPosOnlyWrapper(gym.ObservationWrapper):
    """Extract only agent_xy from the 5-dim state observation.

    Useful for BC policies trained on 2-dim agent position only.
    Must be applied AFTER FixedObsNormWrapper.
    """

    def __init__(self, env):
        super().__init__(env)
        # Update observation space to 2-dim
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=np.zeros(2, dtype=np.float32),
            high=np.ones(2, dtype=np.float32),
            dtype=np.float32,
        )

    def observation(self, obs):
        return obs[:2].astype(np.float32)


def make_env(seed=42, obs_type="state", obs_norm=True, reward_shaping=False, action_space="delta",
             curriculum=False, delta_max=100.0, max_steps=300, render_mode=None):
    """Create a PushT environment with optional wrappers."""
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type=obs_type,
        max_episode_steps=max_steps,
        render_mode=render_mode,
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if curriculum:
        env = CurriculumWrapper(env)
    if action_space == "delta":
        env = DeltaActionWrapper(env, delta_max=delta_max)
    elif action_space == "abs":
        env = AbsActionWrapper(env)
    if reward_shaping:
        env = RewardShapingWrapper(env)
    if obs_norm:
        env = FixedObsNormWrapper(env)
    env.reset(seed=seed)
    return env


def make_vec_env(num_envs=4, seed=42, obs_type="state", obs_norm=True, reward_shaping=False,
                 action_space="delta", curriculum=False, delta_max=100.0, max_steps=300):
    """Create vectorized PushT environments (sync, stable)."""
    from gymnasium.vector import SyncVectorEnv

    def make_fn(i):
        def _make():
            env = gym.make(
                "gym_pusht/PushT-v0",
                obs_type=obs_type,
                max_episode_steps=max_steps,
            )
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if curriculum:
                env = CurriculumWrapper(env)
            if action_space == "delta":
                env = DeltaActionWrapper(env, delta_max=delta_max)
            elif action_space == "abs":
                env = AbsActionWrapper(env)
            if reward_shaping:
                env = RewardShapingWrapper(env)
            if obs_norm:
                env = FixedObsNormWrapper(env)
            env.reset(seed=seed + i)
            return env

        return _make

    envs = SyncVectorEnv([make_fn(i) for i in range(num_envs)])
    return envs
