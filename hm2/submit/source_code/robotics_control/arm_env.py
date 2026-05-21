"""2-link Planar Arm."""

import sys

import numpy as np
import gymnasium as gym
import gymnasium.spaces

# Gymnasium Viewer sets `isopen` only after `get_window` succeeds. If window creation
# fails, __del__ still calls close(), which touches `isopen` and raises AttributeError.
_GYM_VIEWER_CLOSE_PATCHED = False


def _patch_gym_viewer_close():
    global _GYM_VIEWER_CLOSE_PATCHED
    if _GYM_VIEWER_CLOSE_PATCHED:
        return
    from gymnasium.envs.classic_control import rendering

    def _safe_close(self):
        if not getattr(self, "isopen", False):
            return
        if sys.meta_path:
            self.window.close()
            self.isopen = False

    rendering.Viewer.close = _safe_close
    rendering.SimpleImageViewer.close = _safe_close
    _GYM_VIEWER_CLOSE_PATCHED = True


class TwoLinkArmEnv(gym.Env):
    DOF = 2
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 1000,
    }

    def __init__(
        self,
        Q=None,
        R=None,
        goal_q=None,
        init_q=None,
        init_dq=None,
        dt=1e-3,
        l1=0.5,
        l2=0.75,
        m1=0.33,
        m2=0.55,
        izz1=15.0,
        izz2=8.0,
        noise_free=True,
        noise_mu=None,
        noise_sigma=None,
        render_mode=None,
    ):
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(
            low=np.array([-np.pi, -np.pi, -np.inf, -np.inf]),
            high=np.array([np.pi, np.pi, np.inf, np.inf]),
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf]), high=np.array([np.inf, np.inf])
        )

        if Q is None:
            self.Q = np.zeros((self.DOF * 2, self.DOF * 2))
            self.Q[: self.DOF, : self.DOF] = np.eye(self.DOF) * 1000.0
        else:
            self.Q = Q

        if R is None:
            self.R = np.eye(self.DOF) * 0.001
        else:
            self.R = R

        self.dt = dt
        self._goal_q = goal_q
        self.goal_dq = np.zeros(self.DOF)
        self.init_q = np.zeros(self.DOF) if init_q is None else init_q
        self.init_dq = np.zeros(self.DOF) if init_dq is None else init_dq

        self.l1 = l1
        self.l2 = l2
        self.m1 = m1
        self.m2 = m2
        self.izz1 = izz1
        self.izz2 = izz2

        self.K1 = (
            1 / 3.0 * self.m1 + self.m2
        ) * self.l1**2.0 + 1 / 3.0 * self.m2 * self.l2**2.0
        self.K2 = self.m2 * self.l1 * self.l2
        self.K3 = 1 / 3.0 * self.m2 * self.l2**2.0
        self.K4 = 1 / 2.0 * self.m2 * self.l1 * self.l2

        # how much noise to add to input signal
        self.noise_free = noise_free
        self.noise_mu = np.zeros((self.DOF,)) if noise_mu is None else noise_mu
        self.noise_sigma = np.ones((self.DOF,)) if noise_sigma is None else noise_sigma

        self.viewer = None
        self._np_random, self._np_random_seed = gym.utils.seeding.np_random(None)
        self.reset()

    @property
    def position(self):
        return np.copy(self.q)

    @property
    def velocity(self):
        return np.copy(self.dq)

    @property
    def state(self):
        return np.hstack((self.q, self.dq))

    @state.setter
    def state(self, value):
        self.q = value[: self.DOF, ...]
        self.dq = value[self.DOF :, ...]

    @property
    def goal(self):
        return np.hstack((self.goal_q, self.goal_dq))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._np_random, self._np_random_seed = gym.utils.seeding.np_random(seed)
        if self._goal_q is None:
            self.goal_q = (2 * np.pi) * self._np_random.random(self.DOF) - np.pi
        else:
            self.goal_q = self._goal_q.copy()
        self.q = self.init_q.copy()
        self.dq = self.init_dq.copy()
        self.t = 0.0
        obs = np.hstack((self.q, self.dq))
        return obs, {}

    def step(self, action, dt=None):
        if dt is None:
            dt = self.dt

        u = np.asarray(action, dtype=float).copy()
        if not self.noise_free:
            u[0] += self._np_random.normal(self.noise_mu[0], self.noise_sigma[0])
            u[1] += self._np_random.normal(self.noise_mu[1], self.noise_sigma[1])

        u = np.clip(u, self.action_space.low, self.action_space.high)

        C2 = np.cos(self.q[1])
        S2 = np.sin(self.q[1])
        M11 = self.K1 + self.K2 * C2
        M12 = self.K3 + self.K4 * C2
        M21 = M12
        M22 = self.K3
        H1 = (
            -self.K2 * S2 * self.dq[0] * self.dq[1]
            - 1 / 2.0 * self.K2 * S2 * self.dq[1] ** 2.0
        )
        H2 = 1 / 2.0 * self.K2 * S2 * self.dq[0] ** 2.0

        ddq1 = (H2 * M11 - H1 * M21 - M11 * u[1] + M21 * u[0]) / (M12**2.0 - M11 * M22)
        ddq0 = (-H2 + u[1] - M22 * ddq1) / M21

        self.dq += np.array([ddq0, ddq1]) * dt
        self.q += self.dq * dt
        self.t += dt

        # calculate the reward
        x_diff = np.hstack((self.q, self.dq)) - np.hstack((self.goal_q, self.goal_dq))

        reward = -x_diff.dot(self.Q).dot(x_diff) - u.dot(self.R).dot(u)
        reward *= self.dt
        terminated = False
        if np.allclose(self.goal_q, self.q, atol=0.01) and np.allclose(
            self.goal_dq, self.dq, atol=0.01
        ):
            terminated = True

        return np.hstack((self.q, self.dq)), reward, terminated, False, {}

    def render(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        mode = self.render_mode or "rgb_array"

        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        max_len = self.l1 + self.l2
        bound = max_len * 1.5
        ax.set_xlim(-bound, bound)
        ax.set_ylim(-bound, bound)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Goal arm (transparent blue)
        gx1 = self.l1 * np.cos(self.goal_q[0])
        gy1 = self.l1 * np.sin(self.goal_q[0])
        gx2 = gx1 + self.l2 * np.cos(self.goal_q[0] + self.goal_q[1])
        gy2 = gy1 + self.l2 * np.sin(self.goal_q[0] + self.goal_q[1])
        ax.plot([0, gx1], [0, gy1], 'b-', linewidth=4, alpha=0.25)
        ax.plot([gx1, gx2], [gy1, gy2], 'b-', linewidth=3, alpha=0.25)

        # Current arm (red)
        x1 = self.l1 * np.cos(self.q[0])
        y1 = self.l1 * np.sin(self.q[0])
        x2 = x1 + self.l2 * np.cos(self.q[0] + self.q[1])
        y2 = y1 + self.l2 * np.sin(self.q[0] + self.q[1])
        ax.plot([0, x1], [0, y1], 'r-', linewidth=4)
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3)

        # Joints
        ax.plot(0, 0, 'ko', markersize=8)
        ax.plot(x1, y1, 'ko', markersize=6)
        ax.plot(x2, y2, 'go', markersize=8)
        ax.plot(gx2, gy2, 'b*', markersize=10, alpha=0.5)

        if mode == "rgb_array":
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]  # RGBA -> RGB
            plt.close(fig)
            return frame
        else:
            plt.close(fig)
            return None

    def close(self):
        pass


class LimitedTorqueTwoLinkArmEnv(TwoLinkArmEnv):
    def __init__(self, max_torques=None, **kwargs):
        super().__init__(**kwargs)

        if max_torques is None:
            max_torques = np.array([10.0, 10.0])

        self.action_space = gym.spaces.Box(low=-max_torques, high=max_torques)
