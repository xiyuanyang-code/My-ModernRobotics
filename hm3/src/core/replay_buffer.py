"""Simple replay buffer for off-policy algorithms (SAC)."""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size ring buffer storing transitions as numpy arrays."""

    def __init__(self, capacity, obs_dim, act_dim, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = action
        self.rew[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": torch.FloatTensor(self.obs[idx]).to(self.device),
            "action": torch.FloatTensor(self.act[idx]).to(self.device),
            "reward": torch.FloatTensor(self.rew[idx]).to(self.device),
            "next_obs": torch.FloatTensor(self.next_obs[idx]).to(self.device),
            "done": torch.FloatTensor(self.done[idx]).to(self.device),
        }

    def __len__(self):
        return self.size
