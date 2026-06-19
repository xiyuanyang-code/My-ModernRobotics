"""Rollout buffer for PPO — stores trajectories and computes GAE."""

import numpy as np
import torch


class RolloutBuffer:
    """Stores a fixed-length rollout from vectorized envs, computes GAE."""

    def __init__(self, num_steps, num_envs, obs_dim, act_dim, device="cpu"):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        self.obs = np.zeros((num_steps, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((num_steps, num_envs, act_dim), dtype=np.float32)
        self.log_probs = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.rewards = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.dones = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.values = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.advantages = np.zeros((num_steps, num_envs), dtype=np.float32)
        self.returns = np.zeros((num_steps, num_envs), dtype=np.float32)

    def compute_gae(self, last_values, gamma, gae_lambda):
        """Compute GAE advantages and returns."""
        last_gae = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        self.returns = self.advantages + self.values

    def get_batches(self, batch_size):
        """Yield shuffled mini-batches for PPO updates."""
        total = self.num_steps * self.num_envs
        obs = self.obs.reshape(total, -1)
        actions = self.actions.reshape(total, -1)
        log_probs = self.log_probs.reshape(total)
        advantages = self.advantages.reshape(total)
        returns = self.returns.reshape(total)

        # Normalize returns for stable value function training
        ret_mean = returns.mean()
        ret_std = returns.std() + 1e-8
        returns_norm = (returns - ret_mean) / ret_std

        indices = np.random.permutation(total)
        for start in range(0, total, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            yield {
                "obs": torch.FloatTensor(obs[idx]).to(self.device),
                "action": torch.FloatTensor(actions[idx]).to(self.device),
                "log_prob": torch.FloatTensor(log_probs[idx]).to(self.device),
                "advantage": torch.FloatTensor(advantages[idx]).to(self.device),
                "return": torch.FloatTensor(returns_norm[idx]).to(self.device),
            }
