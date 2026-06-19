"""MLP action-chunking policy for behavior cloning."""

import copy
import numpy as np
import torch
import torch.nn as nn


class BCMLPPolicy(nn.Module):
    """MLP that maps observation → flattened action chunk.

    Architecture: obs → Linear → ReLU → ... → Linear → (K * act_dim)

    Args:
        obs_dim: Observation dimension (2 for agent_xy, 5 for full state).
        act_dim: Action dimension per step (2 for PushT).
        chunk_length: Number of actions per chunk (K).
        hidden_dim: Hidden layer width.
        num_layers: Number of hidden layers.
        activation: Activation function name.
    """

    def __init__(self, obs_dim=2, act_dim=2, chunk_length=1,
                 hidden_dim=256, num_layers=2, activation="relu"):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.chunk_length = chunk_length
        self.output_dim = chunk_length * act_dim

        act_fn = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}[activation]

        layers = []
        in_dim = obs_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(act_fn())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

        # Orthogonal init (standard for RL/BC)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2.0))
                nn.init.constant_(m.bias, 0.0)
        # Final layer with small init
        nn.init.orthogonal_(self.net[-1].weight, gain=0.01)

    def forward(self, obs):
        """Predict flattened action chunk.

        Args:
            obs: (batch, obs_dim) — normalized observation.

        Returns:
            (batch, K * act_dim) — predicted action chunk (normalized if training with normalized actions).
        """
        return self.net(obs)

    def predict_chunk(self, obs):
        """Predict and reshape to (batch, K, act_dim)."""
        flat = self.forward(obs)
        return flat.view(-1, self.chunk_length, self.act_dim)


class EMAPolicy:
    """Exponential Moving Average wrapper for a BC policy.

    Maintains a shadow copy of the policy parameters:
        θ_EMA ← α * θ_EMA + (1 - α) * θ

    Args:
        policy: The BCMLPPolicy to track.
        decay: EMA decay rate α (e.g. 0.99 or 0.999).
    """

    def __init__(self, policy, decay=0.999):
        self.decay = decay
        self.ema_params = copy.deepcopy(policy.state_dict())
        self.policy = policy

    @torch.no_grad()
    def update(self):
        """Update EMA parameters from current policy."""
        for key, param in self.policy.state_dict().items():
            self.ema_params[key].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def forward(self, obs):
        """Forward pass using EMA parameters."""
        # Temporarily swap to EMA params
        original_params = {k: v.clone() for k, v in self.policy.state_dict().items()}
        self.policy.load_state_dict(self.ema_params)
        output = self.policy(obs)
        self.policy.load_state_dict(original_params)
        return output

    def predict_chunk(self, obs):
        """Predict chunk using EMA parameters."""
        flat = self.forward(obs)
        return flat.view(-1, self.policy.chunk_length, self.policy.act_dim)

    def eval(self):
        """No-op for interface compatibility."""
        return self

    def train(self):
        """No-op for interface compatibility."""
        return self

    def to(self, device):
        """Move EMA params to device."""
        for key in self.ema_params:
            self.ema_params[key] = self.ema_params[key].to(device)
        return self

    def state_dict(self):
        return copy.deepcopy(self.ema_params)

    def load_state_dict(self, state_dict):
        self.ema_params = state_dict
