"""Neural network architectures for PPO and SAC."""

import torch
import torch.nn as nn
import numpy as np


def layer_init(layer, std=np.sqrt(2), bias=0.0):
    """Orthogonal initialization (standard for RL)."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class MLPEncoder(nn.Module):
    """Shared MLP backbone."""

    def __init__(self, input_dim, hidden_dim, num_layers=2):
        super().__init__()
        layers = [layer_init(nn.Linear(input_dim, hidden_dim)), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.extend([layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh()])
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ========================== PPO Networks ==========================


class PPOActor(nn.Module):
    """Gaussian actor for PPO with tanh squashing → output in [-1, 1]."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, hidden_dim, num_layers)
        self.mean_head = layer_init(nn.Linear(hidden_dim, act_dim), std=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim) * 0.5)  # start with std≈1.65

    def forward(self, obs):
        features = self.encoder(obs)
        mean = self.mean_head(features)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_action(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        x = dist.sample()
        action = torch.tanh(x)  # squash to [-1, 1]
        # Log prob with tanh correction (numerically stable)
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - x - nn.functional.softplus(-2 * x))).sum(dim=-1)
        return action, log_prob

    def get_log_prob(self, obs, action):
        # action is in [-1, 1], inverse tanh to get pre-squash value
        x = torch.atanh(action.clamp(-0.999, 0.999))
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(x).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - x - nn.functional.softplus(-2 * x))).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

    def get_deterministic(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class PPOCritic(nn.Module):
    """Value function for PPO."""

    def __init__(self, obs_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, hidden_dim, num_layers)
        self.value_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs):
        features = self.encoder(obs)
        return self.value_head(features).squeeze(-1)


class PPOAgent(nn.Module):
    """Combined actor-critic for PPO."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.actor = PPOActor(obs_dim, act_dim, hidden_dim, num_layers)
        self.critic = PPOCritic(obs_dim, hidden_dim, num_layers)
        self.act_dim = act_dim

    def get_action_and_value(self, obs):
        action, log_prob = self.actor.get_action(obs)
        value = self.critic(obs)
        return action, log_prob, value

    def evaluate_actions(self, obs, action):
        log_prob, entropy = self.actor.get_log_prob(obs, action)
        value = self.critic(obs)
        return log_prob, entropy, value

    def get_value(self, obs):
        return self.critic(obs)

    def get_deterministic_action(self, obs):
        return self.actor.get_deterministic(obs)


# ========================== SAC Networks ==========================


class SACActor(nn.Module):
    """Gaussian actor for SAC — outputs mean and log_std, tanh-squashed."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, act_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim, hidden_dim, num_layers)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        features = self.encoder(obs)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_action(self, obs, deterministic=False):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        if deterministic:
            action = mean
        else:
            action = dist.rsample()
        # Tanh squashing
        squashed = torch.tanh(action)
        # Log prob with correction for tanh
        log_prob = dist.log_prob(action).sum(dim=-1)
        log_prob -= (2 * (np.log(2) - action - nn.functional.softplus(-2 * action))).sum(dim=-1)
        return squashed, log_prob

    def get_deterministic_action(self, obs):
        mean, _ = self.forward(obs)
        return torch.tanh(mean)


class SACQNetwork(nn.Module):
    """Q-network for SAC (single)."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.encoder = MLPEncoder(obs_dim + act_dim, hidden_dim, num_layers)
        self.q_head = layer_init(nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q_head(self.encoder(x)).squeeze(-1)


class SACAgent(nn.Module):
    """Combined SAC agent: actor + twin Q-networks + target Q-networks."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, num_layers=2):
        super().__init__()
        self.actor = SACActor(obs_dim, act_dim, hidden_dim, num_layers)
        self.q1 = SACQNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.q2 = SACQNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.q1_target = SACQNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        self.q2_target = SACQNetwork(obs_dim, act_dim, hidden_dim, num_layers)
        # Copy weights to targets
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.act_dim = act_dim

    def get_deterministic_action(self, obs):
        return self.actor.get_deterministic_action(obs)

    def soft_update(self, tau):
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
