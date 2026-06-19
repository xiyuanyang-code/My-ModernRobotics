from dataclasses import dataclass


@dataclass
class PPOConfig:
    # Algorithm
    lr: float = 3e-4
    lr_schedule: str = "none"  # "none" or "linear"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 1.0
    ent_coef: float = 0.005
    max_grad_norm: float = 0.5
    epochs_per_update: int = 10
    num_envs: int = 4
    rollout_steps: int = 128
    batch_size: int = 64

    # Network
    hidden_dim: int = 256
    num_layers: int = 2

    # Training
    max_steps: int = 500_000
    eval_freq: int = 5_000
    eval_episodes: int = 10
    save_freq: int = 10_000

    # Environment
    obs_norm: bool = True
    reward_shaping: bool = False
    curriculum: bool = False
    action_scale: float = 512.0

    # Misc
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs/ppo"
    log_freq: int = 100
