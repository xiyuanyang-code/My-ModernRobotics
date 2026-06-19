from dataclasses import dataclass


@dataclass
class SACConfig:
    # Algorithm
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    target_entropy: float = -2.0  # -act_dim

    # Network
    hidden_dim: int = 256
    num_layers: int = 2

    # Replay buffer
    buffer_size: int = 1_000_000
    batch_size: int = 256
    warmup_steps: int = 1000

    # Training
    max_steps: int = 500_000
    eval_freq: int = 5_000
    eval_episodes: int = 10
    save_freq: int = 10_000
    updates_per_step: int = 1

    # Environment
    obs_norm: bool = True
    reward_shaping: bool = False
    action_scale: float = 512.0

    # Misc
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs/sac"
    log_freq: int = 100
