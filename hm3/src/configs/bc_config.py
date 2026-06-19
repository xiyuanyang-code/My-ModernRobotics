from dataclasses import dataclass


@dataclass
class BCConfig:
    # Algorithm
    lr: float = 1e-3
    lr_schedule: str = "none"  # "none" or "cosine"
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    max_epochs: int = 200
    batch_size: int = 256

    # Network (matching PPO defaults)
    obs_dim: int = 5       # full state: agent_xy + block_xy + block_angle
    act_dim: int = 2       # delta target_xy in [-1, 1]
    chunk_length: int = 8  # K
    hidden_dim: int = 256
    num_layers: int = 2
    activation: str = "tanh"

    # Normalization
    obs_norm: bool = True     # divide by [512,512,512,512,2pi]
    action_norm: bool = True  # map actions to [-1, 1]

    # Observation source
    use_full_state: bool = True  # use reconstructed 5-dim state

    # EMA
    ema_decay: float = 0.999  # 0 to disable

    # Evaluation
    eval_freq: int = 10
    eval_episodes: int = 10
    early_stop_patience: int = 0  # 0 to disable

    # Saving
    save_freq: int = 50

    # Misc
    seed: int = 42
    device: str = "auto"
    output_dir: str = "outputs/bc"
    val_ratio: float = 0.2
