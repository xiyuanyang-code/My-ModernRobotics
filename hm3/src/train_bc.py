"""MSE Behavior Cloning training script for PushT with action chunking.

Aligned with PPO interface:
- Input: 5-dim normalized state (agent_xy + block_xy + block_angle)
- Output: 2-dim action in [-1, 1] (DeltaActionWrapper converts to displacement)
"""

import argparse
import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from src.configs import BCConfig
from src.core.bc_dataset import make_bc_dataloaders
from src.core.bc_network import BCMLPPolicy
from src.core.bc_trainer import BCTrainer
from src.core.env import make_env
from src.utils.logger import CSVLogger, TensorboardLogger, ConsoleLogger
from src.utils.plotting import plot_reward_curve, plot_loss_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Train MSE Behavior Cloning on PushT")
    parser.add_argument("--chunk_length", type=int, default=None, help="Action chunk size K")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--activation", type=str, default=None, choices=["relu", "tanh", "gelu"])
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--lr_schedule", type=str, default=None, choices=["none", "cosine"])
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--obs_norm", type=int, default=None, help="1 for on, 0 for off")
    parser.add_argument("--action_norm", type=int, default=None, help="1 for on, 0 for off")
    parser.add_argument("--use_full_state", type=int, default=None, help="1 for 5-dim, 0 for 2-dim")
    parser.add_argument("--ema_decay", type=float, default=None, help="EMA decay, 0 to disable")
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--eval_episodes", type=int, default=None)
    parser.add_argument("--early_stop_patience", type=int, default=None)
    parser.add_argument("--delta_max", type=float, default=None, help="Delta max for action wrapper")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--data_dir", type=str, default="data/lerobot/pusht",
                        help="Path to LeRobot PushT dataset")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt")
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = BCConfig()
    if args.chunk_length is not None:
        config.chunk_length = args.chunk_length
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.activation is not None:
        config.activation = args.activation
    if args.lr is not None:
        config.lr = args.lr
    if args.lr_schedule is not None:
        config.lr_schedule = args.lr_schedule
    if args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if args.grad_clip is not None:
        config.grad_clip = args.grad_clip
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.max_epochs is not None:
        config.max_epochs = args.max_epochs
    if args.obs_norm is not None:
        config.obs_norm = bool(args.obs_norm)
    if args.action_norm is not None:
        config.action_norm = bool(args.action_norm)
    if args.use_full_state is not None:
        config.use_full_state = bool(args.use_full_state)
    if args.ema_decay is not None:
        config.ema_decay = args.ema_decay
    if args.eval_freq is not None:
        config.eval_freq = args.eval_freq
    if args.eval_episodes is not None:
        config.eval_episodes = args.eval_episodes
    if args.early_stop_patience is not None:
        config.early_stop_patience = args.early_stop_patience
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    delta_max = args.delta_max if args.delta_max is not None else 100.0

    # Update obs_dim based on use_full_state
    if config.use_full_state:
        config.obs_dim = 5
    else:
        config.obs_dim = 2

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    state_tag = "5d" if config.use_full_state else "2d"
    run_dir = os.path.join(args.output_dir, f"bc_k{config.chunk_length}_{state_tag}_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    video_dir = os.path.join(run_dir, "videos")

    config.output_dir = ckpt_dir

    # Seed
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Device
    device = "cuda" if config.device == "auto" and torch.cuda.is_available() else config.device
    if device == "auto":
        device = "cpu"
    print(f"Device: {device}")
    print(f"Output: {run_dir}")

    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Build datasets
    data_dir = os.path.join(PROJECT_ROOT, args.data_dir)
    print(f"\nLoading dataset from {data_dir}...")
    train_loader, val_loader, dataset_info = make_bc_dataloaders(
        data_dir=data_dir,
        chunk_length=config.chunk_length,
        batch_size=config.batch_size,
        obs_norm=config.obs_norm,
        action_norm=config.action_norm,
        val_ratio=config.val_ratio,
        seed=config.seed,
        use_full_state=config.use_full_state,
        delta_max=delta_max,
    )

    print(f"  Dataset: {dataset_info['num_episodes']} episodes, "
          f"{dataset_info['num_frames']} frames, "
          f"{dataset_info['num_pairs']} (obs, chunk) pairs")
    print(f"  Obs dim: {dataset_info['obs_dim']}, "
          f"Full state: {dataset_info['use_full_state']}")
    print(f"  Train: {dataset_info['train_pairs']} pairs ({dataset_info['train_episodes']} episodes)")
    print(f"  Val:   {dataset_info['val_pairs']} pairs ({dataset_info['val_episodes']} episodes)")

    # Build policy
    policy = BCMLPPolicy(
        obs_dim=config.obs_dim,
        act_dim=config.act_dim,
        chunk_length=config.chunk_length,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        activation=config.activation,
    )
    print(f"  Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    # Resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint["policy"])
        print(f"Resumed from: {args.resume}")

    # Save config
    config_dict = {**vars(config), **dataset_info}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Create evaluation environment (matching PPO: DeltaActionWrapper + FixedObsNormWrapper)
    print("\nCreating evaluation environment (DeltaActionWrapper + obs_norm)...")
    eval_env = make_env(
        seed=config.seed + 1000,
        obs_norm=config.obs_norm,
        action_space="delta",
        delta_max=delta_max,
        render_mode="rgb_array",
    )

    # If using 2-dim obs, wrap eval env to extract agent_xy only
    if not config.use_full_state:
        from src.core.env import AgentPosOnlyWrapper
        eval_env = AgentPosOnlyWrapper(eval_env)
        print("  Wrapped eval env with AgentPosOnlyWrapper (2-dim obs)")

    # Trainer
    trainer = BCTrainer(policy, config, device)

    # Loggers
    csv_logger = CSVLogger(os.path.join(run_dir, "train_log.csv"))
    tb_logger = TensorboardLogger(os.path.join(run_dir, "tb"))
    console = ConsoleLogger(prefix="BC")

    # Train
    print("\nStarting BC training...")
    best_reward = trainer.train(
        train_loader, val_loader,
        eval_env=eval_env,
        logger=csv_logger,
        tb_logger=tb_logger,
        console=console,
        video_dir=video_dir,
    )
    csv_logger.close()
    tb_logger.close()

    # Plots
    log_path = os.path.join(run_dir, "train_log.csv")
    plot_loss_curve(log_path, os.path.join(run_dir, "train_loss.png"), loss_key="train_loss")
    plot_loss_curve(log_path, os.path.join(run_dir, "val_loss.png"), loss_key="val_loss")
    plot_reward_curve(log_path, os.path.join(run_dir, "eval_reward.png"))

    print(f"\nDone! Best reward: {best_reward:.4f}")
    print(f"Outputs: {run_dir}")


if __name__ == "__main__":
    main()
