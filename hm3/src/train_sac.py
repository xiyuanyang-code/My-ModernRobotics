"""Main SAC training script for PushT."""

import argparse
import os
import sys
import json
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from src.configs import SACConfig
from src.core.env import make_env
from src.core.networks import SACAgent
from src.core.sac import SAC
from src.utils.logger import CSVLogger, ConsoleLogger, TensorboardLogger
from src.utils.plotting import plot_reward_curve


def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC on PushT")
    parser.add_argument("--lr_actor", type=float, default=None)
    parser.add_argument("--lr_critic", type=float, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--eval_freq", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--obs_norm", type=int, default=None)
    parser.add_argument("--reward_shaping", type=int, default=None, help="1 for on, 0 for off")
    parser.add_argument("--delta_max", type=float, default=None, help="Max delta displacement per step")
    parser.add_argument("--action_space", type=str, default=None, choices=["abs", "delta"], help="Action space type")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt to resume from")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # Build config
    config = SACConfig()
    if args.lr_actor is not None:
        config.lr_actor = args.lr_actor
    if args.lr_critic is not None:
        config.lr_critic = args.lr_critic
    if args.hidden_dim is not None:
        config.hidden_dim = args.hidden_dim
    if args.num_layers is not None:
        config.num_layers = args.num_layers
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.buffer_size is not None:
        config.buffer_size = args.buffer_size
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.tau is not None:
        config.tau = args.tau
    if args.eval_freq is not None:
        config.eval_freq = args.eval_freq
    if args.seed is not None:
        config.seed = args.seed
    if args.obs_norm is not None:
        config.obs_norm = bool(args.obs_norm)
    if args.reward_shaping is not None:
        config.reward_shaping = bool(args.reward_shaping)
    if args.device is not None:
        config.device = args.device
    delta_max = args.delta_max if args.delta_max is not None else 100.0
    action_space = args.action_space if args.action_space is not None else "delta"

    # Output: outputs/sac_<timestamp>/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"sac_{timestamp}")
    ckpt_dir = os.path.join(run_dir, "checkpoints")
    video_dir = os.path.join(run_dir, "videos")

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

    # Environment
    print("Creating training environment...")
    train_env = make_env(seed=config.seed, obs_norm=config.obs_norm, reward_shaping=config.reward_shaping, action_space=action_space, delta_max=delta_max)

    print("Creating evaluation environment...")
    eval_env = make_env(
        seed=config.seed + 1000,
        obs_norm=True,
        reward_shaping=config.reward_shaping,
        action_space=action_space,
        render_mode="rgb_array",
        delta_max=delta_max,
    )

    obs_dim = train_env.observation_space.shape[0]
    act_dim = train_env.action_space.shape[0]
    print(f"  obs_dim={obs_dim}, act_dim={act_dim}")

    config.target_entropy = -float(act_dim)

    # Agent
    agent = SACAgent(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    print(f"  Agent params: {sum(p.numel() for p in agent.parameters()):,}")

    # Resume from checkpoint
    start_step = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        agent.load_state_dict(checkpoint["agent"])
        print(f"Resumed agent from: {args.resume}")

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(config), f, indent=2)

    # Trainer (checkpoints go to ckpt_dir)
    config.output_dir = ckpt_dir
    trainer = SAC(agent, config, device)

    # Loggers
    csv_logger = CSVLogger(os.path.join(run_dir, "train_log.csv"))
    tb_logger = TensorboardLogger(os.path.join(run_dir, "tb"))
    console = ConsoleLogger(prefix="SAC")

    # Train
    print("\nStarting SAC training...")
    best_reward = trainer.train(train_env, eval_env, logger=csv_logger, tb_logger=tb_logger, console=console, video_dir=video_dir, start_step=start_step)
    csv_logger.close()
    tb_logger.close()

    # Plot
    plot_reward_curve(
        os.path.join(run_dir, "train_log.csv"),
        os.path.join(run_dir, "reward_curve.png"),
    )

    print(f"\nDone! Best reward: {best_reward:.4f}")
    print(f"Outputs: {run_dir}")


if __name__ == "__main__":
    main()
