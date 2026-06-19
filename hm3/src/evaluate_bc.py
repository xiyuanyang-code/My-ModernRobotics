"""Evaluate a saved BC checkpoint on PushT.

Uses the same eval env setup as PPO (DeltaActionWrapper + FixedObsNormWrapper).
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from src.core.env import make_env
from src.core.bc_network import BCMLPPolicy
from src.utils.video import record_episodes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained BC agent on PushT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--chunk_length", type=int, required=True, help="Action chunk size K")
    parser.add_argument("--obs_dim", type=int, default=5, help="Observation dim (5 for full state)")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--obs_norm", type=int, default=1)
    parser.add_argument("--delta_max", type=float, default=100.0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true", help="Record evaluation videos")
    parser.add_argument("--video_dir", type=str, default=None)
    parser.add_argument("--use_ema", action="store_true", help="Use EMA parameters")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # Create environment (matching PPO: DeltaActionWrapper + FixedObsNormWrapper)
    env = make_env(
        seed=args.seed,
        obs_norm=bool(args.obs_norm),
        action_space="delta",
        delta_max=args.delta_max,
        render_mode="rgb_array",
    )

    # Create policy
    policy = BCMLPPolicy(
        obs_dim=args.obs_dim,
        act_dim=2,
        chunk_length=args.chunk_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        activation=args.activation,
    )

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    if args.use_ema and "ema" in checkpoint:
        policy.load_state_dict(checkpoint["ema"])
        print("Using EMA parameters")
    else:
        policy.load_state_dict(checkpoint["policy"])
    policy.to(args.device)
    policy.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    rewards = []
    for i in range(args.episodes):
        obs, info = env.reset(seed=args.seed + i)
        episode_reward = 0.0
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
            with torch.no_grad():
                chunk = policy.predict_chunk(obs_t).cpu().numpy()[0]

            for k in range(args.chunk_length):
                if done:
                    break
                action = np.clip(chunk[k], -1.0, 1.0).astype(np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                done = terminated or truncated

        rewards.append(episode_reward)
        print(f"  Episode {i}: reward={episode_reward:.4f}")

    print(f"\nMean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Record videos
    if args.record:
        video_dir = args.video_dir or os.path.join(os.path.dirname(args.checkpoint), "videos")
        print(f"\nRecording {args.episodes} videos to {video_dir}...")

        class BCAgent:
            def __init__(self, policy, chunk_length, device):
                self.policy = policy
                self.chunk_length = chunk_length
                self.device = device

            def get_deterministic_action(self, obs_t):
                with torch.no_grad():
                    chunk = self.policy.predict_chunk(obs_t)
                return chunk[0, 0, :].unsqueeze(0)

            def eval(self):
                self.policy.eval()
                return self

        agent = BCAgent(policy, args.chunk_length, args.device)
        record_episodes(env, agent, os.path.join(video_dir, "eval.mp4"), args.episodes, device=args.device)


if __name__ == "__main__":
    main()
