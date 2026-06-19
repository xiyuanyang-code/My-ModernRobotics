"""Evaluate a saved RL checkpoint on PushT."""

import argparse
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import numpy as np

from src.core.env import make_env
from src.core.networks import PPOAgent, SACAgent
from src.utils.video import record_episodes


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained RL agent on PushT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file")
    parser.add_argument("--algo", type=str, choices=["ppo", "sac"], required=True)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--record", action="store_true", help="Record evaluation videos")
    parser.add_argument("--video_dir", type=str, default=None, help="Directory for videos")
    parser.add_argument("--reward_shaping", type=int, default=0)
    parser.add_argument("--delta_max", type=float, default=100.0, help="Max delta displacement per step")
    parser.add_argument("--action_space", type=str, default="delta", choices=["abs", "delta"], help="Action space type")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    env = make_env(seed=args.seed, obs_norm=True, reward_shaping=bool(args.reward_shaping), action_space=args.action_space, delta_max=args.delta_max, render_mode="rgb_array")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Create agent
    if args.algo == "ppo":
        agent = PPOAgent(obs_dim, act_dim, args.hidden_dim, args.num_layers)
    else:
        agent = SACAgent(obs_dim, act_dim, args.hidden_dim, args.num_layers)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    agent.load_state_dict(checkpoint["agent"])
    agent.to(args.device)
    agent.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Evaluate
    rewards = []
    for i in range(args.episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(args.device)
            with torch.no_grad():
                if args.algo == "ppo":
                    action = agent.get_deterministic_action(obs_t)
                else:
                    action = agent.actor.get_deterministic_action(obs_t)
            action_np = action.cpu().numpy()[0]
            # DeltaActionWrapper handles [-1, 1] → delta conversion

            obs, reward, terminated, truncated, info = env.step(action_np)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)
        print(f"  Episode {i}: reward={episode_reward:.4f}")

    print(f"\nMean reward: {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")

    # Record videos
    if args.record:
        video_dir = args.video_dir or os.path.join(os.path.dirname(args.checkpoint), "videos")
        print(f"\nRecording {args.episodes} videos to {video_dir}...")
        record_episodes(env, agent, os.path.join(video_dir, "eval.mp4"), args.episodes, device=args.device)


if __name__ == "__main__":
    main()
