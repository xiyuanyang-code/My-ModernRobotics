"""Video recording utilities."""

import os
import numpy as np


def record_episode(env, agent, path, max_steps=300, device="cpu", obs_norm=True):
    """Run one episode with deterministic policy, save as mp4."""
    os.makedirs(os.path.dirname(path), exist_ok=True)

    import cv2

    frames = []
    obs, info = env.reset()
    total_reward = 0.0

    for _ in range(max_steps):
        obs_t = np.expand_dims(obs, axis=0).astype(np.float32)
        with __import__("torch").no_grad():
            action = agent.get_deterministic_action(
                __import__("torch").FloatTensor(obs_t).to(device)
            )
            action_np = action.cpu().numpy()[0]

        # DeltaActionWrapper handles [-1, 1] → delta conversion

        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward

        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if terminated or truncated:
            break

    # Write video
    if frames:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    return total_reward


def record_episodes(env, agent, path, num_episodes=3, max_steps=300, device="cpu"):
    """Record multiple evaluation episodes."""
    rewards = []
    for i in range(num_episodes):
        episode_path = path.replace(".mp4", f"_ep{i}.mp4")
        r = record_episode(env, agent, episode_path, max_steps, device)
        rewards.append(r)
        print(f"  Recorded episode {i}: reward={r:.4f}")
    return rewards
