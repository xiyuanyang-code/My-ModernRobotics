import argparse
import os
from pathlib import Path


def configure_headless() -> None:
    """Set SDL/XDG defaults that work on SSH/headless servers."""
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

    if not os.environ.get("XDG_RUNTIME_DIR"):
        runtime_dir = Path(f"/tmp/xdg-runtime-{os.getuid()}")
        runtime_dir.mkdir(parents=True, exist_ok=True)
        runtime_dir.chmod(0o700)
        os.environ["XDG_RUNTIME_DIR"] = str(runtime_dir)


configure_headless()

import cv2
import gymnasium as gym

import gym_pusht  # noqa: F401 - registers gym_pusht/PushT-v0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PushT rollout to an mp4 file.")
    parser.add_argument("--output", "-o", default="pusht_rollout.mp4", help="Output video path.")
    parser.add_argument("--steps", type=int, default=1000, help="Number of environment steps to render.")
    parser.add_argument("--fps", type=float, default=10.0, help="Output video frames per second.")
    parser.add_argument("--width", type=int, default=680, help="Rendered frame width.")
    parser.add_argument("--height", type=int, default=680, help="Rendered frame height.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducible rollouts.")
    return parser.parse_args()


def make_writer(output_path: Path, frame_shape: tuple[int, int, int], fps: float) -> cv2.VideoWriter:
    height, width = frame_shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")
    return writer


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)

    env = gym.make(
        "gym_pusht/PushT-v0",
        render_mode="rgb_array",
        visualization_width=args.width,
        visualization_height=args.height,
    )
    env.action_space.seed(args.seed)

    writer = None
    try:
        env.reset(seed=args.seed)
        frame = env.render()
        writer = make_writer(output_path, frame.shape, args.fps)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        for _ in range(args.steps):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            frame = env.render()
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if terminated or truncated:
                env.reset()
    finally:
        if writer is not None:
            writer.release()
        env.close()

    print(f"Wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
