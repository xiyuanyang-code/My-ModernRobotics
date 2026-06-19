"""Behavior Cloning dataset: loads LeRobot PushT demonstration data.

Supports two observation modes:
- 2-dim: agent_xy only (from original dataset)
- 5-dim: [agent_xy, block_xy, block_angle] (reconstructed from video frames)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class PushTBCDataset(Dataset):
    """Dataset of (observation, action_chunk) pairs from PushT demonstrations.

    Loads the official LeRobot PushT parquet dataset and builds
    fixed-length action chunks for behavior cloning training.

    Args:
        data_path: Path to the parquet file.
        chunk_length: Number of future actions per chunk (K).
        obs_norm: If True, normalize observations by dividing by [512,512,512,512,2pi].
        action_norm: If True, normalize actions to [-1, 1] to match PPO output.
        use_full_state: If True, use reconstructed 5-dim state from video frames.
        states_path: Path to reconstructed states .npy file.
    """

    def __init__(self, data_path, chunk_length=1, obs_norm=True, action_norm=True,
                 use_full_state=False, states_path=None, delta_max=100.0):
        import pandas as pd

        self.chunk_length = chunk_length
        self.obs_norm = obs_norm
        self.action_norm = action_norm
        self.use_full_state = use_full_state
        self.delta_max = delta_max

        # Normalization constants (matching PPO's FixedObsNormWrapper)
        self.OBS_HIGH = np.array([512.0, 512.0, 512.0, 512.0, 2 * np.pi], dtype=np.float32)

        # Load parquet
        df = pd.read_parquet(data_path)

        # Extract observations
        all_obs_raw = np.array(df["observation.state"].tolist(), dtype=np.float32)
        all_actions_abs = np.array(df["action"].tolist(), dtype=np.float32)
        all_episode_idx = df["episode_index"].values

        # Load reconstructed 5-dim states if available
        if use_full_state and states_path and os.path.exists(states_path):
            reconstructed = np.load(states_path).astype(np.float32)
            all_obs = np.column_stack([
                all_obs_raw,           # agent_xy from dataset (accurate)
                reconstructed[:, 2:4], # block_xy from video
                reconstructed[:, 4:5]  # block_angle from video
            ])
            print(f"Using full 5-dim state (agent from dataset, block from video)")
        elif use_full_state:
            raise FileNotFoundError(f"Reconstructed states not found: {states_path}")
        else:
            all_obs = all_obs_raw

        self.obs_dim = all_obs.shape[1]

        # Convert absolute actions to delta actions (matching PPO's DeltaActionWrapper)
        # delta = target - current_agent_pos
        # normalized_action = delta / delta_max (maps to [-1, 1] for typical deltas)
        all_delta_actions = all_actions_abs - all_obs_raw[:, :2]

        # Build valid (obs, action_chunk) pairs
        self.observations = []
        self.action_chunks = []
        self.episode_indices = []

        episodes = np.unique(all_episode_idx)
        for ep in episodes:
            ep_mask = all_episode_idx == ep
            ep_obs = all_obs[ep_mask]
            ep_deltas = all_delta_actions[ep_mask]
            ep_len = len(ep_obs)

            for t in range(ep_len - chunk_length + 1):
                if t + chunk_length <= ep_len:
                    self.observations.append(ep_obs[t])
                    self.action_chunks.append(ep_deltas[t : t + chunk_length])
                    self.episode_indices.append(ep)

        self.observations = np.array(self.observations, dtype=np.float32)
        self.action_chunks = np.array(self.action_chunks, dtype=np.float32)
        self.episode_indices = np.array(self.episode_indices)

        # Store stats
        self.num_episodes = len(episodes)
        self.num_frames = len(all_obs)
        self.num_pairs = len(self.observations)

    def __len__(self):
        return self.num_pairs

    def __getitem__(self, idx):
        obs = self.observations[idx].copy()
        chunk = self.action_chunks[idx].copy()

        # Normalize obs: divide by [512,512,512,512,2pi] (matching PPO's FixedObsNormWrapper)
        if self.obs_norm:
            obs = obs / self.OBS_HIGH[:self.obs_dim]

        # Normalize delta actions: divide by delta_max → [-1, 1] (matching PPO's tanh output)
        if self.action_norm:
            chunk = chunk / self.delta_max

        return (
            torch.FloatTensor(obs),
            torch.FloatTensor(chunk.flatten()),  # flattened (K*act_dim,)
        )

    def get_info(self):
        """Return dataset statistics for logging."""
        return {
            "num_episodes": self.num_episodes,
            "num_frames": self.num_frames,
            "num_pairs": self.num_pairs,
            "chunk_length": self.chunk_length,
            "obs_dim": self.obs_dim,
            "use_full_state": self.use_full_state,
        }


def build_bc_dataset(data_dir, chunk_length=1, obs_norm=True, action_norm=True,
                      val_ratio=0.2, seed=42, use_full_state=False, delta_max=100.0):
    """Build train and validation BC datasets from the LeRobot PushT data.

    Splits by episode (not by frame) to avoid data leakage.

    Returns:
        train_dataset, val_dataset, info_dict
    """
    parquet_path = os.path.join(data_dir, "data", "chunk-000", "file-000.parquet")
    states_path = os.path.join(data_dir, "reconstructed_states.npy")

    dataset = PushTBCDataset(
        parquet_path, chunk_length, obs_norm, action_norm,
        use_full_state=use_full_state, states_path=states_path,
        delta_max=delta_max
    )

    # Split by episode
    rng = np.random.RandomState(seed)
    unique_eps = np.unique(dataset.episode_indices)
    rng.shuffle(unique_eps)
    n_val = max(1, int(len(unique_eps) * val_ratio))
    val_eps = set(unique_eps[:n_val])
    train_eps = set(unique_eps[n_val:])

    train_mask = np.array([ep in train_eps for ep in dataset.episode_indices])
    val_mask = ~train_mask

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    info = dataset.get_info()
    info["train_pairs"] = len(train_indices)
    info["val_pairs"] = len(val_indices)
    info["train_episodes"] = len(train_eps)
    info["val_episodes"] = len(val_eps)

    return train_dataset, val_dataset, info


def make_bc_dataloaders(data_dir, chunk_length=1, batch_size=64,
                         obs_norm=True, action_norm=True, val_ratio=0.2,
                         seed=42, num_workers=0, use_full_state=False, delta_max=100.0):
    """Convenience: build train/val DataLoaders."""
    train_ds, val_ds, info = build_bc_dataset(
        data_dir, chunk_length, obs_norm, action_norm, val_ratio, seed,
        use_full_state=use_full_state, delta_max=delta_max
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
    )

    return train_loader, val_loader, info
