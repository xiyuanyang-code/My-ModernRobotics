# RL and Imitation Learning on PushT

Implementation of **PPO**, **SAC**, and **MSE Behavior Cloning with Action Chunks** for the PushT 2D manipulation environment.

## Project Structure

```
├── src/
│   ├── train_ppo.py          # PPO training entry point
│   ├── train_sac.py          # SAC training entry point
│   ├── train_bc.py           # Behavior Cloning training entry point
│   ├── evaluate.py           # RL checkpoint evaluation & video recording
│   ├── evaluate_bc.py        # BC checkpoint evaluation & video recording
│   ├── run_ablations.py      # RL ablation runner
│   ├── run_bc_ablations.py   # BC ablation runner
│   ├── plot_results.py       # Result plotting utility
│   ├── configs/              # Dataclass configs (PPOConfig, SACConfig, BCConfig)
│   ├── core/                 # Algorithm implementations
│   │   ├── ppo.py            # PPO trainer (GAE, clipped surrogate)
│   │   ├── sac.py            # SAC trainer (twin Q, auto-alpha)
│   │   ├── bc_trainer.py     # BC trainer (MSE loss, EMA)
│   │   ├── bc_dataset.py     # PushT BC dataset (action chunking)
│   │   ├── bc_network.py     # BC MLP policy
│   │   ├── networks.py       # PPO/SAC actor-critic networks
│   │   ├── env.py            # Env wrappers (delta action, reward shaping, obs norm)
│   │   ├── rollout.py        # Rollout buffer for PPO
│   │   └── replay_buffer.py  # Replay buffer for SAC
│   └── utils/                # Logging, plotting, video recording
├── scripts/                  # Shell scripts (main entry points)
│   ├── train_ppo_delta.sh    # Train PPO with delta actions
│   ├── train_ppo_abs.sh      # Train PPO with absolute actions
│   ├── train_sac_delta.sh    # Train SAC with delta actions
│   ├── train_sac_abs.sh      # Train SAC with absolute actions
│   ├── train_bc.sh           # Train BC with action chunking
│   ├── evaluate.sh           # Evaluate a checkpoint
│   ├── run_ablations.sh      # Run all RL ablations
│   ├── run_bc_ablations.sh   # Run all BC ablations
│   ├── rl_ablation/          # Individual RL ablation scripts
│   └── il_ablation/          # Individual BC ablation scripts
├── gym_pusht/                # PushT gymnasium environment
├── data/                     # Dataset (LeRobot PushT)
├── outputs/                  # Training outputs (checkpoints, logs, TB events)
└── papers/                   # LaTeX paper and figures
```

## Installation

```bash
# Python 3.10+
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install torch tensorboard matplotlib
```

## Quick Start

### Train PPO (recommended)

```bash
bash scripts/train_ppo_delta.sh
```

### Train SAC

```bash
bash scripts/train_sac_delta.sh
```

### Train Behavior Cloning

```bash
bash scripts/train_bc.sh                          # default K=8
bash scripts/train_bc.sh --chunk_length 16        # K=16
bash scripts/train_bc.sh --chunk_length 1 --obs_dim 2  # 2D obs, K=1
```

### Evaluate a Checkpoint

```bash
bash scripts/evaluate.sh --checkpoint outputs/ppo/best.pt --algo ppo
bash scripts/evaluate.sh --checkpoint outputs/bc/best.pt --algo bc
```

## Ablation Studies

### RL Ablations

Run all 4 ablation groups (algorithm, action space, reward shaping, obs normalization):

```bash
bash scripts/run_ablations.sh
```

Or run individual ablations:

```bash
bash scripts/rl_ablation/01_algo_ppo.sh
bash scripts/rl_ablation/01_algo_sac.sh
bash scripts/rl_ablation/02_action_ppo_delta.sh
bash scripts/rl_ablation/02_action_ppo_abs.sh
# ... etc
```

### BC Ablations

Run all 3 ablation groups (chunk length, obs dimension, EMA):

```bash
bash scripts/run_bc_ablations.sh
```

Or run individual ablations:

```bash
bash scripts/il_ablation/ablation_chunk_k1.sh
bash scripts/il_ablation/ablation_chunk_k8.sh
bash scripts/il_ablation/ablation_obsdim_5d.sh
bash scripts/il_ablation/ablation_ema_on.sh
# ... etc
```

## Plotting

Generate all paper figures from TensorBoard logs:

```bash
python papers/plot_scripts/plot_all.py
```

Figures are saved to `papers/figures/` as PDFs.

## Key Hyperparameters

| Parameter | PPO | SAC | BC |
|---|---|---|---|
| Network | 4-layer MLP, dim 1024 | 4-layer MLP, dim 1024 | 2-layer MLP, dim 256 |
| Learning rate | 3e-4 | 3e-4 | 1e-3 |
| Training steps | 1M env steps | 1M env steps | 200 epochs |
| Action space | Delta (δ_max=100) | Delta (δ_max=100) | Delta (normalized) |
| Chunk length | — | — | K=8 (default) |
| EMA decay | — | — | α=0.999 |

## Results

| Method | Best Eval Reward | Final Eval Reward |
|---|---|---|
| SAC | **292.9** | **281.5** |
| PPO | 50.2 | 23.5 |
| BC K=16 EMA | 35.3 | 35.3 |
| BC K=8 EMA | 29.0 | 29.0 |
