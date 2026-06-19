# Modern Robotics

Introduction to Intelligent Robotics (SJTU, Spring 2026) — Programming Homework and Course Project.

## Programming Homework

### HM1 — Robot Kinematics

Implementation of forward kinematics functions for robotic manipulators, including product-of-exponentials (PoE) formulation, FK for open chains, and kinematic analysis of a 2D/3D linkage system.

- **Key files:** `kin_func_skeleton.py`, `problem_2.py`, `problem_3.py`

### HM2 — Robot Control

Two classical control algorithms:

| Task | Algorithm | Environment |
|------|-----------|-------------|
| Part 1 | PID Controller | CartPole |
| Part 2 | iterative LQR (iLQR) | 2-Link Planar Arm |

- **Key files:** `cartpole_entry.py`, `dualarm_entry.py`, `robotics_control/`

### HM3 — RL & Imitation Learning

Training and comparing reinforcement learning and imitation learning policies on the PushT 2D manipulation environment.

| Method | Algorithm |
|--------|-----------|
| RL | PPO, SAC |
| Imitation Learning | MSE Behavior Cloning with Action Chunks |

- **Key files:** `src/train_ppo.py`, `src/train_sac.py`, `src/train_bc.py`
- **Environment:** [gym-pusht](https://github.com/huggingface/gym-pusht)
- **Dataset:** [LeRobot PushT](https://huggingface.co/datasets/lerobot/pusht)

### HM4 — Camera Calibration and Stereo Vision

A complete camera calibration and stereo depth estimation pipeline:

1. Single camera calibration (intrinsics, distortion, extrinsics)
2. Stereo camera calibration and image rectification
3. Disparity and depth estimation

- **Key files:** `robotics_perception/`, `scripts/single_camera_entry.py`, `scripts/stereo_entry.py`

## Course Project

Code and papers to be released.