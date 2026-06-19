"""MSE Behavior Cloning trainer with action chunking and evaluation.

The BC policy is trained to match PPO's interface:
- Input: normalized 5-dim state [agent_xy, block_xy, block_angle] / [512,512,512,512,2pi]
- Output: 2-dim action in [-1, 1] (like PPO's tanh squashed output)
- Eval env uses DeltaActionWrapper + FixedObsNormWrapper (same as PPO)
"""

import os
import numpy as np
import torch
import torch.nn as nn


class BCTrainer:
    """Trains an MLP action-chunking policy with MSE loss."""

    def __init__(self, policy, config, device="cpu"):
        self.policy = policy
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            policy.parameters(), lr=config.lr, weight_decay=config.weight_decay
        )

        if config.lr_schedule == "cosine":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config.max_epochs
            )
        else:
            self.scheduler = None

        self.ema = None
        if config.ema_decay > 0:
            from .bc_network import EMAPolicy
            self.ema = EMAPolicy(policy, decay=config.ema_decay)

    def _sync_ema_device(self):
        """Move EMA params to same device as policy."""
        if self.ema is not None:
            self.ema.policy = self.policy
            policy_device = next(self.policy.parameters()).device
            for key in self.ema.ema_params:
                self.ema.ema_params[key] = self.ema.ema_params[key].to(policy_device)

    def train_epoch(self, train_loader):
        """Run one training epoch, return average loss."""
        self.policy.train()
        total_loss = 0.0
        num_batches = 0

        for obs, target_chunk in train_loader:
            obs = obs.to(self.device)
            target_chunk = target_chunk.to(self.device)

            pred_chunk = self.policy(obs)
            loss = nn.functional.mse_loss(pred_chunk, target_chunk)

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.grad_clip)

            self.optimizer.step()

            if self.ema is not None:
                self.ema.update()

            total_loss += loss.item()
            num_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()

        return total_loss / max(num_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader):
        """Run validation, return average MSE loss."""
        self.policy.eval()
        total_loss = 0.0
        num_batches = 0

        for obs, target_chunk in val_loader:
            obs = obs.to(self.device)
            target_chunk = target_chunk.to(self.device)

            pred_chunk = self.policy(obs)
            loss = nn.functional.mse_loss(pred_chunk, target_chunk)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def evaluate_in_env(self, env, num_episodes=10, use_ema=False):
        """Evaluate policy in PushT environment with action chunking.

        The eval env should have DeltaActionWrapper + FixedObsNormWrapper,
        matching PPO's evaluation setup. The policy outputs actions in [-1, 1],
        which DeltaActionWrapper converts to displacements.

        Args:
            env: Wrapped PushT env (with DeltaActionWrapper + FixedObsNormWrapper).
            num_episodes: Number of evaluation episodes.
            use_ema: If True, use EMA policy parameters.

        Returns:
            mean_reward, std_reward
        """
        self.policy.eval()
        use_ema_policy = use_ema and self.ema is not None

        rewards = []
        for ep_idx in range(num_episodes):
            obs, info = env.reset(seed=self.config.seed + ep_idx)
            episode_reward = 0.0
            done = False

            while not done:
                # obs is already normalized by FixedObsNormWrapper
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

                # Predict action chunk (output in [-1, 1])
                with torch.no_grad():
                    if use_ema_policy:
                        chunk = self.ema.predict_chunk(obs_t)
                    else:
                        chunk = self.policy.predict_chunk(obs_t)
                    chunk = chunk.cpu().numpy()[0]  # (K, act_dim)

                # Execute chunk open-loop
                for k in range(self.config.chunk_length):
                    if done:
                        break
                    # Action in [-1, 1], DeltaActionWrapper handles conversion
                    action = np.clip(chunk[k], -1.0, 1.0).astype(np.float32)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += reward
                    done = terminated or truncated

            rewards.append(episode_reward)

        return np.mean(rewards), np.std(rewards)

    def _record_video(self, env, path, use_ema=False, max_steps=300):
        """Record one evaluation video."""
        from src.utils.video import record_episode

        self.policy.eval()
        use_ema_policy = use_ema and self.ema is not None
        trainer = self

        class BCAgentWrapper:
            """Wraps BC policy to match the RL agent interface for video recording."""
            def __init__(self):
                pass

            def get_deterministic_action(self, obs_t):
                # obs_t: (1, obs_dim) — normalized by FixedObsNormWrapper
                with torch.no_grad():
                    if use_ema_policy:
                        chunk = trainer.ema.predict_chunk(obs_t)
                    else:
                        chunk = trainer.policy.predict_chunk(obs_t)
                    action = chunk[0, 0, :]  # first action of chunk
                return action.unsqueeze(0)

            def eval(self):
                return self

        wrapper = BCAgentWrapper()
        reward = record_episode(env, wrapper, path, max_steps, self.device)
        return reward

    def save(self, path, epoch=0, best_reward=-float("inf")):
        """Save checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            "policy": self.policy.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch,
            "best_reward": float(best_reward),
        }
        if self.ema is not None:
            state["ema"] = self.ema.state_dict()
        if self.scheduler is not None:
            state["scheduler"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load(self, path):
        """Load checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(checkpoint["policy"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if "ema" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        if "scheduler" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        return checkpoint.get("epoch", 0), checkpoint.get("best_reward", -float("inf"))

    def train(self, train_loader, val_loader, eval_env=None,
              logger=None, tb_logger=None, console=None, video_dir=None):
        """Full training loop."""
        cfg = self.config
        self.policy.to(self.device)
        self._sync_ema_device()

        best_reward = -float("inf")
        best_val_loss = float("inf")
        no_improve_count = 0

        print(f"BC training on {self.device}")
        print(f"  chunk_length={cfg.chunk_length}, hidden_dim={cfg.hidden_dim}, "
              f"num_layers={cfg.num_layers}, lr={cfg.lr}")
        print(f"  max_epochs={cfg.max_epochs}, batch_size={cfg.batch_size}, "
              f"ema_decay={cfg.ema_decay}, grad_clip={cfg.grad_clip}")
        print(f"  obs_norm={cfg.obs_norm}, action_norm={cfg.action_norm}, "
              f"use_full_state={cfg.use_full_state}")

        for epoch in range(1, cfg.max_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            log_row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
            if self.scheduler is not None:
                log_row["lr"] = self.scheduler.get_last_lr()[0]

            # TensorBoard: train/val loss every epoch
            if tb_logger:
                tb_logger.log_scalars("loss", {
                    "train": train_loss,
                    "val": val_loss,
                }, epoch)
                if self.scheduler is not None:
                    tb_logger.log_scalar("lr", self.scheduler.get_last_lr()[0], epoch)

            # Evaluate in environment
            if eval_env is not None and epoch % cfg.eval_freq == 0:
                mean_r, std_r = self.evaluate_in_env(eval_env, cfg.eval_episodes)
                log_row["eval_reward"] = mean_r
                log_row["eval_std"] = std_r

                ema_reward = None
                if self.ema is not None:
                    ema_mean, ema_std = self.evaluate_in_env(
                        eval_env, cfg.eval_episodes, use_ema=True
                    )
                    log_row["ema_eval_reward"] = ema_mean
                    log_row["ema_eval_std"] = ema_std
                    ema_reward = ema_mean

                # TensorBoard: eval metrics
                if tb_logger:
                    tb_logger.log_scalars("eval", {
                        "reward": mean_r,
                        "reward_std": std_r,
                        "ema_reward": ema_reward,
                    }, epoch)

                effective_reward = ema_reward if ema_reward is not None else mean_r
                if effective_reward > best_reward:
                    best_reward = effective_reward
                    self.save(os.path.join(cfg.output_dir, "best.pt"), epoch, best_reward)
                    if video_dir:
                        self._record_video(
                            eval_env,
                            os.path.join(video_dir, f"best_epoch{epoch}.mp4"),
                            use_ema=(ema_reward is not None and ema_reward > mean_r),
                        )
                    print(f"  → New best: {best_reward:.4f}")

                if console:
                    msg = (f"[BC] Epoch {epoch:>4d} | "
                           f"train={train_loss:.6f} val={val_loss:.6f} | "
                           f"rew={mean_r:.4f}±{std_r:.4f}")
                    if ema_reward is not None:
                        msg += f" ema={ema_reward:.4f}"
                    print(msg)

            elif console:
                print(f"[BC] Epoch {epoch:>4d} | train={train_loss:.6f} val={val_loss:.6f}")

            if logger:
                logger.log(log_row)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                no_improve_count = 0
            else:
                no_improve_count += 1

            if cfg.early_stop_patience > 0 and no_improve_count >= cfg.early_stop_patience:
                print(f"  Early stopping at epoch {epoch}")
                break

            if epoch % cfg.save_freq == 0:
                self.save(os.path.join(cfg.output_dir, f"ckpt_epoch{epoch}.pt"), epoch, best_reward)

        self.save(os.path.join(cfg.output_dir, "final.pt"), epoch, best_reward)

        if eval_env is not None and video_dir:
            self._record_video(eval_env, os.path.join(video_dir, "final.mp4"))

        print(f"\nTraining done. Best reward: {best_reward:.4f}")
        return best_reward
