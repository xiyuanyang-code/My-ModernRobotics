"""SAC (Soft Actor-Critic) algorithm implementation."""

import os
import numpy as np
import torch
import torch.nn as nn

from .replay_buffer import ReplayBuffer


class SAC:
    """SAC trainer with twin Q-networks, auto-tuned alpha, and evaluation."""

    def __init__(self, agent, config, device="cpu"):
        self.agent = agent
        self.config = config
        self.device = device

        self.actor_optimizer = torch.optim.Adam(agent.actor.parameters(), lr=config.lr_actor, eps=1e-5)
        self.q_optimizer = torch.optim.Adam(
            list(agent.q1.parameters()) + list(agent.q2.parameters()),
            lr=config.lr_critic, eps=1e-5
        )

        # Auto alpha tuning
        if config.auto_alpha:
            self.target_entropy = config.target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_alpha)
        else:
            self.log_alpha = torch.tensor(np.log(config.alpha), device=device)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def train_step(self, batch):
        """One SAC update step: Q-networks, policy, alpha."""
        obs = batch["obs"]
        actions = batch["action"]
        rewards = batch["reward"]
        next_obs = batch["next_obs"]
        dones = batch["done"]

        # --- Update Q-networks ---
        with torch.no_grad():
            next_actions, next_log_probs = self.agent.actor.get_action(next_obs)
            q1_next = self.agent.q1_target(next_obs, next_actions)
            q2_next = self.agent.q2_target(next_obs, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q = rewards + self.config.gamma * (1 - dones) * q_next

        q1 = self.agent.q1(obs, actions)
        q2 = self.agent.q2(obs, actions)
        q1_loss = nn.functional.mse_loss(q1, target_q)
        q2_loss = nn.functional.mse_loss(q2, target_q)
        q_loss = q1_loss + q2_loss

        self.q_optimizer.zero_grad()
        q_loss.backward()
        q_grad_norm = nn.utils.clip_grad_norm_(
            list(self.agent.q1.parameters()) + list(self.agent.q2.parameters()), float("inf")
        )
        self.q_optimizer.step()

        # --- Update policy ---
        new_actions, log_probs = self.agent.actor.get_action(obs)
        q1_new = self.agent.q1(obs, new_actions)
        q2_new = self.agent.q2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(self.agent.actor.parameters(), float("inf"))
        self.actor_optimizer.step()

        # --- Update alpha ---
        alpha_loss = 0.0
        if self.config.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha_loss = alpha_loss.item()

        # --- Soft update targets ---
        self.agent.soft_update(self.config.tau)

        return {
            "q_loss": q_loss.item(),
            "q1": q1.mean().item(),
            "q2": q2.mean().item(),
            "policy_loss": policy_loss.item(),
            "alpha": self.alpha,
            "alpha_loss": alpha_loss,
            "q_grad_norm": q_grad_norm.item() if torch.is_tensor(q_grad_norm) else q_grad_norm,
            "actor_grad_norm": actor_grad_norm.item() if torch.is_tensor(actor_grad_norm) else actor_grad_norm,
        }

    def evaluate(self, env, num_episodes=10):
        """Run deterministic evaluation episodes."""
        self.agent.eval()
        rewards = []

        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.agent.actor.get_deterministic_action(obs_t)
                action_np = action.cpu().numpy()[0]
                # DeltaActionWrapper handles [-1, 1] → delta conversion
                action_env = action_np

                obs, reward, terminated, truncated, info = env.step(action_env)
                episode_reward += reward
                done = terminated or truncated

            rewards.append(episode_reward)

        return np.mean(rewards), np.std(rewards)

    def _record_video(self, eval_env, path, max_steps=300, num_candidates=4):
        """Record multiple episodes, keep the best one."""
        import shutil
        from src.utils.video import record_episode
        self.agent.eval()

        # Record num_candidates episodes
        candidates = []
        for i in range(num_candidates):
            tmp_path = path.replace(".mp4", f"_cand{i}.mp4")
            reward = record_episode(eval_env, self.agent, tmp_path, max_steps, self.device)
            candidates.append((tmp_path, reward))

        # Keep the best
        best_path, best_reward = max(candidates, key=lambda x: x[1])
        shutil.move(best_path, path)

        # Delete the rest
        for tmp_path, _ in candidates:
            if tmp_path != best_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

        return best_reward

    def train(self, train_env, eval_env, logger=None, tb_logger=None, console=None, video_dir=None, start_step=0):
        """Main SAC training loop."""
        import time

        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Determine device
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        self.agent.to(self.device)
        self.log_alpha = self.log_alpha.to(self.device)

        print(f"SAC training on {self.device}")
        print(f"  max_steps={cfg.max_steps}, buffer={cfg.buffer_size}, batch={cfg.batch_size}")
        if start_step > 0:
            print(f"  resuming from step {start_step}")

        obs_dim = train_env.observation_space.shape[0]
        act_dim = train_env.action_space.shape[0]
        buffer = ReplayBuffer(cfg.buffer_size, obs_dim, act_dim, self.device)

        obs, _ = train_env.reset()
        total_steps = start_step
        best_reward = -float("inf")
        t_start = time.time()
        ep_reward = 0.0
        ep_len = 0

        while total_steps < cfg.max_steps:
            total_steps += 1

            # Select action
            self.agent.eval()
            if total_steps < cfg.warmup_steps:
                action_env = train_env.action_space.sample()
                action_raw = action_env
            else:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action, _ = self.agent.actor.get_action(obs_t)
                action_raw = action.cpu().numpy()[0]
                action_env = action_raw

            # Step environment
            next_obs, reward, terminated, truncated, info = train_env.step(action_env)
            done = terminated or truncated
            buffer.add(obs, action_raw, reward, next_obs, float(done))
            obs = next_obs
            ep_reward += reward
            ep_len += 1

            if done:
                obs, _ = train_env.reset()
                ep_reward_val = ep_reward
                ep_len_val = ep_len
                ep_reward = 0.0
                ep_len = 0
            else:
                ep_reward_val = None
                ep_len_val = None

            # Train
            stats = {}
            if total_steps >= cfg.warmup_steps:
                self.agent.train()
                for _ in range(cfg.updates_per_step):
                    batch = buffer.sample(cfg.batch_size)
                    stats = self.train_step(batch)

            # FPS
            fps = total_steps / (time.time() - t_start + 1e-8)

            # --- Logging ---
            should_log = total_steps % cfg.log_freq == 0

            if logger and should_log and stats:
                logger.log({
                    "step": total_steps,
                    "q_loss": stats["q_loss"],
                    "q1": stats["q1"],
                    "q2": stats.get("q2"),
                    "policy_loss": stats["policy_loss"],
                    "alpha": stats["alpha"],
                    "q_grad_norm": stats.get("q_grad_norm"),
                    "actor_grad_norm": stats.get("actor_grad_norm"),
                    "fps": fps,
                })

            if tb_logger and should_log and stats:
                tb_logger.log_scalars("train", {
                    "q_loss": stats["q_loss"],
                    "q1": stats["q1"],
                    "q2": stats.get("q2"),
                    "policy_loss": stats["policy_loss"],
                    "alpha": stats["alpha"],
                    "alpha_loss": stats.get("alpha_loss"),
                    "q_grad_norm": stats.get("q_grad_norm"),
                    "actor_grad_norm": stats.get("actor_grad_norm"),
                    "buffer_size": len(buffer),
                }, total_steps)
                tb_logger.log_scalar("perf/fps", fps, total_steps)

            if console and should_log and stats:
                print(f"[SAC] Step {total_steps:>8d} | "
                      f"q={stats['q_loss']:.4f} pol={stats['policy_loss']:.4f} "
                      f"α={stats['alpha']:.4f} | "
                      f"buf={len(buffer)} fps={fps:.0f}")

            # Log completed episodes
            if tb_logger and ep_reward_val is not None:
                tb_logger.log_scalars("ep", {
                    "reward": ep_reward_val,
                    "length": ep_len_val,
                }, total_steps)

            # Evaluate
            if total_steps % cfg.eval_freq == 0:
                mean_r, std_r = self.evaluate(eval_env, cfg.eval_episodes)
                print(f"  [Eval] step={total_steps}  reward={mean_r:.4f} ± {std_r:.4f}")

                if logger:
                    logger.log({"step": total_steps, "eval_reward": mean_r, "eval_std": std_r})
                if tb_logger:
                    tb_logger.log_scalars("eval", {
                        "reward": mean_r,
                        "reward_std": std_r,
                    }, total_steps)

                # Save best + video
                if mean_r > best_reward:
                    best_reward = mean_r
                    self.save(os.path.join(cfg.output_dir, "best.pt"))
                    if video_dir:
                        self._record_video(eval_env, os.path.join(video_dir, f"best_step{total_steps}.mp4"))
                    print(f"  → New best: {best_reward:.4f}")

            # Periodic save + video
            if total_steps % cfg.save_freq == 0:
                self.save(os.path.join(cfg.output_dir, f"ckpt_{total_steps}.pt"))
                if video_dir:
                    self._record_video(eval_env, os.path.join(video_dir, f"step_{total_steps}.mp4"))

        # Final save + video
        self.save(os.path.join(cfg.output_dir, "final.pt"))
        if video_dir:
            self._record_video(eval_env, os.path.join(video_dir, "final.mp4"))
        print(f"\nTraining done. Best reward: {best_reward:.4f}")
        return best_reward

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "agent": self.agent.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "log_alpha": self.log_alpha,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer"])
        self.log_alpha = checkpoint["log_alpha"].to(self.device)
