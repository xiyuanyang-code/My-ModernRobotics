"""PPO (Proximal Policy Optimization) algorithm implementation."""

import os
import numpy as np
import torch
import torch.nn as nn

from .rollout import RolloutBuffer


class PPO:
    """PPO trainer with GAE, clipped surrogate loss, and evaluation."""

    def __init__(self, agent, config, device="cpu"):
        self.agent = agent
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(agent.parameters(), lr=config.lr, eps=1e-5)

    def collect_rollouts(self, env, rollout_buffer, obs):
        """Run policy in environment, fill rollout buffer. Returns latest obs and stats."""
        cfg = self.config
        self.agent.eval()

        all_rewards = []
        all_actions = []
        all_values = []
        episode_lengths = []
        episode_rewards = []
        current_ep_reward = np.zeros(rollout_buffer.num_envs)
        current_ep_len = np.zeros(rollout_buffer.num_envs)

        for step in range(rollout_buffer.num_steps):
            obs_t = torch.FloatTensor(obs).to(self.device)

            with torch.no_grad():
                action, log_prob, value = self.agent.get_action_and_value(obs_t)

            action_np = action.cpu().numpy()
            action_env = action_np

            next_obs, rewards, terminateds, truncateds, infos = env.step(action_env)
            dones = np.logical_or(terminateds, truncateds).astype(np.float32)

            rollout_buffer.obs[step] = obs
            rollout_buffer.actions[step] = action_np
            rollout_buffer.log_probs[step] = log_prob.cpu().numpy()
            rollout_buffer.rewards[step] = rewards
            rollout_buffer.dones[step] = dones
            rollout_buffer.values[step] = value.cpu().numpy()

            all_rewards.append(rewards)
            all_actions.append(action_np)
            all_values.append(value.cpu().numpy())

            current_ep_reward += rewards
            current_ep_len += 1

            # Track completed episodes
            for i in range(rollout_buffer.num_envs):
                if dones[i]:
                    episode_rewards.append(current_ep_reward[i])
                    episode_lengths.append(current_ep_len[i])
                    current_ep_reward[i] = 0
                    current_ep_len[i] = 0

            obs = next_obs

        # Bootstrap value for GAE
        with torch.no_grad():
            last_values = self.agent.get_value(torch.FloatTensor(obs).to(self.device)).cpu().numpy()

        rollout_buffer.compute_gae(last_values, cfg.gamma, cfg.gae_lambda)

        # Collect stats
        all_rewards = np.array(all_rewards)  # (num_steps, num_envs)
        all_actions = np.array(all_actions)  # (num_steps, num_envs, act_dim)
        all_values = np.array(all_values)    # (num_steps, num_envs)

        stats = {
            "rollout_reward_mean": all_rewards.mean(),
            "rollout_reward_max": all_rewards.max(),
            "rollout_reward_min": all_rewards.min(),
            "rollout_reward_std": all_rewards.std(),
            "action_mean": all_actions.mean(),
            "action_std": all_actions.std(),
            "value_mean": all_values.mean(),
            "num_episodes": len(episode_rewards),
        }
        if episode_rewards:
            stats["ep_reward_mean"] = np.mean(episode_rewards)
            stats["ep_reward_max"] = np.max(episode_rewards)
            stats["ep_reward_min"] = np.min(episode_rewards)
            stats["ep_len_mean"] = np.mean(episode_lengths)

        return obs, stats

    def update(self, rollout_buffer):
        """Run PPO update epochs on collected rollout."""
        cfg = self.config
        self.agent.train()

        total_pg_loss = 0
        total_vf_loss = 0
        total_entropy = 0
        total_clip_fraction = 0
        num_updates = 0

        for _ in range(cfg.epochs_per_update):
            for batch in rollout_buffer.get_batches(cfg.batch_size):
                obs = batch["obs"]
                actions = batch["action"]
                old_log_probs = batch["log_prob"]
                advantages = batch["advantage"]
                returns = batch["return"]

                # Normalize advantages (divide by std only, keep sign)
                advantages = advantages / (advantages.std() + 1e-8)

                # Get current policy evaluation
                new_log_probs, entropy, values = self.agent.evaluate_actions(obs, actions)

                # Policy loss (clipped surrogate)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Clip fraction: how many ratios hit the clip boundary
                clip_fraction = ((ratio - 1.0).abs() > cfg.clip_eps).float().mean()

                # Value loss
                vf_loss = nn.functional.mse_loss(values, returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + cfg.vf_coef * vf_loss + cfg.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += entropy.mean().item()
                total_clip_fraction += clip_fraction.item()
                num_updates += 1

        # Explained variance: how well value function predicts returns
        # EV = 1 - Var(returns - values) / Var(values); 1 = perfect, 0 = mean predictor
        all_returns = rollout_buffer.returns.reshape(-1)
        all_values = rollout_buffer.values.reshape(-1)
        explained_variance = 1.0 - np.var(all_returns - all_values) / (np.var(all_values) + 1e-8)

        current_lr = self.optimizer.param_groups[0]["lr"]

        return {
            "pg_loss": total_pg_loss / num_updates,
            "vf_loss": total_vf_loss / num_updates,
            "entropy": total_entropy / num_updates,
            "clip_fraction": total_clip_fraction / num_updates,
            "explained_variance": explained_variance,
            "grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
            "lr": current_lr,
            "total_loss": (total_pg_loss + cfg.vf_coef * total_vf_loss) / num_updates,
        }

    def evaluate(self, env, num_episodes=10):
        """Run deterministic evaluation episodes, return mean reward."""
        self.agent.eval()
        rewards = []

        for _ in range(num_episodes):
            obs, info = env.reset()
            episode_reward = 0
            done = False

            while not done:
                obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.agent.get_deterministic_action(obs_t)
                action_np = action.cpu().numpy()[0]
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
        """Main PPO training loop."""
        import time

        cfg = self.config
        os.makedirs(cfg.output_dir, exist_ok=True)

        # Determine device
        if cfg.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = cfg.device
        self.agent.to(self.device)

        print(f"PPO training on {self.device}")
        print(f"  max_steps={cfg.max_steps}, num_envs={cfg.num_envs}, rollout={cfg.rollout_steps}")
        if start_step > 0:
            print(f"  resuming from step {start_step}")

        # Rollout buffer
        obs_dim = train_env.single_observation_space.shape[0]
        act_dim = train_env.single_action_space.shape[0]
        buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, obs_dim, act_dim, self.device)

        # Get initial obs
        obs, _ = train_env.reset()
        total_steps = start_step
        best_reward = -float("inf")
        t_start = time.time()

        # Helper: update curriculum progress in all sub-envs
        def set_curriculum_progress(p):
            if hasattr(train_env, 'envs'):
                for e in train_env.envs:
                    _set_curriculum(e, p)
            else:
                _set_curriculum(train_env, p)

        def _set_curriculum(env, p):
            if hasattr(env, 'set_progress'):
                env.set_progress(p)
            elif hasattr(env, 'env'):
                _set_curriculum(env.env, p)

        while total_steps < cfg.max_steps:
            # Collect rollouts
            obs, rollout_stats = self.collect_rollouts(train_env, buffer, obs)
            total_steps += cfg.rollout_steps * cfg.num_envs

            # Update curriculum progress (linear from easy to hard)
            set_curriculum_progress(total_steps / cfg.max_steps)

            # Update
            stats = self.update(buffer)

            # Linear lr decay
            if cfg.lr_schedule == "linear":
                frac = 1.0 - total_steps / cfg.max_steps
                new_lr = cfg.lr * frac
                for pg in self.optimizer.param_groups:
                    pg["lr"] = max(new_lr, 1e-6)

            # FPS
            fps = total_steps / (time.time() - t_start + 1e-8)

            # --- Logging ---
            should_log = total_steps % cfg.log_freq < cfg.rollout_steps * cfg.num_envs

            if logger and should_log:
                log_row = {
                    "step": total_steps,
                    "pg_loss": stats["pg_loss"],
                    "vf_loss": stats["vf_loss"],
                    "entropy": stats["entropy"],
                    "clip_fraction": stats["clip_fraction"],
                    "explained_variance": stats["explained_variance"],
                    "grad_norm": stats["grad_norm"],
                    "lr": stats["lr"],
                    "fps": fps,
                    **rollout_stats,
                }
                logger.log(log_row)

            if tb_logger and should_log:
                # Training metrics
                tb_logger.log_scalars("train", {
                    "pg_loss": stats["pg_loss"],
                    "vf_loss": stats["vf_loss"],
                    "total_loss": stats["total_loss"],
                    "entropy": stats["entropy"],
                    "clip_fraction": stats["clip_fraction"],
                    "explained_variance": stats["explained_variance"],
                    "grad_norm": stats["grad_norm"],
                    "lr": stats["lr"],
                }, total_steps)
                # Rollout metrics
                tb_logger.log_scalars("rollout", {
                    "reward_mean": rollout_stats["rollout_reward_mean"],
                    "reward_max": rollout_stats["rollout_reward_max"],
                    "reward_min": rollout_stats["rollout_reward_min"],
                    "reward_std": rollout_stats["rollout_reward_std"],
                    "action_mean": rollout_stats["action_mean"],
                    "action_std": rollout_stats["action_std"],
                    "value_mean": rollout_stats["value_mean"],
                }, total_steps)
                # Episode metrics (if any episodes completed)
                if rollout_stats.get("num_episodes", 0) > 0:
                    tb_logger.log_scalars("ep", {
                        "reward_mean": rollout_stats["ep_reward_mean"],
                        "reward_max": rollout_stats["ep_reward_max"],
                        "reward_min": rollout_stats["ep_reward_min"],
                        "len_mean": rollout_stats["ep_len_mean"],
                    }, total_steps)
                # Performance
                tb_logger.log_scalar("perf/fps", fps, total_steps)

            if console and should_log:
                print(f"[PPO] Step {total_steps:>8d} | "
                      f"pg={stats['pg_loss']:.4f} vf={stats['vf_loss']:.4f} "
                      f"ent={stats['entropy']:.3f} clip={stats['clip_fraction']:.2f} "
                      f"ev={stats['explained_variance']:.3f} | "
                      f"rew={rollout_stats['rollout_reward_mean']:.3f} "
                      f"[{rollout_stats['rollout_reward_min']:.2f}, {rollout_stats['rollout_reward_max']:.2f}] "
                      f"act_std={rollout_stats['action_std']:.3f} "
                      f"ep={rollout_stats.get('num_episodes', 0)} "
                      f"ep_rew={rollout_stats.get('ep_reward_mean', 0):.2f} "
                      f"fps={fps:.0f}")

            # Evaluate
            if total_steps % cfg.eval_freq < cfg.rollout_steps * cfg.num_envs:
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
            if total_steps % cfg.save_freq < cfg.rollout_steps * cfg.num_envs:
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
            "optimizer": self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint["agent"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
