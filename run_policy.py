#!/usr/bin/env python3
import os
import torch
import imageio
import gymnasium as gym
from density_matching_v1 import CFG, GMMPolicy  
import numpy
def load_and_record(
        model_path="best_policy.pt",
        env_name="Humanoid-v4",
        device="cuda:0",
        save_path="policy_rollout.mp4",
        fps=30
    ):

    # -------------------------------
    # 1. Headless EGL mode
    # -------------------------------
    os.environ["MUJOCO_GL"] = "egl"

    # -------------------------------
    # 2. Make env (rgb_array mode)
    # -------------------------------
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()

    # -------------------------------
    # 3. Load Policy
    # -------------------------------
    cfg = CFG()
    cfg.env_name = env_name
    cfg.device = device
    cfg.obs_dim = obs.shape[0]
    cfg.act_dim = env.action_space.shape[0]

    policy = GMMPolicy(cfg, cfg.act_dim).to(device)
    policy.load_state_dict(torch.load(model_path, map_location=device))
    policy.eval()

    print(f"🎉 Loaded policy: {model_path}")

    frames = []
    total_return = 0
    done = False

    # -------------------------------
    # 4. Rollout loop
    # -------------------------------
    while not done:
        # capture frame
        frame = env.render()
        frames.append(frame)

        obs_t = torch.tensor(obs, device=device).float().unsqueeze(0)

        # critic-free action
        with torch.no_grad():
            act = policy.act_no_critic(obs_t)[0].cpu().numpy()

        obs, r, done, trunc, _ = env.step(act)
        total_return += r

        if trunc:
            break

    env.close()

    # -------------------------------
    # 5. Save video
    # -------------------------------
    print(f"🎥 Saving video → {save_path}")
    imageio.mimsave(save_path, frames, fps=fps)

    print(f"✨ Episode Return = {total_return:.1f}")
    print(f"🎬 Video saved: {save_path}")


if __name__ == "__main__":
    load_and_record()
