"""
Example of how to run the ccbench environment using a random sparse policy.

(c) 2026 Siemens AG

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import numpy as np
import pandas as pd

from gymnasium.envs.registration import make
from matplotlib import pyplot as plt

from ibs.envs.ccbench.ccbench_env import CCBenchEnv

ENV_ID = "ibs.envs.ccbench:CCBench-v1"
NUM_EPISODES = 1
MAX_EPISODE_STEPS = 100
KEEP_ACTIONS_ZERO_PERCENTAGE = 0.95
SEED = 42
DESIRED_GOAL = {"OD": 7.5, "WT": 0.4}

if __name__ == "__main__":
    env = make(
        ENV_ID,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    assert isinstance(env.unwrapped, CCBenchEnv)
    # separate seeded random number generators for the actions and their sparsity make the example reproducible
    env.action_space.seed(SEED)
    rng = np.random.default_rng(SEED)

    for episode in range(NUM_EPISODES):
        observations, actions, rewards, infos = [], [], [], []
        terminated = truncated = False
        # seeding the first reset is sufficient, further resets continue the random number stream
        obs, info = env.reset(seed=SEED if episode == 0 else None, options={"desired_goal": DESIRED_GOAL})
        while not (terminated or truncated):
            action = env.action_space.sample()
            action = {k: action[k] if rng.random() >= KEEP_ACTIONS_ZERO_PERCENTAGE else 0.0 for k in action}

            obs, reward, terminated, truncated, info = env.step(action)
            # obs and action are dictionaries, so we store them as they are for now and will convert to DataFrame later
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)

        df_observations = pd.DataFrame(map(lambda el: el["observation"], observations))
        df_goals = pd.DataFrame(map(lambda el: el["desired_goal"], observations)).rename(
            columns=env.unwrapped.target_obj_map
        )
        df_actions = pd.DataFrame(actions)
        df_rewards = pd.Series(rewards, name="Reward")

        df_env = pd.concat([df_observations, df_goals, df_actions, df_rewards], axis=1)

        # Group observations and objectives according to env.unwrapped.target_obj_map
        observation_names = list(env.unwrapped.observation_space["observation"].spaces.keys())  # pyright: ignore[reportIndexIssue]
        plot_columns = [
            obs
            for obs in observation_names
            if obs not in env.unwrapped.target_obj_map.values() and obs not in env.unwrapped.temp_channels
        ] + ["Reward"]
        axes = df_env[plot_columns].plot(subplots=True, figsize=(12, 10))
        for obs, obj in env.unwrapped.target_obj_map.items():
            if obs in plot_columns and obj in df_env.columns:
                # Find the subplot index for the objective
                idx = plot_columns.index(obs)
                df_env[obj].plot(ax=axes[idx], label=obj, c="red", linestyle="--")
                axes[idx].legend()
        plt.tight_layout()
        plt.show()
