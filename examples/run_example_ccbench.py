"""Example of how to run the distillation process benchmark using a pre-trained MOOSE policy."""

import numpy as np
import pandas as pd

from gymnasium.envs.registration import make
from matplotlib import pyplot as plt

from ibs.envs.ccbench.ccbench_env import CCBenchEnv

ENV_ID = "ibs.envs.ccbench:CCBench-v0"
NUM_EPISODES = 1
MAX_EPISODE_STEPS = 100
KEEP_ACTIONS_ZERO_PERCENTAGE = 0.95

if __name__ == "__main__":
    env = make(
        ENV_ID,
        max_episode_steps=MAX_EPISODE_STEPS,
    )
    assert isinstance(env.unwrapped, CCBenchEnv)

    for _ in range(NUM_EPISODES):
        observations, actions, rewards, infos = [], [], [], []
        terminated = False
        obs, info = env.reset()
        step_count = 0
        while not terminated:
            action = env.action_space.sample()
            action = {k: action[k] if np.random.rand() >= KEEP_ACTIONS_ZERO_PERCENTAGE else 0.0 for k in action}
            
            obs, reward, terminated, _, info = env.step(action)
            # obs and action are dictionaries, so we store them as they are for now and will convert to DataFrame later
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            # if RENDER_SCENE:
            #     env.render()
            step_count += 1

        df_observations = pd.DataFrame(map(lambda el: el["observation"], observations))
        df_goals = pd.DataFrame(map(lambda el: el["desired_goal"], observations)).rename(columns=env.unwrapped.target_obj_map)
        df_actions = pd.DataFrame(actions)
        df_rewards = pd.Series(rewards, name="Reward")

        df_env = pd.concat([df_observations, df_goals, df_actions, df_rewards], axis=1)

        # Group observations and objectives according to env.unwrapped.target_obj_map
        observation_names = list(env.unwrapped.observation_space["observation"].spaces.keys()) # pyright: ignore[reportIndexIssue]
        axes = df_env[
            [obs for obs in observation_names if obs not in env.unwrapped.target_obj_map.values()] + ["Reward"]
        ].plot(subplots=True, figsize=(12, 10))
        for obs, obj in env.unwrapped.target_obj_map.items():
            if obs in df_env.columns and obj in df_env.columns:
                # Find the subplot index for the objective
                idx = observation_names.index(obs)
                df_env[obj].plot(ax=axes[idx], label=obj, c="red", linestyle="--")
                axes[idx].legend()
        plt.tight_layout()
        plt.show()
