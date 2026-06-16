"""Example of how to run the Industrial Benchmark using the registered Gymnasium environment."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from gymnasium.envs.registration import make

import ibs  # noqa: F401 – triggers environment registration

ENV_ID = "IndustrialBenchmark-v0"
NUM_EPISODES = 5
MAX_EPISODE_STEPS = 1000
SETPOINT = 70

if __name__ == "__main__":
    env = make(
        ENV_ID,
        setpoint=SETPOINT,
        reset_after_timesteps=MAX_EPISODE_STEPS,
    )

    for _ in range(NUM_EPISODES):
        observations, actions, rewards = [], [], []
        terminated = False
        obs, info = env.reset()
        step_count = 0
        while not terminated:
            action = env.action_space.sample()
            obs, reward, terminated, _, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            step_count += 1

        obs_names = ["Setpoint", "Velocity", "Gain", "Shift", "Fatigue", "Consumption"]
        action_names = ["Action: Delta Velocity", "Action: Delta Gain", "Action: Delta Shift"]

        res = np.concatenate(
            [
                observations,
                actions,
                np.array(rewards).reshape(len(rewards), 1),
            ],
            axis=1,
        )
        df_env = pd.DataFrame(res, columns=obs_names + action_names + ["Reward"])

        axes = df_env[obs_names + ["Reward"]].plot(subplots=True, figsize=(12, 10), grid=True)

        plt.tight_layout()
        plt.show()
