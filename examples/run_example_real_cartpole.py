"""Example of how to run the real-cartpole benchmark using a pre-trained MOOSE policy."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from ibs.envs.real_cartpole.real_cartpole_env import (
    CONFIG_LSTM,
    CONFIG_MLP,
    RealCartPoleEnv,
)

NUM_EPISODES = 5
MAX_EPISODE_STEPS = 200
RENDER_SCENE = True

if __name__ == "__main__":
    config = {**CONFIG_MLP}  # CONFIG_LSTM, CONFIG_MLP
    config.update(
        {
            "max_episode_steps": MAX_EPISODE_STEPS,
            "init_method": "from_data",  # "fixed", "from_data"
        }
    )
    env = RealCartPoleEnv(config)

    buffer_size_policy = 8
    policy_model = tf.saved_model.load(
        Path(
            "../resources/envs/real_cartpole/policy_models/moose_realcartpole_upswing_linux_2/"
            "checkpoint_000017"
        ).resolve()
    )

    for _ in range(NUM_EPISODES):
        observations, actions, rewards, infos = [], [], [], []
        rewards_from_models = []
        terminated = False
        obs, _ = env.reset()
        t_start = time.time()
        step_count = 0
        while not terminated:
            policy_input = np.hstack(
                list(env._model_input_buffer)[-buffer_size_policy:]
            )
            action = policy_model([policy_input])[0]
            # action = np.array([0.5], dtype=np.float32)  # for testing purpose only
            # action = env.action_space.sample()  # for testing purpose only
            obs, reward, terminated, _, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            rewards_from_models.append(info["mean_reward_from_models"])
            infos.append(info)
            if RENDER_SCENE:
                env.render()
            step_count += 1

        t_stop = time.time()
        duration = t_stop - t_start
        print(
            f"Total execution time for {step_count} steps: "
            f"{duration:.2f}s ({duration / step_count:.3f}s per step)"
        )

        res = np.concatenate(
            [
                observations,
                np.array(actions).reshape(len(actions), 1),
                np.array(rewards).reshape(len(rewards), 1),
                np.array(rewards_from_models).reshape(len(rewards_from_models), 1),
            ],
            axis=1,
        )
        df_env = pd.DataFrame(
            res,
            columns=["Position", "Velocity", "Angle", "Angle Velocity"]
            + ["Action"]
            + ["Reward"]
            + ["Reward (modelled)"],
        )

        df_env.plot(subplots=True, figsize=(10, 8))

        plt.tight_layout()
        plt.show()
