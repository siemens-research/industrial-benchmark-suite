"""Example of how to run the real-cartpole benchmark using a pre-trained MOOSE policy.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT
"""

import time

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from gymnasium.envs.registration import make

from ibs.common.utils import resolve_path

ENV_ID = "RealCartPole-v0"
NUM_EPISODES = 5
MAX_EPISODE_STEPS = 200
INIT_METHOD = "from_data"  # "fixed" or "from_data"
RENDER_SCENE = False

PATH_TO_POLICY_MODEL = resolve_path(
    "../resources/envs/real_cartpole/policy_models/moose_realcartpole_upswing_linux_8/checkpoint_000020", __file__
)
POLICY_BUFFER_SIZE = 10


if __name__ == "__main__":
    env = make(
        ENV_ID,
        init_method=INIT_METHOD,
        max_episode_length=MAX_EPISODE_STEPS,
    )
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    for _ in range(NUM_EPISODES):
        observations, actions, rewards, infos = [], [], [], []
        rewards_from_models = []
        terminated = truncated = False
        obs, info = env.reset()
        model_input_buffer = info["model_input_buffer"]
        t_start = time.time()
        step_count = 0
        while not (terminated or truncated):
            policy_input = np.hstack(model_input_buffer[-POLICY_BUFFER_SIZE:])
            action = policy_model([policy_input])[0]
            # action = np.array([0.5], dtype=np.float32)  # for testing purpose only
            # action = env.action_space.sample()  # for testing purpose only
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            rewards_from_models.append(info["mean_reward_from_models"])
            infos.append(info)
            model_input_buffer = info["model_input_buffer"]
            if RENDER_SCENE:
                env.render()
            step_count += 1

        t_stop = time.time()
        duration = t_stop - t_start
        print(f"Total execution time for {step_count} steps: {duration:.2f}s ({duration / step_count:.3f}s per step)")

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

        plt.grid(True)
        plt.tight_layout()
        plt.show()
