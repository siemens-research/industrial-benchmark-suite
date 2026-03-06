"""Example of how to run the distillation process benchmark using a pre-trained MOOSE policy."""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
from gymnasium.envs.registration import make

import ibs.envs.distillation
from ibs.common.utils import resolve_path

ENV_ID = "Distillation-v0"
NUM_EPISODES = 5
MAX_EPISODE_STEPS = 100
INIT_METHOD = "from_data"  # "fixed" or "from_data"
RENDER_SCENE = False

PATH_TO_POLICY_MODEL = resolve_path(
    "../resources/envs/distillation/policy_models/run_33/checkpoint_000020", __file__
)
POLICY_BUFFER_SIZE = 20

if __name__ == "__main__":
    env = make(
        ENV_ID,
        init_method=INIT_METHOD,
        max_episode_length=MAX_EPISODE_STEPS,
    )
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    for _ in range(NUM_EPISODES):
        observations, actions, rewards, infos = [], [], [], []
        terminated = False
        obs, info = env.reset()
        model_input_buffer = info["model_input_buffer"]
        step_count = 0
        while not terminated:
            policy_input = np.hstack(model_input_buffer[-POLICY_BUFFER_SIZE:])
            action = policy_model([policy_input])[0]
            # action = np.array(5*[0.0], dtype=np.float32)  # for testing purpose only
            # action = env.action_space.sample()  # for testing purpose only
            obs, reward, terminated, _, info = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            infos.append(info)
            model_input_buffer = info["model_input_buffer"]
            # if RENDER_SCENE:
            #     env.render()
            step_count += 1

        res = np.concatenate(
            [
                observations,
                actions,
                np.array(rewards).reshape(len(rewards), 1),
            ],
            axis=1,
        )
        df_env = pd.DataFrame(res, columns=env.unwrapped.obs_names + env.unwrapped.actions_names + ["Reward"])

        # Group observations and objectives according to env.unwrapped.obs_objectives_mapping
        axes = df_env[
            [obs for obs in env.unwrapped.obs_names if obs not in env.unwrapped.objective_names] + ["Reward"]
        ].plot(subplots=True, figsize=(12, 10))
        for obs, obj in env.unwrapped.obs_objectives_mapping.items():
            if obs in df_env.columns and obj in df_env.columns:
                # Find the subplot index for the objective
                idx = env.unwrapped.obs_names.index(obs)
                df_env[obj].plot(ax=axes[idx], label=obj, c="red", linestyle="--")
                axes[idx].legend()

        plt.tight_layout()
        plt.show()
