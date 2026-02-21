"""Example of how to run the 4-tank process benchmark using a pre-trained MOOSE policy."""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

from ibs.envs.four_tank.four_tank_env import FourTankEnv, DEFAULT_CONFIG

MAX_EPISODE_STEPS = 100
# RENDER_SCENE = True

if __name__ == "__main__":
    config = {**DEFAULT_CONFIG}
    config.update(
        {
            "max_episode_steps": MAX_EPISODE_STEPS,
            "init_method": "fixed",
        }
    )
    env = FourTankEnv(config)

    buffer_size_policy = 12
    policy_model = tf.saved_model.load(
        Path(
            "../resources/envs/four_tank/policy_models/run_05/" "checkpoint_000020"
        ).resolve()
    )

    observations, actions, rewards, infos = [], [], [], []
    terminated = False
    obs, _ = env.reset()
    t_start = time.time()
    step_count = 0
    while not terminated:
        policy_input = np.hstack(list(env._model_input_buffer)[-buffer_size_policy:])
        action = policy_model([policy_input])[0]
        # action = np.array(5*[0.0], dtype=np.float32)  # for testing purpose only
        # action = env.action_space.sample()  # for testing purpose only
        obs, reward, terminated, _, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        # if RENDER_SCENE:
        #     env.render()
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
            actions,
            np.array(rewards).reshape(len(rewards), 1),
        ],
        axis=1,
    )
    df_env = pd.DataFrame(res, columns=env.obs_names + env.actions_names + ["Reward"])

    # Group observations and objectives according to env.obs_objectives_mapping
    axes = df_env[
        [obs for obs in env.obs_names if obs not in env.objective_names] + ["Reward"]
    ].plot(subplots=True, figsize=(12, 10))
    for obs, obj in env.obs_objectives_mapping.items():
        if obs in df_env.columns and obj in df_env.columns:
            # Find the subplot index for the objective
            idx = env.obs_names.index(obs)
            df_env[obj].plot(ax=axes[idx], label=obj, c="red", linestyle="--")
            axes[idx].legend()

    plt.tight_layout()
    plt.show()
