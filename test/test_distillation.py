"""Test functionality of distillation benchmark."""

import numpy as np
import tensorflow as tf
from gymnasium.envs.registration import make

from ibs.common.utils import resolve_path
from ibs.envs.distillation.distillation_env import DEFAULT_CONFIG, DistillationEnv

MAX_EPISODE_STEPS = 10

PATH_TO_POLICY_MODEL = resolve_path(
    "../resources/envs/distillation/policy_models/run_33/checkpoint_000020", __file__
)
POLICY_BUFFER_SIZE = 20


def test_env_fixed_init_with_policy():
    """Test functionality of env using make() and a pre-trained policy with fixed initialisation."""
    env = make(
        "Distillation-v0",
        init_method="fixed",
        max_episode_length=MAX_EPISODE_STEPS,
    )
    policy_model = tf.saved_model.load(PATH_TO_POLICY_MODEL)

    observations, actions, rewards, infos = [], [], [], []
    terminated = False
    obs, info = env.reset()
    model_input_buffer = info["model_input_buffer"]
    step_count = 0
    while not terminated:
        policy_input = np.hstack(model_input_buffer[-POLICY_BUFFER_SIZE:])
        action = policy_model([policy_input])[0]
        obs, reward, terminated, _, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        model_input_buffer = info["model_input_buffer"]
        step_count += 1

    assert np.isclose(np.sum(rewards), -1.325, atol=0.001)


def test_env_from_data_init():
    """Test functionality of env when applying random action, make sure return is below 0."""
    config = {**DEFAULT_CONFIG}
    config.update(
        {
            "max_episode_steps": MAX_EPISODE_STEPS,
            "init_method": "from_data",
        }
    )
    env = DistillationEnv(config)

    observations, actions, rewards, infos = [], [], [], []
    terminated = False
    obs, _ = env.reset()
    step_count = 0
    while not terminated:
        action = env.action_space.sample()  # for testing purpose only
        obs, reward, terminated, _, info = env.step(action)
        observations.append(obs)
        actions.append(action)
        rewards.append(reward)
        infos.append(info)
        step_count += 1

    assert np.sum(rewards) < 0.0
