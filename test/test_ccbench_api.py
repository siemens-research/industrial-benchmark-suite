"""Tests for the Gymnasium API conformance and the goal configuration of the CCBench environment.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT
"""

import warnings

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.ccbench_env import create_ccbench_env


def _assert_float_observation(env: CCBenchEnv, obs: dict) -> None:
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    for part in obs.values():
        assert all(isinstance(v, float) for v in part.values())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # gymnasium warns about casting scalar floats to arrays
        assert env.observation_space.contains(obs)


def test_observations_are_float_dicts_for_dict_and_array_actions():
    env = create_ccbench_env()
    obs, _info = env.reset(seed=0)
    _assert_float_observation(env, obs)
    env.action_space.seed(0)
    obs, reward, _terminated, _truncated, _info = env.step(env.action_space.sample())
    _assert_float_observation(env, obs)
    assert isinstance(reward, float)
    obs, *_ = env.step(np.array([0.5, -0.5, 0.01]))
    _assert_float_observation(env, obs)
    with pytest.raises(ValueError):
        env.step(np.array([0.5, -0.5]))
    with pytest.raises(ValueError):
        env.step({"BP_action": 0.0})


def test_time_limit_is_truncation():
    env = create_ccbench_env(episode_length=3)
    env.reset(seed=0)
    results = [env.step(env.action_space.sample())[2:4] for _ in range(3)]
    assert results == [(False, False), (False, False), (False, True)]
    assert env.step_count == 3


def test_legacy_time_limit_is_termination():
    env = create_ccbench_env(episode_length=3, truncate_on_time_limit=False)
    env.reset(seed=0)
    results = [env.step(env.action_space.sample())[2:4] for _ in range(3)]
    assert results == [(False, False), (False, False), (True, False)]


def test_warmup_gives_finite_initial_observation():
    env = create_ccbench_env()
    obs, info = env.reset(seed=0)
    assert env.step_count == 0
    assert info["step"] == 0
    assert info["warmup_steps"] > 0
    assert all(np.isfinite(v) for part in obs.values() for v in part.values())
    for _ in range(4):
        env.step(env.action_space.sample())
    assert len(env.get_episode_data()) == 5
    assert len(env.get_episode_data(include_warmup=True)) == 5 + info["warmup_steps"]
    assert env.get_episode_data().index[0] == info["timestamp"]


def test_fixed_warmup_steps():
    env = create_ccbench_env(warmup_steps=2)
    obs, info = env.reset(seed=0)
    assert info["warmup_steps"] == 2
    assert np.isnan(obs["observation"]["WT"])

    env = create_ccbench_env(warmup_steps=0)
    _obs, info = env.reset(seed=0)
    assert info["warmup_steps"] == 0
    assert info["timestamp"] == env.start_timestamp


def test_temp_channels_hidden_by_default():
    env = create_ccbench_env()
    obs, _info = env.reset(seed=0)
    assert "wave" not in obs["observation"]
    assert "wave" not in env.observation_space["observation"].spaces
    assert "wave" not in env.channel_metadata
    assert "wave" in env.history

    env = create_ccbench_env(expose_temp_channels=True)
    obs, _info = env.reset(seed=0)
    assert "wave" in obs["observation"]
    assert "wave" in env.observation_space["observation"].spaces


def test_info_keys():
    env = create_ccbench_env()
    _obs, info = env.reset(seed=0)
    assert set(info) == {"step", "timestamp", "warmup_steps", "channels"}
    assert info["channels"]["OD"]["unit"] == "mm"
    assert info["channels"]["BP_action"]["unit"] == "mbar/step"
    assert info["channels"]["DS"]["action_channel"] == "DS_action"
    obs, reward, _terminated, _truncated, info = env.step(env.action_space.sample())
    assert set(info) == {"step", "timestamp", "reward_terms"}
    assert info["step"] == 1
    assert set(info["reward_terms"]) == {"OD", "WT"}
    assert reward == pytest.approx(sum(info["reward_terms"].values()))
    assert reward == env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info)


def test_action_spaces_are_bounded_float64():
    env = create_ccbench_env()
    for name, space in env.action_space.spaces.items():
        assert space.dtype == np.float64, name
        assert np.all(np.isfinite(space.low)) and np.all(np.isfinite(space.high)), name


def test_render_mode_is_rejected():
    with pytest.raises(ValueError):
        create_ccbench_env(render_mode="human")
    assert create_ccbench_env().render() is None


def test_gymnasium_env_checker():
    env = create_ccbench_env(episode_length=20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        check_env(env, skip_render_check=True)


def test_config_desired_goal():
    env = create_ccbench_env(desired_goal={"OD": 5.0})
    obs, _info = env.reset(seed=0)
    assert obs["desired_goal"] == {"OD": 5.0, "WT": pytest.approx(0.605)}


def test_option_desired_goal_overrides_config_for_one_episode():
    env = create_ccbench_env(desired_goal={"OD": 5.0})
    obs, _info = env.reset(seed=0, options={"desired_goal": {"OD": 9.0, "WT": 0.3}})
    assert obs["desired_goal"] == {"OD": 9.0, "WT": 0.3}
    obs, _info = env.reset(seed=0)
    assert obs["desired_goal"]["OD"] == 5.0


def test_desired_goal_changes_reward():
    action = {"BP_action": 0.0, "DS_action": 0.0, "HE_action": 0.0}
    rewards = []
    for goal in (5.0, 9.0):
        env = create_ccbench_env(desired_goal={"OD": goal})
        env.reset(seed=0)
        rewards.append(env.step(action)[1])
    assert rewards[0] != rewards[1]


def test_desired_goal_schedule():
    env = create_ccbench_env(desired_goal={"OD": [[0, 5.0], [3, 8.0]]})
    obs, _info = env.reset(seed=0)
    goals = [obs["desired_goal"]["OD"]]
    for _ in range(4):
        obs, *_ = env.step(env.action_space.sample())
        goals.append(obs["desired_goal"]["OD"])
    assert goals == [5.0, 5.0, 5.0, 8.0, 8.0]


def test_step_desired_goal_takes_precedence_and_is_kept():
    env = create_ccbench_env(desired_goal={"OD": [[0, 5.0], [2, 8.0]]})
    env.reset(seed=0)
    obs, *_ = env.step(env.action_space.sample(), desired_goal={"OD": 6.0})
    assert obs["desired_goal"]["OD"] == 6.0
    obs, *_ = env.step(env.action_space.sample())
    assert obs["desired_goal"]["OD"] == 6.0
    with pytest.raises(ValueError):
        env.step(env.action_space.sample(), desired_goal={"XX": 1.0})
    obs, _info = env.reset(seed=0)
    assert obs["desired_goal"]["OD"] == 5.0


def test_invalid_desired_goal_raises():
    with pytest.raises(ValueError, match="outside of the channel limits"):
        create_ccbench_env(desired_goal={"OD": 20.0})
    env = create_ccbench_env()
    with pytest.raises(ValueError, match="not a target channel"):
        env.reset(seed=0, options={"desired_goal": {"AT": 1.0}})
