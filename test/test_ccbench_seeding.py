"""Tests for seeding and reproducibility of the CCBench environment, exploration agent and data generation.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.exploration_agent import ExplorationAgent
from ibs.envs.ccbench.generate_data import generate_data

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "resources/envs/ccbench/config/ccbench_default.yaml"


def _default_config() -> dict:
    with open(_CONFIG_PATH, "rt", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def _make_env(**overrides) -> CCBenchEnv:
    return CCBenchEnv(config_dict={**_default_config(), **overrides})


def _fixed_action(step: int) -> dict:
    return {
        "BP_action": 2.0 * np.sin(0.3 * step),
        "DS_action": np.cos(0.2 * step),
        "HE_action": 0.05 * np.sin(0.1 * step),
    }


def _run_episode(env: CCBenchEnv, seed=None, steps: int = 30, options=None) -> pd.DataFrame:
    env.reset(seed=seed, options=options)
    for step in range(1, steps + 1):
        env.step(_fixed_action(step))
    return env.get_episode_data(include_temporary=True, include_intermediate=True, include_warmup=True)


def test_same_seed_gives_identical_trajectory():
    pd.testing.assert_frame_equal(_run_episode(_make_env(), seed=3), _run_episode(_make_env(), seed=3))


def test_different_seeds_give_different_trajectories():
    first = _run_episode(_make_env(), seed=3)
    second = _run_episode(_make_env(), seed=4)
    assert not np.allclose(first["AT"].to_numpy(), second["AT"].to_numpy())


def test_trajectory_ignores_global_state_and_earlier_episodes():
    expected = _run_episode(_make_env(), seed=3)

    np.random.seed(1)
    env = _make_env()
    _run_episode(env, seed=11)
    _run_episode(env)
    np.random.seed(2)
    pd.testing.assert_frame_equal(_run_episode(env, seed=3), expected)


def test_unseeded_reset_continues_stream():
    def two_episodes():
        env = _make_env()
        first = _run_episode(env, seed=3)
        return first, _run_episode(env)

    first_a, second_a = two_episodes()
    first_b, second_b = two_episodes()
    pd.testing.assert_frame_equal(second_a, second_b)
    assert not np.allclose(first_a["AT"].to_numpy(), second_a["AT"].to_numpy())
    assert second_a.index[0] == pd.Timestamp("2000-01-02T00:00:00Z")


def test_episode_start_default_and_option():
    env = _make_env(warmup_steps=0)
    _obs, info = env.reset(seed=0)
    assert info["timestamp"] == pd.Timestamp("2000-01-01T00:00:00Z")

    _obs, info = env.reset(seed=0, options={"episode_start": "2021-06-01T08:00:00"})
    assert info["timestamp"] == pd.Timestamp("2021-06-01T08:00:00Z")

    env = _make_env(warmup_steps=0, start_timestamp="2010-03-04T05:06:07+01:00")
    _obs, info = env.reset(seed=0)
    assert info["timestamp"] == pd.Timestamp("2010-03-04T04:06:07Z")


# Golden trajectory for seed 7 and the `_fixed_action` sequence with the legacy flags `warmup_steps=0` and
# `expose_temp_channels=True` (observations of reset and after steps 1, 10 and 30, channels in alphabetical order).
# It pins the random number streams of all channel generators. Only regenerate it (print the observations of this
# setup with `repr`) for intended numerical changes, and mention the change in the commit message.
_GOLDEN_CHANNELS = ["AT", "BP", "DS", "HE", "OD", "WT", "wave"]
_GOLDEN_OBSERVATIONS = {
    0: [3.766379650870017, 12.0, 7.000763732837336, 10.972138009695755, np.nan, np.nan, 0.0],
    1: [
        3.766379650870017,
        12.591040413322679,
        7.980830310678577,
        10.977129680528096,
        np.nan,
        np.nan,
        0.5562064600695253,
    ],
    10: [
        3.7652396500588625,
        25.308087457254114,
        10.824012378641841,
        11.22283205874494,
        2.008307788761078,
        0.10645380701290696,
        -0.3777061650515335,
    ],
    30: [
        3.70480543120584,
        25.057286742103805,
        5.588431417714258,
        11.969832956430393,
        11.96101749057708,
        0.7853810509537285,
        -0.7965129471558949,
    ],
}
_GOLDEN_REWARDS = {10: -6.040238404226015, 30: -4.591398541530808}


def test_golden_trajectory():
    env = _make_env(warmup_steps=0, expose_temp_channels=True)
    obs, _info = env.reset(seed=7)
    observations = {0: obs["observation"]}
    rewards = {}
    for step in range(1, 31):
        obs, reward, _terminated, _truncated, _info = env.step(_fixed_action(step))
        observations[step] = obs["observation"]
        rewards[step] = reward
    for step, expected in _GOLDEN_OBSERVATIONS.items():
        actual = [observations[step][channel] for channel in _GOLDEN_CHANNELS]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, err_msg=f"step {step}")
    for step, expected_reward in _GOLDEN_REWARDS.items():
        assert rewards[step] == pytest.approx(expected_reward, rel=1e-12)


def _agent_actions(agent: ExplorationAgent, steps: int = 50) -> list:
    env = _make_env()
    obs, _info = env.reset(seed=0)
    actions = []
    for _ in range(steps):
        action = agent.act(observation=obs["observation"], desired_goal=obs["desired_goal"], timestep=env.cur_timestep)
        actions.append([float(action[k]) for k in sorted(action)])
        obs, *_ = env.step(action)
    return actions


def test_exploration_agent_is_reproducible():
    action_spec = _default_config()["action_spec"]
    first = _agent_actions(ExplorationAgent(action_spec, seed=5))
    np.testing.assert_array_equal(first, _agent_actions(ExplorationAgent(action_spec, seed=5)))
    assert not np.array_equal(first, _agent_actions(ExplorationAgent(action_spec, seed=6)))


def test_exploration_agent_reset_reseeds():
    action_spec = _default_config()["action_spec"]
    agent = ExplorationAgent(action_spec, seed=1)
    _agent_actions(agent)
    agent.reset(seed=5)
    np.testing.assert_array_equal(_agent_actions(agent), _agent_actions(ExplorationAgent(action_spec, seed=5)))


def _generate(data_dir: Path) -> pd.DataFrame:
    generate_data(
        config_file=_CONFIG_PATH,
        data_dir=str(data_dir),
        episodes=2,
        episode_length=20,
        show_plots=False,
        save_plots=False,
        return_plots=False,
    )
    return pd.read_parquet(data_dir / "gen_process_data.parquet")


def test_generate_data_is_reproducible(tmp_path: Path):
    first = _generate(tmp_path / "first")
    second = _generate(tmp_path / "second")
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 2 * 21
    assert first.index[21] - first.index[0] == pd.Timedelta(days=1)
