"""Tests for the CCBench environment and CLI."""

import subprocess
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import yaml

import ibs.envs  # noqa: F401  # pylint: disable=unused-import  # registers the environments
from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.ccbench_env import create_ccbench_env

_ZERO_ACTION = {"BP_action": 0.0, "DS_action": 0.0, "HE_action": 0.0}


def _load_default_ccbench_config() -> dict:
    config_path = Path(__file__).resolve().parents[1] / "resources/envs/ccbench/config/ccbench_default.yaml"
    with open(config_path, "rt", encoding="utf8") as handle:
        return yaml.safe_load(handle)


def _assert_basic_observation(observation: dict) -> None:
    assert "observation" in observation
    assert "achieved_goal" in observation
    assert "desired_goal" in observation
    assert isinstance(observation["observation"], dict)
    assert isinstance(observation["achieved_goal"], dict)
    assert isinstance(observation["desired_goal"], dict)


def test_ccbench_env_with_config_dict():
    config = _load_default_ccbench_config()
    env = CCBenchEnv(
        config_dict={
            "process_spec": config["process_spec"],
            "action_spec": config["action_spec"],
            "keep_intermediate_series": False,
        }
    )

    obs, _info = env.reset()
    _assert_basic_observation(obs)

    action = env.action_space.sample()
    obs, _reward, _terminated, _truncated, _info = env.step(action)
    _assert_basic_observation(obs)


def test_ccbench_env_with_config_file(tmp_path: Path):
    config = _load_default_ccbench_config()
    config_file = tmp_path / "ccbench_config.yaml"
    with open(config_file, "wt", encoding="utf8") as handle:
        yaml.safe_dump(
            {
                "process_spec": config["process_spec"],
                "action_spec": config["action_spec"],
            },
            handle,
            sort_keys=False,
        )

    env = CCBenchEnv(config_file=config_file)
    obs, _info = env.reset()
    _assert_basic_observation(obs)


def test_ccbench_env_with_defaults():
    env = CCBenchEnv()
    obs, _info = env.reset()
    _assert_basic_observation(obs)


@pytest.mark.filterwarnings("ignore:.*Casting input x to numpy array:UserWarning")  # scalar float observations
def test_gym_make_v1():
    env = gym.make("CCBench-v1", episode_length=3)
    obs, info = env.reset(seed=0)
    assert "wave" not in obs["observation"]
    assert info["warmup_steps"] > 0
    results = [env.step(_ZERO_ACTION)[2:4] for _ in range(3)]
    assert results[-1] == (False, True)


@pytest.mark.filterwarnings("ignore::UserWarning")  # NaN observations and rewards without warm-up are expected in v0
@pytest.mark.filterwarnings("ignore:.*CCBench-v0 is out of date:DeprecationWarning")
def test_gym_make_v0_keeps_legacy_flags():
    env = gym.make("CCBench-v0", episode_length=3)
    obs, info = env.reset(seed=0)
    assert "wave" in obs["observation"]
    assert info["warmup_steps"] == 0
    results = [env.step(_ZERO_ACTION)[2:4] for _ in range(3)]
    assert results[-1] == (True, False)

    env = gym.make("CCBench-v0", config_dict={"warmup_steps": "auto"})
    _obs, info = env.reset(seed=0)
    assert info["warmup_steps"] > 0


def _zero_action_episode(seed: int = 0, steps: int = 200, **kwargs) -> dict:
    env = create_ccbench_env(**kwargs)
    obs, _info = env.reset(seed=seed)
    observations = [obs["observation"]]
    for _ in range(steps):
        obs, *_ = env.step(_ZERO_ACTION)
        observations.append(obs["observation"])
    return {channel: np.array([o[channel] for o in observations]) for channel in observations[0]}


def test_initial_control_override_keeps_other_channel_settings():
    env = create_ccbench_env(config_dict={"process_spec": {"DS": {"cfa_init": 4.0}}})
    assert env.process_spec["DS"]["limits"] == [1.0, 12.0]
    obs, _info = env.reset(seed=0)
    assert obs["observation"]["DS"] == 4.0
    assert obs["observation"]["BP"] == 12.0
    assert 2.0 <= obs["observation"]["HE"] <= 12.0


def test_disturbance_offset_override_shifts_ambient_temperature():
    base = _zero_action_episode()
    shifted = _zero_action_episode(config_dict={"process_spec": {"AT": {"rwo_offset": 25.0}}})
    assert np.median(shifted["AT"] - base["AT"]) == pytest.approx(20.0)


def test_noise_override_removes_target_noise():
    base = _zero_action_episode()
    noiseless = _zero_action_episode(config_dict={"process_spec": {"OD": {"wn_scale": 0.0}}})
    np.testing.assert_array_equal(noiseless["AT"], base["AT"])
    assert np.std(noiseless["OD"] - base["OD"]) == pytest.approx(0.1, rel=0.2)


@pytest.mark.parametrize("seed", range(5))
def test_long_episodes_stay_finite(seed: int):
    episode = _zero_action_episode(seed=seed, episode_length=200)
    assert all(np.all(np.isfinite(values)) for values in episode.values())


def test_ccbench_cli_generates_small_dataset(tmp_path: Path):
    config = _load_default_ccbench_config()
    config.update(
        {
            "episodes": 1,
            "episode_length": 2,
            "show_plots": False,
            "save_plots": False,
            "return_plots": False,
            "plot_with_intermediates": False,
            "plot_with_temporaries": False,
        }
    )
    config_file = tmp_path / "ccbench_small.yaml"
    with open(config_file, "wt", encoding="utf8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    # Run CLI in a subprocess to avoid sys.stdout/sys.stderr file handle conflicts
    # that arise when tqdm and logging.StreamHandler still hold references to the
    # CliRunner's closed capture buffers after invoke() returns.
    base_cmd = [sys.executable, "-m", "ibs"]

    result_without_subcommand = subprocess.run(
        base_cmd
        + [
            "--data-dir",
            str(tmp_path),
            "--config-file",
            str(config_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result_without_subcommand.returncode != 0

    result = subprocess.run(
        base_cmd
        + [
            "ccbench",
            "--data-dir",
            str(tmp_path),
            "--config-file",
            str(config_file),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr

    data_file = tmp_path / "gen_process_data.parquet"
    assert data_file.exists()
    assert data_file.stat().st_size > 0
