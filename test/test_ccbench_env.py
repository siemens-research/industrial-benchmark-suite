"""Tests for the CCBench environment and CLI."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from ibs.__main__ import app
from ibs.envs.ccbench.ccbench_env import CCBenchEnv


def _load_default_ccbench_config() -> dict:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "resources/envs/ccbench/config/ccbench_default.yaml"
    )
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

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "ccbench",
            "--data-dir",
            str(tmp_path),
            "--config-file",
            str(config_file),
        ],
    )
    assert result.exit_code == 0, result.stdout

    data_file = tmp_path / "gen_process_data.parquet"
    assert data_file.exists()
    assert data_file.stat().st_size > 0