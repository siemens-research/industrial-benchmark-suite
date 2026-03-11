"""Tests for the CCBench environment and CLI."""

import subprocess
import sys
from pathlib import Path

import yaml

from ibs.envs.ccbench.ccbench_env import CCBenchEnv


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
