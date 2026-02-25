"""Helpers for CCBench configuration handling."""

import datetime as dt
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Union

from ibs.common.config import load_config


def build_ccbench_config(
    defaults: Mapping[str, Any],
    config_file: Optional[Union[str, Path]] = None,
    config_dict: Optional[Mapping[str, Any]] = None,
    config_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    data_dir_default: Optional[Path] = None,
) -> Dict[str, Any]:
    """Build a merged configuration for data generation.

    Args:
        defaults: Default configuration values.
        config_file: Optional YAML file path to load.
        config_dict: Optional mapping merged after file values.
        config_kwargs: Optional mapping merged last.
        data_dir_default: Fallback directory for output data.

    Returns:
        Merged configuration dictionary.
    """
    config_dict = {} if config_dict is None else dict(config_dict)
    config_kwargs = {} if config_kwargs is None else dict(config_kwargs)
    if "data_dir" in config_dict and config_dict["data_dir"] is None:
        config_dict.pop("data_dir")
    if "data_dir" in config_kwargs and config_kwargs["data_dir"] is None:
        config_kwargs.pop("data_dir")

    config = load_config(config_file=config_file, config_dict=config_dict, config_kwargs=config_kwargs)
    if "data_dir" not in config or config.get("data_dir") is None:
        config["data_dir"] = data_dir_default or (Path(__file__).resolve().parents[3] / "data")

    now = dt.datetime.now()
    if "date" not in config:
        config["date"] = now
    if "id" not in config:
        config["id"] = defaults.get("id", "data").format(**config)
    else:
        config["id"] = config["id"].format(**config)

    merged: Dict[str, Any] = dict(defaults)
    merged.update(config)
    return merged
