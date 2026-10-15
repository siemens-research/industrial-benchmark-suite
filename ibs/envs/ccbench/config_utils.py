"""
Helpers for CCBench configuration handling.

(c) 2026 Siemens AG

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import datetime as dt
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Union

import numpy as np

from ibs.common.config import load_config

_CALLBACK_KEYS = ("fc_callback", "af_kernel")


def deep_merge(base: Mapping[str, Any], override: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Recursively merge two configuration mappings without modifying them.

    Nested mappings are merged key by key, all other values (including lists) of `override` replace those of
    `base`. A value of None in `override` removes the corresponding key from the result.

    Args:
        base: Mapping with the base configuration.
        override: Mapping whose values take precedence over those in `base`; may be None.

    Returns:
        New dictionary containing the merged configuration.
    """
    merged: Dict[str, Any] = {k: _copy_value(v) for k, v in base.items()}
    if override is None:
        return merged
    for key, value in override.items():
        if value is None:
            merged.pop(key, None)
        elif isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = _copy_value(value)
    return merged


def _copy_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _copy_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy_value(v) for v in value]
    return value


def resolve_callbacks(spec: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Turn callback strings of a channel specification into callables.

    The values of `fc_callback` and `af_kernel` given as strings are evaluated as Python expressions with a
    restricted namespace (`np`, `signal`, `norm_filter`, `soft_hinge`, no builtins). This is not a sandbox:
    configuration files containing such strings are trusted code and must not come from untrusted sources.

    Args:
        spec: Mapping from channel names to channel specifications (e.g. `process_spec` or `action_spec`).

    Returns:
        New dictionary with copied channel specifications, callback strings replaced by the evaluated objects.
    """
    from scipy import signal

    from ibs.envs.ccbench.functions import norm_filter
    from ibs.envs.ccbench.functions import soft_hinge

    callbacks_namespace = {
        "np": np,
        "signal": signal,
        "norm_filter": norm_filter,
        "soft_hinge": soft_hinge,
    }

    def resolve_callback(callback_spec: str):
        safe_globals: Dict[str, Any] = {"__builtins__": {}}
        safe_globals.update(callbacks_namespace)
        return eval(callback_spec, safe_globals, {})  # pylint: disable=eval-used

    resolved = {k: dict(v) for k, v in spec.items()}
    for series_spec in resolved.values():
        for key in _CALLBACK_KEYS:
            if isinstance(series_spec.get(key), str):
                series_spec[key] = resolve_callback(series_spec[key])
    return resolved


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
