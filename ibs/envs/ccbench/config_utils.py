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
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

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


CHANNEL_SEMANTICS = ("action", "control", "state", "context", "target", "temp")
VOP_TYPES = ("setpoint", "upper_limit", "lower_limit")
BOOL_CONFIG_KEYS = ("truncate_on_time_limit", "expose_temp_channels", "keep_intermediate_series")


def parse_freq(freq: Any) -> pd.Timedelta:
    """Convert a step duration specification to a `pd.Timedelta`.

    Args:
        freq: Step duration as number of seconds, as pandas timedelta string (e.g. "1s", "5min"), as
            `datetime.timedelta`/`pd.Timedelta`, or as fixed-frequency pandas offset; None means one second.

    Returns:
        Positive step duration.
    """
    if freq is None:
        return pd.Timedelta(seconds=1)
    if isinstance(freq, bool):
        raise ValueError(f"freq must be a positive duration, got {freq!r}")
    if isinstance(freq, (int, float, np.integer, np.floating)):
        result = pd.Timedelta(seconds=float(freq))
    else:
        result = pd.Timedelta(freq)
    if result is pd.NaT or result <= pd.Timedelta(0):
        raise ValueError(f"freq must be a positive duration, got {freq!r}")
    return result


def channel_metadata(
    process_spec: Mapping[str, Mapping[str, Any]], freq: Any = None, include_temp: bool = False
) -> Dict[str, Dict[str, Any]]:
    """Describe the channels of a CCBench process specification.

    Args:
        process_spec: Channel specifications including the action channels (e.g. `CCBenchEnv.process_spec`).
        freq: Step duration, see `parse_freq`.
        include_temp: Also describe channels with semantics `temp`.

    Returns:
        Mapping from channel name to a dict with `semantics`, `limits` (`[low, high]`, infinite if unbounded),
        `unit` (None if not specified) and `step_seconds`; control channels additionally contain `action_channel`,
        target channels `vop_type`. Objective channels are not included.
    """
    step_seconds = parse_freq(freq).total_seconds()
    result: Dict[str, Dict[str, Any]] = {}
    for name, spec in process_spec.items():
        semantics = spec.get("semantics")
        if semantics not in CHANNEL_SEMANTICS or (semantics == "temp" and not include_temp):
            continue
        limits = spec.get("limits", [-np.inf, np.inf])
        entry: Dict[str, Any] = {
            "semantics": semantics,
            "limits": [float(limits[0]), float(limits[1])],
            "unit": spec.get("unit"),
            "step_seconds": step_seconds,
        }
        if semantics == "control":
            entry["action_channel"] = spec.get("assoc_action_channel")
        if semantics == "target":
            entry["vop_type"] = spec.get("vop_type")
        result[name] = entry
    return result


def validate_ccbench_config(config: Mapping[str, Any]) -> None:
    """Check a CCBench environment configuration for consistency.

    Only keys used by the environment are checked, data generation keys (e.g. `episodes`, plot settings) are
    ignored. All detected problems are collected and reported together.

    Args:
        config: Environment configuration with `process_spec`, optional `action_spec` and further environment keys.

    Raises:
        ValueError: if the configuration is invalid; the message lists all detected problems.
    """
    errors: List[str] = []
    process_spec = config.get("process_spec")
    action_spec = config.get("action_spec")
    if not isinstance(process_spec, Mapping) or len(process_spec) == 0:
        errors.append("process_spec must be a non-empty mapping of channel specifications")
        process_spec = {}
    if action_spec is None:
        action_spec = {}
    elif not isinstance(action_spec, Mapping):
        errors.append("action_spec must be a mapping of channel specifications")
        action_spec = {}

    for spec_name, spec in (("process_spec", process_spec), ("action_spec", action_spec)):
        for channel, channel_spec in spec.items():
            errors.extend(_validate_channel(f"{spec_name}.{channel}", channel_spec, spec_name == "action_spec"))

    channels = {k: v for k, v in {**action_spec, **process_spec}.items() if isinstance(v, Mapping)}
    action_channels = {k for k, v in channels.items() if v.get("semantics") == "action"}
    targets = {k: v for k, v in process_spec.items() if isinstance(v, Mapping) and v.get("semantics") == "target"}
    for channel, spec in process_spec.items():
        if not isinstance(spec, Mapping):
            continue
        if spec.get("semantics") == "control":
            assoc = spec.get("assoc_action_channel")
            if assoc not in action_channels:
                errors.append(f"process_spec.{channel}.assoc_action_channel must name an action channel, got {assoc!r}")
            errors.extend(_validate_value_in_limits(f"process_spec.{channel}.cfa_init", spec.get("cfa_init"), spec))
        if spec.get("semantics") == "target" and spec.get("vop_type") not in VOP_TYPES:
            errors.append(f"process_spec.{channel}.vop_type must be one of {VOP_TYPES}, got {spec.get('vop_type')!r}")

    errors.extend(_validate_desired_goal(config.get("desired_goal"), targets))

    if config.get("freq") is not None:
        try:
            parse_freq(config["freq"])
        except (ValueError, TypeError) as e:
            errors.append(f"freq: {e}")
    episode_length = config.get("episode_length")
    if episode_length is not None and (not _is_int(episode_length) or episode_length <= 0):
        errors.append(f"episode_length must be a positive integer, got {episode_length!r}")
    warmup_steps = config.get("warmup_steps")
    if warmup_steps is not None and warmup_steps != "auto" and (not _is_int(warmup_steps) or warmup_steps < 0):
        errors.append(f"warmup_steps must be 'auto' or a non-negative integer, got {warmup_steps!r}")
    for key in BOOL_CONFIG_KEYS:
        if config.get(key) is not None and not isinstance(config[key], bool):
            errors.append(f"{key} must be a boolean, got {config[key]!r}")
    if config.get("start_timestamp") is not None:
        try:
            if pd.Timestamp(config["start_timestamp"]) is pd.NaT:
                raise ValueError("NaT")
        except (ValueError, TypeError):
            errors.append(f"start_timestamp is not a valid timestamp: {config['start_timestamp']!r}")

    if errors:
        raise ValueError("Invalid CCBench configuration:\n- " + "\n- ".join(errors))


def _is_int(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _validate_channel(name: str, spec: Any, is_action_spec: bool) -> List[str]:
    from ibs.envs.ccbench import generators

    if not isinstance(spec, Mapping):
        return [f"{name} must be a mapping"]
    errors = []
    allowed_semantics = ("action", "temp") if is_action_spec else CHANNEL_SEMANTICS
    if spec.get("semantics") not in allowed_semantics:
        errors.append(f"{name}.semantics must be one of {allowed_semantics}, got {spec.get('semantics')!r}")
    types = spec.get("type")
    if isinstance(types, str):
        types = [types]
    if not isinstance(types, (list, tuple)):
        errors.append(f"{name}.type must be a generator name or a list of generator names, got {types!r}")
    else:
        available = {
            k
            for k, v in vars(generators).items()
            if isinstance(v, type)
            and issubclass(v, generators.ChannelGenerator)
            and v is not generators.ChannelGenerator
        }
        unknown = [t for t in types if t not in available]
        if unknown:
            errors.append(f"{name}.type contains unknown generators {unknown}")
    if "limits" in spec:
        limits = spec["limits"]
        if (
            not isinstance(limits, (list, tuple))
            or len(limits) != 2
            or not all(_is_number(v) for v in limits)
            or not limits[0] < limits[1]
        ):
            errors.append(f"{name}.limits must be two numbers [low, high] with low < high, got {limits!r}")
    if "unit" in spec and not isinstance(spec["unit"], str):
        errors.append(f"{name}.unit must be a string, got {spec['unit']!r}")
    return errors


def _valid_limits(spec: Mapping[str, Any]) -> Tuple[float, float]:
    limits = spec.get("limits")
    if isinstance(limits, (list, tuple)) and len(limits) == 2 and all(_is_number(v) for v in limits):
        return float(limits[0]), float(limits[1])
    return -np.inf, np.inf


def _validate_value_in_limits(name: str, value: Any, spec: Mapping[str, Any]) -> List[str]:
    if value is None:
        return []
    low, high = _valid_limits(spec)
    if isinstance(value, (list, tuple)):
        if len(value) != 2 or not all(_is_number(v) for v in value) or value[0] > value[1]:
            return [f"{name} must be a number or a range [low, high] with low <= high, got {value!r}"]
        values = list(value)
    elif _is_number(value):
        values = [value]
    else:
        return [f"{name} must be a number or a range [low, high], got {value!r}"]
    if any(not low <= v <= high for v in values):
        return [f"{name}={value!r} is outside of the channel limits [{low}, {high}]"]
    return []


def _validate_desired_goal(desired_goal: Any, targets: Mapping[str, Mapping[str, Any]]) -> List[str]:
    if desired_goal is None:
        return []
    if not isinstance(desired_goal, Mapping):
        return [f"desired_goal must be a mapping from target channel names to values, got {desired_goal!r}"]
    errors = []
    for target, goal in desired_goal.items():
        name = f"desired_goal.{target}"
        if target not in targets:
            errors.append(f"{name}: {target!r} is not a target channel (targets: {sorted(targets)})")
            continue
        if _is_number(goal):
            errors.extend(_validate_value_in_limits(name, goal, targets[target]))
            continue
        if not isinstance(goal, (list, tuple)) or len(goal) == 0:
            errors.append(f"{name} must be a number or a non-empty schedule [[step, value], ...], got {goal!r}")
            continue
        steps = []
        for entry in goal:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2 or not _is_int(entry[0]):
                errors.append(f"{name} schedule entries must be [step, value] with integer step, got {entry!r}")
                break
            steps.append(entry[0])
            errors.extend(_validate_value_in_limits(f"{name}[step {entry[0]}]", entry[1], targets[target]))
        else:
            if steps[0] != 0:
                errors.append(f"{name} schedule must start at step 0, got {steps[0]}")
            if any(b <= a for a, b in zip(steps, steps[1:])):
                errors.append(f"{name} schedule steps must be strictly increasing, got {steps}")
    return errors
