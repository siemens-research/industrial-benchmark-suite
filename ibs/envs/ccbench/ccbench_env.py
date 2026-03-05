"""
Generation of artificial multivariate time-series data, suitable for use as OpenAI Gym environment.

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
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

import gymnasium as gym
import numpy as np
import pandas as pd

from ibs.common.config import load_config
from ibs.common.goal_env import GoalEnv
from ibs.common.sequences import is_sequence

from . import generators


class CCBenchEnv(GoalEnv):
    """Goal-based synthetic time-series environment driven by a channel specification."""
    def __init__(
        self,
        config_file: Optional[Path | str] = None,
        config_dict: Optional[Mapping[str, Any]] = None,
        *args,
        **kwargs,
    ):
        """Initialize the environment from a channel specification.

        Args:
            config_file: Optional YAML file providing configuration values.
            config_dict: Optional mapping of configuration values.
        """
        super().__init__(*args, **kwargs)
        if config_file is None and config_dict is None:
            config_file = (
                Path(__file__).resolve().parents[3]
                / "resources/envs/ccbench/config/ccbench_default.yaml"
            )
        config = load_config(config_file=config_file, config_dict=config_dict)

        process_spec = config.get("process_spec")
        if process_spec is None:
            raise ValueError("process_spec must be provided directly or via config")

        start_timestamp = None
        if "start_timestamp" in config and config["start_timestamp"] is not None:
            start_timestamp = _coerce_timestamp(config["start_timestamp"])
        freq = config.get("freq")
        episode_step = config.get("episode_step")
        render_file = config.get("render_file")
        keep_intermediate_series = config.get("keep_intermediate_series", True)
        max_episode_steps = config.get("episode_length", None)

        if "action_spec" in config:
            process_spec = _apply_action_limits(process_spec, config["action_spec"])
        process_spec = _resolve_process_callbacks(process_spec)

        self.process_spec = {k: dict(v) for k, v in process_spec.items()}
        self.cur_timestep = None
        self.next_timestep = None
        self.timestep_size = freq or pd.Timedelta(seconds=1)
        self.episode_step_size = episode_step or pd.DateOffset(days=1, normalize=True)
        self.keep_intermediate_series = keep_intermediate_series
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        self.action_control_map: Dict[str, Optional[str]] = {
            k: None for k, spec in self.process_spec.items() if spec["semantics"] == "action"
        }
        for k, spec in self.process_spec.items():
            if spec["semantics"] == "control":
                self.action_control_map[spec["assoc_action_channel"]] = k
        self.target_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "target"]
        self.state_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "state"]
        self.context_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "context"]
        self.temp_channels = [k for k, spec in self.process_spec.items() if spec["semantics"] == "temp"]
        self.target_obj_map = {k: f"{k}_obj" for k, spec in self.process_spec.items() if spec["semantics"] == "target"}

        self.metadata["render.modes"] = ["human"]
        self.render_file = render_file
        if self.render_file is not None:
            self.metadata["render.modes"].append("file")
        self.reward_range = (-float("inf"), 0.0)

        def get_box_space(channel):
            if channel in self.process_spec and "limits" in self.process_spec[channel]:
                return gym.spaces.Box(
                    low=np.array(self.process_spec[channel]["limits"][0]),
                    high=np.array(self.process_spec[channel]["limits"][1]),
                    shape=None,
                    dtype=np.float32,
                )
            else:
                return gym.spaces.Box(low=np.array(-np.inf), high=np.array(np.inf), shape=None, dtype=np.float32)

        state_space: Dict[str, gym.spaces.Space] = {
            k: get_box_space(k)
            for k in (list(self.process_spec.keys()) + list(self.target_obj_map.values()))
            if (k not in self.action_control_map or self.action_control_map[k] is None) and (k not in self.target_obj_map.values())
        }
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Dict(state_space),
                "achieved_goal": gym.spaces.Dict({k: get_box_space(k) for k in self.target_channels}),
                "desired_goal": gym.spaces.Dict({k: get_box_space(k) for k in self.target_channels}),
            }
        )
        self.action_space = gym.spaces.Dict({k: get_box_space(k) for k, c in self.action_control_map.items()})

        self.index = []
        self.desired_goal = None
        self.history = {}
        self.replay = None
        self.reset(episode_start=start_timestamp, perform_step=False)

        self.intermediate_map = {
            k: (
                []
                if not isinstance(spec["type"], Sequence)
                else [f"{k}_{gen.info_str(verbosity=2)}" for gen in spec["generator"][:-1]]
            )
            for k, spec in self.process_spec.items()
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
        episode_start: Optional[dt.datetime] = None,
        perform_step: bool = True,
        replay: Optional[pd.DataFrame] = None,
    ) -> Tuple[dict[str, Any], dict[str, Any]]:
        """Reset the environment state and optionally perform the first step.

        Args:
            seed: RNG seed forwarded to the base environment.
            options: Optional reset options passed to the base class.
            episode_start: Explicit start timestamp. If omitted, a suitable day
                boundary is chosen.
            perform_step: If True, performs one step with NaN actions to return
                a valid initial observation.
            replay: Optional DataFrame indexed by timestamps to replay channels.
                If provided, `episode_start` is derived from its index.
        """
        super().reset(seed=seed, options=dict(options) if options is not None else None)
        
        self.step_count = 0

        # find specified or suitable episode start timestamp
        if replay is not None:
            episode_start = replay.index.min()
            self.replay = replay
        else:
            if episode_start is None and self.cur_timestep is None:
                episode_start = pd.Timestamp.now(tz="UTC").floor(freq="1D")
            self.replay = None

        if len(self.index) > 0 or len(self.history) < 1 or replay is not None:
            # reset is necessary because old episode exists, or nothing has been setup at all so far, or
            # a new replay data set has been specified
            base_ts = episode_start or self.cur_timestep
            if base_ts is None:
                base_ts = pd.Timestamp.now(tz="UTC").floor(freq="1D")
            self.next_timestep = base_ts
            if episode_start is None:
                self.next_timestep = base_ts + self.episode_step_size
            self.cur_timestep = None
            self.index = []

            # setup empty history for all channels
            self.history = {
                k: pd.Series([], dtype=np.float32, name=k, index=pd.DatetimeIndex([], tz="utc"))
                for k, spec in self.process_spec.items()
            }
            for k, c in self.action_control_map.items():
                if k not in self.history:
                    self.history[k] = pd.Series([], dtype=np.float32, name=k, index=pd.DatetimeIndex([], tz="utc"))
                    self.process_spec[k] = dict(type=[], semantics="action")
            for k, o in self.target_obj_map.items():
                if o not in self.history:
                    self.history[o] = pd.Series([], dtype=np.float32, name=k, index=pd.DatetimeIndex([], tz="utc"))
                    self.process_spec[o] = dict(
                        type=[],
                        semantics="objective",
                        assoc_target_channel=k,
                        limits=self.process_spec[k].get("limits", [-np.inf, np.inf]),
                    )

            # create individual channel generator instances
            for k, spec in self.process_spec.items():
                if not is_sequence(spec["type"]):
                    spec["type"] = [spec["type"]]
                non_available = [typ for typ in spec["type"] if typ not in generators.__dict__]
                if len(non_available) > 0:
                    raise ValueError(
                        f"The following generators are not available (key 'type' of specification for '{k}'): {non_available}"
                    )
                spec["generator"] = [
                    generators.__dict__[typ](
                        channel_name=k,
                        spec=spec,
                        replay=(
                            replay[k]
                            if ((i == len(spec["type"]) - 1) and replay is not None and k in replay.columns)
                            else None
                        ),
                    )
                    for i, typ in enumerate(spec["type"])
                ]

        if perform_step:
            # observation, _, terminated, truncated, info = self.step({k: np.nan for k in self.action_control_map.keys()})
            observation, _, terminated, truncated, info = self.step({k: 0.0 for k in self.action_control_map.keys()})
            if terminated or truncated:
                info.setdefault("terminal_reason", "reset_step")
            return observation, info
        return {}, {}

    def step(
        self,
        action: Union[Mapping[str, float], np.ndarray],
        desired_goal: Optional[Mapping[str, float]] = None,
    ) -> Tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Advance one timestep using the provided action.

        Args:
            action: Either a mapping of action channels to values or a numpy array
                aligned with `action_control_map` order.
            desired_goal: Optional goal override for target channels. If omitted,
                goals may be derived or replayed.
        """
        observation = {}
        terminated = False
        truncated = False
        info = {}

        self.step_count += 1
        self.cur_timestep = self.next_timestep
        if self.cur_timestep is None:
            raise RuntimeError("cur_timestep is None at step entry")
        cur_ts = self.cur_timestep
        self.index.append(cur_ts)

        def get_replay_goal():
            if self.replay is None:
                return None
            if cur_ts not in self.replay.index:
                return None
            goal = {}
            for t, o in self.target_obj_map.items():
                if o not in self.replay.columns:
                    return None
                goal[t] = float(self.replay.at[cur_ts, o])  # type: ignore[arg-type]
            return goal

        self.desired_goal = desired_goal or get_replay_goal() or self.desired_goal or self.derive_desired_goal()
        action_is_dict = not isinstance(action, (np.ndarray,))
        sample_values_dict: Dict[str, float] = {}
        sample_values_list: List[float] = []
        goal_values: Dict[str, float] = {}
        # iterate generation of current channel values for each channel
        for k, spec in self.process_spec.items():
            if not is_sequence(spec["type"]):
                spec["type"] = [spec["type"]]
            new_val = None
            if spec["semantics"] == "action":
                new_val = action[k] if action_is_dict else action[list(self.action_control_map.keys()).index(k)]
            if spec["semantics"] == "control":
                new_val = (
                    action[spec["assoc_action_channel"]]
                    if action_is_dict
                    else action[list(self.action_control_map.keys()).index(spec["assoc_action_channel"])]
                )
            if spec["semantics"] == "objective":
                new_val = self.desired_goal[spec["assoc_target_channel"]]
            # compute chain of channel generators for the current channel
            for j, gen in enumerate(spec["generator"]):
                old_val = new_val
                new_val = gen.step(timestep=self.cur_timestep, history=self.history, cur_val=new_val)
                if not isinstance(new_val, (float, int)):
                    raise ValueError(
                        f"{gen.__class__.__name__}.step() (spec {k}, module {j}) did not return a scalar "
                        f"number, but {str(new_val)} when called at timestep {self.cur_timestep} "
                        f"with cur_val={str(old_val)}"
                    )
                last = j == len(spec["generator"]) - 1
                if not last and self.keep_intermediate_series:
                    # key = f"{k}_{gen.info_str()}"
                    key = self.intermediate_map[k][j]
                    if key not in self.history:
                        self.history[key] = pd.Series(
                            [], dtype=np.float32, name=key, index=pd.DatetimeIndex([], tz="utc")
                        )
                    self.history[key].loc[cur_ts] = cast(float, new_val)  # type: ignore[index]
            if "limits" in spec:
                low, high = spec["limits"]
                new_val = float(np.clip(float(new_val), low, high))  # type: ignore[arg-type]
            if spec["semantics"] in ("action", "control"):
                # actions and control values are "input" and belong to the previous timestep
                # if last_timestep is not None: self.history[k][last_timestep] = new_val
                # self.history[k][self.cur_timestep] = np.nan
                self.history[k].loc[cur_ts] = cast(float, new_val)  # type: ignore[index]
            else:
                self.history[k].loc[cur_ts] = cast(float, new_val)  # type: ignore[index]
            if spec["semantics"] not in ("objective",) and (
                spec["semantics"] not in ("action",) or self.action_control_map[k] is None
            ):
                if action_is_dict:
                    sample_values_dict[k] = cast(float, new_val)
                else:
                    sample_values_list.append(cast(float, new_val))
            if spec["semantics"] == "target":
                goal_values[k] = cast(float, new_val)

        observation["observation"] = sample_values_dict if action_is_dict else np.array(sample_values_list)
        observation["achieved_goal"] = goal_values
        observation["desired_goal"] = self.desired_goal
        reward = self.compute_reward(achieved_goal=goal_values, desired_goal=self.desired_goal, info=info)
        
        terminated = (
            self.step_count >= self.max_episode_steps if self.max_episode_steps is not None else False
        )

        cur_ts = self.cur_timestep
        self.next_timestep = cur_ts + self.timestep_size
        return observation, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment (currently a no-op placeholder)."""
        pass

    def compute_reward(
        self, achieved_goal: Mapping[str, float], desired_goal: Mapping[str, float], info: Mapping
    ) -> float:
        """Compute reward from achieved vs desired goals using per-target `vop_type`."""
        reward = 0.0
        for target, objective in self.target_obj_map.items():
            vop_type = self.process_spec[target]["vop_type"]
            if vop_type == "setpoint":
                reward -= np.abs(achieved_goal[target] - desired_goal[target])
            elif vop_type == "upper_limit":
                reward += np.minimum(desired_goal[target] - achieved_goal[target], 0.0)
            elif vop_type == "lower_limit":
                reward += np.minimum(achieved_goal[target] - desired_goal[target], 0.0)
            else:
                raise ValueError(f"unknown vop_type specified for channel {target}: {vop_type}")
        return reward

    def derive_desired_goal(self):
        """Derive a default goal (midpoint of limits or 0.0) for target channels."""
        return {
            k: spec["limits"][0] + 0.5 * (spec["limits"][1] - spec["limits"][0]) if has_limits(spec) else 0.0
            for k, spec in self.process_spec.items()
            if spec["semantics"] == "target"
        }

    def get_episode_data(self, include_temporary=False, include_intermediate=False) -> pd.DataFrame:
        """Return the current episode history as a DataFrame.

        Args:
            include_temporary: Include channels tagged as `temp`.
            include_intermediate: Include intermediate generator outputs.
        """
        intermediates = [
            intermediate for sens, intermediates in self.intermediate_map.items() for intermediate in intermediates
        ]
        res = {
            sens: (
                series
                if sens not in self.process_spec or self.process_spec[sens]["semantics"] not in ("control", "action")
                else series.shift(-1)
            )
            for sens, series in self.history.items()
            if (include_temporary or sens not in self.temp_channels)
            and (include_intermediate or sens not in intermediates)
        }
        return pd.DataFrame(res, index=pd.DatetimeIndex(self.index))


def has_limits(d):
    """Return True if `limits` exist and are finite at both ends."""
    return "limits" in d and (not np.isinf(d["limits"][0])) and (not np.isinf(d["limits"][1]))


def _apply_action_limits(
    process_spec: Mapping[str, Mapping[str, Any]], action_spec: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    merged = {k: dict(v) for k, v in process_spec.items()}
    for action_name, action_cfg in action_spec.items():
        if action_name in merged:
            if "limits" not in merged[action_name] and "limits" in action_cfg:
                merged[action_name]["limits"] = action_cfg["limits"]
            continue
        merged[action_name] = {
            "type": [],
            "semantics": "action",
        }
        if "limits" in action_cfg:
            merged[action_name]["limits"] = action_cfg["limits"]
    return merged


def _resolve_process_callbacks(process_spec: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    from ibs.envs.ccbench.functions import norm_filter, soft_hinge
    from scipy import signal

    callbacks_namespace = {
        "np": np,
        "signal": signal,
        "norm_filter": norm_filter,
        "soft_hinge": soft_hinge,
    }

    def resolve_callback(callback_spec: str):
        safe_globals = {"__builtins__": {}}
        safe_globals.update(callbacks_namespace)
        return eval(callback_spec, safe_globals, {})  # pylint: disable=eval-used

    resolved = {k: dict(v) for k, v in process_spec.items()}
    for _, series_spec in resolved.items():
        if "fc_callback" in series_spec and isinstance(series_spec["fc_callback"], str):
            series_spec["fc_callback"] = resolve_callback(series_spec["fc_callback"])
        if "af_kernel" in series_spec and isinstance(series_spec["af_kernel"], str):
            series_spec["af_kernel"] = resolve_callback(series_spec["af_kernel"])
    return resolved


def _coerce_timestamp(value: Any) -> dt.datetime:
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, str):
        return pd.Timestamp(value).to_pydatetime()
    raise TypeError(f"Unsupported start_timestamp type: {type(value).__name__}")


def create_ccbench_env(**kwargs) -> CCBenchEnv:
    """Factory function to create real cartpole environment with specified config."""

    return CCBenchEnv(
        config_file=Path(__file__).resolve().parents[3] / "resources" / "envs" / "ccbench" / "config" / "ccbench_default.yaml",
        **kwargs
    )
