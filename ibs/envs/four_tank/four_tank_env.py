"""Gymnasium environment of a 4-tank process demonstrator.

In the 4-tank system, water can be pumped in between four interconnected tanks. The goal is to control
the water levels in the tanks by applying appropriate valve opening levels.
This benchmark environment provides a realistic simulation of the system dynamics based on pre-trained dynamic models
derived from real-world measurements.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT

Author:
    Marc Weber (marc-weber@siemens.com)
"""

from collections import deque
from pathlib import Path
from typing import Dict

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from ibs.common.utils import resolve_path


DEFAULT_CONFIG = {
    "tf_models_dir": "../../../resources/envs/four_tank/dynamic_models/lstm_run_05",
    "tf_model_names": [
        "model_1",
        "model_2",
        "model_3",
        "model_4",
    ],
    "observation_buffer_size": 12,
    "init_method": "from_data",  # "from_data", "fixed"
    "init_data_file": "../../../resources/envs/four_tank/data_recordings/initial_observation_samples_2025-05-14__buffersize_12.csv",
    "fixed_init_observation": [500.0, 500.0, 500.0, 80.0, 80.0, 80.0, 15.0, 12.5, 9.3, 12.0, 10.0, 8.5],
    "max_episode_length": 100,
}


class FourTankEnv(gym.Env):
    """Gym environment class of the data-modeled 4-tank process demonstrator.

    ## Description

    In the 4-tank system, water can be pumped in between 3 interconnected tanks, and the 4th tank
    serves as a central reservoir.
    The goal is to control the water levels in the 3 main tanks by applying appropriate valve opening levels.
    This benchmark environment provides a realistic simulation of the system dynamics based on
    pre-trained dynamic models derived from real-world measurements.

    ## State Space

    The state space of the 4-tank system is defined as follows. The first 6 variables correspond to the
    current valve opening levels, the next 3 variables correspond to the current water levels in the main tanks
    (as measured by level meters) and the last 3 variables are their corresponding objectives (external inputs).

    | Num | Observation                | Min   | Max   |
    |-----|----------------------------|-------|-------|
    | 0   | Inlet Valve Tank 1         | 0.0   | 700.0 |
    | 1   | Inlet Valve Tank 2         | 0.0   | 700.0 |
    | 2   | Inlet Valve Tank 3         | 0.0   | 700.0 |
    | 3   | Connecting Valve Tank 1-2  | 0.0   | 100.0 |
    | 4   | Connecting Valve Tank 3-4  | 0.0   | 100.0 |
    | 5   | Connecting Valve Tank 2-3  | 0.0   | 100.0 |
    | 6   | Level Meter Tank 1         | 0.0   | 21.0  |
    | 7   | Level Meter Tank 2         | 0.0   | 21.0  |
    | 8   | Level Meter Tank 3         | 0.0   | 21.0  |
    | 9   | Objective Level Tank 1     | 0.0   | 21.0  |
    | 10  | Objective Level Tank 2     | 0.0   | 21.0  |
    | 11  | Objective Level Tank 3     | 0.0   | 21.0  |

    ## Action Space

    The system is controlled by the following continuous (floating point) control actions.
    Each action corresponds to the setpoint of a valve opening level, which is applied at each step.

    | Num | Action                     | Min   | Max   |
    |-----|------------------------ ---|-------|-------|
    | 0   | Control Valve 1            | 0.0   | 700.0 |
    | 1   | Control Valve 2            | 0.0   | 700.0 |
    | 2   | Control Valve 3            | 0.0   | 700.0 |
    | 3   | Control Valve 1-2          | 0.0   | 100.0 |
    | 4   | Control Valve 3-4          | 0.0   | 100.0 |

    ## Reward Function

    The reward at each step is the sum of two components:

    1. Setpoint adherence – the negative absolute deviation of each level meter
       from its corresponding objective, weighted by 1.0:
       ``r_setpoint = -1.0 * |Level Meter Tank 1 - Objective Level Tank 1|
                      -1.0 * |Level Meter Tank 2 - Objective Level Tank 2|
                      -1.0 * |Level Meter Tank 3 - Objective Level Tank 3|``

    2. Action regularization – a penalty on the 1-step change of each valve
       observation (weights: 0.001 for inlet valves, 0.01 for connecting valves):
       ``r_action = -0.001 * |Inlet Valve Tank k(t) - Inlet Valve Tank k(t-1)|  (k=1,2,3)
                   -0.01  * |Connecting Valve j(t) - Connecting Valve j(t-1)|   (j=1-2,3-4)``

    The total reward is ``r = r_setpoint + r_action``.  The maximum achievable
    reward is 0, attained when all water levels match their objectives and no
    valve changes are applied.

    ## Starting State

    Depending on the config option "init_method", the starting state is either drawn from a set list
    of random initial states ("from_data") or always using the same initial sample provided by the
    "initial_observation" option ("fixed").

    ## Episode End

    Depending on the config field 'max_episode_steps', the episode is either terminated when the
    maximum step limit is reached (default: 100) or is run infinitely (None).
    """

    metadata = {
        "render_modes": ["human"],
        "render_fps": 30,
    }

    obs_names = [
        "Inlet Valve Tank 1",
        "Inlet Valve Tank 2",
        "Inlet Valve Tank 3",
        "Connecting Valve Tank 1-2",
        "Connecting Valve Tank 3-4",
        "Connecting Valve Tank 2-3",
        "Level Meter Tank 1",
        "Level Meter Tank 2",
        "Level Meter Tank 3",
        "Objective Level Tank 1",
        "Objective Level Tank 2",
        "Objective Level Tank 3",
    ]

    actions_names = [
        "Inlet Valve Tank 1",
        "Inlet Valve Tank 2",
        "Inlet Valve Tank 3",
        "Connecting Valve Tank 1-2",
        "Connecting Valve Tank 3-4",
    ]

    objective_names = [
        "Objective Level Tank 1",
        "Objective Level Tank 2",
        "Objective Level Tank 3",
    ]

    obs_objectives_mapping = {
        "Level Meter Tank 1": "Objective Level Tank 1",  # CV1 Objective
        "Level Meter Tank 2": "Objective Level Tank 2",  # CV2 Objective
        "Level Meter Tank 3": "Objective Level Tank 3",  # CV3 Objective
    }

    obs_limits = np.array(
        [
            [0.0, 700.0],
            [0.0, 700.0],
            [0.0, 700.0],
            [0.0, 100.0],
            [0.0, 100.0],
            [0.0, 100.0],
            [0.0, 21.0],
            [0.0, 21.0],
            [0.0, 21.0],
            [0.0, 21.0],
            [0.0, 21.0],
            [0.0, 21.0],
        ]
    )
    action_limits = np.array(
        [
            [0.0, 700.0],
            [0.0, 700.0],
            [0.0, 700.0],
            [0.0, 100.0],
            [0.0, 100.0],
        ]
    )

    def __init__(self, env_config: Dict = None):
        """Initialize environment with pre-trained models and define action/observation space.

        Args:
            env_config: Dictionary with (additional) environment specifications
            (if not provided, use parameters from DEFAULT_CONFIG)
        """
        self.config = {**DEFAULT_CONFIG}
        if env_config is not None:
            assert isinstance(env_config, dict), "env_config must be a dict"
            self.config.update(env_config)

        model_dirs = [
            resolve_path(Path(self.config["tf_models_dir"], model_name), __file__)
            for model_name in self.config["tf_model_names"]
        ]
        self.dynamic_models = [
            tf.saved_model.load(model_dir)  # load from tf.SavedModel (saved as graph)
            # tf.keras.models.load_model(model_dir, compile=False)  # load from keras SavedModel
            for model_dir in model_dirs
        ]

        self.observation_buffer_size = self.config["observation_buffer_size"]

        self.observation_space = gym.spaces.Box(self.obs_limits[:, 0], self.obs_limits[:, 1], dtype=np.float32)

        self.action_space = gym.spaces.Box(self.action_limits[:, 0], self.action_limits[:, 1], dtype=np.float32)

        self._model_input_buffer = deque([], maxlen=self.config["observation_buffer_size"])
        self.current_objectives = []

        self.state = None
        self.max_episode_steps = self.config.get("max_episode_length", None)
        self.step_count = 0
        self.reset_count = 0

        if self.config["init_method"] == "from_data":
            assert resolve_path(self.config["init_data_file"], __file__).exists(), (
                "Initial observation samples file does not exist."
            )
            self.initial_observations = np.loadtxt(
                resolve_path(self.config["init_data_file"], __file__),
                delimiter=",",
                dtype=np.float32,
            )

        self.render_mode = self.config.get("render_mode", "human")

        # Render resources are created lazily on the first render() call
        self._render_fig = None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _init_render(self):
        """Create the matplotlib figure, axes, and line artists used by render().

        All observations are displayed in individual subplots sharing the x-axis.
        Where an obs_objectives_mapping entry exists the corresponding objective
        is drawn as a dashed tomato-coloured line in the same subplot so that
        tracking performance is immediately visible.  Axes y-limits are set from
        obs_limits.  All plot chrome (spines, tick params, grid, margins) is
        configured here once so that render() only has to update line data.
        """
        # Ensure interactive mode is enabled before creating figures
        plt.ion()

        # Number of subplots = number of obs variables that are NOT pure objective
        # entries (objectives share the subplot of their paired state variable)
        objective_set = set(self.objective_names)
        self._render_obs_indices = [i for i, name in enumerate(self.obs_names) if name not in objective_set]
        n_plots = len(self._render_obs_indices)

        self._render_fig, self._render_axes = plt.subplots(
            n_plots,
            1,
            sharex=True,
            figsize=(14, 1.4 * n_plots),
            gridspec_kw={"hspace": 0.05},
            constrained_layout=True,
        )
        self._render_fig.suptitle("Four-Tank Environment", fontsize=11)

        # Ensure axes is always a list even for n_plots == 1
        if n_plots == 1:
            self._render_axes = [self._render_axes]

        self._render_state_lines = []  # one per subplot
        self._render_obj_lines = []  # one per subplot (None if no objective)
        self._render_history = []  # list of state vectors appended each step

        for ax, obs_idx in zip(self._render_axes, self._render_obs_indices):
            obs_name = self.obs_names[obs_idx]
            ymin, ymax = self.obs_limits[obs_idx]

            # State variable line
            (state_line,) = ax.plot([], [], color="C0", linewidth=1.2, label=obs_name)
            ax.set_ylabel(obs_name, fontsize=7, labelpad=3)
            ax.set_ylim(ymin - 0.05 * (ymax - ymin), ymax + 0.05 * (ymax - ymin))
            ax.tick_params(labelsize=7)
            ax.grid(True, alpha=0.35, linewidth=0.5)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)

            # Objective line (dashed tomato) if a mapping exists for this obs
            obj_line = None
            if obs_name in self.obs_objectives_mapping:
                obj_obs_name = self.obs_objectives_mapping[obs_name]
                (obj_line,) = ax.plot(
                    [],
                    [],
                    color="tomato",
                    linewidth=1.0,
                    linestyle="--",
                    label=obj_obs_name,
                )
                ax.legend(
                    loc="upper left",
                    fontsize=6,
                    framealpha=0.5,
                    borderpad=0.3,
                    handlelength=1.2,
                    handletextpad=0.4,
                )
            else:
                ax.legend(
                    loc="upper left",
                    fontsize=6,
                    framealpha=0.5,
                    borderpad=0.3,
                    handlelength=1.2,
                    handletextpad=0.4,
                )

            self._render_state_lines.append(state_line)
            self._render_obj_lines.append(obj_line)

        self._render_axes[-1].set_xlabel("Step", fontsize=8)

    def step(self, action):
        """Iterate simulated DAL 4-tank benchmark process by one step using pre-trained dynamics model.

        Args:
            action: control action vector of shape (3, ) containing floating point action signal

        Returns:
            observation, reward, terminated, done, info: tuple of signals describing the next
            observable state of the environment.
        """
        # Build flattened input vector from current state buffer to be used as model input
        current_input_state = np.hstack(self._model_input_buffer)
        action = np.array(action, dtype=np.float32)

        # Execute tf dynamic models using current state and provided action
        dynamic_models_outputs = [
            dynamic_model(
                [
                    current_input_state.reshape(1, current_input_state.shape[0]),
                    action.reshape(1, action.shape[0]),
                ]
            )
            for dynamic_model in self.dynamic_models
        ]

        # Process tf model output and split into next_state/reward information
        state_outputs_by_model = [
            dynamic_models_output[0][0].numpy() for dynamic_models_output in dynamic_models_outputs
        ]
        reward_outputs_by_model = [
            dynamic_models_output[1][0].numpy() for dynamic_models_output in dynamic_models_outputs
        ]

        next_state_by_model = [
            state_output.reshape(
                self.observation_buffer_size,
                state_output.shape[0] // self.observation_buffer_size,
            )[-1]
            for state_output in state_outputs_by_model
        ]
        reward_by_model = [reward_output[0] for reward_output in reward_outputs_by_model]

        # Compute mean over model ensemble members to be used as resulting state/reward
        mean_next_state_from_models = np.mean(next_state_by_model, axis=0)
        mean_reward_from_models = np.mean(reward_by_model, axis=0)

        # Update next state with current objectives
        for i, obj_var in enumerate(self.objective_names):
            mean_next_state_from_models[self.obs_names.index(obj_var)] = self.current_objectives[i]

        # Update internal state representation applying mean over forecast model results
        self._model_input_buffer.append(mean_next_state_from_models)
        self.state = mean_next_state_from_models

        # Compute reward signal based on current state and state history
        reward, reward_components = self._compute_reward()

        self.step_count += 1

        terminated = (
            not self.observation_space.contains(self.state) or self.step_count >= self.max_episode_steps
            if self.max_episode_steps is not None
            else False
        )

        return (
            self.state,
            reward,
            terminated,
            False,
            {
                "model_input_buffer": list(self._model_input_buffer),
                "reward_components": reward_components,
                "reward_from_models": mean_reward_from_models,
            },
        )

    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state for a new episode.

        Safely closes the previous render figure if it exists and creates a fresh one
        on the next render() call.
        """
        # Safely close the old render figure if it exists
        if self._render_fig is not None:
            plt.close(self._render_fig)
            self._render_fig = None

        if self.config["init_method"] == "fixed":
            initial_observation_vector = np.array(self.config["fixed_init_observation"], dtype=np.float32)
            for _ in range(self.observation_buffer_size):
                self._model_input_buffer.append(initial_observation_vector)
            self.state = initial_observation_vector
        elif self.config["init_method"] == "from_data":
            initial_model_inputs = self.initial_observations[self.reset_count % len(self.initial_observations)]
            initial_model_inputs_reshaped = initial_model_inputs.reshape(
                -1, initial_model_inputs.shape[0] // self.observation_buffer_size
            )
            self._model_input_buffer.extend(initial_model_inputs_reshaped)
            self.state = initial_model_inputs_reshaped[-1]
        else:
            raise NotImplementedError("Unknown initialization method.")

        self.step_count = 0
        self.reset_count += 1

        self.current_objectives = [self.state[self.obs_names.index(obj_var)] for obj_var in self.objective_names]

        return self.state, {"model_input_buffer": list(self._model_input_buffer)}

    def _compute_reward(self):
        """Compute default benchmark reward based on available state (history) information.

        The reward consists of two separate components:
        1. Setpoint adherence: negative weighted absolute deviation of each level
           meter from its objective (weights = 1.0 each).
        2. Action regularization: negative weighted absolute 1-step change of all
           6 valve observations

        Returns:
            total_reward: aggregated reward signal
            reward_components: dictionary of reward contributing components (for logging/analysis purposes)
        """
        setpoint_weights = [1.0] * len(self.obs_objectives_mapping)
        action_reg_weights = [0.001, 0.001, 0.001, 0.01, 0.01]

        # --- Setpoint adherence term ---
        reward_setpoint_components = {}
        reward_diff_setpoint = 0.0
        for w, (lv, obj) in zip(setpoint_weights, self.obs_objectives_mapping.items()):
            diff = -w * np.abs(self.state[self.obs_names.index(lv)] - self.state[self.obs_names.index(obj)])
            reward_diff_setpoint += diff
            reward_setpoint_components[lv] = diff

        # --- action regularisation term ---
        reward_action_reg = 0.0
        for w, action_name in zip(action_reg_weights, self.actions_names):
            idx = self.obs_names.index(action_name)
            diff = -w * np.abs(self._model_input_buffer[-1][idx] - self._model_input_buffer[-2][idx])
            reward_action_reg += diff

        total_reward = reward_diff_setpoint + reward_action_reg

        reward_components = {
            **reward_setpoint_components,
            "action_regularization": reward_action_reg,
        }

        return total_reward, reward_components

    def update_objectives(self, new_objectives: dict):
        """Update the current objectives based on the provided new objectives."""
        for obj_var, new_value in new_objectives.items():
            if obj_var in self.obs_names:
                self.current_objectives[self.objective_names.index(obj_var)] = new_value
            else:
                raise ValueError(f"Objective variable {obj_var} not found in state variables.")

    def render(self):
        """Update the live matplotlib visualisation with the current environment state.

        All figure geometry and artist objects are created once in _init_render()
        (called lazily on the first render() invocation); this method only pushes
        new data into the existing line artists and redraws, keeping invocation
        overhead minimal.
        """
        if self.state is None:
            return

        # Lazily create the render figure on the first call
        if self._render_fig is None:
            self._init_render()

        # Accumulate history
        self._render_history.append(self.state.copy())
        data = np.array(self._render_history)  # shape (T, n_obs)
        x = np.arange(data.shape[0])

        for ax, obs_idx, state_line, obj_line in zip(
            self._render_axes,
            self._render_obs_indices,
            self._render_state_lines,
            self._render_obj_lines,
        ):
            state_line.set_data(x, data[:, obs_idx])
            ax.set_xlim(0, max(10, len(self._render_history)))

            if obj_line is not None:
                obs_name = self.obs_names[obs_idx]
                obj_obs_name = self.obs_objectives_mapping[obs_name]
                obj_idx = self.obs_names.index(obj_obs_name)
                obj_line.set_data(x, data[:, obj_idx])
                # Zoom y-axis around the current objective ± margin
                current_obj = data[-1, obj_idx]
                margin = 3.0
                ymin_lim = max(self.obs_limits[obj_idx, 0], current_obj - margin)
                ymax_lim = min(self.obs_limits[obj_idx, 1], current_obj + margin)
                if ymax_lim > ymin_lim:
                    ax.set_ylim(ymin_lim, ymax_lim)

        self._render_fig.canvas.draw_idle()
        self._render_fig.canvas.flush_events()


def create_four_tank_env(**kwargs) -> FourTankEnv:
    """Factory function to create real cartpole environment with specified config."""

    config = {**DEFAULT_CONFIG}
    config.update(kwargs)

    return FourTankEnv(config)
