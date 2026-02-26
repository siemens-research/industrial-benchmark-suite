"""Gymnasium environment of a 4-tank process demonstrator.

In the 4-tank system, water can be pumped in between four interconnected tanks. The goal is to control
the water levels in the tanks by applying appropriate valve opening levels.
This benchmark environment provides a realistic simulation of the system dynamics based on pre-trained dynamic models
derived from real-world measurements.

Author: Marc Weber (marc-weber@siemens.com)

Copyright (C) 2026 Siemens AG.
"""

from collections import deque
from pathlib import Path
from typing import Dict

import gymnasium as gym
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

    The reward is computed as the sum over the negative absolute deviations of the current water levels
    in the 3 main tanks from their corresponding objectives (demand levels) minus a regularization term on
    the action changes (sum of negative absolute value of the 1-step differences over all 3 actions).
    Hence, the maximum reward is 0, which is achieved when the water levels in all 3 main tanks perfectly
    match their objectives and no more action changes needed to be applied.

    ## Starting State

    Depending on the config field 'init_method', the starting state is either drawn as a 'random'
    sample or set by a 'fixed' sample (as defined by the 'fixed_init_observation' field).

    ## Episode End

    Depending on the config field 'max_episode_steps', the episode is either terminated when the
    maximum step limit is reached or is run infinitely (default).
    """

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
        "Control Valve 1",
        "Control Valve 2",
        "Control Valve 3",
        "Control Valve 1-2",
        "Control Valve 3-4",
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
            assert resolve_path(
                self.config["init_data_file"], __file__
            ).exists(), "Initial observation samples file does not exist."
            self.initial_observations = np.loadtxt(
                resolve_path(self.config["init_data_file"], __file__),
                delimiter=",",
                dtype=np.float32,
            )

        self.render_mode = self.config.get("render_mode", "human")

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
        reward = mean_reward_from_models

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
            {"model_input_buffer": list(self._model_input_buffer)},
        )

    def reset(self, seed=None, options=None):
        if self.config["init_method"] == "fixed":
            initial_observation_vector = np.array(
                self.config["fixed_init_observation"], dtype=np.float32
            )
            for _ in range(self.observation_buffer_size):
                self._model_input_buffer.append(
                    initial_observation_vector
                )
            self.state = initial_observation_vector
        elif self.config["init_method"] == "from_data":
            initial_model_inputs = self.initial_observations[
                self.reset_count % len(self.initial_observations)
            ]
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

    def update_objectives(self, new_objectives: dict):
        """Update the current objectives based on the provided new objectives."""
        for obj_var, new_value in new_objectives.items():
            if obj_var in self.obs_names:
                index = self.obs_names.index(obj_var)
                self.current_objectives[self.objective_names.index(obj_var)] = new_value
            else:
                raise ValueError(f"Objective variable {obj_var} not found in state variables.")


def create_four_tank_env(**kwargs) -> FourTankEnv:
    """Factory function to create real cartpole environment with specified config."""

    config = {**DEFAULT_CONFIG}
    config.update(kwargs)

    return FourTankEnv(config)
