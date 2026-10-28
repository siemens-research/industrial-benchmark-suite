"""Gymnasium environment of the CEBIT 2018 real cartpole system.

The dynamics of the real cartpole system, originally designed and built for the Cebit 2018,
are simulated by running pre-trained Tensorflow surrogate models.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT

Author: Marc Weber (marc-weber@siemens.com)

Parts of the source code are taken from the original Cartpole
environment as provided by the Farama Foundation (under MIT license, see license text below).

The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

from collections import deque
from pathlib import Path
from typing import Dict

import gymnasium as gym
import numpy as np
import tensorflow as tf

from ibs.common.utils import resolve_path

DEFAULT_CONFIG_MLP = {
    "tf_models_dir": "../../../resources/envs/real_cartpole/dynamic_models/mlp",
    "tf_model_names": [
        "model_1",
        "model_2",
        "model_3",
        "model_4",
    ],
    "observation_buffer_size": 8,  # 8
    "init_method": "from_data",  # "fixed"
    "fixed_init_observation": [325.0, 0.0, 3.1, 0.0],
    "init_data_file": "../../../resources/envs/real_cartpole/data_recordings/initial_observation_samples__2024-02-01_15-10-13__buffersize_8.csv",
    "max_episode_length": 500,
}

DEFAULT_CONFIG_LSTM = {
    "tf_models_dir": "../../../resources/envs/real_cartpole/dynamic_models/lstm_2025-05-30",
    "tf_model_names": [
        "model_1",
        "model_2",
        "model_3",
        "model_4",
    ],
    "observation_buffer_size": 10,  # 8
    "init_method": "from_data",  # "fixed"
    "fixed_init_observation": [325.0, 0.0, 3.1, 0.0],
    "init_data_file": "../../../resources/envs/real_cartpole/data_recordings/initial_observation_samples__2024-02-01_15-10-13__buffersize_10.csv",
    "max_episode_length": 500,
}


class RealCartPoleEnv(gym.Env):
    """Gym environment class of the simulated Cebit 2018 'real' cartpole system.

    ## Description

    The environment provides a simulation of the Cebit 2018 cartpole system
    currently located at the Siemens site in Munich/Neuperlach ("Real-Cartpole")
    based on pre-trained dynamic models. It includes a realistic representation
    ("digital twin") of the actual physical system, including friction and solid body
    dynamics, amongst others.

    ## State Space

    The state space of the real cartpole system is defined as follows:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | 0 mm                | 600 mm            |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | -pi rad (-180°)     | +pi rad (180°)    |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    The angular coordinates are defined such that the upright pole position is given by 0 rad and
    downward position by +/- pi rad. Be aware that position space is not well explored
    and modelled near the outer edges (Cart Position < 150mm and  Cart Position > 450mm).

    ## Action Space

    The system is controlled by a 1-dimensional, continuous (floating point) action in
    [-1, 1] which is equivalent to the motor current (force signal) applied to the cart.

    ## Rewards

    The continuous reward signal is composed of 3 components that are linearly weighted:
    reward = 0.01 * (
        - min(abs(CartPosition - 325.0) / 10.0, 30.0)
        - min(abs(PoleAngle) / 5.0, 36.0)
        - min(abs(PoleAngularVelocity) / 10.0, 3.0) * (-15.0 < PoleAngle < 15.0)
    )

    The maximum achievable reward is 0 (if all three criteria are perfectly met).
    Any deviation from the individual targets result in a negative reward signal.
    If the epsisode is terminated before the maximum steps number is reached,
    a negative reward of -1.*(steps_missed) is returned, where steps_missed correpsonds
    to the remaining number of steps.

    ## Starting State

    Depending on the config field 'init_method', the starting state is either drawn as a 'random'
    sample or set by a 'fixed' sample (as defined by the 'initial_observation' field).

    ## Episode End

    Depending on the config field 'max_episode_steps', the episode is either terminated when the
    maximum step limit is reached or is run infinitely (default). Also, the episode is terminated
    if the cart moves outside the allowed position interval between 0 and 600mm.
    """

    # Static class properties used by render() function
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    length = 0.5
    x_threshold = 2.4
    screen_width = 600
    screen_height = 400
    screen = None
    clock = None

    def __init__(self, env_config: Dict = None):
        """Initialize environment with pre-trained models and define action/observation space.

        Args:
            env_config: Dictionary with (additional) environment specifications
            (if not provided, use parameters from DEFAULT_CONFIG)
        """
        self.config = {**DEFAULT_CONFIG_MLP}
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

        # Note: comment-in these lines if you wish to convert from keras model saving to
        # TF SavedModel (i.e. graph export)
        # for model_path, model in zip(model_dirs, self.dynamic_models):
        #     tf.saved_model.save(model, Path(model_path, "export_tf_graph"))

        self.observation_buffer_size = self.config["observation_buffer_size"]

        obs_low = np.array(
            [
                0.0,  # Cart position x
                -np.finfo(np.float32).max,  # Cart velocity v
                -np.pi,  # Pole position (angle)
                -np.finfo(np.float32).max,  # Pole (angular) velocity
            ],
        )
        obs_high = np.array(
            [
                600.0,  # Cart position x
                np.finfo(np.float32).max,  # Cart velocity v
                np.pi,  # Pendulum position (angle theta)
                np.finfo(np.float32).max,  # Pendulum (angular) velocity (dtheta/dt)
            ],
        )
        self.observation_space = gym.spaces.Box(obs_low, obs_high, dtype=np.float32)

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

        self._model_input_buffer = deque([], maxlen=self.config["observation_buffer_size"])

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

    def step(self, action):
        """Iterate simulated cartpole model by one step using pre-trained dynamics model.

        Args:
            action: 1d action vector of shape (1, ) containing floating point action signal
            in [-1, 1]

        Returns:
            observation, reward, terminated, done, info: tuple of signals describing the next
            observable state of the environment.
        """
        # Build flattened input vector from current state buffer to be used as model input
        current_input_state = np.hstack(self._model_input_buffer)
        action = np.array(action)

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

        # Update internal state representation applying mean over forecast model results
        self._model_input_buffer.append(mean_next_state_from_models)

        self.state = self._model_output_vector_to_observation(mean_next_state_from_models)

        reward = self._compute_reward()

        self.step_count += 1

        terminated = (
            not self.observation_space.contains(self.state) or self.step_count >= self.max_episode_steps
            if self.max_episode_steps is not None
            else False
        )
        if terminated:
            reward += -1.0 * np.clip((self.max_episode_steps - self.step_count), 0, None)

        return (
            self.state,
            reward,
            terminated,
            False,
            {"mean_reward_from_models": mean_reward_from_models, "model_input_buffer": list(self._model_input_buffer)},
        )

    def reset(self, seed=None, options=None):
        if self.config["init_method"] == "fixed":
            initial_observation_vector = np.array(self.config["fixed_init_observation"], dtype=np.float32)
            for _ in range(self.observation_buffer_size):
                self._model_input_buffer.append(self._observation_to_model_input_vector(initial_observation_vector))
            self.state = initial_observation_vector
        elif self.config["init_method"] == "from_data":
            initial_model_inputs = self.initial_observations[self.reset_count % len(self.initial_observations)]
            initial_model_inputs_reshaped = initial_model_inputs.reshape(
                -1, initial_model_inputs.shape[0] // self.observation_buffer_size
            )
            self._model_input_buffer.extend(initial_model_inputs_reshaped)
            self.state = self._model_output_vector_to_observation(initial_model_inputs_reshaped[-1])
        else:
            raise NotImplementedError("Unknown initialization method.")

        self.step_count = 0
        self.reset_count += 1

        return self.state, {"model_input_buffer": list(self._model_input_buffer)}

    def _compute_reward(self):
        """Compute default benchmark reward based on current state information."""
        pos_x = self.state[0]
        angle = self.state[2] * 180.0 / np.pi
        angle_vel = self.state[3] * 180.0 / np.pi

        reward = 0.01 * (
            -min(abs(pos_x - 325.0) / 10.0, 30.0)
            - min(abs(angle) / 5.0, 36.0)
            - min(abs(angle_vel) / 10.0, 3.0) * (-15.0 < angle < 15.0)
        )

        return reward

    @staticmethod
    def _observation_to_model_input_vector(obs, action_value=0.0):
        """Format observation vector to match input specification of dynamic models."""
        return np.array(
            [obs[0], obs[1], np.sin(obs[2]), np.cos(obs[2]), obs[3], action_value],
            dtype=np.float32,
        )

    @staticmethod
    def _model_output_vector_to_observation(out):
        """Reduce extended output of dynamic models to observation vector representation.

        Returns observation vector with the following features: [pos_x, vel_x, pos_ang, vel_ang].
        """
        return np.array([out[0], out[1], np.arctan2(out[2], out[3]), out[4]], dtype=np.float32)

    # pylint: disable=inconsistent-return-statements
    def render(self):
        """Render the environment in either of the speficied render modes 'human' or 'rgb_array'.

        This function is taken and modified from the classic cartpole implementation
        provided by the Farama Foundation (https://www.farama.org) at the following URL:

        https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/

        The orginal (i.e. non-modified) version of this function is licensed by

        The MIT License

        Copyright (c) 2016 OpenAI
        Copyright (c) 2022 Farama Foundation

        Permission is hereby granted, free of charge, to any person obtaining a copy
        of this software and associated documentation files (the "Software"), to deal
        in the Software without restriction, including without limitation the rights
        to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
        copies of the Software, and to permit persons to whom the Software is
        furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in
        all copies or substantial portions of the Software.

        """
        # pylint: disable=import-outside-toplevel, no-member, c-extension-no-member
        # pylint: disable=too-many-statements

        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise gym.error.DependencyNotInstalled(
                'pygame is not installed, run `pip install "gymnasium[classic-control]"`'
            ) from e

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state
        pos_x = x[0] * 2.4 / 300.0 - 2.4

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        left, right, top, bottom = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = pos_x * scale + self.screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(left, bottom), (left, top), (right, top), (right, bottom)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 153, 153))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 153, 153))

        left, right, top, bottom = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(left, bottom), (left, top), (right, top), (right, bottom)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (150, 150, 150))
        gfxdraw.filled_polygon(self.surf, pole_coords, (150, 150, 150))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (0, 0, 0),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (0, 0, 0),
        )

        gfxdraw.hline(self.surf, 0, self.screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))


def create_real_cartpole_env_v0(**kwargs) -> RealCartPoleEnv:
    """Factory function to create real cartpole environment v0 (default)."""

    config = {
        "tf_models_dir": "../../../resources/envs/real_cartpole/dynamic_models/lstm_2025-05-30",
        "tf_model_names": [
            "model_1",
            "model_2",
            "model_3",
            "model_4",
        ],
        "observation_buffer_size": 10,
        "init_method": "from_data",
        "init_data_file": "../../../resources/envs/real_cartpole/data_recordings/initial_observation_samples__2024-02-01_15-10-13__buffersize_10.csv",
        "max_episode_length": 500,
    }
    config.update(kwargs)

    return RealCartPoleEnv(config)
