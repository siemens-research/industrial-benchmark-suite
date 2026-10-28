"""
Data generation and visualization functions used by the CCBench environment.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT

Author:
    Holger Schoener <holger.schoener@siemens.com>
"""

import datetime as dt
import os
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
from typing import cast

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from ibs.envs.ccbench.generators import get_num_plots
from ibs.envs.ccbench.ccbench_env import CCBenchEnv
from ibs.envs.ccbench.exploration_agent import ExplorationAgent
from ibs.common.logging import configure_logger
from ibs.common.sequences import DictNamespace
from ibs.envs.ccbench.config_utils import build_ccbench_config
from ibs.envs.ccbench.config_utils import resolve_callbacks

# configuration keys forwarded unchanged to the environment, if present in the data generation config
ENV_CONFIG_KEYS = (
    "start_timestamp",
    "freq",
    "desired_goal",
    "warmup_steps",
    "expose_temp_channels",
    "truncate_on_time_limit",
)


def generate_data(
    config_file: Optional[Union[str, Path]] = None,
    config_dict: Optional[Mapping] = None,
    **config_kwargs,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Sequence[Figure]]]:
    """
    Generate an artificial data set, according to configuration provided in config_file and/or config (latter taking precedence).

    Results are written to files as configured in the configuration. For possible config settings see the annotated
    YAML file provided in the repository (data/generatedata.yaml).

    Args:
        config_file: str or Path with the path of a YAML file containing the configuration of data generation; can be None if
            config_dict or config_kwargs are provided.
        config_dict: dict or other Mapping containing the configuration of data generation; can be None if config_file
            or config_kwargs are provided. Configuration settings from config_dict take precedence over those
            provided in the config_file.
        config_kwargs: further key-value pairs to be used for configuration, overwriting config_file and config_dict
            settings.
    """
    c = DictNamespace()
    c.id = "artificial_process_data"
    c.episodes = 1
    c.episode_length = 1000
    c.export_filename = "{data_dir}/gen_process_data.parquet"
    c.include_temporary = False
    c.include_intermediate = False
    c.seed = 12345
    c.show_plots = False
    c.save_plots = True
    c.return_plots = False
    c.plot_with_intermediates = False
    c.plot_with_temporaries = False
    c.plot_width = 25
    c.plot_height = 3
    c.dpi = 150
    c.process_spec = {}
    c.action_spec = {}

    config = build_ccbench_config(
        defaults=c,
        config_file=config_file,
        config_dict=config_dict,
        config_kwargs=config_kwargs,
    )
    c.update(config)

    c.export_filename = c.export_filename.format(**c)
    root, ext = os.path.splitext(c.export_filename)
    if ext.lower() in {".h5", ".hdf5"}:
        c.export_filename = root + ".parquet"
    if not os.path.exists(os.path.dirname(c.export_filename)):
        os.makedirs(os.path.dirname(c.export_filename), exist_ok=True)

    logger = configure_logger(__file__, c)
    logger.info("Generating new dataset")

    c.process_spec = resolve_callbacks(c.process_spec)
    c.action_spec = resolve_callbacks(c.action_spec)

    # independent, reproducible per-episode seeds for environment and agent, all derived from c.seed
    env_seed_seq, agent_seed_seq = np.random.SeedSequence(c.seed).spawn(2)
    env_seeds = env_seed_seq.generate_state(max(c.episodes, 1))
    agent_seeds = agent_seed_seq.generate_state(max(c.episodes, 1))

    c.plotargs = {}
    c.num_plots = get_num_plots(
        c.process_spec,
        with_intermediates=c.plot_with_intermediates and c.include_intermediate,
        with_temporaries=c.plot_with_temporaries and c.include_temporary,
    )
    c.rel_plot_height = 1.0 / c.num_plots
    c.rel_unit_height = 1.0 / (c.num_plots * c.plot_height)

    c.cur_plot_num = 0
    c.cur_episode = 0
    c.fig = plt.figure(figsize=(c.plot_width, c.num_plots * c.plot_height), dpi=c.dpi, clear=True)
    c.figs = []

    if c.show_plots:
        plt.show()

    # import traceback
    import warnings

    warnings.simplefilter("ignore", UserWarning)
    # from armor.utils.logging import enable_warning_traces
    # enable_warning_traces(True)
    # orig_warnings_handler = warnings.warn
    # def warn(*args, **kwargs):
    #     tb = traceback.extract_stack()
    #     orig_warnings_handler(*args, **kwargs)
    #     print("".join(traceback.format_list(tb)[:-1]))
    # warnings.warn = warn

    def plot_series(label, data, nesting=0, episode=0):
        if not c.show_plots and not c.save_plots and not c.return_plots:
            return
        if nesting > 1:
            return
        if episode != c.cur_episode:
            if c.save_plots:
                c.fig.savefig(
                    os.path.splitext(c.export_filename)[0] + f"_ep-{c.cur_episode}.png",
                    dpi=c.dpi,
                    facecolor=None,
                    transparent=True,
                )
            plt.close(c.fig)
            if c.return_plots:
                c.figs.append(c.fig)
            c.fig = plt.figure(figsize=(c.plot_width, c.num_plots * c.plot_height), dpi=c.dpi, clear=True)
            c.cur_episode = episode
            c.cur_plot_num = 0
        indent = 0.1 * nesting
        axes_rect = (
            0.03 + indent,
            1.0 + (0.4 * c.rel_unit_height) - (c.cur_plot_num + 1) * c.rel_plot_height,
            0.96 - indent,
            c.rel_plot_height - (0.7 * c.rel_unit_height),
        )
        ax = c.fig.add_axes(axes_rect)
        if isinstance(data, list):
            for d in data:
                ax.plot(d.index, d.values, **c.plotargs)
                # d.plot.line(ax=ax, **c.plotargs)
        else:
            ax.plot(data.index, data.values, **c.plotargs)
            # data.plot.line(ax=ax, **c.plotargs)
        ax.set_title(label)
        if c.show_plots:
            plt.show()
        # ax.set_position([0.05 + 0.1 * nesting, 0.05, 0.9 - (0.1 * nesting), 0.9])
        # ax2 = ax.figure.add_axes([0, 0, 0.01, 1])
        # ax2.set_frame_on(False)
        # ax2.set_axis_off()
        c.cur_plot_num += 1

    def plot_episode(env, episode_data, episode=0, with_intermediates=True):
        logger.info(f"Plotting data for episode {episode}")

        def plot_for_intermediates(sens, prefix=None):
            if not c.include_intermediate or not with_intermediates:
                return
            if prefix is None:
                prefix = f"sensor {sens}: "
            for k in env.intermediate_map.get(sens, []):
                plot_series(f"{prefix}intermediate {k}", env.history[k], nesting=1, episode=episode)

        for action, control in env.action_control_map.items():
            if control is not None:
                plot_for_intermediates(control, f"control {control}: ")
                plot_series(
                    f"episode {episode} - control {control} (with action {action})",
                    [episode_data[action], episode_data[control]],
                    nesting=0,
                    episode=episode,
                )
            else:
                plot_series(f"episode {episode} - action {action}", episode_data[action], nesting=0, episode=episode)
        for target in env.target_channels:
            plot_for_intermediates(target, f"episode {episode} - target {target}: ")
            plot_series(f"episode {episode} - target {target}", episode_data[target], nesting=0, episode=episode)
        for state in env.state_channels:
            plot_for_intermediates(state, f"episode {episode} - state {state}: ")
            plot_series(f"episode {episode} - state {state}", episode_data[state], nesting=0, episode=episode)
        for context in env.context_channels:
            plot_for_intermediates(context, f"episode {episode} - context {context}: ")
            plot_series(f"episode {episode} - context {context}", episode_data[context], nesting=0, episode=episode)
        if c.plot_with_temporaries and c.include_temporary:
            for temp in env.temp_channels:
                plot_for_intermediates(temp, f"episode {episode} - temporary {temp}: ")
                plot_series(f"episode {episode} - temporary {temp}", episode_data[temp], nesting=0, episode=episode)

    # logger.info(f"Generating data according to\n{pp.pformat(process_spec)}")

    # get / initialize environment
    env_config: Dict[str, Any] = {
        "process_spec": c.process_spec,
        "action_spec": c.action_spec,
        "freq": dt.timedelta(seconds=1),
        "episode_step": pd.DateOffset(days=1, normalize=True),
        "keep_intermediate_series": True,
        "render_file": os.path.splitext(c.export_filename)[0] + ".png",
    }
    env_config.update({k: c[k] for k in ENV_CONFIG_KEYS if k in c})
    env = CCBenchEnv(config_dict=env_config)

    # get / initialize default agent
    default_agent = ExplorationAgent(c.action_spec)
    # for each requested episode
    data = []
    for episode in range(c.episodes):
        logger.info(f"Generating episode {episode}")
        # reset agent/environment, get initial observation
        default_agent.reset(seed=int(agent_seeds[episode]))
        # seeded resets start at the configured start timestamp, so explicitly continue after the last episode
        options = None if episode == 0 else {"episode_start": env.cur_timestep + env.episode_step_size}
        obs_raw, _ = env.reset(seed=int(env_seeds[episode]), options=options)
        obs: Dict[str, Any] = cast(Dict[str, Any], obs_raw)
        # run a loop over timesteps
        # for step in tqdm.tqdm(range(c.episode_length), desc="step", ncols=100, position=4, mininterval=1.0):
        for _ in tqdm.tqdm(range(c.episode_length), desc="step", ncols=100, mininterval=1.0):
            # get action from agent using observation
            actions = default_agent.act(
                observation=obs["observation"], desired_goal=obs["desired_goal"], timestep=env.cur_timestep
            )
            # perform an environment step using the action, to obtain next observation, reward, terminal state indicator
            obs_raw, reward, terminated, truncated, _info = env.step(actions)
            obs = cast(Dict[str, Any], obs_raw)
            # feed back reward into agent
            default_agent.observe(reward)
            if terminated or truncated:
                break
        episode_data = env.get_episode_data(
            include_temporary=c.include_temporary, include_intermediate=c.include_intermediate
        )
        data.append(episode_data)
        plot_episode(env, episode_data, episode, with_intermediates=c.plot_with_intermediates)
    data = pd.concat(data, axis=0)
    # visualize generated episodes
    if c.save_plots:
        plot_filename = os.path.splitext(c.export_filename)[0] + f"_ep-{c.cur_episode}.png"
        c.fig.savefig(plot_filename, dpi=c.dpi, facecolor=None, transparent=True)
        logger.info(f"written timeseries visualization file {plot_filename}")
    if c.return_plots:
        c.figs.append(c.fig)
    with pd.option_context("display.max_columns", 40, "display.width", 400):
        logger.info(f"generated data.head()/tail():\n{data.head(20)}\n{data.tail(20)}")
    logger.info(f"Storing generated data in {c.export_filename}")
    # store generated data
    data.to_parquet(c.export_filename, compression="zstd", engine="pyarrow", index=True)
    if c.return_plots:
        return data, c.figs
    return data
