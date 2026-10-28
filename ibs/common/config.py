"""Utilities for loading and merging YAML configuration.

(C) Copyright Siemens AG 2026

SPDX-License-Identifier: MIT
"""

import logging
from pathlib import Path
from typing import Any
from typing import Dict
from typing import Mapping
from typing import Optional
from typing import Union

import yaml

logger = logging.getLogger(__name__)


def load_config(
    config_file: Optional[Union[str, Path]] = None,
    config_dict: Optional[Mapping[str, Any]] = None,
    config_kwargs: Optional[Mapping[str, Any]] = None,
    *,
    require_any: bool = False,
) -> Dict[str, Any]:
    """Load YAML configuration and merge with additional mappings.

    Args:
        config_file: Optional YAML file path to load.
        config_dict: Optional mapping merged after file values.
        config_kwargs: Optional mapping merged last.
        require_any: Raise ValueError when all inputs are None.

    Returns:
        Merged configuration dictionary.
    """
    if require_any and config_file is None and config_dict is None and config_kwargs is None:
        raise ValueError("config_file or config_dict or config_kwargs have to be specified, but all are None")

    config: Dict[str, Any] = {}
    if config_file is not None:
        logger.debug("Reading config file %s ...", config_file)
        with open(config_file, "rt", encoding="utf8") as f:
            loaded = yaml.safe_load(f)
            if loaded is not None:
                config.update(loaded)
    if config_dict is not None:
        config.update(dict(config_dict))
    if config_kwargs is not None:
        config.update(dict(config_kwargs))
    return config
