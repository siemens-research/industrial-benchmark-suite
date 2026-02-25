"""Utility functions for yaml file options handling."""

from typing import Any
from typing import Mapping
from typing import Optional

from ibs.common.config import load_config


def merge_options(
    config_file: Optional[str] = None,
    config_dict: Optional[Mapping[str,Any]] = None,
    config_kwargs: Optional[Mapping[str,Any]] = None,
) -> dict:
    """
    Merge options from different sources, to be used as utility function by other functions needing this common processing.
    
    A yaml file name, an explicit configuration dictionary, and/or a dictionary collected from remaining key-value pairs of a
    function can be provided, and they are read from file and merged, with entries from later arguments overwriting those
    of previous ones.
    :param config_file:
        str or None, file name of a YAML file to parse with arbitrary configuration to be interpreted by caller.
    :param config_dict:
        dict or other mapping with configuration settings to be merged with other argument's configuration
        information (with `config_dict` taking precedence).
    :param config_kwargs:
        dict or other mapping with configuration settings to be merged with other argument's configuration
        information. This is intended to be provided from a ", **kwargs)" last argument of the calling function.
        This has highest priority and overwrites configuration values with same keys provided in the other arguments.
    :return:
        dict, the result of merging the other dictionaries in argument provision order.
    """
    return load_config(
        config_file=config_file,
        config_dict=config_dict,
        config_kwargs=config_kwargs,
        require_any=True,
    )
