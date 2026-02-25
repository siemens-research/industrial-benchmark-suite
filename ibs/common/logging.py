"""Utilities for logging management."""

import datetime as dt
import logging
import logging.config
import os
import sys
import textwrap
import traceback
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Union

import yaml

from ibs.common.sequences import addtodict


class DummyLogger:
    @classmethod
    def info(cls, msg, exc_info=False, *args, **kwargs):
        sys.stdout.write(str(msg))
        sys.stdout.write("\n")
        if exc_info:
            sys.stdout.write(traceback.format_exc())
            sys.stdout.write("\n")
    debug = info
    warning = info
    error = info


def get_logging_config_str(name: str, id_str: Optional[str]=None, logfiles: Union[str, Sequence[str]] | None = None):
    """
    Construct a YAML string usable as argument to logging.config.dictConfig(), when parsed by yaml.full_load().
    
    The logging configuration information could be directly constructed as Python dictionary, but this is intended to be
    easily replaced by logging configuration specified directly in a YAML configuration file in the caller.
    :param name: name by which the configured logger can be retrieved later: use e.g. logging.getLogger(name).
    :param id_str: optional str to be used as logging source information in logging messages; if None the `name` is used.
    :param logfiles: name or sequence of names of files where logs should be written to, in addition to the console.
    :return: the constructed YAML string.
    """
    res = textwrap.dedent(f"""\
        version: 1
        formatters:
          module:
            format: '%(asctime)s | %(levelname)s | {id_str if id_str else '%(name)s'} | %(message)s'
            datefmt: '%Y-%m-%d %H:%M:%S'
        handlers:
          console: {{class: logging.StreamHandler, level: INFO, formatter: module, stream: 'ext://sys.stdout'}}
        """)
    files = ""
    if isinstance(logfiles, str):
        res += textwrap.indent(textwrap.dedent(f"""\
          file:
            class: logging.FileHandler
            filename: {logfiles.format(**locals())}
            level: INFO
            formatter: module
          """), "  ")
        files = ', file'
    elif logfiles is not None:
        for i, logfile in enumerate(logfiles):
            res += textwrap.indent(textwrap.dedent(f"""\
              file{i}:
                class: logging.FileHandler
                filename: {logfile.format(**locals())}
                level: INFO
                formatter: module
              """), "  ")
        files = ', ' + ', '.join([f'file{i}' for i in range(len(logfiles))])
    res += textwrap.dedent(f"""\
        loggers:
          {name}: {{level: INFO, handlers: [console{files}], propagate: no}}
        """)
    return res


def configure_logger(name: str, config: Mapping):
    """
    Configure and return a logger according to configuration in the provided `config` dictionary.
     
    `config` is typically resulting from reading a YAML configuration file, but could also be set up directly from code.
    Elements in `config` necessary or optionally used are:
    * id: optional str to be used as logging source information in logging messages; if None the `name` is used.
    * log_file: name or sequence of names of files where logs should be written to, in addition to the console.
    * other key-value pairs can be referenced in `log_file` name(s) by python string format() syntax and are inserted there.
    :param name: name by which the configured logger can be retrieved later: use e.g. logging.getLogger(name).
    :param config: configuration information with information for constructing the logger.
    :return: the readily configured logging.Logger instance.
    """
    logfile = (
        config["log_file"].format(**addtodict({"date": dt.datetime.now()}, config))
        if "log_file" in config else
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+".log"
    )
    logging_config_yaml = get_logging_config_str(name, config.get("id", ), logfiles=logfile)
    logging.config.dictConfig(yaml.full_load(logging_config_yaml))
    logger = logging.getLogger(name)
    return logger


orig_warnings_handler = None
def enable_warning_traces(enable=True):
    import traceback
    import warnings
    global orig_warnings_handler
    
    if enable and orig_warnings_handler is None:
        orig_warnings_handler = warnings.warn
        def warn(*args, **kwargs):
            tb = traceback.extract_stack()
            assert orig_warnings_handler is not None
            orig_warnings_handler(*args, **kwargs)
            print("".join(traceback.format_list(tb)[:-1]))
        warnings.warn = warn
    elif not enable and orig_warnings_handler is not None:
        warnings.warn = orig_warnings_handler
        orig_warnings_handler = None
