"""
This module provides utility functions for logging, notably a function to set up a logger with a specified name and
logging level. It also defines an enumeration for logging levels to facilitate easy configuration of the logger.

Source: https://github.com/tmartinezML/LOFAR-Diffusion/blob/master/src/utils/logging.py
"""
import logging
from enum import Enum


class LoggingLevels(Enum):
    """
    An enum to represent the different logging levels. This is used to set the logging level for the logger.
    """
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


def get_logger(name: str, level: int = LoggingLevels.INFO.value) -> logging.Logger:
    """
    Set up a logger with the given name and level. If the logger already has
    handlers, clear and reset them.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level, by default LoggingLevels.INFO.value

    Returns
    -------
    logging.Logger
        The logger with the given name and level.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # Check if the logger already has handlers
        logger.handlers.clear()  # Clear the default handlers
    logger.setLevel(level)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s (%(name)s): %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
