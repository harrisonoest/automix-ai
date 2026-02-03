"""Logging configuration for AutoMix AI."""

import logging
import sys


def setup_logging(verbose: bool = False):
    """Configure logging for the application.

    Args:
        verbose: Enable debug logging if True.
    """
    level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    # Silence noisy libraries
    logging.getLogger("librosa").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("soundfile").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (typically __name__).

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
