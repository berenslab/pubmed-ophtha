"""Logging configuration for the pubmed_ophtha package."""

import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with a console handler and standard format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s")
    )
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
