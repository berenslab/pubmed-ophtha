"""Utility functions for computing resources."""

import os


def get_cpu_count() -> int:
    """
    Get the number of available CPU cores.

    Uses SLURM_CPUS_ON_NODE if available, otherwise falls back to os.sched_getaffinity
    or os.cpu_count.

    Returns:
        int: Number of CPU cores.

    """
    if "SLURM_CPUS_ON_NODE" in os.environ:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1
