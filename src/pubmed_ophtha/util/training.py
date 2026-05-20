"""Module for training utilities."""

import random


def create_seed() -> int:
    """
    Generate a random seed for the model training.

    Returns:
        int: Random seed between 0 and 2**32 - 1, compatible with numpy and pytorch.

    """
    random_seed = random.randint(0, 2**32 - 1)  # numpy compatible
    random.seed(random_seed)

    # Needs to fulfill the range of both numpy and pytorch
    # Due to detectron
    seed = random.randint(0, 2**32 - 1)
    # For pytorch, we can use a larger range, but it is not necessary
    # random.randint(-9223372036854775808, 18446744073709551615)

    return seed
