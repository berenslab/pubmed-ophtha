"""Module containing base functions for Label Studio datasets in Detectron2 format."""

import json
import os
from collections.abc import Callable
from typing import Any

from pubmed_ophtha.const.paths import (
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    RectangleLabelEnum,
)

from .dataset_conversion import create_dataset

DEFAULT_IMAGES_PATH = os.path.join(LABEL_STUDIO_BASE_FOLDER, LABEL_STUDIO_IMAGE_PATH)


def build_loading_function(
    dataset_path: str,
    annotation_path: str,
    category_map: list[str],
    filter_fn: Callable[[list[RectangleLabelEnum]], int | list[int]],
    images_path: str = DEFAULT_IMAGES_PATH,
    force_reload: bool = False,
    test_size: float = 0.2,
    split_file_path: str | None = None,
) -> Callable[[str], list[dict[str, Any]]]:
    """
    Build a loading function for a Label Studio dataset in Detectron2 format.

    Args:
        dataset_path (str): Path to the detectron formatted dataset.
        annotation_path (str): Path to the Label Studio annotations.
        category_map (list[str]): List of category names.
        filter_fn (Callable[[list[RectangleLabelEnum]], int  |  list[int]]): Function
            to filter and convert labels.
        images_path (str, optional): Path to the folder containing the images. Defaults
            to DEFAULT_IMAGES_PATH.
        force_reload (bool, optional): If True force to reconvert the dataset. Defaults
            to False.
        test_size (float, optional): Test set ratio. Defaults to 0.2.
        split_file_path (str | None, optional): Path to the train/test split file. If
            None, a new split will be created. Defaults to None.

    Returns:
        Callable[[str], list[dict[str, Any]]]: Function to load the dataset split.

    """

    def load_dataset(
        split,
    ):
        if force_reload or not os.path.exists(dataset_path):
            # Create dataset from label studio annotations
            create_dataset(
                dataset_path=dataset_path,
                annotation_path=annotation_path,
                image_path=images_path,
                category_info=category_map,
                category_mapping_fn=filter_fn,
                test_size=test_size,
                split_file_path=split_file_path,
            )

        with open(dataset_path) as f:
            data = json.load(f)

        return data[split]

    return load_dataset
