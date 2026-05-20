"""
Module for loading the PanelSeg dataset in Detectron2 format.

The PanelSeg dataset is taken from:
Zou et al. (2020), "Unified deep neural network for segmentation and labeling \
of multipanel biomedical figures", Journal of the Association for Information Science \
and Technology
"""

import json
import os

from .image_clef import (
    convert_to_detectron_labels,
)
from .meta import (
    DETECTRON_DATASET_REGISTRY,
    register_dataset,
)

DEFAULT_PANELSEG_PATH = os.path.join("datasets", "panel_seg_and_custom")
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_PANELSEG_PATH, "detectron_labels.json")
DEFAULT_COCO_PATH = os.path.join(DEFAULT_PANELSEG_PATH, "coco_format")


@DETECTRON_DATASET_REGISTRY.register("panel_detection")
def add_dataset():
    """Create and register the custom panel segmentation dataset."""
    class_map = ["panel", "label"]

    loading_fn = _load_dataset

    register_dataset(
        dataset_base_name="panel_detection",
        loading_fn=loading_fn,
        class_list=class_map,
    )


def _load_dataset(split: str):
    """Load and register the dataset in Detectron2 format."""
    # figure_splitting_train
    if not os.path.exists(DEFAULT_DATASET_PATH):
        convert_to_detectron_labels(DEFAULT_DATASET_PATH, DEFAULT_COCO_PATH)

    with open(DEFAULT_DATASET_PATH) as f:
        data = json.load(f)

    return data[split]
