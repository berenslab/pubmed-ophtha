"""Module defining imaging type detection dataset for Detectron2 training."""

import os
import warnings

from pubmed_ophtha.const.paths import (
    DETECTRON_CONVERTED_DATASET_PATH,
    IMAGE_GT_FILE,
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    TRAIN_TEST_SPLIT_FILE,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    ImagingTypeEnum,
    PanelTypeEnum,
    RectangleLabelEnum,
)

from .meta import (
    DETECTRON_DATASET_REGISTRY,
    register_dataset,
)
from .meta.label_studio.dataset_base import (
    build_loading_function,
)

DEFAULT_DATASET_PATH = os.path.join(
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_ANNOTATION_PATH,
    DETECTRON_CONVERTED_DATASET_PATH,
    "detectron_imaging_type_fundus_labels.json",
)
DEFAULT_ANNOTATION_PATH = os.path.join(
    LABEL_STUDIO_BASE_FOLDER, LABEL_STUDIO_ANNOTATION_PATH, IMAGE_GT_FILE
)

BASE_DATASET_NAME = "imaging_type_detection"


@DETECTRON_DATASET_REGISTRY.register(BASE_DATASET_NAME)
def add_dataset(
    dataset_path: str = DEFAULT_DATASET_PATH,
    annotation_path: str = DEFAULT_ANNOTATION_PATH,
    dataset_name: str = BASE_DATASET_NAME,
):
    """
    Register the dataset in the detectron dataset registry.

    Args:
        dataset_path (str, optional): Path to the detectron converted dataset. Will get
            created if not provided. Defaults to DEFAULT_DATASET_PATH.
        annotation_path (str, optional): Path to the Label Studio annotation file.
            Defaults to DEFAULT_ANNOTATION_PATH.
        dataset_name (str, optional): Name of the dataset. Defaults to
            BASE_DATASET_NAME.

    """
    # The dataset contains the annotated images and ignores panels and panel labels
    category_map = ["CFP", "OCT", "Retinal Imaging", "Other"]

    def filter_and_convert_labels(labels: list[RectangleLabelEnum]) -> int | list[int]:
        # Check if Panel or Label is present
        is_panel = PanelTypeEnum.PANEL in labels
        is_label = PanelTypeEnum.LABEL in labels
        is_image = PanelTypeEnum.IMAGE in labels

        if is_panel or is_label or not is_image:
            return -1  # Ignore non-image panels

        relevant_labels = [
            label for label in labels if isinstance(label, ImagingTypeEnum)
        ]

        if len(relevant_labels) == 0:
            return -1  # Ignore

        if len(relevant_labels) > 1:
            warnings.warn(
                "Annotation has multiple imaging type labels. This is unexpected. "
                "Skipping annotation."
            )
            return -1  # Invalid case

        return category_map.index(relevant_labels[0].value)

    loading_fn = build_loading_function(
        dataset_path=dataset_path,
        annotation_path=annotation_path,
        category_map=category_map,
        filter_fn=filter_and_convert_labels,
        test_size=0.1,
        split_file_path=os.path.join(LABEL_STUDIO_BASE_FOLDER, TRAIN_TEST_SPLIT_FILE),
    )

    register_dataset(
        dataset_base_name=dataset_name,
        loading_fn=loading_fn,
        class_list=category_map,
    )


@DETECTRON_DATASET_REGISTRY.register(f"{BASE_DATASET_NAME}_kfold_dummy")
def add_kfold_dataset():
    """Register k folds of the dataset as dummy."""
    # Register k folds of the dataset
    for fold in range(5):
        dataset_name = f"{BASE_DATASET_NAME}_fold_{fold}"
        dataset_file_path = os.path.join(
            os.path.dirname(DEFAULT_DATASET_PATH),
            "k_fold",
            f"fold_{fold}",
            os.path.basename(DEFAULT_DATASET_PATH),
        )

        add_dataset(
            dataset_path=dataset_file_path,
            annotation_path=None,  # pyright: ignore[reportArgumentType]
            dataset_name=dataset_name,
        )
