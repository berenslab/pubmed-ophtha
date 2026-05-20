"""Module for loading and joining the GT annotations from Label Studio."""

import os
import warnings
from typing import TypedDict

from pubmed_ophtha.const.paths import (
    CAPTION_GT_FILE,
    FIGURE_GT_FILE,
    IMAGE_GT_FILE,
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    PANEL_GT_FILE,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    Sample,
    parse_label_studio_annotations,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_figure_annotations import (
    FigureSample,
    parse_pdf_annotations,
)

GroundTruthSample = TypedDict(
    "GroundTruthSample",
    {
        "panel": Sample | None,
        "image": Sample | None,
        "caption": Sample | None,
        "figure": FigureSample | None,
    },
)


def is_captioned_sample(sample: Sample) -> bool:
    """
    Check if a sample has a caption.

    Args:
        sample (Sample): Sample to check for caption presence.

    Raises:
        ValueError: If multiple finished annotations are found.

    Returns:
        bool: True if the sample has a caption, False otherwise.

    """
    if sample.was_cancelled or not sample._has_annotations(sample.annotations):
        return False

    if (len(sample.annotations) - sample.cancelled_annotations) != 1:
        raise ValueError(
            f"Expected exactly one annotation, found {len(sample.annotations)}"
            + f" (sample.id={sample.id})"
        )

    selected_annotations = sample.finished_annotations[0]

    if (
        selected_annotations.bounding_boxes is None
        or len(selected_annotations.bounding_boxes) == 0
    ):
        return False

    relevant_boxes = [
        box
        for box in selected_annotations.bounding_boxes
        if (box.text is not None and len(box.text) > 0)
    ]

    if sum([1 for box in relevant_boxes if box.name is None]) > 1:
        # raise ValueError(
        warnings.warn(
            f"Found more than one bounding box without a label (sample.id={sample.id})"
        )
        return False

    if len(relevant_boxes) == 0:
        return False

    if sum([1 for box in relevant_boxes if box.name is None]) > 1:
        # raise ValueError(
        warnings.warn(
            f"Found more than one bounding box without a label (sample.id={sample.id})"
        )
        return False

    return True


def load_gt_annotations(
    panel_data_gt_file: str,
    image_data_gt_file: str,
    caption_data_gt_file: str,
    figure_data_gt_file: str,
    local_image_base_path: str | None = None,
) -> dict[str, GroundTruthSample]:
    """
    Load the ground truth from label studio annotations.

    Args:
        panel_data_gt_file (str): Path to the panel annotation file exported from Label
            Studio.
        image_data_gt_file (str): Path to the image annotation file exported from Label
            Studio.
        caption_data_gt_file (str): Path to the caption annotation file exported from
            Label Studio.
        figure_data_gt_file (str): Path to the figure annotation file exported from
            Label Studio.
        local_image_base_path (str | None, optional): Path to the folder containing
            labeled png images. Defaults to None.

    Returns:
        dict[str, GroundTruthSample]: Dictionary with figure UUID as keys and
            GroundTruthSample dicts as values, containing the loaded annotations for
            panel, image, caption, and figure.

    """

    def filter_missing_annotations(annotations: list[Sample]) -> list[Sample]:
        return [
            sample
            for sample in annotations
            if not sample.has_meta
            and sample.has_annotations
            and not sample.was_cancelled
            and sample.finished_annotations[0].bounding_boxes is not None
        ]

    panel_annotations = filter_missing_annotations(
        parse_label_studio_annotations(
            panel_data_gt_file, local_image_base_path=local_image_base_path
        )
    )

    image_annotations = filter_missing_annotations(
        parse_label_studio_annotations(
            image_data_gt_file, local_image_base_path=local_image_base_path
        )
    )

    caption_annotations = list(
        filter(
            is_captioned_sample,
            parse_label_studio_annotations(
                caption_data_gt_file, local_image_base_path=local_image_base_path
            ),
        )
    )

    figure_annotations = parse_pdf_annotations(figure_data_gt_file)

    gt_data = {}

    for sample in panel_annotations:
        if sample.image_id in gt_data:
            raise ValueError(
                f"Duplicate image_id {sample.image_id} found in panel annotations"
            )
        gt_data[sample.image_id] = {"panel": sample}

    for sample in image_annotations:
        if sample.image_id in gt_data:
            if "image" in gt_data[sample.image_id]:
                raise ValueError(
                    f"Duplicate image_id {sample.image_id} found in image annotations"
                )
            gt_data[sample.image_id]["image"] = sample
        else:
            gt_data[sample.image_id] = {"image": sample}

    for sample in caption_annotations:
        if sample.image_id in gt_data:
            if "caption" in gt_data[sample.image_id]:
                raise ValueError(
                    f"Duplicate image_id {sample.image_id} found in caption annotations"
                )
            gt_data[sample.image_id]["caption"] = sample
        else:
            gt_data[sample.image_id] = {"caption": sample}

    for sample in figure_annotations:
        if sample.image_id in gt_data:
            if "figure" in gt_data[sample.image_id]:
                raise ValueError(
                    f"Duplicate image_id {sample.image_id} found in figure annotations"
                )
            gt_data[sample.image_id]["figure"] = sample
        else:
            gt_data[sample.image_id] = {"figure": sample}

    # Set the missing keys to None
    for image_id, annotations in gt_data.items():
        if "panel" not in annotations:
            gt_data[image_id]["panel"] = None
        if "image" not in annotations:
            gt_data[image_id]["image"] = None
        if "caption" not in annotations:
            gt_data[image_id]["caption"] = None
        if "figure" not in annotations:
            gt_data[image_id]["figure"] = None

    return gt_data


def get_default_gt_files(
    root_folder: str | None = None,
) -> dict[str, str]:
    """
    Fill with default paths to GT files.

    Args:
        ground_truth_files (dict[str, str  |  None] | None, optional): Dictionary with
            GT paths. None entries are filled with the default path. Defaults to None.
        root_folder (str | None, optional): Root folder for GT files. If given,
            the default paths are constructed based on the label studio root folder.
            Defaults to None.

    Returns:
        dict[str, str]: Dictionary with paths to GT files, where missing entries are
            filled with default paths.

    """
    ground_truth_files: dict[str, str] = {}

    if root_folder is None:
        root_folder = os.path.join(
            LABEL_STUDIO_BASE_FOLDER, LABEL_STUDIO_ANNOTATION_PATH
        )

    if ground_truth_files["caption_data_gt_file"] is None:
        ground_truth_files["caption_data_gt_file"] = os.path.join(
            root_folder, CAPTION_GT_FILE
        )

    if ground_truth_files["figure_data_gt_file"] is None:
        ground_truth_files["figure_data_gt_file"] = os.path.join(
            root_folder, FIGURE_GT_FILE
        )

    if ground_truth_files["image_data_gt_file"] is None:
        ground_truth_files["image_data_gt_file"] = os.path.join(
            root_folder, IMAGE_GT_FILE
        )

    if ground_truth_files["panel_data_gt_file"] is None:
        ground_truth_files["panel_data_gt_file"] = os.path.join(
            root_folder, PANEL_GT_FILE
        )

    return ground_truth_files
