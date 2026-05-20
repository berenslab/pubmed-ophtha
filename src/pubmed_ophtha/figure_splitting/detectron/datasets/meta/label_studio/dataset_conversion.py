"""Module for converting label studio annotations to Detectron2 format."""

import json
import logging
import os
import random
from typing import Any, Callable, Sequence

from detectron2.structures import BoxMode
from PIL import Image
from tqdm.auto import tqdm

from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    RectangleLabelEnum,
    parse_label_studio_annotations,
)

logger = logging.getLogger(__name__)


def check_bbox_bounds(x0: float, y0: float, width: float, height: float) -> bool:
    """
    Check whether the bounding box is within the image bounds.

    The bounding box is defined by the top left corner (x0, y0) and the width \
        and height.
    Each value is in the range [0, 100] which equals the percentage of the \
        image width/height.

    Args:
        x0 (float): X coordinate of the top left corner of the bounding box.
        y0 (float): Y coordinate of the top left corner of the bounding box.
        width (float): Width of the bounding box.
        height (float): Height of the bounding box.

    Returns:
        bool: True if the bounding box is within the image bounds, False otherwise.

    """
    x1 = x0 + width
    y1 = y0 + height

    # Check x0
    if x0 < 0 or x0 > 100:
        return False

    # Check y0
    if y0 < 0 or y0 > 100:
        return False

    # Check width
    if width <= 0 or width > 100:
        return False

    # Check height
    if height <= 0 or height > 100:
        return False

    # Check x1
    if x1 <= 0 or x1 > 100:
        return False

    # Check y1
    if y1 <= 0 or y1 > 100:
        return False

    return True


def convert_annotations(
    annotation_file_path: str,
    image_path: str,
    category_mapping_fn: Callable[[list[RectangleLabelEnum]], int | list[int]]
    | None = None,
) -> dict[str, Any]:
    """
    Convert the label studio annotations to Detectron2 format.

    Args:
        annotation_file_path (str): Path to the annotation file.
        image_path (str): Path to the images.
        category_mapping_fn (Callable[[list[str]], int | list[int]]): Function to map
            the list of labels to a category ID or list of category IDs.
            If multiple category IDs are returned, multiple annotations will be created
            for the same bounding box. If a number less than 0 is returned, the
            bounding box will be skipped.
            Defaults to `convert_labels_to_category_id`.

    Returns:
        dict: Dictionary of annotations in Detectron2 format.
            The key is the image path and the value is a dictionary with the
            following keys:
            - file_name: Path to the image file.
            - image_id: Unique image ID.
            - height: Height of the image.
            - width: Width of the image.
            - annotations: List of dictionaries of annotations for the image.
                - bbox: Bounding box coordinates in [x0, y0, x1, y1] format.
                - bbox_mode: Bounding box mode (XYXY_ABS).
                - category_id: Category ID.
                - iscrowd: 0 for non-crowd objects.

    """
    if category_mapping_fn is None:
        raise ValueError("category_mapping_fn must be provided")

    annotations = parse_label_studio_annotations(
        annotation_file_path, local_image_base_path=image_path
    )
    annotation_dict = {}

    for entry_ind, sample in enumerate(tqdm(annotations)):
        if sample.has_meta or not sample.has_annotations or sample.was_cancelled:
            # Ignore samples with meta information, no annotations or cancelled
            continue

        # Assert so type checker does not complain
        assert sample.finished_annotations is not None
        if len(sample.finished_annotations) > 1:
            raise ValueError(
                "Multiple annotations per sample are not supported. "
                "Please ensure that each sample has only one annotation."
            )

        if sample.finished_annotations[0].bounding_boxes is None:
            # No annotations found
            continue

        # Convert to Detectron2 format
        c_image_path = os.path.join(image_path, sample.base_image_path)
        record = {}
        record["file_name"] = c_image_path
        record["image_id"] = sample.finished_annotations[0].id
        record["annotations"] = []

        im_width = None
        im_height = None

        is_error = False

        for r in sample.finished_annotations[0].bounding_boxes:
            x0 = r.x0
            y0 = r.y0
            x1 = r.x1
            y1 = r.y1

            if not r.check_bounds():
                # Wrong formatting of the bounding box
                # Clip boxes to the image size
                new_bounds = r.clip_to_bounds()

                if new_bounds is None:
                    # Box is empty
                    continue

                # Update values
                x0 = new_bounds[0]
                y0 = new_bounds[1]
                x1 = new_bounds[2]
                y1 = new_bounds[3]

            if im_width is None or im_height is None:
                im_width = r.original_width
                im_height = r.original_height

            box_labels: Sequence[RectangleLabelEnum] = []

            box_labels.extend(r.labels.copy())
            if r.box_type is not None:
                box_labels.append(r.box_type)

            if len(box_labels) == 0:
                continue

            try:
                detected_category_ids = category_mapping_fn(box_labels)

                if isinstance(detected_category_ids, int):
                    if detected_category_ids < 0:
                        # Signal to skip
                        continue

                    detected_category_ids = [detected_category_ids]

                for detected_category_id in detected_category_ids:
                    if detected_category_id < 0:
                        # Signal to skip
                        continue

                    record["annotations"].append(
                        {
                            "bbox": [
                                x0,
                                y0,
                                x1,
                                y1,
                            ],
                            "bbox_mode": BoxMode.XYXY_ABS,
                            "category_id": detected_category_id,
                            "iscrowd": 0,
                        }
                    )
            except ValueError as e:
                # Skip this image if there is an error in the labels
                logger.error(f"Error in sample {sample.id}: {e}")
                is_error = True
                break
        if is_error:
            continue

        if len(record["annotations"]) == 0:
            # No valid annotations found
            logger.debug(
                f"No valid annotations found for sample {sample.id}. Skipping."
            )
            continue

        if im_width is None or im_height is None:
            # Load the image to get the width and height
            try:
                with Image.open(c_image_path) as img:
                    im_width, im_height = img.size
            except Exception as e:
                logger.error(f"Error loading image {c_image_path}: {e}")
                continue

        record["height"] = im_height
        record["width"] = im_width

        annotation_dict[c_image_path] = record

    return annotation_dict


def create_train_test_split(
    file_list: list[str], test_size: float = 0.2, seed: int = 0
) -> tuple[list[str], list[str]]:
    """
    Split the file list into train and test sets.

    Args:
        file_list (list[str]): List of file paths.
        test_size (float, optional): Percentage of test files. Defaults to 0.2.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        tuple[list[str], list[str]]: List of train and test file paths.

    """
    random.seed(seed)

    num_test = int(len(file_list) * test_size)
    test_files = random.sample(file_list, num_test)
    train_files = [f for f in file_list if f not in test_files]
    return train_files, test_files


def create_dataset(
    dataset_path: str,
    annotation_path: str,
    image_path: str,
    category_mapping_fn: Callable[[list[RectangleLabelEnum]], int | list[int]]
    | None = None,
    category_info: list[str] | None = None,
    test_size: float = 0.2,
    split_file_path: str | None = None,
):
    """
    Create the fundus labels dataset from the label studio annotations.

    Args:
        dataset_path (str, optional): Path to the fundus labels dataset in detectron
            format.
        annotation_path (str, optional): Path to the label studio annotations.
        image_path (str, optional): Path to the images.
        category_mapping_fn (Callable[[list[str]], int | list[int]], optional):
            Function to map the list of labels to a category ID or list of category IDs.
            If multiple category IDs are returned, multiple annotations will be created
            for the same bounding box. If a number less than 0 is returned, the
            bounding box will be skipped. Defaults to `convert_labels_to_category_id`.
        category_info (list[str] | None, optional): List of category names. Defaults to
            None.
        test_size (float, optional): Percentage of test files. Defaults to 0.2.
        split_file_path (str | None, optional): Path to the train/test split file. If
            None, a new split will be created. Defaults to None.

    Raises:
        FileNotFoundError: If the annotation file does not exist.

    """
    if not os.path.exists(annotation_path):
        raise FileNotFoundError(f"Dataset not found at {annotation_path}")

    if category_mapping_fn is None:
        raise ValueError("category_mapping_fn must be provided")

    # Convert annotations to Detectron2 format
    annotation_dict = convert_annotations(
        annotation_path, image_path, category_mapping_fn=category_mapping_fn
    )

    # Create train/test split
    file_list = list(annotation_dict.keys())

    if split_file_path is not None and os.path.exists(split_file_path):
        with open(split_file_path) as f:
            split_data = json.load(f)

        train_files = split_data["train"]
        test_files = split_data["test"]
    else:
        train_files, test_files = create_train_test_split(
            file_list, test_size=test_size
        )

    dataset = {
        "train": [annotation_dict[f] for f in train_files if f in annotation_dict],
        "test": [annotation_dict[f] for f in test_files if f in annotation_dict],
    }

    if category_info is not None:
        dataset["categories"] = category_info

    os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)
