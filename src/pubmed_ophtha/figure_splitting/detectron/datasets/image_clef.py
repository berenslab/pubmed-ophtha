"""Module for loading the ImageCLEF 2016 dataset in Detectron2 format."""

import json
import os
from typing import Any

from detectron2.structures import BoxMode

from pubmed_ophtha.const.paths import DATASET_FOLDER
from pubmed_ophtha.figure_splitting.dataset_preprocessing.dataset_conversion import (
    convert_to_coco_format,
)
from pubmed_ophtha.figure_splitting.detectron.datasets.meta import (
    DETECTRON_DATASET_REGISTRY,
    register_dataset,
)

DEFAULT_IMAGECLEF_PATH = os.path.join(DATASET_FOLDER, "image_clef")
DEFAULT_DATASET_PATH = os.path.join(DEFAULT_IMAGECLEF_PATH, "detectron_labels.json")
DEFAULT_COCO_PATH = os.path.join(DEFAULT_IMAGECLEF_PATH, "coco_format")


@DETECTRON_DATASET_REGISTRY.register("image_clef_2016")
def add_dataset():
    """Create and register the ImageCLEF 2016 dataset."""
    class_map = ["panel"]

    loading_fn = _load_dataset

    register_dataset(
        dataset_base_name="image_clef_2016",
        loading_fn=loading_fn,
        class_list=class_map,
    )


def _load_dataset(split: str):
    """Load and register the ImageCLEF 2016 dataset in Detectron2 format."""
    # figure_splitting_train
    if not os.path.exists(DEFAULT_DATASET_PATH):
        if (
            not os.path.exists(os.path.join(DEFAULT_COCO_PATH, "annotations"))
            or not os.path.exists(os.path.join(DEFAULT_COCO_PATH, "test"))
            or not os.path.exists(os.path.join(DEFAULT_COCO_PATH, "train"))
            or not os.path.exists(
                os.path.join(DEFAULT_COCO_PATH, "annotations", "train.json")
            )
            or not os.path.exists(
                os.path.join(DEFAULT_COCO_PATH, "annotations", "test.json")
            )
        ):
            # Convert dataset
            convert_to_coco_format(DEFAULT_IMAGECLEF_PATH, DEFAULT_COCO_PATH)

        convert_to_detectron_labels(DEFAULT_DATASET_PATH, DEFAULT_COCO_PATH)

    with open(DEFAULT_DATASET_PATH) as f:
        data = json.load(f)

    return data[split]


def _convert_coco_to_detectron(
    coco_data: dict[str, list[dict[str, Any]]], image_path: str
) -> list[dict[str, Any]]:
    """
    Convert the COCO dataset format to Detectron2 format.

    Converts one loaded annotation file in COCO format to the Detectron2 format.
    Uses the path to the images to point to the correct image files.

    Args:
        coco_data (dict): COCO formatted dataset with keys "images" and "annotations".
        image_path (str): Path to the directory containing the images.

    Returns:
        list[dict]: List of records in Detectron2 format, each containing image \
            metadata and annotations.

    """
    detectron_data = []

    coco_annotations = {}

    for annotation in coco_data["annotations"]:
        image_id = annotation["image_id"]
        if image_id not in coco_annotations:
            coco_annotations[image_id] = []
        coco_annotations[image_id].append(annotation)

    for image in coco_data["images"]:
        image_id = image["id"]

        record = {
            "file_name": os.path.join(image_path, image["file_name"]),
            "image_id": image_id,
            "height": image["height"],
            "width": image["width"],
            "annotations": [],
        }

        for annotation in coco_annotations.get(image_id, []):
            bbox = annotation["bbox"]
            category_id = annotation["category_id"]

            bbox_x0 = bbox[0]
            bbox_y0 = bbox[1]
            bbox_w = bbox[2]
            bbox_h = bbox[3]

            bbox_x1 = bbox_x0 + bbox_w
            bbox_y1 = bbox_y0 + bbox_h

            if bbox_x1 > record["width"] or bbox_y1 > record["height"]:
                raise ValueError(
                    f"Bounding box {bbox} in image {image['file_name']} exceeds image "
                    f"dimensions ({record['width']}x{record['height']})."
                )

            record["annotations"].append(
                {
                    "bbox": [bbox[0], bbox[1], bbox_w, bbox_h],
                    "bbox_mode": BoxMode.XYWH_ABS,
                    "category_id": category_id,
                    "iscrowd": annotation.get("iscrowd", 0),
                    "segmentation": annotation.get("segmentation", []),
                }
            )

        detectron_data.append(record)

    return detectron_data


def convert_to_detectron_labels(dataset_path: str, coco_path: str):
    """
    Load the dataset in COCO format and convert it to Detectron2 format.

    The converted dataset will be saved as a JSON file at the specified \
    `dataset_path`. The COCO formatted dataset should contain the annotations \
    in the "annotations" directory, with separate files for training and testing \
    data. The images should be located in the "train" and "test" directories \
    within the `coco_path`.

    Args:
        dataset_path (str): Location to save the converted dataset in Detectron2 \
            format (JSON file).
        coco_path (str): Path to the COCO formatted version of the ImageCLEF 2016 \
            dataset.

    """
    with open(os.path.join(coco_path, "annotations", "train.json")) as f:
        train_data = json.load(f)
    with open(os.path.join(coco_path, "annotations", "test.json")) as f:
        test_data = json.load(f)

    detectron_data = {
        "train": _convert_coco_to_detectron(
            train_data, os.path.join(coco_path, "train")
        ),
        "test": _convert_coco_to_detectron(test_data, os.path.join(coco_path, "test")),
    }

    with open(dataset_path, "w") as f:
        json.dump(detectron_data, f, indent=4, ensure_ascii=False)
