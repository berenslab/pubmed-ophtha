"""Functions that convert the dataset from the ImageCLEF to other formats."""

from __future__ import annotations

import json
import logging
import os
import shutil
import xml.etree.ElementTree as ET

import cv2
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

IMAGECLEF_XML_PATHS = {
    "train": "FigureSeparationTraining2016-GT.xml",
    "test": "FigureSeparationTest2016GT.xml",
}
IMAGECLEF_FOLDER_NAMES = {
    "train": "FigureSeparationTraining2016",
    "test": "FigureSeparationTest2016",
}

IMAGE_CLEF_CAPTION_FILES = {
    "train": "CompoundFigureDetectionTraining2016-Captions.csv",
    "test": "CompoundFigureDetectionTest2016-Captions.csv",
}

logger = logging.getLogger(__name__)


def _convert_set_to_yolo_format(
    input_xml_file: str,
    input_image_folder: str,
    annotation_output_folder: str,
    image_output_folder: str,
):
    """
    Convert a set in ImageCLEF (train/test) to YOLO format.

    Args:
        input_xml_file (str): Path to the XML file containing the annotations.
        input_image_folder (str): Path to the folder containing the images.
        annotation_output_folder (str): Folder to save the annotations to.
        image_output_folder (str): Folder to save the training images in.

    """
    tree = ET.parse(input_xml_file)
    root = tree.getroot()

    for annotation_item in tqdm(root.findall("./annotation")):
        filename_item = annotation_item.find("./filename")

        if filename_item is None or filename_item.text is None:
            continue

        image_path = os.path.join(input_image_folder, f"{filename_item.text}.jpg")

        if not os.path.exists(image_path):
            logger.warning(f"Image '{image_path}' does not exist")
            continue

        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image '{image_path}'")
        height, width, _ = image.shape

        object_annotations = []

        for object_item in annotation_item.findall("./object"):
            coordinates = object_item.findall("./point")
            if len(coordinates) == 4:
                x_coords = [int(point.attrib["x"]) for point in coordinates]
                y_coords = [int(point.attrib["y"]) for point in coordinates]

                obj_width = max(x_coords) - min(x_coords)
                obj_height = max(y_coords) - min(y_coords)
                x_mid = (min(x_coords) + obj_width / 2) / width
                y_mid = (min(y_coords) + obj_height / 2) / height

                object_annotations.append(
                    (0, x_mid, y_mid, obj_width / width, obj_height / height)
                )

        if len(object_annotations) > 0:
            label_path = os.path.join(
                annotation_output_folder, f"{filename_item.text}.txt"
            )

            with open(label_path, "w") as f:
                for obj in object_annotations:
                    f.write(f"{obj[0]} {obj[1]} {obj[2]} {obj[3]} {obj[4]}\n")

        image_path = os.path.join(image_output_folder, f"{filename_item.text}.jpg")
        cv2.imwrite(image_path, image)


def convert_to_yolo_format(image_clef_folder: str, output_folder: str):
    """
    Convert the ImageCLEF dataset to the YOLO format.

    The resulting dataset has only one category: 0. This category indicates a panel.

    Args:
        image_clef_folder (str): Path to the folder containing the ImageCLEF dataset.\
            Expects the following folder contents:
            ├── FigureSeparationTraining2016 (folder containing training images)
            ├── FigureSeparationTraining2016-GT.xml
            ├── FigureSeparationTest2016 (folder containing test images)
            └── FigureSeparationTest2016GT.xml
        output_folder (str): Folder containing the output images and labels.

    """
    set_xml_paths = {
        set_name: os.path.join(image_clef_folder, xml_path)
        for set_name, xml_path in IMAGECLEF_XML_PATHS.items()
    }
    set_image_folder_paths = {
        set_name: os.path.join(image_clef_folder, folder_name)
        for set_name, folder_name in IMAGECLEF_FOLDER_NAMES.items()
    }

    image_output_folders = {
        "train": os.path.join(output_folder, "images", "train"),
        "test": os.path.join(output_folder, "images", "test"),
    }
    annotation_output_folders = {
        "train": os.path.join(output_folder, "labels", "train"),
        "test": os.path.join(output_folder, "labels", "test"),
    }

    for set_name, xml_path in set_xml_paths.items():
        if not os.path.exists(xml_path):
            logger.warning(f"File '{xml_path}' does not exist")
            continue

        # Get the image folder
        image_folder = set_image_folder_paths[set_name]
        if not os.path.exists(image_folder):
            logger.warning(f"Folder '{image_folder}' does not exist")
            continue

        # Create the output folders
        image_output_folder = image_output_folders[set_name]
        annotation_output_folder = annotation_output_folders[set_name]

        os.makedirs(image_output_folder, exist_ok=True)
        os.makedirs(annotation_output_folder, exist_ok=True)

        # Call conversion on the set
        _convert_set_to_yolo_format(
            xml_path, image_folder, annotation_output_folder, image_output_folder
        )


def _convert_set_to_coco_format(
    input_xml_file: str,
    input_image_folder: str,
    output_folder: str,
    output_annotation_file: str,
    base_id: int = 0,
    add_segmentation: bool = True,
    caption_file: str | None = None,
) -> int:
    """
    Convert the given set into COCO format.

    Args:
        input_xml_file (str): Path to the XML file containing the annotations.
        input_image_folder (str): Path to the folder containing the images.
        output_folder (str): Folder to save the images to.
        output_annotation_file (str): File where the COCO annotations will be saved.
        base_id (int, optional): In the COCO annotation format each image needs a \
            unique identifier. These will be sequential starting from base_id. \
            Defaults to 0.
        add_segmentation (bool, optional): If True, additionally add the bounding \
            boxes as segmentation. Defaults to True.
        caption_file (str | None, optional): Path to the caption file. If provided, \
            the captions will be added to the images in the COCO format. \
            Defaults to None.

    Returns:
        int: The next base_id to use for the next set.

    """
    tree = ET.parse(input_xml_file)
    root = tree.getroot()

    annotation_dict = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "panel", "supercategory": "none"}],
    }

    caption_dict = None

    if caption_file is not None:
        with open(caption_file) as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines]
            file_names = [line.split("\t")[0] for line in lines]
            caption_dict = {
                file_name: ". ".join(
                    line.removeprefix(file_name).strip().split(" . ")
                ).strip()
                for file_name, line in zip(file_names, lines)
            }

    for annotation_item in tqdm(root.findall("./annotation")):
        filename_item = annotation_item.find("./filename")

        if filename_item is None or filename_item.text is None:
            continue

        image_path = os.path.join(input_image_folder, f"{filename_item.text}.jpg")

        if not os.path.exists(image_path):
            logger.warning(f"Image '{image_path}' does not exist")
            continue

        output_file_name = f"{os.path.basename(filename_item.text)}.jpg"
        image_id = base_id + len(annotation_dict["images"])
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to read image '{image_path}'")

        height, width, _ = image.shape

        annotation_dict["images"].append(
            {
                "id": image_id,
                "file_name": output_file_name,
                "height": height,
                "width": width,
            }
        )

        if caption_dict is not None:
            if caption_dict.get(filename_item.text) is not None:
                annotation_dict["images"][-1]["_image_caption"] = caption_dict[
                    filename_item.text
                ]

        for object_item in annotation_item.findall("./object"):
            coordinates = object_item.findall("./point")
            if len(coordinates) == 4:
                x_coords = [int(point.attrib["x"]) for point in coordinates]
                y_coords = [int(point.attrib["y"]) for point in coordinates]

                x0 = min(x_coords)
                y0 = min(y_coords)
                x1 = max(x_coords)
                y1 = max(y_coords)
                bbox_width = x1 - x0
                bbox_height = y1 - y0

                annotation_dict["annotations"].append(
                    {
                        "iscrowd": 0,
                        "ignore": 0,
                        "image_id": image_id,
                        "category_id": 0,
                        "bbox": [x0, y0, bbox_width, bbox_height],
                        "area": bbox_width * bbox_height,
                        "segmentation": []
                        if not add_segmentation
                        else [[x0, y0, x0, y1, x1, y1, x1, y0]],
                        "id": len(annotation_dict["annotations"]),
                    }
                )

        image_path = os.path.join(output_folder, output_file_name)
        cv2.imwrite(image_path, image)

    with open(output_annotation_file, "w") as f:
        json.dump(annotation_dict, f, ensure_ascii=False, indent=4)

    return base_id + len(annotation_dict["images"])


def convert_to_coco_format(
    image_clef_folder: str, output_folder: str, add_segmentation: bool = True
):
    """
    Convert the ImageCLEF dataset to the COCO format.

    The resulting dataset has only one category: "panel".

    Args:
        image_clef_folder (str): Path to the folder containing the ImageCLEF dataset.\
            Expects the following folder contents:
            ├── FigureSeparationTraining2016 (folder containing training images)
            ├── FigureSeparationTraining2016-GT.xml
            ├── FigureSeparationTest2016 (folder containing test images)
            ├── FigureSeparationTest2016GT.xml
            ├── CompoundFigureDetectionTraining2016-Captions.csv (optional)
            └── CompoundFigureDetectionTest2016-Captions.csv (optional)

        output_folder (str): Folder containing the output images and labels.
        add_segmentation (bool, optional): If True, additionally add the bounding \
            boxes as segmentation. Defaults to True.

    """
    # Expects the follow folder structure:
    # image_clef_folder
    # ├── FigureSeparationTraining2016
    # ├── FigureSeparationTraining2016-GT.xml
    # ├── FigureSeparationTest2016
    # ├── FigureSeparationTest2016GT.xml
    # ├── CompoundFigureDetectionTraining2016-Captions.csv
    # └── CompoundFigureDetectionTest2016-Captions.csv

    set_xml_paths = {
        set_name: os.path.join(image_clef_folder, xml_path)
        for set_name, xml_path in IMAGECLEF_XML_PATHS.items()
    }
    set_image_folder_paths = {
        set_name: os.path.join(image_clef_folder, folder_name)
        for set_name, folder_name in IMAGECLEF_FOLDER_NAMES.items()
    }

    set_caption_file_paths = {
        set_name: os.path.join(image_clef_folder, caption_file)
        for set_name, caption_file in IMAGE_CLEF_CAPTION_FILES.items()
    }

    image_output_folders = {
        "train": os.path.join(output_folder, "train"),
        "test": os.path.join(output_folder, "test"),
    }
    annotation_output_files = {
        "train": os.path.join(output_folder, "annotations", "train.json"),
        "test": os.path.join(output_folder, "annotations", "test.json"),
    }

    os.makedirs(os.path.join(output_folder, "annotations"), exist_ok=True)
    base_id = 0
    for set_name, xml_path in set_xml_paths.items():
        if not os.path.exists(xml_path):
            logger.warning(f"File '{xml_path}' does not exist")
            continue

        # Get the image folder
        image_folder = set_image_folder_paths[set_name]
        if not os.path.exists(image_folder):
            logger.warning(f"Folder '{image_folder}' does not exist")
            continue

        # Create the output folders
        image_output_folder = image_output_folders[set_name]
        annotation_file = annotation_output_files[set_name]

        caption_file = set_caption_file_paths[set_name]
        if not os.path.exists(caption_file):
            caption_file = None

        os.makedirs(image_output_folder, exist_ok=True)

        # Call conversion on the set
        base_id = _convert_set_to_coco_format(
            xml_path,
            image_folder,
            image_output_folder,
            annotation_file,
            base_id=base_id,
            add_segmentation=add_segmentation,
            caption_file=caption_file,
        )


def _convert_panel_seg_sample_to_coco(
    sample_df: pd.DataFrame, image_dir: str, image_id: int, annotation_start_id: int
) -> tuple[dict[str, str | int], list[dict[str, int | str | list[float] | None]]]:
    """
    Convert a single sample from the PanelSeg dataset to COCO format.

    Args:
        sample_df (pd.DataFrame): DataFrame containing the bounding box annotations \
            for a single image.
        image_dir (str): Directory where the image will be saved.
        image_id (int): Sequential ID for the image in the COCO format.
        annotation_start_id (int): Starting ID for the annotations in the COCO format.

    Returns:
        tuple[dict, list[dict]]: A tuple containing the image information and a list \
            of annotations in COCO format.

    """
    annotations = []
    image_path = sample_df["on_disk_path"].iloc[0]

    # copy image to the appropriate directory
    shutil.copyfile(image_path, os.path.join(image_dir, os.path.basename(image_path)))
    with Image.open(image_path) as im:
        width, height = im.size

    image_info = {
        "id": image_id,
        "file_name": os.path.basename(image_path),
        "width": width,
        "height": height,
    }

    ann_start_id = annotation_start_id

    for _, row in sample_df.iterrows():
        annotation = {
            "id": ann_start_id,
            "image_id": image_id,
            "category_id": 0,
            "bbox": [
                row["panel_x0"],
                row["panel_y0"],
                row["panel_x1"] - row["panel_x0"],
                row["panel_y1"] - row["panel_y0"],
            ],
            "area": (row["panel_x1"] - row["panel_x0"])
            * (row["panel_y1"] - row["panel_y0"]),
            "iscrowd": 0,
            "segmentation": [
                [
                    row["panel_x0"],
                    row["panel_y0"],
                    row["panel_x1"],
                    row["panel_y0"],
                    row["panel_x1"],
                    row["panel_y1"],
                    row["panel_x0"],
                    row["panel_y1"],
                ]
            ],
            "__label_name__": None,
        }
        annotations.append(annotation)
        ann_start_id += 1

        if not pd.isna(row["label_x0"]):
            label_name = None

            if not pd.isna(row["label_name"]):
                label_name = row["label_name"].strip()

                annotations[-1]["__label_name__"] = label_name

            annotation = {
                "id": ann_start_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [
                    row["label_x0"],
                    row["label_y0"],
                    row["label_x1"] - row["label_x0"],
                    row["label_y1"] - row["label_y0"],
                ],
                "area": (row["label_x1"] - row["label_x0"])
                * (row["label_y1"] - row["label_y0"]),
                "iscrowd": 0,
                "segmentation": [
                    [
                        row["label_x0"],
                        row["label_y0"],
                        row["label_x1"],
                        row["label_y0"],
                        row["label_x1"],
                        row["label_y1"],
                        row["label_x0"],
                        row["label_y1"],
                    ]
                ],
                "__label_name__": label_name,
            }
            annotations.append(annotation)
            ann_start_id += 1

    return image_info, annotations


def _convert_panel_seg_split(
    split_df: pd.DataFrame, annotation_file: str, image_folder: str
):
    """
    Convert a split of the PanelSeg dataset to COCO format.

    Args:
        split_df (pd.DataFrame): DataFrame containing the split data.
        annotation_file (str): Path to save the COCO annotations.
        image_folder (str): Folder to save the images to.

    """
    assert annotation_file.endswith(
        ".json"
    ), f"Annotation file '{annotation_file}' must be a JSON file."

    output_dict = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 0, "name": "panel"},
            {"id": 1, "name": "label"},
        ],
    }
    for index, sub_df in tqdm(split_df.groupby("file_name", as_index=False)):
        image_id = len(output_dict["images"]) + 1
        annotation_start_id = len(output_dict["annotations"]) + 1
        image_info, annotations = _convert_panel_seg_sample_to_coco(
            sub_df, image_folder, image_id, annotation_start_id
        )
        output_dict["images"].append(image_info)
        output_dict["annotations"].extend(annotations)
    with open(annotation_file, "w") as f:
        json.dump(output_dict, f, indent=4, ensure_ascii=False)


def create_updated_csvs(
    panel_seg_dataset_path: str,
    image_clef_coco_dataset_path: str,
    output_path: str | None = None,
):
    """
    Create updated CSV files for the PanelSeg dataset.

    These csvs resolve the overlap between the train set of the PanelSeg dataset \
    and the test set of the ImageCLEF dataset as well as the train set of the \
    ImageCLEF dataset and the eval set of the PanelSeg dataset.

    The updated CSVs also remove duplicate entries that were manually selected.

    Args:
        panel_seg_dataset_path (str): Base path to the PanelSeg dataset.
        image_clef_coco_dataset_path (str): Base path of the COCO version of the \
            ImageCLEF dataset.
        output_path (str | None, optional): Path to save the updated CSVs to. If None, \
            defaults to `panel_seg_dataset_path/data/updated_csvs`. Defaults to None.

    """
    if output_path is None:
        output_path = os.path.join(panel_seg_dataset_path, "data", "updated_csvs")

    ignore_file_paths = [
        "\\".join(["PanelSeg", "data", "8", "PMC1550422_1471-2407-6-199-2.jpg"]),
        "\\".join(["PanelSeg", "data", "0", "1477-7819-7-10-1.jpg"]),
        "\\".join(["PanelSeg", "data", "0", "1757-1626-1-309-2.jpg"]),
        "\\".join(["PanelSeg", "data", "0", "cc2360-2.jpg"]),
    ]

    eval_df = pd.read_csv(
        os.path.join(panel_seg_dataset_path, "data", "eval.csv"), header=None
    )
    train_df = pd.read_csv(
        os.path.join(panel_seg_dataset_path, "data", "train.csv"), header=None
    )

    with open(
        os.path.join(image_clef_coco_dataset_path, "annotations", "train.json")
    ) as file:
        image_clef_train_file_names = [
            e["file_name"].removesuffix(".jpg") for e in json.load(file)["images"]
        ]

    with open(
        os.path.join(image_clef_coco_dataset_path, "annotations", "test.json")
    ) as file:
        image_clef_test_file_names = [
            e["file_name"].removesuffix(".jpg") for e in json.load(file)["images"]
        ]

    panel_seg_train_image_clef_test_overlap = {}

    for file_name in train_df[0].unique().tolist():
        if file_name is None or not isinstance(file_name, str):
            raise ValueError(f"Invalid file name '{file_name}' in train_df")
        base_name = os.path.basename(file_name.replace("\\", "/"))
        pmc_prefix = base_name.split("_")[0] + "_"
        base_name = base_name.removeprefix(pmc_prefix).removesuffix(".jpg")
        if (base_name.startswith("gr") and len(base_name) == 3) or base_name.startswith(
            "IPC-11-1"
        ):
            base_name = file_name

        if base_name in image_clef_test_file_names:
            panel_seg_train_image_clef_test_overlap[file_name] = base_name

    panel_seg_test_image_clef_train_overlap = {}

    for file_name in eval_df[0].unique().tolist():
        if file_name is None or not isinstance(file_name, str):
            raise ValueError(f"Invalid file name '{file_name}' in eval_df")
        base_name = os.path.basename(file_name.replace("\\", "/"))
        pmc_prefix = base_name.split("_")[0] + "_"
        base_name = base_name.removeprefix(pmc_prefix).removesuffix(".jpg")
        if (base_name.startswith("gr") and len(base_name) == 3) or base_name.startswith(
            "IPC-11-1"
        ):
            base_name = file_name

        if base_name in image_clef_train_file_names:
            panel_seg_test_image_clef_train_overlap[file_name] = base_name

    move_from_eval_to_train = eval_df[
        eval_df[0].isin(list(panel_seg_test_image_clef_train_overlap.keys()))
    ]

    move_from_train_to_eval = train_df[
        train_df[0].isin(list(panel_seg_train_image_clef_test_overlap.keys()))
    ]

    updated_train_df = pd.concat(
        [
            train_df[
                (~train_df[0].isin(move_from_train_to_eval[0]))
                & (~train_df[0].isin(ignore_file_paths))
            ],
            move_from_eval_to_train,
        ],
        ignore_index=True,
    )

    updated_eval_df = pd.concat(
        [
            eval_df[
                (~eval_df[0].isin(move_from_eval_to_train[0]))
                & (~eval_df[0].isin(ignore_file_paths))
            ],
            move_from_train_to_eval,
        ],
        ignore_index=True,
    )

    os.makedirs(output_path, exist_ok=True)

    updated_train_df.to_csv(
        os.path.join(output_path, "train.csv"), index=False, header=False
    )
    updated_eval_df.to_csv(
        os.path.join(output_path, "eval.csv"), index=False, header=False
    )


def convert_panel_seg_to_coco_format(
    dataset_base_path: str, output_path: str, updated_csv_path: str | None = None
):
    """
    Convert the PanelSeg dataset to COCO format.

    Note that the following file paths are duplicates and will be ignored:
        - PanelSeg/data/8/PMC1550422_1471-2407-6-199-2.jpg
        - PanelSeg/data/0/1477-7819-7-10-1.jpg
        - PanelSeg/data/0/1757-1626-1-309-2.jpg
        - PanelSeg/data/0/cc2360-2.jpg

    Args:
        dataset_base_path (str): Path to the base directory of the PanelSeg dataset. \
            Expects the following folder contents:
            ├── data
            │   ├── 0
            |   |   └── <file_name>.jpg
            │   ├── 4
            │   ├── 5
            │   ├── 6
            │   ├── 7
            │   ├── 8
            │   ├── 9
            │   ├── 10
            │   ├── train.csv
            │   └── eval.csv

        output_path (str): Path to save the converted dataset. The output will be \
            saved in the following structure:
            ├── train
            │   ├── <file_name>.jpg
            ├── test
            │   ├── <file_name>.jpg
            ├── annotations
            │   ├── train.json
            │   └── test.json
        updated_csv_path (str): Path to the folder containing the updated CSV files \
            (train.csv and eval.csv). If missing, the function will raise an error. \
            These CSVs are necessary since the train set of the PanelSeg dataset and \
            the test set of ImageCLEF overlap. Use the function \
            `create_updated_csvs()` to create these CSVs. If None, defaults to \
            `dataset_base_path/data/updated_csvs`. Defaults to None.

    Raises:
        FileNotFoundError: If the dataset base path or the CSV files do not exist.

    """
    if not os.path.exists(dataset_base_path):
        raise FileNotFoundError(
            f"Dataset base path '{dataset_base_path}' does not exist"
        )

    if updated_csv_path is None:
        updated_csv_path = os.path.join(dataset_base_path, "data", "updated_csvs")

    if not os.path.exists(updated_csv_path):
        raise FileNotFoundError(f"Updated CSV path '{updated_csv_path}' does not exist")

    train_csv_path = os.path.join(updated_csv_path, "train.csv")
    test_csv_path = os.path.join(updated_csv_path, "eval.csv")

    if not os.path.exists(train_csv_path):
        raise FileNotFoundError(f"Train CSV file '{train_csv_path}' does not exist")
    if not os.path.exists(test_csv_path):
        raise FileNotFoundError(f"Test CSV file '{test_csv_path}' does not exist")

    column_names = [
        "file_name",
        "panel_x0",
        "panel_y0",
        "panel_x1",
        "panel_y1",
        "panel_name",
        "label_x0",
        "label_y0",
        "label_x1",
        "label_y1",
        "label_name",
    ]

    train_df = pd.read_csv(train_csv_path, header=None, names=column_names)
    eval_df = pd.read_csv(test_csv_path, header=None, names=column_names)

    # Convert Windows-style paths to current system paths
    eval_df["file_name"] = eval_df["file_name"].apply(
        lambda x: os.path.join(*x.split("\\"))
    )
    train_df["file_name"] = train_df["file_name"].apply(
        lambda x: os.path.join(*x.split("\\"))
    )

    eval_df["base_file_name"] = eval_df["file_name"].apply(os.path.basename)
    train_df["base_file_name"] = train_df["file_name"].apply(os.path.basename)

    # Remove PMC<id> prefix and file extension
    eval_df["processed_base_file_name"] = eval_df["base_file_name"].apply(
        lambda x: x.removeprefix(f"{x.split('_')[0]}_").removesuffix(
            f".{x.split('.')[-1]}"
        )
    )
    train_df["processed_base_file_name"] = train_df["base_file_name"].apply(
        lambda x: x.removeprefix(f"{x.split('_')[0]}_").removesuffix(
            f".{x.split('.')[-1]}"
        )
    )

    eval_df["on_disk_path"] = eval_df["file_name"].apply(
        lambda x: x.replace(
            os.path.join("PanelSeg", "data"), os.path.join(dataset_base_path, "data")
        )
    )
    train_df["on_disk_path"] = train_df["file_name"].apply(
        lambda x: x.replace(
            os.path.join("PanelSeg", "data"), os.path.join(dataset_base_path, "data")
        )
    )

    if not eval_df["on_disk_path"].apply(os.path.exists).all():
        missing_files = (
            eval_df[~eval_df["on_disk_path"].apply(os.path.exists)]["on_disk_path"]
            .unique()
            .tolist()
        )
        raise FileNotFoundError(
            "Some test images do not exist on disk. Please check the dataset path. "
            f"The missing files are:\n{missing_files}"
        )
    if not train_df["on_disk_path"].apply(os.path.exists).all():
        missing_files = (
            train_df[~train_df["on_disk_path"].apply(os.path.exists)]["on_disk_path"]
            .unique()
            .tolist()
        )
        raise FileNotFoundError(
            "Some training images do not exist on disk. Please check the dataset path. "
            f"The missing files are:\n{missing_files}"
        )

    # There are duplicate entries in the dataset, so we need to remove them
    # These were selected manually
    ignore_file_paths = [
        os.path.join("PanelSeg", "data", "8", "PMC1550422_1471-2407-6-199-2.jpg"),
        os.path.join("PanelSeg", "data", "0", "1477-7819-7-10-1.jpg"),
        os.path.join("PanelSeg", "data", "0", "1757-1626-1-309-2.jpg"),
        os.path.join("PanelSeg", "data", "0", "cc2360-2.jpg"),
    ]

    train_df = train_df[~train_df["file_name"].isin(ignore_file_paths)]
    eval_df = eval_df[~eval_df["file_name"].isin(ignore_file_paths)]

    # Create output folders
    image_dirs = {
        "train": os.path.join(output_path, "train"),
        "test": os.path.join(output_path, "test"),
    }
    annotation_paths = {
        "train": os.path.join(output_path, "annotations", "train.json"),
        "test": os.path.join(output_path, "annotations", "test.json"),
    }
    for split, image_dir in image_dirs.items():
        os.makedirs(image_dir, exist_ok=True)

    for split, annotation_path in annotation_paths.items():
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)

    logger.info(f"Converting {len(train_df)} training samples")
    _convert_panel_seg_split(train_df, annotation_paths["train"], image_dirs["train"])
    logger.info(f"Converting {len(eval_df)} evaluation samples")
    _convert_panel_seg_split(eval_df, annotation_paths["test"], image_dirs["test"])
