"""Automatic panel-to-sub-caption assembly using OCR-based label matching."""

import json
import logging
import multiprocessing
import os
import sqlite3

import cv2
import easyocr
import numpy as np
import pandas as pd
import torch
from nltk import edit_distance
from PIL import Image
from tqdm.auto import tqdm

from pubmed_ophtha.const.thresholds import DETECTION_BUFFER
from pubmed_ophtha.panel_assembly.db_loading import (
    get_base_article_data,
)
from pubmed_ophtha.util.computing import get_cpu_count
from pubmed_ophtha.util.database_interface import (
    database_writer,
    get_database_connection_context,
)

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.3
OVERLAP_THRESHOLD = 0.7


class BiDirectionalMap:
    """Bidirectional mapping that supports O(1) lookup in both directions."""

    def __init__(self) -> None:
        """Initialize empty forward and backward dicts."""
        self.forward: dict = {}
        self.backward: dict = {}

    def add(self, key: object, value: object) -> None:
        """
        Insert a key-value pair and its inverse into the map.

        Args:
            key (object): Forward lookup key.
            value (object): Forward lookup value (also used as the backward key).

        """
        self.forward[key] = value
        self.backward[value] = key

    def get_by_key(self, key: object) -> object | None:
        """
        Retrieve the value associated with a key.

        Args:
            key (object): Forward lookup key.

        Returns:
            object | None: Associated value, or None if the key is absent.

        """
        return self.forward.get(key, None)

    def get_by_value(self, value: object) -> object | None:
        """
        Retrieve the key associated with a value.

        Args:
            value (object): Backward lookup key.

        Returns:
            object | None: Associated key, or None if the value is absent.

        """
        return self.backward.get(value, None)


def stem_subcaption_name(name: str) -> str:
    """
    Strip non-alphabetic characters from a sub-caption name and lowercase it.

    Args:
        name (str): Raw sub-caption name string.

    Returns:
        str: Lowercase alphabetic-only version of the name.

    """
    # Remove non-alphabetic characters and convert to lowercase
    return "".join(filter(str.isalpha, name)).lower()


def can_automatically_assign_captions(s_caps: dict[str, str]) -> bool:
    """
    Check whether all sub-caption names reduce to single letters after stemming.

    Only single-letter labels can be reliably matched via OCR without an LLM.

    Args:
        s_caps (dict[str, str]): Mapping of sub-caption names to their text.

    Returns:
        bool: True if every stemmed name is exactly one character long.

    """
    stemmed_names = [stem_subcaption_name(name) for name in s_caps.keys()]
    return all(0 < len(name) < 2 for name in stemmed_names)


def calculate_overlap_fraction(box_a: pd.Series, box_b: pd.Series) -> float:
    """
    Compute what fraction of box_a's area is covered by box_b.

    Args:
        box_a (pd.Series): Row with keys predicted_boxes_x0/y0/x1/y1 (reference
            box).
        box_b (pd.Series): Row with keys predicted_boxes_x0/y0/x1/y1 (covering
            box).

    Returns:
        float: Intersection area divided by box_a area; 0.0 if box_a has zero
            area.

    """
    intersection_x0 = max(box_a["predicted_boxes_x0"], box_b["predicted_boxes_x0"])
    intersection_y0 = max(box_a["predicted_boxes_y0"], box_b["predicted_boxes_y0"])
    intersection_x1 = min(box_a["predicted_boxes_x1"], box_b["predicted_boxes_x1"])
    intersection_y1 = min(box_a["predicted_boxes_y1"], box_b["predicted_boxes_y1"])

    box_a_area = (box_a["predicted_boxes_x1"] - box_a["predicted_boxes_x0"]) * (
        box_a["predicted_boxes_y1"] - box_a["predicted_boxes_y0"]
    )

    intersection_area = max(0, intersection_x1 - intersection_x0) * max(
        0, intersection_y1 - intersection_y0
    )

    return intersection_area / box_a_area


def calculate_iou(box_a: pd.Series, box_b: pd.Series) -> float:
    """
    Compute the Intersection-over-Union between two bounding boxes.

    Args:
        box_a (pd.Series): Row with keys predicted_boxes_x0/y0/x1/y1.
        box_b (pd.Series): Row with keys predicted_boxes_x0/y0/x1/y1.

    Returns:
        float: IoU score in [0, 1]; 0.0 if the union area is zero.

    """
    intersection_x0 = max(box_a["predicted_boxes_x0"], box_b["predicted_boxes_x0"])
    intersection_y0 = max(box_a["predicted_boxes_y0"], box_b["predicted_boxes_y0"])
    intersection_x1 = min(box_a["predicted_boxes_x1"], box_b["predicted_boxes_x1"])
    intersection_y1 = min(box_a["predicted_boxes_y1"], box_b["predicted_boxes_y1"])

    intersection_area = max(0, intersection_x1 - intersection_x0) * max(
        0, intersection_y1 - intersection_y0
    )

    box_a_area = (box_a["predicted_boxes_x1"] - box_a["predicted_boxes_x0"]) * (
        box_a["predicted_boxes_y1"] - box_a["predicted_boxes_y0"]
    )
    box_b_area = (box_b["predicted_boxes_x1"] - box_b["predicted_boxes_x0"]) * (
        box_b["predicted_boxes_y1"] - box_b["predicted_boxes_y0"]
    )

    union_area = box_a_area + box_b_area - intersection_area

    return intersection_area / union_area if union_area > 0 else 0


def get_article_images_data(
    cursor: sqlite3.Cursor, article_images_id: int, model_version: str
) -> dict | None:
    """
    Load figure data needed for automatic panel assembly from the database.

    Retrieves split captions, NMS-filtered predictions, and the article image
    for a given article_images record.

    Args:
        cursor (sqlite3.Cursor): Active read-only SQLite cursor.
        article_images_id (int): Primary key of the article_images row.
        model_version (str): Model version string used to filter
            image_predictions rows.

    Returns:
        dict | None: Dict with keys article_id, image_cluster_id, split_captions,
            predictions_df, full_caption_id, full_caption_text,
            sub_caption_texts, and article_image; or None if captions cannot be
            loaded.

    """
    base = get_base_article_data(cursor, article_images_id, model_version)
    if base is None:
        return None

    entries = base["split_captions_entries"]  # (id, name, text, is_full_caption)
    split_captions_dict = {
        name: s_id for s_id, name, _, is_full in entries if not is_full
    }
    full_caption_id = next(s_id for s_id, _, _, is_full in entries if is_full)
    full_caption_text = next(text for _, _, text, is_full in entries if is_full)
    sub_caption_texts = {
        name: text for _, name, text, is_full in entries if not is_full
    }

    predictions_df = base["predictions_df"]
    predictions_df.set_index("id", inplace=True)
    predictions_df = predictions_df[predictions_df["survives_nms"].astype(bool)]

    return {
        "article_id": base["article_id"],
        "image_cluster_id": base["image_cluster_id"],
        "split_captions": split_captions_dict,
        "predictions_df": predictions_df,
        "full_caption_id": full_caption_id,
        "full_caption_text": full_caption_text,
        "sub_caption_texts": sub_caption_texts,
        "article_image": base["article_image"],
    }


def get_panel_hierarchy(predicted_boxes_df: pd.DataFrame) -> dict[int, dict]:
    """
    Build a hierarchy mapping each panel to its associated labels and images.

    For each Panel box, collects Label and image boxes that overlap it
    sufficiently (by IoU or containment) and expands the panel content box
    to encompass all associated detections.

    Args:
        predicted_boxes_df (pd.DataFrame): NMS-filtered predictions with columns
            predicted_labels, predicted_boxes_x0/y0/x1/y1.

    Returns:
        dict[int, dict]: Maps panel row index to a dict with keys labels (list
            of label row indices), images (list of image row indices), and
            content_box (x0, y0, x1, y1 tuple).

    """
    panel_mask = predicted_boxes_df["predicted_labels"] == "Panel"
    label_mask = predicted_boxes_df["predicted_labels"] == "Label"
    image_mask = ~(panel_mask | label_mask)

    hierarchy = {}

    for panel_id, panel_df in predicted_boxes_df[panel_mask].iterrows():
        panel_data = {
            "labels": [],
            "images": [],
            "content_box": (
                panel_df["predicted_boxes_x0"],
                panel_df["predicted_boxes_y0"],
                panel_df["predicted_boxes_x1"],
                panel_df["predicted_boxes_y1"],
            ),
        }

        for label_id, label_df in predicted_boxes_df[label_mask].iterrows():
            panel_iou = calculate_iou(panel_df, label_df)
            if panel_iou >= IOU_THRESHOLD:
                panel_data["labels"].append(label_id)
                panel_data["content_box"] = (
                    min(panel_data["content_box"][0], label_df["predicted_boxes_x0"]),
                    min(panel_data["content_box"][1], label_df["predicted_boxes_y0"]),
                    max(panel_data["content_box"][2], label_df["predicted_boxes_x1"]),
                    max(panel_data["content_box"][3], label_df["predicted_boxes_y1"]),
                )
            else:
                # check for overlaps
                label_overlaps = calculate_overlap_fraction(label_df, panel_df)
                if label_overlaps >= OVERLAP_THRESHOLD:
                    panel_data["labels"].append(label_id)
                    panel_data["content_box"] = (
                        min(
                            panel_data["content_box"][0], label_df["predicted_boxes_x0"]
                        ),
                        min(
                            panel_data["content_box"][1], label_df["predicted_boxes_y0"]
                        ),
                        max(
                            panel_data["content_box"][2], label_df["predicted_boxes_x1"]
                        ),
                        max(
                            panel_data["content_box"][3], label_df["predicted_boxes_y1"]
                        ),
                    )

        for image_id, image_df in predicted_boxes_df[image_mask].iterrows():
            panel_iou = calculate_iou(panel_df, image_df)
            if panel_iou >= IOU_THRESHOLD:
                panel_data["images"].append(image_id)
                panel_data["content_box"] = (
                    min(panel_data["content_box"][0], image_df["predicted_boxes_x0"]),
                    min(panel_data["content_box"][1], image_df["predicted_boxes_y0"]),
                    max(panel_data["content_box"][2], image_df["predicted_boxes_x1"]),
                    max(panel_data["content_box"][3], image_df["predicted_boxes_y1"]),
                )
            else:
                # check for overlaps
                image_overlaps = calculate_overlap_fraction(image_df, panel_df)
                if image_overlaps >= OVERLAP_THRESHOLD:
                    panel_data["images"].append(image_id)
                    panel_data["content_box"] = (
                        min(
                            panel_data["content_box"][0],
                            image_df["predicted_boxes_x0"],
                        ),
                        min(
                            panel_data["content_box"][1],
                            image_df["predicted_boxes_y0"],
                        ),
                        max(
                            panel_data["content_box"][2],
                            image_df["predicted_boxes_x1"],
                        ),
                        max(
                            panel_data["content_box"][3],
                            image_df["predicted_boxes_y1"],
                        ),
                    )

        hierarchy[panel_id] = panel_data

    return hierarchy


def detect_text(
    reader: easyocr.Reader,
    input_image: Image.Image,
    score_threshold: float = 0.8,
) -> list | None:
    """
    Run OCR on an image crop and return results above the confidence threshold.

    Falls back to a binarized, upscaled version of the crop when no confident
    detection is found on the raw image.

    Args:
        reader (easyocr.Reader): Initialized EasyOCR Reader instance.
        input_image (Image.Image): PIL image crop to read text from.
        score_threshold (float): Minimum confidence score to accept a detection.

    Returns:
        list | None: List of (bbox, text, score) tuples sorted by descending
            score, or None if no text is detected above the threshold in either
            pass.

    """
    # Convert PIL image to OpenCV format
    im_cv = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    result = reader.readtext(im_cv)
    # Sort result by score
    result = sorted(result, key=lambda x: x[2], reverse=True)  # pyright: ignore[reportArgumentType]

    if len(result) == 0 or result[0][2] < score_threshold:  # pyright: ignore[reportOperatorIssue, reportArgumentType]
        # Fall back to thresholding
        upscale_factor = 3
        gray = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=upscale_factor, fy=upscale_factor)
        _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        new_result = reader.readtext(bw, detail=1)

        if len(new_result) == 0 and len(result) == 0:
            return None

        elif len(new_result) > 0 and (
            len(result) == 0
            or max([r[2] for r in new_result]) > max([r[2] for r in result])  # pyright: ignore[reportArgumentType]
        ):
            rescaled_result = []
            for detected_box, detected_text, detection_score in new_result:
                rescaled_box = [
                    (point[0] / upscale_factor, point[1] / upscale_factor)  # pyright: ignore[reportOperatorIssue]
                    for point in detected_box
                ]
                rescaled_result.append((rescaled_box, detected_text, detection_score))

            # Sort result by score
            result = sorted(rescaled_result, key=lambda x: x[2], reverse=True)
    return result


def can_assign(
    input_text: str,
    sub_caption_name: str,
    use_lower_case_matching: bool = True,
    use_special_character_removal_matching: bool = True,
    use_number_removal_matching: bool = True,
    use_lower_special_character_removal_matching: bool = True,
    use_lower_number_removal_matching: bool = True,
) -> bool:
    """
    Check whether OCR-detected text matches a sub-caption name.

    Tries strategies in order: exact, lowercase, alphanumeric-only, alpha-only,
    and lowercase variants of the latter two.

    Args:
        input_text (str): Text string detected by OCR.
        sub_caption_name (str): Sub-caption name to match against.
        use_lower_case_matching (bool): Enable case-insensitive comparison.
        use_special_character_removal_matching (bool): Enable comparison after
            stripping non-alphanumeric characters.
        use_number_removal_matching (bool): Enable comparison after stripping
            digits.
        use_lower_special_character_removal_matching (bool): Enable lowercase
            comparison after stripping non-alphanumeric characters.
        use_lower_number_removal_matching (bool): Enable lowercase comparison
            after stripping digits.

    Returns:
        bool: True if any enabled matching strategy produces a match.

    """
    # First try exact match
    if input_text.strip() == sub_caption_name.strip():
        return True

    if use_lower_case_matching and (
        input_text.strip().lower() == sub_caption_name.strip().lower()
    ):
        return True

    if use_special_character_removal_matching and "".join(
        filter(str.isalnum, input_text.strip())
    ) == "".join(filter(str.isalnum, sub_caption_name.strip())):
        return True

    if use_number_removal_matching and "".join(
        filter(str.isalpha, input_text.strip())
    ) == "".join(filter(str.isalpha, sub_caption_name.strip())):
        return True

    if use_lower_special_character_removal_matching and (
        "".join(filter(str.isalnum, input_text.strip())).lower()
        == "".join(filter(str.isalnum, sub_caption_name.strip())).lower()
    ):
        return True

    if use_lower_number_removal_matching and (
        "".join(filter(str.isalpha, input_text.strip())).lower()
        == "".join(filter(str.isalpha, sub_caption_name.strip())).lower()
    ):
        return True
    return False


def assign_text_to_labels(
    detected_label_texts: dict[int, dict],
    detected_missing_labels: dict[int, dict],
    sub_captions: dict[str, str],
) -> BiDirectionalMap:
    """
    Build a one-to-one mapping between detected label boxes and sub-caption names.

    First assigns labels detected directly on label boxes, then falls back to
    labels detected within the panel crop for any unmatched sub-captions.

    Args:
        detected_label_texts (dict[int, dict]): Maps label row index to a dict
            with keys text and confidence.
        detected_missing_labels (dict[int, dict]): Maps panel row index to a
            dict of synthesized label entries (same structure as
            detected_label_texts).
        sub_captions (dict[str, str]): Maps sub-caption name to its database ID.

    Returns:
        BiDirectionalMap: Mapping from sub-caption ID to label row index.

    """
    # Create a 1 to 1 mapping between detected labels and subcaption names.
    # First assign detected_label_texts, then detected_missing_labels if any remain.

    label_assignments = BiDirectionalMap()

    # Check which comparisons can be used
    use_lower_case_matching = set(sub_captions.keys()) == {
        name.lower() for name in sub_captions.keys()
    }
    use_special_character_removal_matching = set(sub_captions.keys()) == {
        "".join(filter(str.isalnum, name)) for name in sub_captions.keys()
    }

    use_number_removal_matching = set(sub_captions.keys()) == {
        "".join(filter(str.isalpha, name)) for name in sub_captions.keys()
    }

    use_lower_special_character_removal_matching = set(sub_captions.keys()) == {
        "".join(filter(str.isalnum, name)).lower() for name in sub_captions.keys()
    }

    use_lower_number_removal_matching = set(sub_captions.keys()) == {
        "".join(filter(str.isalpha, name)).lower() for name in sub_captions.keys()
    }

    for label_id, detected_data in detected_label_texts.items():
        for sub_caption_name, sub_caption_id in sub_captions.items():
            if label_assignments.get_by_key(sub_caption_id) is not None:
                continue

            # Check if we can assign the label to the sub caption
            if can_assign(
                detected_data["text"],
                sub_caption_name,
                use_lower_case_matching=use_lower_case_matching,
                use_special_character_removal_matching=use_special_character_removal_matching,
                use_number_removal_matching=use_number_removal_matching,
                use_lower_special_character_removal_matching=use_lower_special_character_removal_matching,
                use_lower_number_removal_matching=use_lower_number_removal_matching,
            ):
                label_assignments.add(sub_caption_id, label_id)
                break

    for _, missing_labels in detected_missing_labels.items():
        for label_id, detected_data in missing_labels.items():
            for sub_caption_name, sub_caption_id in sub_captions.items():
                if label_assignments.get_by_key(sub_caption_id) is not None:
                    continue

                # Check if we can assign the label to the sub caption
                if can_assign(
                    detected_data["text"],
                    sub_caption_name,
                    use_lower_case_matching=use_lower_case_matching,
                    use_special_character_removal_matching=use_special_character_removal_matching,
                    use_number_removal_matching=use_number_removal_matching,
                    use_lower_special_character_removal_matching=use_lower_special_character_removal_matching,
                    use_lower_number_removal_matching=use_lower_number_removal_matching,
                ):
                    label_assignments.add(sub_caption_id, label_id)
                    break

    return label_assignments


def create_final_hierarchy(
    hierarchy: dict[int, dict],
    detected_label_texts: dict[int, dict],
    detected_missing_labels: dict[int, dict],
    predicted_boxes_df: pd.DataFrame,
    label_assignments: BiDirectionalMap,
) -> list[dict]:
    """
    Produce the final per-panel assignment records from detection and label data.

    For each panel, collects assigned and unassigned label entries, flags panels
    that need human review (shared labels, unresolved labels, or missing labels),
    and marks single clean assignments as final.

    Args:
        hierarchy (dict[int, dict]): Panel hierarchy from ``get_panel_hierarchy``.
        detected_label_texts (dict[int, dict]): Maps label row index to OCR
            result dict.
        detected_missing_labels (dict[int, dict]): Maps panel row index to
            synthesized label dicts for panels with no directly detected labels.
        predicted_boxes_df (pd.DataFrame): NMS-filtered predictions DataFrame.
        label_assignments (BiDirectionalMap): Mapping from sub-caption ID to
            label row index.

    Returns:
        list[dict]: Panel assignment dicts, each with keys panel_id, labels,
            images, has_estimated_labels, needs_review, has_shared_labels,
            panel_content_box, unassigned, and assigned_subcaption.

    """
    # Create final assignments, mark panels for further review

    final_assignments = []

    assigned_labels = []
    shared_labels = []

    for panel_id, panel_data in hierarchy.items():
        panel_assignment = {
            "panel_id": panel_id,
            "labels": {},
            "images": panel_data["images"],
            "has_estimated_labels": False,
            "needs_review": False,
            "has_shared_labels": False,
            "panel_content_box": panel_data["content_box"],
            "unassigned": {},
            "assigned_subcaption": None,
        }

        for label_id in panel_data["labels"]:
            if label_id in detected_label_texts:
                sub_caption_id = label_assignments.get_by_value(label_id)
                label_box = [
                    e.item() if isinstance(e, np.generic) else e
                    for e in predicted_boxes_df.loc[
                        label_id,
                        [
                            "predicted_boxes_x0",
                            "predicted_boxes_y0",
                            "predicted_boxes_x1",
                            "predicted_boxes_y1",
                        ],
                    ].values
                ]

                confidence_value = detected_label_texts[label_id]["confidence"]
                if isinstance(confidence_value, np.generic):
                    confidence_value = confidence_value.item()

                if sub_caption_id is not None:
                    panel_assignment["labels"][label_id] = {
                        "sub_caption_id": sub_caption_id,
                        "text": detected_label_texts[label_id]["text"],
                        "confidence": confidence_value,
                        "bbox": label_box,
                    }

                    if label_id in assigned_labels:
                        panel_assignment["has_shared_labels"] = True
                        shared_labels.append(label_id)
                    else:
                        assigned_labels.append(label_id)
                else:
                    panel_assignment["unassigned"][label_id] = {
                        "text": detected_label_texts[label_id]["text"],
                        "confidence": confidence_value,
                        "bbox": label_box,
                    }
        if panel_id in detected_missing_labels:
            for label_id, detected_data in detected_missing_labels[panel_id].items():
                sub_caption_id = label_assignments.get_by_value(label_id)
                if sub_caption_id is not None:
                    confidence_value = detected_data["confidence"]
                    if isinstance(confidence_value, np.generic):
                        confidence_value = confidence_value.item()

                    panel_assignment["labels"][label_id] = {
                        "sub_caption_id": sub_caption_id,
                        "text": detected_data["text"],
                        "confidence": confidence_value,
                        "bbox": [
                            e.item() if isinstance(e, np.generic) else e
                            for e in detected_data["bbox"]
                        ],
                    }
                    panel_assignment["has_estimated_labels"] = True

        if (
            len(panel_assignment["labels"]) == 0
            or len(panel_assignment["unassigned"]) > 0
            or len(panel_assignment["labels"]) != len(panel_data["labels"])
        ):
            panel_assignment["needs_review"] = True

        final_assignments.append(panel_assignment)

    for panel_assignment in final_assignments:
        # Update shared labels info
        if not panel_assignment["has_shared_labels"] and any(
            [label_id in shared_labels for label_id in panel_assignment["labels"]]
        ):
            panel_assignment["has_shared_labels"] = True

        if (
            len(panel_assignment["labels"]) == 1
            and len(panel_assignment["unassigned"]) == 0
            and not panel_assignment["has_estimated_labels"]
            and not panel_assignment["has_shared_labels"]
        ):
            panel_assignment["assigned_subcaption"] = list(
                panel_assignment["labels"].values()
            )[0]["sub_caption_id"]

    return final_assignments


def detect_label_texts(
    article_image: Image.Image,
    predicted_boxes_df: pd.DataFrame,
    hierarchy: dict[int, dict],
    sub_captions: dict[str, str],
) -> tuple[dict[int, dict], dict[int, dict]]:
    """
    Run OCR on label boxes and on panel crops when no label yielded a result.

    Only runs OCR when sub-captions can be automatically assigned (i.e. all
    names are single-letter after stemming).

    Args:
        article_image (Image.Image): Full article image from which crops are
            extracted.
        predicted_boxes_df (pd.DataFrame): NMS-filtered predictions DataFrame.
        hierarchy (dict[int, dict]): Panel hierarchy from ``get_panel_hierarchy``.
        sub_captions (dict[str, str]): Maps sub-caption name to its database ID.

    Returns:
        tuple[dict[int, dict], dict[int, dict]]: Tuple of
            (detected_label_texts, detected_missing_labels).
            detected_label_texts maps label row index to a dict with keys text
            and confidence, sorted by descending confidence.
            detected_missing_labels maps panel row index to a dict of
            synthesized label entries for panels where no label box yielded a
            detection.

    """
    detected_label_texts = {}
    detected_missing_labels = {}

    label_mask = predicted_boxes_df["predicted_labels"] == "Label"

    if can_automatically_assign_captions(sub_captions):
        reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

        for label_id, label_data in predicted_boxes_df[label_mask].iterrows():
            buf = DETECTION_BUFFER * max(
                label_data["predicted_boxes_x1"] - label_data["predicted_boxes_x0"],
                label_data["predicted_boxes_y1"] - label_data["predicted_boxes_y0"],
            )
            box = (
                label_data["predicted_boxes_x0"] - buf,
                label_data["predicted_boxes_y0"] - buf,
                label_data["predicted_boxes_x1"] + buf,
                label_data["predicted_boxes_y1"] + buf,
            )

            im_cp = article_image.crop(box)

            result = detect_text(reader, im_cp, score_threshold=0.8)
            if result is None or len(result) == 0:
                continue

            # only use the text with the highest confidence
            best_entry = result[0]

            detected_label_texts[label_id] = {
                "text": best_entry[1],
                "confidence": best_entry[2],
            }

        # Sort by confidence
        detected_label_texts = dict(
            sorted(
                detected_label_texts.items(),
                key=lambda item: item[1]["confidence"],
                reverse=True,
            )
        )

        num_new_labels = 0
        for panel_id, panel_data in hierarchy.items():
            if len(panel_data["labels"]) > 0 and any(
                [label in detected_label_texts for label in panel_data["labels"]]
            ):
                # found at least one label with detected text
                continue

            im_cp = article_image.crop(panel_data["content_box"])

            result = detect_text(reader, im_cp, score_threshold=0.8)
            if result is None or len(result) == 0:
                continue

            new_panel_data = {}
            for detected_box, detected_text, detection_score in result:
                if detection_score < 0.8:
                    continue

                label_id = -1 * num_new_labels - 1

                # Move box to global position
                bbox = (
                    detected_box[0][0] + panel_data["content_box"][0],
                    detected_box[0][1] + panel_data["content_box"][1],
                    detected_box[2][0] + panel_data["content_box"][0],
                    detected_box[2][1] + panel_data["content_box"][1],
                )
                num_new_labels += 1

                new_panel_data[label_id] = {
                    "bbox": bbox,
                    "text": detected_text,
                    "confidence": detection_score,
                }

            # Sort by confidence
            new_panel_data = dict(
                sorted(
                    new_panel_data.items(),
                    key=lambda item: item[1]["confidence"],
                    reverse=True,
                )
            )
            detected_missing_labels[panel_id] = new_panel_data
    return detected_label_texts, detected_missing_labels


def assign_captions_automatically(
    figure_data: dict, _article_images_id: int
) -> list[dict]:
    """
    Automatically assign sub-captions to detected panels for a single figure.

    Handles the single-caption case (no split needed), the irrelevant-image
    case (no fundus modality detected), and the general multi-panel case using
    OCR-based label matching.

    Args:
        figure_data (dict): Dict returned by ``get_article_images_data``.
        article_images_id (int): Primary key of the article_images row (used
            for logging only).

    Returns:
        list[dict]: Panel assignment dicts as produced by
            ``create_final_hierarchy``, with irrelevant panels marked via a
            relevance key set to False.

    """
    sub_captions = figure_data["split_captions"]
    predicted_boxes_df = figure_data["predictions_df"]
    article_image = figure_data["article_image"]

    # Check if further processing is needed
    stop_processing = (
        not predicted_boxes_df["predicted_labels"]
        .apply(lambda x: x in ["OCT", "CFP", "Retinal Imaging"])
        .any()
    )
    panel_mask = predicted_boxes_df["predicted_labels"] == "Panel"
    label_mask = predicted_boxes_df["predicted_labels"] == "Label"
    image_mask = ~(panel_mask | label_mask)

    if stop_processing:
        return [
            {"panel_id": panel_id, "relevance": False}
            for panel_id in predicted_boxes_df[panel_mask].index.tolist()
        ]

    hierarchy = get_panel_hierarchy(predicted_boxes_df)

    num_panels = len(predicted_boxes_df[panel_mask])

    if len(sub_captions) == 1:
        if (
            "None" in sub_captions
            or None in sub_captions
            or edit_distance(
                figure_data["full_caption_text"].split(),
                list(figure_data["sub_caption_texts"].values())[0].split(),
            )
            < 4
        ):
            # No captions to assign
            image_box_ids = predicted_boxes_df[image_mask].index.tolist()

            panel_box_id = None
            if num_panels > 0:
                panel_box_id = (
                    predicted_boxes_df[panel_mask]
                    .sort_values(by="predicted_scores", ascending=False)
                    .index.tolist()[0]
                )
            return [
                {
                    "panel_id": panel_box_id,
                    "labels": {},
                    "images": image_box_ids,
                    "has_estimated_labels": False,
                    "needs_review": False,
                    "has_shared_labels": False,
                    "panel_content_box": (
                        0,
                        0,
                        article_image.width,
                        article_image.height,
                    ),
                    "unassigned": {},
                    "assigned_subcaption": figure_data["full_caption_id"],
                },
            ] + [
                {"panel_id": panel_id, "relevance": False}
                for panel_id in predicted_boxes_df[panel_mask].index.tolist()
                if panel_box_id is None or panel_id != panel_box_id
            ]
        else:
            # Mark all as irrelevant
            return [
                {"panel_id": panel_id, "relevance": False}
                for panel_id in predicted_boxes_df[panel_mask].index.tolist()
            ]

    detected_label_texts, detected_missing_labels = detect_label_texts(
        article_image, predicted_boxes_df, hierarchy, sub_captions
    )

    label_assignments = assign_text_to_labels(
        detected_label_texts, detected_missing_labels, sub_captions
    )

    final_hierarchy = create_final_hierarchy(
        hierarchy,
        detected_label_texts,
        detected_missing_labels,
        predicted_boxes_df,
        label_assignments,
    )

    missing_panel_ids = {p_id for p_id in predicted_boxes_df[panel_mask].index} - {
        entry["panel_id"] for entry in final_hierarchy
    }
    for panel_id in missing_panel_ids:
        final_hierarchy.append({"panel_id": panel_id, "relevance": False})

    return final_hierarchy


def create_tables(write_cur: sqlite3.Cursor) -> None:
    """
    Create the label_assignments and panel_assignments tables if they do not exist.

    Args:
        write_cur (sqlite3.Cursor): Writable SQLite cursor.

    """
    write_cur.execute("""
    CREATE TABLE IF NOT EXISTS label_assignments (
        id INTEGER PRIMARY KEY ON CONFLICT REPLACE,
        sub_caption_id INTEGER,
        text TEXT,
        confidence FLOAT,
        bbox_x0 FLOAT,
        bbox_y0 FLOAT,
        bbox_x1 FLOAT,
        bbox_y1 FLOAT,
        panel_id INTEGER,
        origin TEXT,
        FOREIGN KEY (panel_id) REFERENCES image_predictions(id),
        FOREIGN KEY (sub_caption_id) REFERENCES parsed_split_captions(id)
    );
    """)

    # Create panel_assignments
    write_cur.execute("""
    CREATE TABLE IF NOT EXISTS panel_assignments (
        panel_id INTEGER PRIMARY KEY,
        label_ids TEXT,
        image_ids TEXT,
        relevance BOOLEAN,
        has_estimated_labels BOOLEAN,
        needs_review BOOLEAN,
        has_shared_labels BOOLEAN,
        panel_content_box_x0 FLOAT,
        panel_content_box_y0 FLOAT,
        panel_content_box_x1 FLOAT,
        panel_content_box_y1 FLOAT,
        unassigned_labels TEXT,
        assigned_subcaption INTEGER,
        origin TEXT,
        FOREIGN KEY (panel_id) REFERENCES image_predictions(id),
        FOREIGN KEY (assigned_subcaption) REFERENCES parsed_split_captions(id)
    );
    """)


def post_process_tables(write_cur: sqlite3.Cursor) -> None:
    """
    Run all pre-assignment table migrations in the correct order.

    Adds the prediction_source column to image_predictions if missing.

    Args:
        write_cur (sqlite3.Cursor): Writable SQLite cursor.

    """
    # Update image_predictions
    # Add prediction_source column to image_predictions if not exists (default: 'model')
    write_cur.execute("""
    PRAGMA table_info(image_predictions);
    """)
    columns = [column[1] for column in write_cur.fetchall()]
    if "prediction_source" not in columns:
        write_cur.execute("""
        ALTER TABLE image_predictions
        ADD COLUMN prediction_source TEXT DEFAULT 'model';
        """)


def write_into_db(
    write_cur: sqlite3.Cursor,
    article_images_id: int,
    final_hierarchy: list[dict],
    current_model_version: str,
    article_id: int,
) -> None:
    """
    Persist panel and label assignment results to the database.

    Inserts any panels that were synthesized during assignment (panel_id is
    None), then inserts label_assignments rows one by one and updates
    panel_assignments rows for all panels in the hierarchy.

    Args:
        write_cur (sqlite3.Cursor): Writable SQLite cursor.
        article_images_id (int): Primary key of the article_images row.
        final_hierarchy (list[dict]): Panel assignment dicts from
            ``assign_captions_automatically``.
        current_model_version (str): Model version string written to new
            predictions.
        article_id (int): Article identifier written to new image_predictions
            rows.

    Raises:
        sqlite3.InterfaceError: If a value cannot be bound during panel
            assignment insertion.
        sqlite3.ProgrammingError: If the panel assignment INSERT statement is
            malformed.

    """
    # Preprocess and find panels and labels that need to be inserted
    batched_label_data = {
        "sub_caption_id": [],
        "text": [],
        "confidence": [],
        "bbox_x0": [],
        "bbox_y0": [],
        "bbox_x1": [],
        "bbox_y1": [],
        "panel_id": [],
        "origin": [],
    }
    for panel_data in final_hierarchy:
        panel_id = panel_data.get("panel_id", None)

        if panel_id is None:
            # Insert new panel into image_predictions
            write_cur.execute(
                """
                INSERT INTO image_predictions (
                    article_images_id,
                    predicted_labels,
                    predicted_boxes_x0,
                    predicted_boxes_y0,
                    predicted_boxes_x1,
                    predicted_boxes_y1,
                    predicted_scores,
                    prediction_source,
                    model_version,
                    article_id,
                    secondary_predicted_labels,
                    secondary_predicted_scores,
                    survives_nms,
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                RETURNING id;
                """,
                (
                    article_images_id,
                    "Panel",
                    panel_data["panel_content_box"][0],
                    panel_data["panel_content_box"][1],
                    panel_data["panel_content_box"][2],
                    panel_data["panel_content_box"][3],
                    1.0,
                    "automatic_caption_assignment",
                    current_model_version,
                    article_id,
                    "None",
                    -1.0,
                    True,
                ),
            )
            panel_id = write_cur.fetchone()[0]
            panel_data["panel_id"] = panel_id

        if "labels" not in panel_data:
            continue

        for label_id, label_info in panel_data["labels"].items():
            if label_id < 0:
                origin = -1

            else:
                origin = label_id

            batched_label_data["sub_caption_id"].append(label_info["sub_caption_id"])
            batched_label_data["text"].append(label_info["text"])
            batched_label_data["confidence"].append(label_info["confidence"])
            batched_label_data["bbox_x0"].append(label_info["bbox"][0])
            batched_label_data["bbox_y0"].append(label_info["bbox"][1])
            batched_label_data["bbox_x1"].append(label_info["bbox"][2])
            batched_label_data["bbox_y1"].append(label_info["bbox"][3])
            batched_label_data["panel_id"].append(panel_id)
            batched_label_data["origin"].append(str(origin))

    # Insert all new labels
    db_label_ids = []
    for i in range(len(batched_label_data["sub_caption_id"])):
        write_cur.execute(
            """
            INSERT INTO label_assignments (
                sub_caption_id,
                text,
                confidence,
                bbox_x0,
                bbox_y0,
                bbox_x1,
                bbox_y1,
                panel_id,
                origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id;
            """,
            (
                batched_label_data["sub_caption_id"][i],
                batched_label_data["text"][i],
                batched_label_data["confidence"][i],
                batched_label_data["bbox_x0"][i],
                batched_label_data["bbox_y0"][i],
                batched_label_data["bbox_x1"][i],
                batched_label_data["bbox_y1"][i],
                batched_label_data["panel_id"][i],
                batched_label_data["origin"][i],
            ),
        )
        db_label_id = write_cur.fetchone()
        db_label_ids.append(db_label_id)

    # Map back the inserted label IDs to the panel data
    label_index = 0

    batched_panel_data = {
        "panel_id": [],
        "label_ids": [],
        "image_ids": [],
        "relevance": [],
        "has_estimated_labels": [],
        "needs_review": [],
        "has_shared_labels": [],
        "panel_content_box_x0": [],
        "panel_content_box_y0": [],
        "panel_content_box_x1": [],
        "panel_content_box_y1": [],
        "unassigned_labels": [],
        "assigned_subcaption": [],
    }
    for panel_data in final_hierarchy:
        if "relevance" in panel_data and not panel_data["relevance"]:
            batched_panel_data["panel_id"].append(panel_data["panel_id"])
            batched_panel_data["label_ids"].append(
                json.dumps([], separators=(",", ":"))
            )
            batched_panel_data["image_ids"].append(
                json.dumps([], separators=(",", ":"))
            )
            batched_panel_data["relevance"].append(False)
            batched_panel_data["has_estimated_labels"].append(False)
            batched_panel_data["needs_review"].append(False)
            batched_panel_data["has_shared_labels"].append(False)
            batched_panel_data["panel_content_box_x0"].append(0)
            batched_panel_data["panel_content_box_y0"].append(0)
            batched_panel_data["panel_content_box_x1"].append(0)
            batched_panel_data["panel_content_box_y1"].append(0)
            batched_panel_data["unassigned_labels"].append(
                json.dumps({}, separators=(",", ":"))
            )
            batched_panel_data["assigned_subcaption"].append("")
            continue

        if "labels" not in panel_data:
            raise ValueError("Panel data missing 'labels' key during DB write.")

        updated_labels = []
        for label_id, label_info in panel_data["labels"].items():
            db_label_id = db_label_ids[label_index][0]
            updated_labels.append(db_label_id)
            label_index += 1

        batched_panel_data["panel_id"].append(panel_data["panel_id"])
        batched_panel_data["label_ids"].append(
            json.dumps(updated_labels, separators=(",", ":"))
        )
        batched_panel_data["image_ids"].append(
            json.dumps(panel_data["images"], separators=(",", ":"))
        )
        batched_panel_data["relevance"].append(panel_data.get("relevance", True))
        batched_panel_data["has_estimated_labels"].append(
            panel_data["has_estimated_labels"]
        )
        batched_panel_data["needs_review"].append(panel_data["needs_review"])
        batched_panel_data["has_shared_labels"].append(panel_data["has_shared_labels"])
        batched_panel_data["panel_content_box_x0"].append(
            panel_data["panel_content_box"][0]
        )
        batched_panel_data["panel_content_box_y0"].append(
            panel_data["panel_content_box"][1]
        )
        batched_panel_data["panel_content_box_x1"].append(
            panel_data["panel_content_box"][2]
        )
        batched_panel_data["panel_content_box_y1"].append(
            panel_data["panel_content_box"][3]
        )
        batched_panel_data["unassigned_labels"].append(
            json.dumps(panel_data["unassigned"], separators=(",", ":"))
        )
        batched_panel_data["assigned_subcaption"].append(
            panel_data["assigned_subcaption"]
        )

    try:
        # Insert all panel data in batch
        write_cur.executemany(
            """
            INSERT INTO panel_assignments (
                panel_id,
                label_ids,
                image_ids,
                relevance,
                has_estimated_labels,
                needs_review,
                has_shared_labels,
                panel_content_box_x0,
                panel_content_box_y0,
                panel_content_box_x1,
                panel_content_box_y1,
                unassigned_labels,
                assigned_subcaption,
                origin
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            list(
                zip(
                    batched_panel_data["panel_id"],
                    batched_panel_data["label_ids"],
                    batched_panel_data["image_ids"],
                    batched_panel_data["relevance"],
                    batched_panel_data["has_estimated_labels"],
                    batched_panel_data["needs_review"],
                    batched_panel_data["has_shared_labels"],
                    batched_panel_data["panel_content_box_x0"],
                    batched_panel_data["panel_content_box_y0"],
                    batched_panel_data["panel_content_box_x1"],
                    batched_panel_data["panel_content_box_y1"],
                    batched_panel_data["unassigned_labels"],
                    batched_panel_data["assigned_subcaption"],
                    ["automatic_caption_assignment"]
                    * len(batched_panel_data["panel_id"]),
                )
            ),
        )
    except sqlite3.InterfaceError as e:
        logger.error(f"Error during label assignment insertion: {e}")
        logger.error(
            f"Batched label data size: {len(batched_label_data['sub_caption_id'])}"
        )
        raise e
    except sqlite3.ProgrammingError as e:
        logger.error(f"Error during panel assignment insertion: {e}")
        logger.error(f"Batched panel data size: {len(batched_panel_data['panel_id'])}")
        raise e
    except sqlite3.IntegrityError as e:
        logger.error(f"Integrity error during panel assignment insertion: {e}")
        raise e


def batched_db_write(connection: sqlite3.Connection, batch: list[dict]) -> None:
    """
    Write a batch of assignment results to the database and commit.

    Args:
        connection (sqlite3.Connection): Open SQLite connection used to obtain
            a write cursor.
        batch (list[dict]): Dicts with keys article_images_id, final_hierarchy,
            current_model_version, and article_id.

    """
    cur = connection.cursor()
    for entry in batch:
        write_into_db(
            cur,
            entry["article_images_id"],
            entry["final_hierarchy"],
            entry["current_model_version"],
            entry["article_id"],
        )
    connection.commit()


def worker_process(
    database_path: str,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    progress_queue: multiprocessing.Queue,
    model_version: str,
    device_name: str | int | None,
) -> None:
    """
    Spawned worker that reads article_images_ids from a queue and runs assignment.

    Consumes tasks until a None poison pill is received, puts results on
    result_queue, and signals progress via progress_queue. A final None on
    progress_queue indicates the worker has finished.

    Args:
        database_path (str): Path to the SQLite database file (read-only).
        task_queue (multiprocessing.Queue): Supplies article_images_id integers;
            None terminates the worker.
        result_queue (multiprocessing.Queue): Receives assignment result dicts.
        progress_queue (multiprocessing.Queue): Receives integer increments
            (1 per completed task) and a final None when the worker exits.
        model_version (str): Forwarded to ``get_article_images_data``.
        device_name (str | int | None): CUDA device index or "cpu" used to set
            CUDA_VISIBLE_DEVICES.

    """
    if device_name is not None and device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device_name is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(device_name)

    try:
        with get_database_connection_context(database_path, read_only=True) as con:
            cur = con.cursor()
            while True:
                task = task_queue.get()
                if task is None:
                    # Poison pill means shutdown
                    break
                article_images_id = task

                try:
                    figure_data = get_article_images_data(
                        cur, article_images_id, model_version
                    )
                except Exception as e:
                    logger.error(
                        f"Error retrieving figure data for"
                        f" article_images_id {article_images_id}: {e}"
                    )
                    progress_queue.put(1)  # Indicate that one task is done
                    continue

                if figure_data is None:
                    logger.debug(
                        f"Figure data for article_images_id"
                        f" {article_images_id} is None, skipping."
                    )
                    progress_queue.put(1)  # Indicate that one task is done
                    continue

                try:
                    assignment = assign_captions_automatically(
                        figure_data, article_images_id
                    )

                    result_queue.put(
                        {
                            "article_images_id": article_images_id,
                            "final_hierarchy": assignment,
                            "current_model_version": model_version,
                            "article_id": figure_data["article_id"],
                        }
                    )

                except Exception as e:
                    logger.error(f"Error processing image {article_images_id}: {e}")
                finally:
                    progress_queue.put(1)  # Indicate that one task is done
    except Exception as e:
        logger.error(f"Error in assignment worker: {e}")
    finally:
        progress_queue.put(None)  # Indicate that this worker is done


def run_automatic_assignment(
    database_path: str,
    model_version: str,
    num_workers: int | None = None,
    database_batch_size: int = 100,
    easy_ocr_estimated_model_size: int = 5,
) -> None:
    """
    Run the full automatic panel assembly pipeline in parallel.

    Performs pre-processing migrations, selects unprocessed article images,
    spawns worker processes (one per available VRAM slot or CPU), collects
    results via a dedicated database-writer process, and tracks progress.

    Args:
        database_path (str): Path to the SQLite database file.
        model_version (str): Model version string used to filter panel
            predictions.
        num_workers (int | None): Number of parallel worker processes. Defaults
            to CPU count minus one; clamped to available GPU slots when CUDA is
            present.
        database_batch_size (int): Number of results to accumulate before
            writing a batch to the database.
        easy_ocr_estimated_model_size (int): Estimated EasyOCR model VRAM usage
            in GB, used to compute how many workers can share a GPU.

    Warning:
        Before dispatching work, this function performs a rollback of partially
        processed articles: any article_image whose panels are not all present in
        ``panel_assignments`` has its existing rows deleted from
        ``panel_assignments``, ``label_assignments``, and ``image_predictions``
        (where ``prediction_source`` is ``'automatic_caption_assignment'``).
        This ensures idempotent re-runs but is destructive — partial results for
        those articles are permanently removed from the database and will be
        recomputed from scratch.

    """
    if num_workers is None:
        num_workers = max(1, get_cpu_count() - 1)

    num_workers = max(num_workers, 1)

    if num_workers > 1:
        # make sure that there are enough devices per worker
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            # Get VRAM on all devices
            vram_sizes = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(num_devices)
            ]
            # Sort devices by VRAM size
            sorted_devices = sorted(
                range(num_devices), key=lambda i: vram_sizes[i], reverse=True
            )

            # Select devices with enough VRAM
            selected_devices = []
            num_possible_workers = 0
            buffer_factor = 1.1
            gb_to_bytes = 1024**3
            easy_ocr_estimated_model_size_bytes = (
                easy_ocr_estimated_model_size * gb_to_bytes
            )
            for device in sorted_devices:
                if (
                    vram_sizes[device]
                    > easy_ocr_estimated_model_size_bytes * buffer_factor
                ):  # 50% buffer
                    selected_devices.append(device)
                    num_possible_workers += int(
                        vram_sizes[device]
                        // (easy_ocr_estimated_model_size_bytes * buffer_factor)
                    )

            if num_possible_workers < num_workers:
                logger.warning(
                    f"Limited VRAM: Reducing workers to {num_possible_workers}."
                )
                num_workers = num_possible_workers

            # Assign workers to devices in round-robin fashion
            worker_device_map = [
                selected_devices[i % len(selected_devices)] for i in range(num_workers)
            ]
        else:
            # All workers on CPU
            worker_device_map = ["cpu"] * num_workers
    else:
        worker_device_map = [
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        ]

    with get_database_connection_context(database_path, read_only=False) as con:
        write_cur = con.cursor()

        post_process_tables(write_cur)
        create_tables(write_cur)

        con.commit()

        # Get all article_images_ids that need processing
        # write_cur.execute("""
        # SELECT id
        # FROM article_images;
        # """)

        # all_article_images_ids = [row[0] for row in write_cur.fetchall()]

        write_cur.execute(
            """
        SELECT id, article_images_id FROM image_predictions
        WHERE predicted_labels='Panel' AND model_version = ?;
        """,
            (model_version,),
        )

        unprocessed_panel_ids = write_cur.fetchall()

        write_cur.execute("""SELECT DISTINCT panel_id FROM panel_assignments;""")
        processed_panel_ids = {row[0] for row in write_cur.fetchall()}

        all_article_images_ids = set()

        # roll back changes if only few of the panels in an image were processed
        for panel_id, article_images_id in tqdm(
            unprocessed_panel_ids, desc="Preparing unprocessed panels"
        ):
            if (
                panel_id not in processed_panel_ids
                and article_images_id not in all_article_images_ids
            ):
                all_article_images_ids.add(article_images_id)

            if article_images_id not in all_article_images_ids:
                continue

            # Drop panel id from panel_assignments, label_assignments
            # and (image_predictions if newly created)
            write_cur.execute(
                "DELETE FROM panel_assignments WHERE panel_id = ?;", (panel_id,)
            )
            write_cur.execute(
                """DELETE FROM label_assignments WHERE panel_id = ?;""",
                (panel_id,),
            )

            write_cur.execute(
                "DELETE FROM image_predictions"
                " WHERE id = ? AND prediction_source = 'automatic_caption_assignment';",
                (panel_id,),
            )
        all_article_images_ids = list(all_article_images_ids)
        con.commit()

    mp_context = multiprocessing.get_context("spawn")

    # Prepare queues for multiprocessing
    result_queue = mp_context.Queue(maxsize=1000)
    task_queue = mp_context.Queue()

    # Queue to track progress
    progress_queue = mp_context.Queue()

    # Prepare process to write into the database
    database_writer_process = mp_context.Process(
        target=database_writer,
        args=(database_path, result_queue, batched_db_write),
        kwargs={
            "batch_size": database_batch_size,
        },
        name="DatabaseWriterProcess",
    )
    database_writer_process.start()

    # Put tasks into the task queue
    for task in all_article_images_ids:
        task_queue.put(task)

    num_devices = len(set(worker_device_map))
    logger.info(
        f"Starting assignment with {num_workers} workers on {num_devices} devices..."
    )

    # Start workers
    worker_processes = []
    for i in range(num_workers):
        p = mp_context.Process(
            target=worker_process,
            args=(
                database_path,
                task_queue,
                result_queue,
                progress_queue,
                model_version,
                worker_device_map[i],
            ),
            name=f"AssignmentWorker-{i}",
        )
        p.start()
        worker_processes.append(p)

    # Add poison pills to stop the workers
    for _ in range(num_workers):
        task_queue.put(None)

    # Track progress
    total_tasks_count = len(all_article_images_ids)
    num_none = 0
    with tqdm(total=total_tasks_count, desc="Processed Entries") as pbar:
        while num_none < num_workers:
            progress = progress_queue.get()
            if progress is None:
                num_none += 1
            else:
                pbar.update(progress)

    # Wait for all workers to finish
    for p in worker_processes:
        p.join()

    # Stop the database writer
    result_queue.put(None)

    # Print remaining tasks in the result queue
    remaining_results = result_queue.qsize()
    if remaining_results > 0:
        logger.info(f"Waiting for {remaining_results} remaining results to be written.")
    database_writer_process.join()
