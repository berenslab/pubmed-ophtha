"""Module for assigning the GT panels to their subcaptions."""

import ast
import asyncio
import json
import logging
import os
from copy import deepcopy

import easyocr
import openai
import pandas as pd
import torch
from tqdm.auto import tqdm

from pubmed_ophtha.caption_splitting.response_models import SplitSubCaptions
from pubmed_ophtha.const.paths import (
    ASSEMBLY_GT_AUTOMATIC_ASSIGNMENT_FILE,
    ASSEMBLY_GT_CAPTION_SPLITTING_FILE,
    ASSEMBLY_GT_FINAL_HIERARCHY_FILE,
    ASSEMBLY_GT_FOLDER,
    ASSEMBLY_GT_IMAGE_PREDICTIONS_FILE,
    ASSEMBLY_GT_LLM_REFINEMENT_FILE,
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import Sample
from pubmed_ophtha.panel_assembly.automatically_assign_panels import (
    assign_captions_automatically,
    detect_text,
    get_panel_hierarchy,
)
from pubmed_ophtha.panel_assembly.llm_refinement import (
    is_partially_assigned,
    refine_panel_assembly,
)
from pubmed_ophtha.panel_assembly.messages import determine_subcaption_type

from .join_with_annotations import (
    GroundTruthSample,
    get_default_gt_files,
    load_gt_annotations,
)

logger = logging.getLogger(__name__)


def automatic_assignment(
    sample: Sample, predictions_df: pd.DataFrame, split_captions_dict: dict[str, str]
) -> list[dict]:
    """
    Create an automatic panel assembly for a sample without GT assignments.

    Args:
        sample (Sample): Panel sample to create the automatic assignment for.
        predictions_df (pd.DataFrame): Image predictions for the sample, including
            predicted boxes and labels.
        split_captions_dict (dict[str, str]): Dictionary mapping sub-caption names to
            their text for the sample.

    Raises:
        ValueError: If the sample does not have a local image path.

    Returns:
        list[dict]: Automatic assignment containing the panel hierarchy and assigned
            sub-captions based on the image predictions and split captions.

    """
    if sample.local_image_base_path is None:
        raise ValueError(f"Sample {sample.id} does not have a local image path.")

    article_image = sample.get_image().convert("RGB")

    article_id = int(sample.data.article_id.removeprefix("PMC"))
    image_cluster_id = (
        sample.base_image_path.removesuffix(".jpg")
        .removesuffix(".png")
        .removesuffix("_updated")
        .split("_", maxsplit=1)[-1]
    )

    full_caption_text = sample.data.caption
    full_caption_id = -1  # Placeholder, as we don't have a caption ID in this context

    split_captions = {key: num for num, key in enumerate(split_captions_dict.keys())}

    article_data = {
        "article_id": article_id,
        "image_cluster_id": image_cluster_id,
        "split_captions": split_captions,
        "predictions_df": predictions_df.set_index("id"),
        "full_caption_id": full_caption_id,
        "full_caption_text": full_caption_text,
        "sub_caption_texts": split_captions_dict,
        "article_image": article_image,
    }

    automatic_assignment = assign_captions_automatically(article_data, -1)

    return automatic_assignment


def automatic_gt_assignment(
    split_caption_dict: dict[str, str],
    image_predictions: pd.DataFrame,
    figure_annotations: GroundTruthSample,
) -> list[dict]:
    """
    Convert GT data to a hierarchy.

    Args:
        split_caption_dict (dict[str, str]): Split caption information for the sample,
            mapping sub-caption names to their text.
        image_predictions (pd.DataFrame): Image predictions for the sample, including
            predicted boxes and labels.
        figure_annotations (GroundTruthSample): Ground truth annotations for the
            sample, containing the panel and caption annotations.

    Raises:
        ValueError: GT data is missing required annotations or has inconsistent
            relationships.

    Returns:
        list[dict]: Sample hierarchy with panel assignments based on GT annotations and
            image predictions.

    """
    panel_sample = figure_annotations["panel"]

    if panel_sample is None:
        raise ValueError("No panel annotations found in GT data for sample.")
    panel_sample_id = panel_sample.id
    if len(split_caption_dict) == 1:
        key_name = list(split_caption_dict.keys())[0]

        if key_name != "None":
            raise ValueError(
                f"GT sample {panel_sample_id} has one sub-caption with name {key_name}"
                ", expected 'None' for unsplit caption."
            )

        panel_prediction = image_predictions[
            image_predictions["predicted_labels"] == "Panel"
        ]
        if len(panel_prediction) != 1:
            raise ValueError(
                f"GT sample {panel_sample_id} has one sub-caption but multiple "
                "predicted panels, expected exactly one predicted panel for unsplit "
                "caption."
            )

        # Calculate panel content box as the union of all predicted label boxes (if any)
        panel_x0 = image_predictions["predicted_boxes_x0"].min().item()
        panel_y0 = image_predictions["predicted_boxes_y0"].min().item()
        panel_x1 = image_predictions["predicted_boxes_x1"].max().item()
        panel_y1 = image_predictions["predicted_boxes_y1"].max().item()
        automatic_hierarchy = [
            {
                "panel_id": panel_prediction.iloc[0]["id"].item(),
                "labels": {},
                "images": image_predictions[
                    (image_predictions["predicted_labels"] != "Panel")
                    & (image_predictions["predicted_labels"] != "Label")
                ]["id"].tolist(),
                "has_estimated_labels": False,
                "needs_review": False,
                "has_shared_labels": False,
                "panel_content_box": (
                    panel_x0,
                    panel_y0,
                    panel_x1,
                    panel_y1,
                ),
                "unassigned": {},
                "assigned_subcaption_name": "None",
                "assigned_subcaption_text": split_caption_dict["None"],
            }
        ]
        return automatic_hierarchy

    caption_sample = figure_annotations["caption"]
    assert caption_sample is not None
    caption_annotations = caption_sample.finished_annotations
    assert caption_annotations is not None
    caption_bounding_boxes = caption_annotations[0].bounding_boxes
    assert caption_bounding_boxes is not None
    panel_id_map = {
        panel.id: {
            "name": panel.name,
            "text": panel.text,
        }
        for panel in caption_bounding_boxes
        if panel.name is not None or (panel.text is not None and len(panel.text) > 0)
    }

    if panel_sample_id == 21548:
        # Error in annotation
        panel_id_map.pop("1")

    _panel_or_image = (
        figure_annotations["panel"]
        if figure_annotations.get("image") is None
        else figure_annotations["image"]
    )
    assert _panel_or_image is not None
    _panel_or_image_annotations = _panel_or_image.finished_annotations
    assert _panel_or_image_annotations is not None
    _panel_or_image_bounding_boxes = _panel_or_image_annotations[0].bounding_boxes
    assert _panel_or_image_bounding_boxes is not None
    panel_box_ids = {panel.id: panel for panel in _panel_or_image_bounding_boxes}

    assert all(panel_id in panel_box_ids for panel_id in panel_id_map.keys())

    if figure_annotations.get("image") is not None and all(
        [
            box.id in panel_id_map or box.parent_id is not None
            for box in panel_box_ids.values()
        ]
    ):
        box_id_to_image_id = {
            row["box_id"]: row["id"] for _, row in image_predictions.iterrows()
        }

        parent_map = {
            box.id: box.parent_id
            for box in panel_box_ids.values()
            if box.parent_id is not None and box.id not in panel_id_map
        }

        # Invert parent map to get children for each parent
        children_map = {}
        for child_id, parent_id in parent_map.items():
            children_map.setdefault(parent_id, []).append(child_id)

        panel_hierarchy = {}

        for panel_id, panel_info in panel_id_map.items():
            children = children_map.get(panel_id, [])
            if len(children) == 0:
                raise ValueError(
                    f"Panel {panel_id} in sample {panel_sample_id} has no children, "
                    "expected at least one child box for panel without name."
                )

            children_image_ids = [box_id_to_image_id[child_id] for child_id in children]
            panel_image_id = box_id_to_image_id[panel_id]

            panel_content_box_df = image_predictions[
                image_predictions["id"].isin([panel_image_id] + children_image_ids)
            ][
                [
                    "predicted_boxes_x0",
                    "predicted_boxes_y0",
                    "predicted_boxes_x1",
                    "predicted_boxes_y1",
                ]
            ]

            panel_hierarchy[panel_image_id] = {
                "labels": image_predictions[
                    (image_predictions["predicted_labels"] == "Label")
                    & (image_predictions["id"].isin(children_image_ids))
                ]["id"].tolist(),
                "images": image_predictions[
                    (image_predictions["predicted_labels"] != "Label")
                    & (image_predictions["predicted_labels"] != "Panel")
                    & (image_predictions["id"].isin(children_image_ids))
                ]["id"].tolist(),
                "content_box": (
                    panel_content_box_df["predicted_boxes_x0"].min().item(),
                    panel_content_box_df["predicted_boxes_y0"].min().item(),
                    panel_content_box_df["predicted_boxes_x1"].max().item(),
                    panel_content_box_df["predicted_boxes_y1"].max().item(),
                ),
            }

    else:
        panel_hierarchy = get_panel_hierarchy(image_predictions)

        has_parent_annotations = any(
            box.parent_id is not None
            for box in panel_box_ids.values()
            if box.id not in panel_id_map
        )

        if has_parent_annotations:
            # Force existing relationships
            for panel_id, panel_data in panel_hierarchy.items():
                panel_box_id = image_predictions[
                    image_predictions["id"] == panel_id
                ].iloc[0]["box_id"]
                for content_label in ["images", "labels"]:
                    content_to_remove = []
                    for image_id in panel_data[content_label]:
                        box_id = image_predictions[
                            image_predictions["id"] == image_id
                        ].iloc[0]["box_id"]

                        if pd.isna(box_id):
                            continue  # Skip if no box_id is associated with the pred

                        if box_id not in panel_box_ids:
                            raise ValueError(
                                f"Image prediction {image_id} in sample "
                                f"{panel_sample_id} has box id {box_id} which is not "
                                "found in GT panel boxes."
                            )

                        if (
                            panel_box_ids[box_id].parent_id is not None
                            and panel_box_ids[box_id].parent_id != panel_box_id
                        ):
                            # Move to the correct panel using GT parent-child relation
                            correct_panel_id = panel_box_ids[box_id].parent_id

                            correct_parent_images_id = image_predictions[
                                image_predictions["box_id"] == correct_panel_id
                            ]
                            if correct_parent_images_id.empty:
                                raise ValueError(
                                    f"Correct parent panel {correct_panel_id} for "
                                    f"image prediction {image_id} in sample "
                                    f"{panel_sample_id} has no associated image "
                                    "predictions."
                                )

                            correct_parent_images_id = correct_parent_images_id.iloc[0][
                                "id"
                            ]

                            if correct_parent_images_id not in panel_hierarchy:
                                raise ValueError(
                                    f"Correct parent panel {correct_panel_id} for "
                                    f"image prediction {image_id} in sample "
                                    f"{panel_sample_id} is not found in the panel "
                                    "hierarchy."
                                )

                            if (
                                image_id
                                not in panel_hierarchy[correct_parent_images_id][
                                    content_label
                                ]
                            ):
                                panel_hierarchy[correct_parent_images_id][
                                    content_label
                                ].append(image_id)
                            if image_id not in content_to_remove:
                                content_to_remove.append(image_id)

                    panel_hierarchy[panel_id][content_label] = [
                        image_id
                        for image_id in panel_data[content_label]
                        if image_id not in content_to_remove
                    ]
        # Assert that all ids are in the panel hierarchy according to GT relationships
        for panel_id, panel_data in panel_hierarchy.items():
            panel_box_id = image_predictions[image_predictions["id"] == panel_id].iloc[
                0
            ]["box_id"]
            for content_label in ["images", "labels"]:
                for image_id in panel_data[content_label]:
                    box_id = image_predictions[
                        image_predictions["id"] == image_id
                    ].iloc[0]["box_id"]

                    if pd.isna(box_id):
                        continue  # Skip if no box_id is associated with the image pred

                    if box_id not in panel_box_ids:
                        raise ValueError(
                            "After enforcing GT relationships, image prediction "
                            f"{image_id} in sample {panel_sample_id} has box id "
                            f"{box_id} which is not found in GT panel boxes."
                        )

                    if (
                        panel_box_ids[box_id].parent_id is not None
                        and panel_box_ids[box_id].parent_id != panel_box_id
                    ):
                        raise ValueError(
                            "After enforcing GT relationships, image prediction "
                            f"{image_id} in sample {panel_sample_id} has box id "
                            f"{box_id} with parent id "
                            f"{panel_box_ids[box_id].parent_id}, expected parent id "
                            f"{panel_id} according to GT relationships."
                        )

    # Assert all ids are assigned
    all_assigned_ids = set()
    for panel_id, panel_data in panel_hierarchy.items():
        all_assigned_ids.update([panel_id])
        all_assigned_ids.update(panel_data["images"])
        all_assigned_ids.update(panel_data["labels"])

    unassigned_ids = set(image_predictions["id"].tolist()) - all_assigned_ids
    if (
        len(unassigned_ids) > 0
        and not (
            image_predictions[image_predictions["id"].isin(unassigned_ids)][
                "prediction_source"
            ]
            == "model"
        ).all()
    ):
        raise ValueError(
            f"Some image predictions in sample {panel_sample_id} are not assigned to "
            f"the panel hierarchy and are not model predictions: {unassigned_ids}"
        )

    # Make sure the hierarchy keeps to the parent_id relationships in the GT data
    automatic_hierarchy = []

    easy_ocr_reader = None

    figure_image = None
    for panel_id, panel_data in panel_hierarchy.items():
        panel_box_id = image_predictions[image_predictions["id"] == panel_id].iloc[0][
            "box_id"
        ]

        # use easy oct on labels

        detected_label_texts = {}
        if len(panel_data["labels"]) > 0:
            if easy_ocr_reader is None:
                easy_ocr_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

            if figure_image is None:
                _panel_for_image = figure_annotations["panel"]
                assert _panel_for_image is not None
                figure_image = _panel_for_image.get_image().convert("RGB")

            for label_id in panel_data["labels"]:
                label_crop = figure_image.crop(
                    (
                        image_predictions[image_predictions["id"] == label_id].iloc[0][
                            "predicted_boxes_x0"
                        ],
                        image_predictions[image_predictions["id"] == label_id].iloc[0][
                            "predicted_boxes_y0"
                        ],
                        image_predictions[image_predictions["id"] == label_id].iloc[0][
                            "predicted_boxes_x1"
                        ],
                        image_predictions[image_predictions["id"] == label_id].iloc[0][
                            "predicted_boxes_y1"
                        ],
                    )
                )

                detected_result = detect_text(
                    easy_ocr_reader, label_crop, score_threshold=0.8
                )

                if detected_result is None or len(detected_result) == 0:
                    detected_label_texts[label_id] = {
                        "sub_caption_text": panel_id_map[panel_box_id]["text"],
                        "text": None,
                        "confidence": None,
                    }
                else:
                    detected_label_texts[label_id] = {
                        "sub_caption_text": panel_id_map[panel_box_id]["text"],
                        "text": detected_result[0][1],
                        "confidence": detected_result[0][2],
                    }

        automatic_hierarchy.append(
            {
                "panel_id": panel_id,
                "labels": detected_label_texts,
                "images": panel_data["images"],
                "has_estimated_labels": False,
                "needs_review": False,
                "has_shared_labels": False,
                "panel_content_box": panel_data["content_box"],
                "unassigned": {},
                "assigned_subcaption_name": panel_id_map[panel_box_id]["name"],
                "assigned_subcaption_text": panel_id_map[panel_box_id]["text"],
            }
        )
    return automatic_hierarchy


async def llm_refinement(
    sample: Sample,
    client: openai.AsyncClient,
    semaphore: asyncio.Semaphore,
    current_assignment: dict,
    predictions_df: pd.DataFrame,
    split_captions: SplitSubCaptions,
) -> dict:
    """
    Run LLM refinement on a single sample.

    Args:
        sample (Sample): Panel sample to refine panel assembly for.
        client (openai.AsyncClient): Client to use for OpenAI API calls.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent API calls.
        current_assignment (dict): Automatic assignment to refine, containing panel
            hierarchy and assigned sub-captions.
        predictions_df (pd.DataFrame): DataFrame containing the image predictions for
            the sample, including predicted boxes and labels.
        split_captions (SplitSubCaptions): Split caption information for the sample,
            including sub-caption texts and types.

    Raises:
        ValueError: Refinement fails or returns invalid results after retries.

    Returns:
        dict: Refinement result containing the unified assignment and any additional
            information returned by the LLM.

    """
    # NOTE positional has to be annotated manually due to processing
    if sample.local_image_base_path is None:
        raise ValueError(f"Sample {sample.id} does not have a local image path.")

    article_image = sample.get_image().convert("RGB")

    article_id = int(sample.data.article_id.removeprefix("PMC"))
    image_cluster_id = (
        sample.base_image_path.removesuffix(".jpg")
        .removesuffix(".png")
        .removesuffix("_updated")
        .split("_", maxsplit=1)[-1]
    )

    full_caption = sample.data.caption

    caption_name_to_id = {
        key: num for num, key in enumerate(split_captions.sub_captions.keys())
    }

    current_panel_data = {
        panel["panel_id"]: {
            "label_ids": [f"a_{lid}" for lid in panel["labels"].keys()],
            "image_ids": panel["images"],
            "relevance": True,
            "has_estimated_labels": panel["has_estimated_labels"],
            "needs_review": panel["needs_review"],
            "has_shared_labels": panel["has_shared_labels"],
            "assigned_subcaption": (
                caption_name_to_id.get(panel["assigned_subcaption_name"])
                if panel["assigned_subcaption_name"] is not None
                else None
            ),
            "unassigned_labels": {
                str(lid): info for lid, info in panel["unassigned"].items()
            },
            "panel_content_box": {
                "x0": panel["panel_content_box"][0],
                "y0": panel["panel_content_box"][1],
                "x1": panel["panel_content_box"][2],
                "y1": panel["panel_content_box"][3],
            },
        }
        for panel in current_assignment
    }

    panel_has_final_assignment = {
        panel_id: (
            panel["assigned_subcaption"] is not None
            and not panel["has_shared_labels"]
            and not panel["needs_review"]
            and not panel["has_estimated_labels"]
        )
        for panel_id, panel in current_panel_data.items()
        if panel is not None
    }

    assigned_label_data = []
    seen_assigned = {}
    for panel in current_assignment:
        panel_id = panel["panel_id"]
        for lid_str, label_info in panel["labels"].items():
            lid = int(lid_str)
            if lid in seen_assigned:
                seen_assigned[lid]["assigned_to"].append(panel_id)
                continue
            if lid not in predictions_df["id"].tolist():
                raise ValueError(
                    f"Label id {lid} assigned to panel {panel_id} in sample "
                    f"{sample.id} is not found in predictions."
                )
            row = predictions_df[predictions_df["id"] == lid].iloc[0]
            entry = {
                "label_id": f"a_{lid}",
                "box": {
                    "x0": row["predicted_boxes_x0"],
                    "y0": row["predicted_boxes_y0"],
                    "x1": row["predicted_boxes_x1"],
                    "y1": row["predicted_boxes_y1"],
                },
                "text": label_info.get("text"),
                "assigned_to": [panel_id],
                "is_final_assignment": panel_has_final_assignment.get(panel_id, False),
                "origin": lid,
            }
            seen_assigned[lid] = entry
            assigned_label_data.append(entry)

    unassigned_label_data = []
    seen_unassigned = {}
    for panel_id, panel in current_panel_data.items():
        for lid_str, info in panel["unassigned_labels"].items():
            lid = int(lid_str)
            if lid < 0:
                continue
            if lid in seen_unassigned:
                seen_unassigned[lid]["assigned_to"].append(panel_id)
                continue
            row = predictions_df[predictions_df["id"] == lid].iloc[0]
            entry = {
                "label_id": f"u_{lid}",
                "box": {
                    "x0": row["predicted_boxes_x0"],
                    "y0": row["predicted_boxes_y0"],
                    "x1": row["predicted_boxes_x1"],
                    "y1": row["predicted_boxes_y1"],
                },
                "text": None,
                "assigned_to": [panel_id],
                "is_final_assignment": False,
            }
            seen_unassigned[lid] = entry
            unassigned_label_data.append(entry)

    joint_label_data = {
        entry["label_id"]: entry
        for entry in assigned_label_data + unassigned_label_data
    }

    image_data = {
        row["id"]: {
            "image_id": row["id"],
            "box": {
                "x0": row["predicted_boxes_x0"],
                "y0": row["predicted_boxes_y0"],
                "x1": row["predicted_boxes_x1"],
                "y1": row["predicted_boxes_y1"],
            },
            "assigned_to": [
                panel_id
                for panel_id, panel in current_panel_data.items()
                if panel is not None and row["id"] in panel["image_ids"]
            ],
        }
        for _, row in predictions_df[
            ~predictions_df["predicted_labels"].isin(["Panel", "Label"])
        ].iterrows()
    }

    extended_panel_data = {
        "article_images_id": sample.id,
        "article_id": article_id,
        "image_cluster_id": image_cluster_id,
        "split_captions": {
            name: sub_caption.text
            for name, sub_caption in split_captions.sub_captions.items()
        },
        "label_data": joint_label_data,
        "full_caption": full_caption,
        "article_image": article_image,
        "panel_is_final": panel_has_final_assignment,
        "image_data": image_data,
        "panel_data": {
            p_id: entry
            for p_id, entry in current_panel_data.items()
            if entry is not None
        },
        "caption_name_to_id": caption_name_to_id,
    }

    result = await refine_panel_assembly(
        extended_panel_data,
        sample.id,  # article_images_id
        client,
        semaphore,
        num_retries=3,
    )

    if result is None:
        raise ValueError(f"LLM refinement failed for sample {sample.id} after retries.")

    # Convert result back into original format
    refined_hierarchy = []

    refined_panel_ids = set()

    for panel_id, panel_assignment in result["unified_assignment"].items():
        if "reject_panel" in panel_assignment and panel_assignment["reject_panel"]:
            raise ValueError(
                f"LLM suggested to reject panel {panel_id} in sample {sample.id}, "
                "which is not supported with gt annotations."
            )

        if (
            "assigned_subcaption" not in panel_assignment
            or panel_assignment["assigned_subcaption"] is None
        ):
            raise ValueError(
                f"LLM refinement result for panel {panel_id} in sample {sample.id} "
                "does not contain an assigned sub-caption, which is required for gt "
                "annotations."
            )

        refined_panel_ids.add(panel_id)
        automatic_panel_data = next(
            (panel for panel in current_assignment if panel["panel_id"] == panel_id),
            None,
        )

        if automatic_panel_data is None:
            raise ValueError(
                f"Panel id {panel_id} in LLM refinement result for sample {sample.id} "
                "is not found in the current assignment."
            )

        label_data = {}

        for label_id in []:
            label_base_name = int(label_id.removeprefix("a_").removeprefix("u_"))
            if label_id.startswith("a_"):
                if label_base_name not in automatic_panel_data["labels"]:
                    label_base_name = str(label_base_name)

                if label_base_name not in automatic_panel_data["labels"]:
                    raise ValueError(
                        f"Assigned label id {label_id} in panel {panel_id} of sample "
                        f"{sample.id} is not found in the current assignment labels."
                    )

                label_data[label_id] = {
                    "text": automatic_panel_data["labels"][label_base_name].get("text"),
                    "sub_caption_text": automatic_panel_data["labels"][
                        label_base_name
                    ].get("sub_caption_text"),
                    "confidence": automatic_panel_data["labels"][label_base_name].get(
                        "confidence"
                    ),
                }
            elif label_id.startswith("u_"):
                if label_base_name not in automatic_panel_data["unassigned"]:
                    label_base_name = str(label_base_name)

                if label_base_name not in automatic_panel_data["unassigned"]:
                    raise ValueError(
                        f"Unassigned label id {label_id} in panel {panel_id} of sample "
                        f"{sample.id} is not found in the current assignment "
                        "unassigned labels."
                    )

                label_data[label_id] = {
                    "text": automatic_panel_data["unassigned"][label_base_name].get(
                        "text"
                    ),
                    "sub_caption_text": panel_assignment["assigned_subcaption"],
                    "confidence": automatic_panel_data["unassigned"][
                        label_base_name
                    ].get("confidence"),
                }

        # Check assigned_subcaption and assigned_images
        refined_panel_data = {
            "panel_id": panel_id,
            "labels": label_data,
            "images": panel_assignment["image_ids"]
            if len(panel_assignment["image_ids"]) == 0
            or not isinstance(panel_assignment["image_ids"][0], list)
            else panel_assignment["image_ids"][0],
            "evidence_data": {
                "evidence": [
                    ev.model_dump() for ev in panel_assignment.get("evidence", [])
                ],
                "confidence": panel_assignment.get("confidence", None),
            },
            "panel_content_box": (
                panel_assignment["content_box"]["x0"],
                panel_assignment["content_box"]["y0"],
                panel_assignment["content_box"]["x1"],
                panel_assignment["content_box"]["y1"],
            ),
            "assigned_subcaption_name": panel_assignment["assigned_subcaption"],
            "assigned_subcaption_text": split_captions.sub_captions[
                panel_assignment["assigned_subcaption"]
            ].text,
        }

        refined_hierarchy.append(refined_panel_data)

    for panel in current_assignment:
        if panel["panel_id"] not in refined_panel_ids:
            refined_hierarchy.append(panel)

    return {
        "figure_uuid": sample.image_id,
        "refined_hierarchy": refined_hierarchy,
        "assignment_method": "llm_refinement"
        if not result["used_adjudication"]
        else "llm_refinement_with_adjudication",
    }


def automatic_assignment_main(
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
):
    """
    Run automatic panel assembly.

    If full ground truth is available build from GT.

    Args:
        gt_assembly_folder (str, optional): Folder to save the automatic hierarchy map
            output. Defaults to ASSEMBLY_GT_FOLDER.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.

    Raises:
        ValueError: If not all unassigned labels are actually in the original labels.

    """
    output_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_AUTOMATIC_ASSIGNMENT_FILE
    )
    caption_splitting_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_CAPTION_SPLITTING_FILE
    )
    image_splitting_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_IMAGE_PREDICTIONS_FILE
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    local_image_base_path = os.path.join(
        label_studio_root_folder, LABEL_STUDIO_IMAGE_PATH
    )

    ground_truth_files = get_default_gt_files(
        os.path.join(label_studio_root_folder, LABEL_STUDIO_ANNOTATION_PATH)
    )

    # Load ground truth data
    logger.info("Loading ground truth annotations...")
    gt_annotations = load_gt_annotations(
        **ground_truth_files,
        local_image_base_path=local_image_base_path,
    )

    logger.info("Loading assembly data...")
    # Load caption splitting assembly data
    caption_data = {}

    with open(caption_splitting_assembly_file) as f:
        for line in f:
            loaded = json.loads(line)
            caption_data[loaded["sample_id"]] = loaded

    # Load image splitting assembly data
    image_data_df = pd.read_csv(image_splitting_assembly_file)

    automatic_hierarchy_map = {}
    processed_figure_uuids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            automatic_hierarchy_map = json.load(f)

        processed_figure_uuids = set(automatic_hierarchy_map.keys())
    for figure_uuid, figure_annotations in tqdm(gt_annotations.items()):
        if figure_uuid in processed_figure_uuids:
            continue

        panel_sample = figure_annotations.get("panel")
        if panel_sample is None:
            logger.warning(f"No panel annotations found for figure {figure_uuid}")
            continue

        panel_sample_id = panel_sample.id

        if panel_sample_id not in caption_data:
            logger.warning(
                f"No caption splitting data found for panel sample {panel_sample_id}"
            )
            continue

        panel_caption_splitting_data = caption_data[panel_sample_id]

        # Find in image_data_df
        image_predictions = image_data_df[image_data_df["sample_id"] == panel_sample_id]

        if image_predictions.empty:
            logger.warning(
                f"No image splitting data found for panel sample {panel_sample_id}"
            )
            continue

        image_predictions = image_predictions[image_predictions["survives_nms"]]

        # Load caption
        loaded_split_caption = SplitSubCaptions.model_validate_json(
            panel_caption_splitting_data["sub_captions"]
        )

        split_caption_dict = {
            name: sub_caption.text
            for name, sub_caption in loaded_split_caption.sub_captions.items()
        }

        is_gt = panel_caption_splitting_data.get("splitting_response") is None

        if is_gt:
            automatic_hierarchy = automatic_gt_assignment(
                split_caption_dict, image_predictions, figure_annotations
            )

            automatic_hierarchy_map[figure_uuid] = {
                "hierarchy": automatic_hierarchy,
                "is_gt": True,
            }

            with open(output_file, "w") as f:
                json.dump(automatic_hierarchy_map, f, indent=4, ensure_ascii=False)

            continue

        automatic_hierarchy = automatic_assignment(
            panel_sample,
            image_predictions,
            split_caption_dict,
        )

        updated_hierarchy = []

        sub_caption_names = list(split_caption_dict.keys())
        for panel_data in automatic_hierarchy:
            panel_data_copy = deepcopy(panel_data)
            if "labels" in panel_data and len(panel_data["labels"]) > 0:
                detected_labels = {
                    label_id: {
                        "sub_caption_text": sub_caption_names[
                            content["sub_caption_id"]
                        ],
                        "text": content["text"],
                        "confidence": content["confidence"],
                    }
                    for label_id, content in panel_data["labels"].items()
                }

                panel_data_copy["labels"] = detected_labels

            if "assigned_subcaption" in panel_data_copy:
                sub_caption_id = panel_data_copy.pop("assigned_subcaption")
                if sub_caption_id is None:
                    panel_data_copy["assigned_subcaption_name"] = None
                    panel_data_copy["assigned_subcaption_text"] = None
                else:
                    panel_data_copy["assigned_subcaption_name"] = sub_caption_names[
                        sub_caption_id
                    ]
                    panel_data_copy["assigned_subcaption_text"] = split_caption_dict[
                        sub_caption_names[sub_caption_id]
                    ]

            if (
                "unassigned" in panel_data_copy
                and len(panel_data_copy["unassigned"]) > 0
            ):
                unassigned_label_ids = set(panel_data_copy["unassigned"].keys())

                # Make sure all unassigned labels are actually in the original labels
                original_label_ids = image_predictions[
                    image_predictions["predicted_labels"] == "Label"
                ]["id"].tolist()

                if not unassigned_label_ids.issubset(set(original_label_ids)):
                    raise ValueError(
                        f"Panel data for panel {panel_data_copy['panel_id']} has "
                        "unassigned content, expected all content to be assigned to a "
                        f"sub-caption: {panel_data_copy['unassigned']}"
                    )

            updated_hierarchy.append(panel_data_copy)

        automatic_hierarchy_map[figure_uuid] = {
            "hierarchy": updated_hierarchy,
            "is_gt": False,
        }

        with open(output_file, "w") as f:
            json.dump(automatic_hierarchy_map, f, indent=4, ensure_ascii=False)


async def llm_refinement_main(
    server_endpoint: str,
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
    num_concurrent_requests: int = 20,
    api_key: str | None = "test",
    completion_endpoint: str = "/v1",
):
    """
    Run LLM refinement.

    Args:
        server_endpoint (str): Bare base URL of the LLM server for refinement
            API calls.
        gt_assembly_folder (str, optional): Folder containing the assembly data and to
            save the LLM refinement output. Defaults to ASSEMBLY_GT_FOLDER.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.
        num_concurrent_requests (int, optional): Number of concurrent requests.
            Defaults to 20.
        api_key (str | None, optional): API key for LLM endpoint. Defaults to "test".
        completion_endpoint (str, optional): Endpoint suffix appended to
            ``server_endpoint`` for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    output_file = os.path.join(gt_assembly_folder, ASSEMBLY_GT_LLM_REFINEMENT_FILE)
    caption_splitting_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_CAPTION_SPLITTING_FILE
    )
    image_splitting_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_IMAGE_PREDICTIONS_FILE
    )
    automatic_assignment_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_AUTOMATIC_ASSIGNMENT_FILE
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    local_image_base_path = os.path.join(
        label_studio_root_folder, LABEL_STUDIO_IMAGE_PATH
    )

    ground_truth_files = get_default_gt_files(
        os.path.join(label_studio_root_folder, LABEL_STUDIO_ANNOTATION_PATH)
    )

    # Load ground truth data
    logger.info("Loading ground truth annotations...")
    gt_annotations = load_gt_annotations(
        **ground_truth_files,
        local_image_base_path=local_image_base_path,
    )

    logger.info("Loading assembly data...")
    # Load caption splitting assembly data
    caption_data = {}

    with open(caption_splitting_assembly_file) as f:
        for line in f:
            loaded = json.loads(line)
            caption_data[loaded["sample_id"]] = loaded

    # Load image splitting assembly data
    image_data_df = pd.read_csv(image_splitting_assembly_file)

    with open(automatic_assignment_assembly_file) as f:
        automatic_hierarchy_map = json.load(f)

    llm_refinement_map = {}

    processed_figure_uuids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            llm_refinement_map = json.load(f)

        processed_figure_uuids = set(llm_refinement_map.keys())

    task_list = []

    client = openai.AsyncClient(
        base_url=server_endpoint + completion_endpoint,
        api_key=api_key,
        timeout=600,  # Timeout 600 seconds for LLM calls
    )

    semaphore = asyncio.Semaphore(num_concurrent_requests)

    for figure_uuid, automatic_assignment_data in tqdm(automatic_hierarchy_map.items()):
        if figure_uuid in processed_figure_uuids:
            continue

        if automatic_assignment_data["is_gt"]:
            llm_refinement_map[figure_uuid] = {
                "refined_hierarchy": automatic_assignment_data["hierarchy"],
                "is_gt": True,
                "assignment_method": "ground_truth",
            }
            continue

        if not is_partially_assigned(
            {
                panel["panel_id"]: {
                    "relevance": panel.get("relevance", True),
                    "needs_review": panel.get("needs_review", False),
                    "has_shared_labels": panel.get("has_shared_labels", False),
                    "has_estimated_labels": panel.get("has_estimated_labels", False),
                    "assigned_subcaption": panel.get("assigned_subcaption_name", None),
                }
                for panel in automatic_assignment_data["hierarchy"]
            }
        ):
            # Can use automatic assignment
            llm_refinement_map[figure_uuid] = {
                "refined_hierarchy": automatic_assignment_data["hierarchy"],
                "is_gt": False,
                "assignment_method": "automatic",
            }

            continue

        panel_sample_llm = (
            gt_annotations[figure_uuid].get("panel")
            if figure_uuid in gt_annotations
            else None
        )
        if panel_sample_llm is None:
            continue

        loaded_split_caption = SplitSubCaptions.model_validate_json(
            caption_data[panel_sample_llm.id]["sub_captions"]
        )

        detected_label_type = determine_subcaption_type(
            list(loaded_split_caption.sub_captions.keys())
        )

        if detected_label_type["positional"] is not None:
            logger.info(
                f"Sample {panel_sample_llm.id} has positional "
                "sub-captions. Skipping LLM refinement as it is not compatible with "
                "annotated positional captions."
            )

            llm_refinement_map[figure_uuid] = {
                "refined_hierarchy": automatic_assignment_data["hierarchy"],
                "is_gt": False,
                "assignment_method": "manual_due_to_positional_captions",
            }
            continue

        # llm_refinement_map[figure_uuid] = {
        #     "refined_hierarchy": automatic_assignment_data["hierarchy"],
        #     "is_gt": False,
        #     "assignment_method": "manual_due_to_reassignment_needed",
        # }
        # continue

        task_list.append(
            asyncio.create_task(
                llm_refinement(
                    panel_sample_llm,
                    client,
                    semaphore,
                    automatic_assignment_data["hierarchy"],
                    image_data_df[image_data_df["sample_id"] == panel_sample_llm.id],
                    loaded_split_caption,
                )
            )
        )

        # llm_refinement_map[figure_uuid] = {
        #     "refined_hierarchy": updated_assignment,
        #     "is_gt": False,
        #     "assignment_method": updated_method,
        # }

    # save intermediate results for already processed samples
    with open(output_file, "w") as f:
        json.dump(llm_refinement_map, f, indent=4, ensure_ascii=False)

    if len(task_list) == 0:
        logger.info("No samples require LLM refinement. Skipping LLM calls.")
        return

    for task in tqdm(asyncio.as_completed(task_list), total=len(task_list)):
        try:
            result = await task
        except ValueError as e:
            logger.error(f"Error processing task: {e}")
            continue

        if result is None:
            continue

        figure_uuid = result["figure_uuid"]
        llm_refinement_map[figure_uuid] = {
            "refined_hierarchy": result["refined_hierarchy"],
            "is_gt": False,
            "assignment_method": result["assignment_method"],
        }

        with open(output_file, "w") as f:
            json.dump(llm_refinement_map, f, indent=4, ensure_ascii=False)


def llm_refinement_main_sync(
    server_endpoint: str,
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
    num_concurrent_requests: int = 20,
    api_key: str | None = "test",
    completion_endpoint: str = "/v1",
):
    """
    Run LLM refinement.

    Args:
        server_endpoint (str): Bare base URL of the LLM server for refinement
            API calls.
        gt_assembly_folder (str, optional): Folder containing the assembly data and to
            save the LLM refinement output. Defaults to ASSEMBLY_GT_FOLDER.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.
        num_concurrent_requests (int, optional): Number of concurrent requests.
            Defaults to 20.
        api_key (str | None, optional): API key for LLM endpoint. Defaults to "test".
        completion_endpoint (str, optional): Endpoint suffix appended to
            ``server_endpoint`` for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    asyncio.run(
        llm_refinement_main(
            server_endpoint,
            gt_assembly_folder=gt_assembly_folder,
            label_studio_root_folder=label_studio_root_folder,
            num_concurrent_requests=num_concurrent_requests,
            api_key=api_key,
            completion_endpoint=completion_endpoint,
        )
    )


def _parse_imaging_type(raw: str | list) -> str | list:
    """
    Parse predicted_labels from CSV.

    Args:
        raw (str | list): Raw value from CSV, can be a string representation of a list
            or a plain string.

    Returns:
        str | list: Loaded list if input was a string representation of a list,
            otherwise returns the original string.

    """
    if isinstance(raw, str) and raw.startswith("["):
        try:
            return ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            pass
    return raw


def _postprocess_panel_boxes(panel: dict, sample_df: pd.DataFrame) -> dict:
    """
    Post process the panel boxes.

    Recalculates the panel content box based on the predicted boxes of the images and
    labels contained in the panel.
    Constructs label and image entries with box and imaging type information.

    Args:
        panel (dict): Panel dictionary containing image and label IDs, and optionally a
            panel content box.
        sample_df (pd.DataFrame): Image and label predictions for the sample, indexed
            by ID.

    Returns:
        dict: Updated panel dictionary with enriched image and label information, and
            updated panel content box if applicable.

    """
    image_ids = panel.get("images", [])
    label_ids = [int(k) for k in panel.get("labels", {}).keys()]
    all_ids = image_ids + label_ids

    if not all_ids or "panel_content_box" not in panel:
        return panel

    rows = sample_df[sample_df.index.isin(all_ids)]
    if rows.empty:
        return panel

    annotated = rows[rows["prediction_source"] == "annotation"]
    has_predicted = (rows["prediction_source"] == "model").any()

    panel = dict(panel)

    if has_predicted and not annotated.empty:
        pcb = [
            float(annotated["predicted_boxes_x0"].min()),
            float(annotated["predicted_boxes_y0"].min()),
            float(annotated["predicted_boxes_x1"].max()),
            float(annotated["predicted_boxes_y1"].max()),
        ]
        panel["panel_content_box"] = pcb
    else:
        pcb = list(panel["panel_content_box"])

    def get_box(row):
        x0, y0 = float(row["predicted_boxes_x0"]), float(row["predicted_boxes_y0"])
        x1, y1 = float(row["predicted_boxes_x1"]), float(row["predicted_boxes_y1"])
        if row["prediction_source"] == "model":
            x0, x1 = max(x0, pcb[0]), min(x1, pcb[2])
            y0, y1 = max(y0, pcb[1]), min(y1, pcb[3])
        return [x0, y0, x1, y1]

    def nullable(val):
        return None if pd.isna(val) else val

    if image_ids:
        enriched = []
        for img_id in image_ids:
            if img_id not in rows.index:
                enriched.append(img_id)
                continue
            row = rows.loc[img_id]
            enriched.append(
                {
                    "id": img_id,
                    "box": get_box(row),
                    "imaging_type": _parse_imaging_type(row["predicted_labels"]),
                    "imaging_type_score": float(row["predicted_scores"]),
                    "mark_status": nullable(row.get("secondary_predicted_labels")),
                    "mark_status_score": nullable(
                        row.get("secondary_predicted_scores")
                    ),
                }
            )
        panel["images"] = enriched

    if label_ids:
        enriched_labels = {}
        for k, v in panel["labels"].items():
            lid = int(k)
            entry = {
                "sub_caption_text": v.get("sub_caption_text"),
                "text": v.get("text"),
                "ocr_confidence": v.get("confidence"),
            }
            if lid in rows.index:
                row = rows.loc[lid]
                entry["box"] = get_box(row)
                entry["score"] = float(row["predicted_scores"].item())
            enriched_labels[k] = entry
        panel["labels"] = enriched_labels

    return panel


def annotation_assembly_post_processing(
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
):
    """
    Post process LLM refinement results.

    Recalculates the panel content box for all samples.

    Args:
        gt_assembly_folder (str, optional): Folder containing the assembly data and to
            save the final hierarchy output. Defaults to ASSEMBLY_GT_FOLDER.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.

    Warning:
        The output file (``ASSEMBLY_GT_FINAL_HIERARCHY_FILE`` inside
        ``gt_assembly_folder``) is overwritten unconditionally on every call.
        There is no dry-run or existence check; any previously saved results
        will be silently replaced.

    """
    output_file = os.path.join(gt_assembly_folder, ASSEMBLY_GT_FINAL_HIERARCHY_FILE)
    caption_splitting_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_CAPTION_SPLITTING_FILE
    )
    image_splitting_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_IMAGE_PREDICTIONS_FILE
    )
    llm_refinement_assembly_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_LLM_REFINEMENT_FILE
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    local_image_base_path = os.path.join(
        label_studio_root_folder, LABEL_STUDIO_IMAGE_PATH
    )

    ground_truth_files = get_default_gt_files(
        os.path.join(label_studio_root_folder, LABEL_STUDIO_ANNOTATION_PATH)
    )

    logger.info("Loading ground truth annotations...")
    gt_annotations = load_gt_annotations(
        **ground_truth_files,
        local_image_base_path=local_image_base_path,
    )
    figure_to_sample_id_all: dict[str, str | int] = {}
    for fig_uuid, fig_anns in gt_annotations.items():
        _panel = fig_anns.get("panel")
        if _panel is not None:
            figure_to_sample_id_all[fig_uuid] = _panel.id

    logger.info("Loading image predictions CSV...")
    image_data_df = pd.read_csv(image_splitting_assembly_file).set_index("id")

    # Load caption splitting data to resolve subcaption text by name
    caption_data: dict[int, dict] = {}
    with open(caption_splitting_assembly_file) as f:
        for line in f:
            loaded = json.loads(line)
            caption_data[loaded["sample_id"]] = loaded

    with open(llm_refinement_assembly_file) as f:
        llm_refinement_map: dict = json.load(f)

    final_hierarchy_map = {}

    for figure_uuid, refinement_data in tqdm(llm_refinement_map.items()):
        updated_hierarchy = deepcopy(refinement_data["refined_hierarchy"])
        is_gt = refinement_data["is_gt"]
        assignment_method = refinement_data["assignment_method"]

        # Enrich images/labels with box coords and type info
        # recalculate panel_content_box
        sample_id = figure_to_sample_id_all.get(figure_uuid)
        if sample_id is not None:
            sample_df = image_data_df[image_data_df["sample_id"] == sample_id]
            updated_hierarchy = [
                _postprocess_panel_boxes(p, sample_df) for p in updated_hierarchy
            ]

        final_hierarchy_map[figure_uuid] = {
            "refined_hierarchy": updated_hierarchy,
            "is_gt": is_gt,
            "assignment_method": assignment_method,
        }

    with open(output_file, "w") as f:
        json.dump(final_hierarchy_map, f, indent=4, ensure_ascii=False)

    logger.info(
        f"Wrote final hierarchy for {len(final_hierarchy_map)} figures to {output_file}"
    )


def run_gt_assembly(
    server_endpoint: str,
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
    api_key: str | None = "test",
    num_concurrent_requests: int = 20,
    completion_endpoint: str = "/v1",
):
    """
    Run the assembly pipeline for GT annotations.

    Args:
        server_endpoint (str): Bare base URL of the LLM refinement API server.
        gt_assembly_folder (str, optional): Folder to save the assembly output.
            Defaults to ASSEMBLY_GT_FOLDER.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.
        api_key (str | None, optional): API key for LLM endpoint. Defaults to "test".
        num_concurrent_requests (int, optional): Number of concurrent requests for LLM
            calls. Defaults to 20.
        completion_endpoint (str, optional): Endpoint suffix appended to
            ``server_endpoint`` for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    automatic_assignment_main(
        gt_assembly_folder=gt_assembly_folder,
        label_studio_root_folder=label_studio_root_folder,
    )

    llm_refinement_main_sync(
        server_endpoint,
        gt_assembly_folder=gt_assembly_folder,
        label_studio_root_folder=label_studio_root_folder,
        api_key=api_key,
        num_concurrent_requests=num_concurrent_requests,
        completion_endpoint=completion_endpoint,
    )

    annotation_assembly_post_processing(
        gt_assembly_folder=gt_assembly_folder,
        label_studio_root_folder=label_studio_root_folder,
    )
