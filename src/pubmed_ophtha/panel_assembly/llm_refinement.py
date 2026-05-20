"""LLM-based refinement and adjudication of panel assembly."""

import asyncio
import json
import logging
import multiprocessing
import queue
import sqlite3
import warnings
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from typing import Any, TypeGuard

import easyocr
import httpx
import openai
import pandas as pd
import pydantic
import torch
from tqdm.auto import tqdm

from pubmed_ophtha.const.thresholds import DETECTION_BUFFER
from pubmed_ophtha.panel_assembly.automatically_assign_panels import detect_text
from pubmed_ophtha.panel_assembly.db_loading import (
    get_base_article_data,
    get_image,
    perform_split_nms,
)
from pubmed_ophtha.panel_assembly.messages import (
    convert_qwen3vl_bbox_to_original,
    prepare_llm_input,
)
from pubmed_ophtha.panel_assembly.response_models import (
    AlphaNumCaptionAssignments,
    CaptionAssignmentPositional,
    ExistingEvidenceLabel,
    NewEvidenceLabel,
    PositionalCaptionAssignments,
    PositionalEvidenceLabel,
)
from pubmed_ophtha.util.computing import get_cpu_count
from pubmed_ophtha.util.database_interface import (
    database_writer,
    get_database_connection_context,
)

logger = logging.getLogger(__name__)


def calculate_iou(
    boxA: Mapping[str, float | int], boxB: Mapping[str, float | int]
) -> float:
    """
    Calculate the IoU between two boxes.

    Args:
        boxA (Mapping[str, float  |  int]): Bounding box with keys x0, y0, x1, y1.
        boxB (Mapping[str, float  |  int]): Bounding box with keys x0, y0, x1, y1.

    Returns:
        float: IoU value between 0 and 1.

    """
    intersection_x0 = max(boxA["x0"], boxB["x0"])
    intersection_y0 = max(boxA["y0"], boxB["y0"])
    intersection_x1 = min(boxA["x1"], boxB["x1"])
    intersection_y1 = min(boxA["y1"], boxB["y1"])

    intersection_width = max(0, intersection_x1 - intersection_x0)
    intersection_height = max(0, intersection_y1 - intersection_y0)

    if intersection_width == 0 or intersection_height == 0:
        return 0.0

    intersection_area = intersection_width * intersection_height

    boxA_area = (boxA["x1"] - boxA["x0"]) * (boxA["y1"] - boxA["y0"])
    boxB_area = (boxB["x1"] - boxB["x0"]) * (boxB["y1"] - boxB["y0"])

    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    return iou


def calculate_overlap_fraction(
    boxA: Mapping[str, float | int], boxB: Mapping[str, float | int]
) -> float:
    """
    Calculate how much box A overlaps with box B.

    Args:
        boxA (Mapping[str, float  |  int]): Bounding box with keys x0, y0, x1, y1.
        boxB (Mapping[str, float  |  int]): Bounding box with keys x0, y0, x1, y1.

    Returns:
        float: Fraction of boxA's area that overlaps with box B.

    """
    intersection_x0 = max(boxA["x0"], boxB["x0"])
    intersection_y0 = max(boxA["y0"], boxB["y0"])
    intersection_x1 = min(boxA["x1"], boxB["x1"])
    intersection_y1 = min(boxA["y1"], boxB["y1"])

    intersection_width = max(0, intersection_x1 - intersection_x0)
    intersection_height = max(0, intersection_y1 - intersection_y0)

    if intersection_width == 0 or intersection_height == 0:
        return 0.0

    intersection_area = intersection_width * intersection_height

    boxA_area = (boxA["x1"] - boxA["x0"]) * (boxA["y1"] - boxA["y0"])

    return intersection_area / float(boxA_area)


def get_article_images_data(
    cursor: sqlite3.Cursor, article_images_id: int, model_version: str
) -> dict | None:
    """
    Load extended panel data needed for LLM-based caption refinement.

    Retrieves panel assignments, label assignments, image predictions, and
    the article image, then assembles joint label data combining assigned and
    unassigned labels.

    Args:
        cursor (sqlite3.Cursor): Active read-only SQLite cursor.
        article_images_id (int): Primary key of the article_images row.
        model_version (str): Model version string used to filter
            image_predictions.

    Returns:
        dict | None: Dict with the following keys, or None if captions
            cannot be loaded:

            - article_images_id
            - article_id
            - image_cluster_id
            - split_captions
            - label_data
            - full_caption
            - article_image
            - panel_is_final
            - image_data
            - panel_data
            - caption_name_to_id

    """
    base = get_base_article_data(cursor, article_images_id, model_version)
    if base is None:
        return None

    article_id = base["article_id"]
    image_cluster_id = base["image_cluster_id"]
    entries = base["split_captions_entries"]  # (id, name, text, is_full_caption)

    split_captions_dict = {
        name: text for _, name, text, is_full in entries if not is_full
    }
    caption_entry = next(text for _, _, text, is_full in entries if is_full)
    caption_name_to_id = {
        name: c_id for c_id, name, _, is_full in entries if not is_full
    }

    # Retrieve panel data for this article image
    cursor.execute(
        """SELECT id
        FROM image_predictions
        WHERE article_images_id = ?
            AND predicted_labels = 'Panel'
            AND model_version = ?;""",
        (
            article_images_id,
            model_version,
        ),
    )
    panel_ids = [row[0] for row in cursor.fetchall()]

    cursor.execute(
        """SELECT panel_id, label_ids, image_ids, relevance,
            has_estimated_labels,
            needs_review,
            has_shared_labels, assigned_subcaption, unassigned_labels,
            panel_content_box_x0,
            panel_content_box_y0, panel_content_box_x1,
            panel_content_box_y1
        FROM panel_assignments WHERE panel_id IN ({});""".format(
            ",".join("?" for _ in panel_ids) if panel_ids else "-1"
        ),
        panel_ids if panel_ids else [],
    )

    current_panel_data = {
        panel_id: {
            "label_ids": json.loads(label_ids_json),
            "image_ids": json.loads(image_ids_json),
            "relevance": bool(relevance),
            "has_estimated_labels": bool(has_estimated_labels),
            "needs_review": bool(needs_review),
            "has_shared_labels": bool(has_shared_labels),
            "assigned_subcaption": assigned_subcaption,
            "unassigned_labels": json.loads(unassigned_labels_json),
            "panel_content_box": {
                "x0": panel_content_box_x0,
                "y0": panel_content_box_y0,
                "x1": panel_content_box_x1,
                "y1": panel_content_box_y1,
            },
        }
        for (
            panel_id,
            label_ids_json,
            image_ids_json,
            relevance,
            has_estimated_labels,
            needs_review,
            has_shared_labels,
            assigned_subcaption,
            unassigned_labels_json,
            panel_content_box_x0,
            panel_content_box_y0,
            panel_content_box_x1,
            panel_content_box_y1,
        ) in cursor.fetchall()
    }

    unassigned_label_ids = set()

    unassigned_label_data_map = {}

    for panel_id, panel in current_panel_data.items():
        if panel is not None:
            for key, value in panel["unassigned_labels"].items():
                if int(key) >= 0:
                    if int(key) not in unassigned_label_data_map:
                        unassigned_label_data_map[int(key)] = value
                    unassigned_label_data_map[int(key)]["assigned_to"] = (
                        unassigned_label_data_map[int(key)].get("assigned_to", [])
                        + [panel_id]
                    )
                    unassigned_label_ids.add(int(key))

    # panel["label_ids"]
    assigned_label_ids = {
        entry
        for panel in current_panel_data.values()
        if panel is not None
        for entry in panel["label_ids"]
    }

    cursor.execute(
        f"""SELECT id, bbox_x0, bbox_y0, bbox_x1, bbox_y1, text, origin, panel_id
        FROM label_assignments
        WHERE id IN ({",".join("?" for _ in assigned_label_ids)});""",
        list(assigned_label_ids),
    )

    panel_has_final_assignment = {
        panel_id: panel["assigned_subcaption"] is not None
        and not panel["has_shared_labels"]
        and not panel["needs_review"]
        and not panel["has_estimated_labels"]
        for panel_id, panel in current_panel_data.items()
        if panel is not None
    }

    assigned_label_data = [
        {
            "label_id": f"a_{row[0]}",
            "box": {"x0": row[1], "y0": row[2], "x1": row[3], "y1": row[4]},
            "text": row[5],
            "assigned_to": [row[7]],
            "is_final_assignment": panel_has_final_assignment.get(row[7], False),
            "origin": int(row[6]),
        }
        for row in cursor.fetchall()
    ]

    origin_to_label_id = {
        entry["origin"]: int(entry["label_id"].removeprefix("a_"))
        for entry in assigned_label_data
        if entry["origin"] >= 0
    }

    cursor.execute(
        f"""SELECT id,
            predicted_boxes_x0,
            predicted_boxes_y0,
            predicted_boxes_x1,
            predicted_boxes_y1
        FROM image_predictions
        WHERE id IN ({",".join("?" for _ in unassigned_label_ids)});""",
        list(unassigned_label_ids),
    )
    unassigned_label_data = [
        {
            "label_id": f"u_{row[0]}"
            if row[0] not in origin_to_label_id
            else f"a_{origin_to_label_id[row[0]]}",
            "box": {"x0": row[1], "y0": row[2], "x1": row[3], "y1": row[4]},
            "assigned_to": [
                panel_id
                for panel_id, panel_data in current_panel_data.items()
                if panel_data is not None
                and str(row[0]) in panel_data["unassigned_labels"]
                or (
                    row[0] in origin_to_label_id
                    and str(origin_to_label_id[row[0]]) in panel_data["label_ids"]
                )
            ],
            "text": unassigned_label_data_map.get(row[0], {}).get("text", None)
            if row[0] not in origin_to_label_id
            else None,
            "confidence": unassigned_label_data_map.get(row[0], {}).get(
                "confidence", None
            )
            if row[0] not in origin_to_label_id
            else None,
            "is_final_assignment": False,
        }
        for row in cursor.fetchall()
    ]

    cursor.execute(
        "SELECT * FROM image_predictions"
        " WHERE article_images_id = ? AND model_version = ?;",
        (
            article_images_id,
            model_version,
        ),
    )

    predictions_df = pd.DataFrame(
        cursor.fetchall(), columns=[desc[0] for desc in cursor.description]
    )

    predictions_df["survives_nms"] = perform_split_nms(
        predictions_df, iou_threshold=0.3, score_threshold=0.25
    )

    all_labels = {
        row["id"]: {
            "x0": row["predicted_boxes_x0"],
            "y0": row["predicted_boxes_y0"],
            "x1": row["predicted_boxes_x1"],
            "y1": row["predicted_boxes_y1"],
        }
        for _, row in predictions_df.iterrows()
        if row["predicted_labels"] == "Label"
    }

    # Set id column as index
    predictions_df = predictions_df[
        (predictions_df["survives_nms"].astype(bool))
        & (~predictions_df["predicted_labels"].isin(["Panel", "Label"]))
    ]

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
                for panel_id, panel_data in current_panel_data.items()
                if panel_data is not None and row["id"] in panel_data["image_ids"]
            ],
        }
        for _, row in predictions_df.iterrows()
    }
    article_image = get_image(cursor, article_images_id)

    joint_label_data = {
        entry["label_id"]: entry
        for entry in assigned_label_data + unassigned_label_data
    }

    reader = None
    assigned_origins = {int(entry["origin"]) for entry in assigned_label_data}
    for panel_id, panel_data in current_panel_data.items():
        if panel_data is None or panel_has_final_assignment.get(panel_id, False):
            # Skip panels with final assignments, as these cannot be changed
            continue

        for label_id, label_box in all_labels.items():
            if label_id in assigned_origins or f"u_{label_id}" in joint_label_data:
                # Do not add
                continue

            overlap_fraction = calculate_overlap_fraction(
                label_box, panel_data["panel_content_box"]
            )

            if overlap_fraction > 0.05:
                if reader is None:
                    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

                if f"n_{label_id}" not in joint_label_data:
                    buf = DETECTION_BUFFER * max(
                        label_box["x1"] - label_box["x0"],
                        label_box["y1"] - label_box["y0"],
                    )

                    detections = detect_text(
                        reader,
                        article_image.crop(
                            (
                                label_box["x0"] - buf,
                                label_box["y0"] - buf,
                                label_box["x1"] + buf,
                                label_box["y1"] + buf,
                            )
                        ),
                    )

                    detected_text = (
                        detections[0][1]
                        if detections is not None and len(detections) > 0
                        else None
                    )

                    detected_confidence = (
                        detections[0][2]
                        if detections is not None and len(detections) > 0
                        else None
                    )

                    joint_label_data[f"n_{label_id}"] = {
                        "label_id": f"n_{label_id}",
                        "box": label_box,
                        "overlaps_with": {
                            panel_id: overlap_fraction,
                        },
                        "text": detected_text,
                        "confidence": detected_confidence,
                        "is_final_assignment": False,
                    }
                else:
                    joint_label_data[f"n_{label_id}"]["overlaps_with"][panel_id] = (
                        overlap_fraction
                    )

    return {
        "article_images_id": article_images_id,
        "article_id": article_id,
        "image_cluster_id": image_cluster_id,
        "split_captions": split_captions_dict,
        "label_data": joint_label_data,
        "full_caption": caption_entry,
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


def is_partially_assigned(panel_details: dict[int, dict | None]) -> bool:
    """
    Determine whether a figure's panels are only partially assigned.

    A figure is considered partially assigned when at least one panel still
    needs review, has shared labels, has estimated labels, or lacks an assigned
    sub-caption — and not all panels are irrelevant or cleanly done.

    Args:
        panel_details (dict[int, dict | None]): Maps panel ID to a detail dict
            with keys relevance, has_shared_labels, needs_review,
            has_estimated_labels, and assigned_subcaption; or None for missing
            panels.

    Returns:
        bool: True if the figure needs further LLM refinement.

    """
    num_irrelevant = sum(
        1
        for panel in panel_details.values()
        if panel is not None and not panel["relevance"]
    )
    num_need_review = sum(
        1
        for panel in panel_details.values()
        if panel is not None
        and (
            panel["has_shared_labels"]
            or panel["needs_review"]
            or panel["has_estimated_labels"]
            or panel["assigned_subcaption"] is None
        )
    )

    num_done = sum(
        1
        for panel in panel_details.values()
        if panel is not None
        and (
            panel["relevance"]
            and not panel["has_shared_labels"]
            and not panel["needs_review"]
            and not panel["has_estimated_labels"]
            and panel["assigned_subcaption"] is not None
        )
    )

    num_panels = sum(1 for panel in panel_details.values() if panel is not None)

    if num_irrelevant == num_panels:
        return False
    if num_irrelevant > 0 and num_done == 1 and num_irrelevant == num_panels - 1:
        return False

    if num_irrelevant > 0 and (num_done + num_irrelevant) == num_panels:
        return False
    if num_need_review == num_panels:
        return True
    if num_need_review > 0:
        return True

    return False


def get_partially_assigned_article_images(
    cursor: sqlite3.Cursor, model_version: str
) -> list[int]:
    """
    Find article_images IDs whose panel assignments are incomplete or conflicted.

    Args:
        cursor (sqlite3.Cursor): Active read-only SQLite cursor.
        model_version (str): Model version string used to filter Panel
            predictions.

    Returns:
        list[int]: article_images IDs that require LLM refinement.

    """
    cursor.execute(
        """SELECT id, article_images_id
        FROM image_predictions
        WHERE predicted_labels='Panel' AND model_version= ?;""",
        (model_version,),
    )

    article_images_id_to_panel_id = {}

    for panel_id, article_images_id in cursor.fetchall():
        article_images_id_to_panel_id.setdefault(article_images_id, []).append(panel_id)

    cursor.execute("""SELECT panel_id, relevance,
            has_estimated_labels,
            needs_review,
            has_shared_labels, assigned_subcaption
        FROM panel_assignments;""")

    detailed_panel_data = {
        panel_id: {
            "relevance": bool(relevance),
            "has_estimated_labels": bool(has_estimated_labels),
            "needs_review": bool(needs_review),
            "has_shared_labels": bool(has_shared_labels),
            "assigned_subcaption": assigned_subcaption,
        }
        for (
            panel_id,
            relevance,
            has_estimated_labels,
            needs_review,
            has_shared_labels,
            assigned_subcaption,
        ) in cursor.fetchall()
    }

    partially_assigned_article_images = []

    for article_images_id, panel_ids in tqdm(article_images_id_to_panel_id.items()):
        details = {}
        for panel_id in panel_ids:
            if panel_id in detailed_panel_data:
                details[panel_id] = detailed_panel_data[panel_id]
        if is_partially_assigned(details):
            partially_assigned_article_images.append(article_images_id)

    return partially_assigned_article_images


async def get_response(
    client: openai.AsyncOpenAI,
    messages: list[dict],
    sem: asyncio.Semaphore,
) -> tuple[str, str, dict]:
    """
    Send a chat completion request and return parsed output.

    Args:
        client (openai.AsyncOpenAI): Async OpenAI client pointed at the
            inference server.
        messages (list[dict]): Chat message dicts to send.
        sem (asyncio.Semaphore): Limits the number of concurrent in-flight
            requests.

    Returns:
        tuple[str, str, dict]: Tuple of (pydantic_json_str, model_reasoning,
            usage_data). pydantic_json_str is the text after the </think> tag.
            usage_data contains prompt_tokens, completion_tokens, total_tokens,
            and response_time_seconds.

    Raises:
        ValueError: If the model returns an empty or missing response.
        ValueError: If the response text does not contain a ``</think>`` tag.
            This function assumes DeepSeek-style chain-of-thought output where
            the model wraps its reasoning in ``<think>…</think>`` and emits the
            final answer after the closing tag. Any model that does not produce
            this tag will cause an unhandled ``ValueError`` from
            ``str.split("</think>")``.

    """
    async with sem:
        prompt_start_time = datetime.now()
        response = await client.chat.completions.create(
            model="Qwen/Qwen3-VL-30B-A3B-Thinking",
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=0.6,
            top_p=0.95,
        )

    prompt_completion_time = datetime.now() - prompt_start_time

    if (
        response is None
        or response.choices is None
        or len(response.choices) == 0
        or response.choices[0].message.content is None
    ):
        raise ValueError("No response from model")

    response_text = response.choices[0].message.content

    usage_data = {
        "prompt_tokens": response.usage.prompt_tokens
        if response.usage is not None
        else None,
        "completion_tokens": response.usage.completion_tokens
        if response.usage is not None
        else None,
        "total_tokens": response.usage.total_tokens
        if response.usage is not None
        else None,
        "response_time_seconds": prompt_completion_time.total_seconds(),
    }
    model_resoning, pydantic_json_str = response_text.split("</think>")
    return pydantic_json_str.strip(), model_resoning.strip(), usage_data


class AssignmentValidationError(Exception):
    """Raised when an LLM response cannot be parsed or validated after all retries."""


async def get_and_validate_panel_assembly(
    client: openai.AsyncOpenAI,
    messages: list[dict],
    article_image_data: dict,
    sem: asyncio.Semaphore,
    target_model: type[AlphaNumCaptionAssignments] | type[PositionalCaptionAssignments],
    num_validation_follow_ups: int = 2,
    timeout_seconds: int = 10,
) -> AlphaNumCaptionAssignments | PositionalCaptionAssignments:
    """
    Obtain and validate a panel assembly from the LLM with follow-up retries.

    Retries on parse errors, invalid panel/label IDs, and invalid sub-caption
    names, feeding error messages back to the model as follow-up turns.

    Args:
        client (openai.AsyncOpenAI): Async OpenAI client pointed at the
            inference server.
        messages (list[dict]): Initial chat message list.
        article_image_data (dict): Dict with panel_data, label_data, image_data,
            and split_captions used for semantic validation.
        sem (asyncio.Semaphore): Limits concurrent requests.
        target_model (type): Pydantic model class to parse the response into.
        num_validation_follow_ups (int): Maximum correction attempts before
            giving up.
        timeout_seconds (int): Seconds to wait between retry attempts.

    Returns:
        AlphaNumCaptionAssignments | PositionalCaptionAssignments: Validated
            assignment object of the target model type.

    Raises:
        AssignmentValidationError: If a valid assignment cannot be obtained
            within num_validation_follow_ups attempts.

    """
    got_valid_assignment = False
    num_tries = 0

    updated_messages = deepcopy(messages)

    panel_assembly = None
    error_message = None

    while not got_valid_assignment and (
        num_tries < num_validation_follow_ups or num_validation_follow_ups <= 0
    ):
        num_tries += 1
        pydantic_json_str, model_resoning, usage_data = await get_response(
            client, updated_messages, sem
        )
        try:
            panel_assembly = target_model.model_validate_json(pydantic_json_str)
        except pydantic.ValidationError as e:
            error_message = (
                "The previous response did not conform to the required format and "
                "could not be parsed. Please provide a new response that adheres to "
                f"the specified JSON schema. The error was: {str(e)}\n\n"
                f"Model reasoning was: {model_resoning[:500]}..."
            )

            updated_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": pydantic_json_str,
                        }
                    ],
                }
            )
            updated_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": error_message,
                        }
                    ],
                }
            )
            await asyncio.sleep(timeout_seconds)  # Short delay before retrying
            continue
        except httpx.HTTPStatusError as http_err:
            if http_err.response.status_code == 503:
                if http_err.response.json().get("detail", "") == "No healthy workers":
                    # extended wait time
                    await asyncio.sleep(300)
                    continue
                else:
                    await asyncio.sleep(60)
                    continue

            if http_err.response.status_code == 500:
                await asyncio.sleep(120)
                continue
            raise http_err
        except openai.APITimeoutError as _:
            await asyncio.sleep(60)  # Try requeue
            continue
        except openai.InternalServerError as _:
            await asyncio.sleep(120)  # Wait and retry
            continue
        except Exception as e:
            raise AssignmentValidationError(
                "An unexpected error occurred while processing the LLM response."
            ) from e

        # Collect all validation errors grouped by category
        validation_errors = []

        # Handle errors in response
        if isinstance(panel_assembly, AlphaNumCaptionAssignments):
            invalid_panel_ids = set()
            invalid_label_ids = set()
            invalid_sub_captions = set()
            removed_final_panel_ids = set()
            none_sub_captions = set()

            # Ensure that all assigned panel ids are actually in the panel data
            valid_panel_ids = set(article_image_data["panel_data"].keys())
            for assignment in panel_assembly.assignments:
                if assignment.panel_id not in valid_panel_ids:
                    invalid_panel_ids.add(assignment.panel_id)

            # Remove reassignments of final panels, as these arent valid
            filtered_assignments = []
            for assignment in panel_assembly.assignments:
                if (
                    assignment.panel_id in article_image_data["panel_is_final"]
                    and article_image_data["panel_is_final"][assignment.panel_id]
                ):
                    # This panel has a final assignment, skip it
                    removed_final_panel_ids.add(assignment.panel_id)
                    continue
                filtered_assignments.append(assignment)
            panel_assembly = AlphaNumCaptionAssignments(
                assignments=filtered_assignments
            )

            # Ensure that all ExistingEvidence Labels refer to valid ids
            valid_label_ids = set(article_image_data["label_data"].keys())
            for assignment in panel_assembly.assignments:
                if assignment.evidence is not None:
                    for evidence_entry in assignment.evidence:
                        if isinstance(evidence_entry, ExistingEvidenceLabel):
                            if evidence_entry.label_id not in valid_label_ids:
                                invalid_label_ids.add(evidence_entry.label_id)

            # ensure that all new evidence labels intersect with the panel content box
            invalid_new_evidence_labels = set()
            for assignment in panel_assembly.assignments:
                if assignment.evidence is not None:
                    panel_content_box = article_image_data["panel_data"].get(
                        assignment.panel_id, {}
                    )["panel_content_box"]
                    if panel_content_box is not None:
                        for evidence_entry in assignment.evidence:
                            if isinstance(evidence_entry, NewEvidenceLabel):
                                evidence_box = evidence_entry.bbox_2d

                                conv_evidence_box = convert_qwen3vl_bbox_to_original(
                                    evidence_box,
                                    image_height=article_image_data[
                                        "article_image"
                                    ].height,
                                    image_width=article_image_data[
                                        "article_image"
                                    ].width,
                                )
                                # Calculate overlap fraction with panel content box
                                iou = calculate_overlap_fraction(
                                    conv_evidence_box, panel_content_box
                                )

                                if iou < 0.05:
                                    invalid_new_evidence_labels.add(assignment.panel_id)

            # Size of new evidence labels is not larger than fraction of content box
            invalid_new_evidence_size = set()
            for assignment in panel_assembly.assignments:
                if assignment.evidence is not None:
                    panel_content_box = article_image_data["panel_data"].get(
                        assignment.panel_id, {}
                    )["panel_content_box"]
                    if panel_content_box is not None:
                        panel_content_area = (
                            panel_content_box["x1"] - panel_content_box["x0"]
                        ) * (panel_content_box["y1"] - panel_content_box["y0"])
                        for evidence_entry in assignment.evidence:
                            if isinstance(evidence_entry, NewEvidenceLabel):
                                evidence_box = evidence_entry.bbox_2d

                                conv_evidence_box = convert_qwen3vl_bbox_to_original(
                                    evidence_box,
                                    image_height=article_image_data[
                                        "article_image"
                                    ].height,
                                    image_width=article_image_data[
                                        "article_image"
                                    ].width,
                                )
                                evidence_area = (
                                    conv_evidence_box["x1"] - conv_evidence_box["x0"]
                                ) * (conv_evidence_box["y1"] - conv_evidence_box["y0"])

                                if evidence_area > 0.2 * panel_content_area:
                                    invalid_new_evidence_size.add(assignment.panel_id)

            # Ensure assigned sub-captions are actually in the sub-caption options
            valid_sub_caption_names = set(article_image_data["split_captions"].keys())
            for assignment in panel_assembly.assignments:
                if (
                    assignment.assigned_subcaption is not None
                    and assignment.assigned_subcaption not in valid_sub_caption_names
                ):
                    invalid_sub_captions.add(assignment.assigned_subcaption)

            for assignment in panel_assembly.assignments:
                if (
                    assignment.assigned_subcaption is None
                    and not assignment.reject_panel
                ):
                    none_sub_captions.add(assignment.panel_id)

            # Build error messages for AlphaNumCaptionAssignments
            if invalid_panel_ids:
                validation_errors.append(
                    f"Invalid panel ids: {sorted(invalid_panel_ids)}. "
                    f"Valid panel ids are: {sorted(valid_panel_ids)}."
                )

            if invalid_label_ids:
                validation_errors.append(
                    f"Invalid label ids in ExistingEvidenceLabel:"
                    f" {sorted(invalid_label_ids)}. "
                    f"Valid label ids are: {sorted(valid_label_ids)}."
                )

            if len(invalid_new_evidence_labels) > 0:
                validation_errors.append(
                    "NewEvidenceLabels with low overlap (less than 5%) to panel "
                    f"content box for panel ids: {sorted(invalid_new_evidence_labels)}."
                    " Ensure that the bounding boxes for new evidence labels intersect "
                    "sufficiently with the panel content box."
                )

            if len(invalid_new_evidence_size) > 0:
                validation_errors.append(
                    "NewEvidenceLabels with size larger than 20% of panel content box "
                    f"for panel ids: {sorted(invalid_new_evidence_size)}. "
                    "Ensure that the bounding boxes for new evidence labels are not "
                    "excessively large compared to the panel content box."
                )

            if invalid_sub_captions:
                validation_errors.append(
                    f"Invalid sub-caption names: {sorted(invalid_sub_captions)}. "
                    f"Valid sub-caption names are: {sorted(valid_sub_caption_names)}."
                )

            if removed_final_panel_ids and (
                invalid_panel_ids or invalid_label_ids or invalid_sub_captions
            ):
                # Only feedback if there are other errors,
                # otherwise just silently remove these invalid reassignments
                validation_errors.append(
                    f"Removed assignments for final panels:"
                    f" {sorted(removed_final_panel_ids)}. "
                    f"These panels cannot be reassigned."
                )

            if none_sub_captions:
                validation_errors.append(
                    f"Assigned None as sub-caption for panels without rejection:"
                    f" {sorted(none_sub_captions)}. "
                    "If you want to reject these panels, please set"
                    " reject_panel to True for these panels."
                )

        elif isinstance(panel_assembly, PositionalCaptionAssignments):
            invalid_image_ids = set()
            invalid_sub_captions = set()
            none_sub_captions = set()

            # Make sure all images in the assignments are actually in the image data
            valid_image_ids = set(article_image_data["image_data"].keys())
            for assignment in panel_assembly.assignments:
                for img_id in assignment.assigned_images:
                    if img_id not in valid_image_ids:
                        invalid_image_ids.add(img_id)

            # Ensure assigned sub-captions are actually in the sub-caption options
            valid_sub_caption_names = set(article_image_data["split_captions"].keys())
            for assignment in panel_assembly.assignments:
                if (
                    assignment.assigned_subcaption is not None
                    and assignment.assigned_subcaption not in valid_sub_caption_names
                ):
                    invalid_sub_captions.add(assignment.assigned_subcaption)

            for assignment in panel_assembly.assignments:
                if assignment.assigned_subcaption is None:
                    none_sub_captions.add(assignment.assigned_subcaption)

            # Build error messages for PositionalCaptionAssignments
            if invalid_image_ids:
                validation_errors.append(
                    f"Invalid image ids: {sorted(invalid_image_ids)}. "
                    f"Valid image ids are: {sorted(valid_image_ids)}."
                )

            if invalid_sub_captions:
                validation_errors.append(
                    f"Invalid sub-caption names: {sorted(invalid_sub_captions)}. "
                    f"Valid sub-caption names are: {sorted(valid_sub_caption_names)}."
                )

            if none_sub_captions:
                validation_errors.append(
                    "Assigned None as sub-caption for some assignments. "
                    "The assigned_subcaption field cannot be None"
                    " for PositionalCaptionAssignments."
                )

        # If there are any errors, send them all at once
        if validation_errors:
            error_message = (
                "The response contains the following validation errors:\n"
                + "\n".join(f"- {error}" for error in validation_errors)
                + "\n\nPlease provide a corrected response."
            )
            updated_messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": pydantic_json_str,
                        }
                    ],
                }
            )
            updated_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": error_message,
                        }
                    ],
                }
            )
            await asyncio.sleep(timeout_seconds)  # Short delay before retrying
            continue

        got_valid_assignment = True

    if not got_valid_assignment:
        raise AssignmentValidationError(
            f"Failed to get a valid panel assembly after {num_tries} tries. "
            "Last error was: {error_message}".format(
                error_message=error_message
                if error_message is not None
                else "No specific error message"
            )
        )

    assert panel_assembly is not None
    return panel_assembly  # , model_resoning, usage_data


async def get_multi_assignments_llm(
    client: openai.AsyncOpenAI,
    article_image_data: dict[str, dict],
    sem: asyncio.Semaphore,
    min_resolution: int = 100,
    max_resolution: int = 768,
    n_polls: int = 2,
    n_retries: int = 2,
    timeout_seconds: int = 10,
) -> list[AlphaNumCaptionAssignments] | list[PositionalCaptionAssignments]:
    """
    Collect multiple independent LLM panel assemblies for a single figure.

    Runs n_polls concurrent assignment requests and returns all successful
    results. Raises if fewer than n_polls responses succeed.

    Args:
        client (openai.AsyncOpenAI): Async OpenAI client.
        article_image_data (dict[str, dict]): Extended panel data dict.
        sem (asyncio.Semaphore): Limits concurrent in-flight requests.
        min_resolution (int): Minimum image pixel dimension for the prompt.
        max_resolution (int): Maximum image pixel dimension for the prompt.
        n_polls (int): Number of independent assignment attempts to collect.
        n_retries (int): Validation follow-up retries per attempt.
        timeout_seconds (int): Seconds to wait between validation retries.

    Returns:
        list[AlphaNumCaptionAssignments] | list[PositionalCaptionAssignments]:
            Validated assignment objects, one per successful poll.

    Raises:
        RuntimeError: If fewer than n_polls successful responses are obtained.

    """
    messages, target_pydantic_model = prepare_llm_input(
        article_image_data, min_resolution=min_resolution, max_resolution=max_resolution
    )

    assignments = []

    tasks = []
    for _ in range(n_polls):
        tasks.append(
            get_and_validate_panel_assembly(
                client,
                messages,
                article_image_data,
                sem,
                target_pydantic_model,
                num_validation_follow_ups=n_retries,
                timeout_seconds=timeout_seconds,
            )
        )

    for task in asyncio.as_completed(tasks):
        try:
            panel_assembly = await task
            assignments.append(panel_assembly)
        except AssignmentValidationError as ave:
            warnings.warn(f"Failed to get valid assignment after retries: {ave}")
        except Exception as e:
            warnings.warn(f"Failed to get valid assignment ({type(e).__name__}): {e}")

    if len(assignments) < n_polls:
        raise RuntimeError(
            f"{article_image_data['article_images_id']}: Could not get enough "
            "successful LLM responses"
        )

    return assignments


def unify_alphanum_assignments(
    assignments: list[AlphaNumCaptionAssignments],
) -> dict | None:
    """
    Merge multiple alphanumeric assignments into a single unified result.

    Returns None if any panel is assigned conflicting sub-captions or
    conflicting evidence types across annotators.

    Args:
        assignments (list[AlphaNumCaptionAssignments]): Independent assignment
            objects to unify.

    Returns:
        dict | None: Maps panel ID to a unified assignment dict, or None if
            the assignments cannot be reconciled.

    """
    # Check if the same panel ids have been rejected
    rejected_panel_ids_by_assignment = []
    for assignment in assignments:
        rejected_panel_ids = set()
        for panel_assignment in assignment.assignments:
            if panel_assignment.reject_panel:
                rejected_panel_ids.add(panel_assignment.panel_id)
        rejected_panel_ids_by_assignment.append(rejected_panel_ids)

    union_rejected_panel_ids = set.union(*rejected_panel_ids_by_assignment)
    intersection_rejected_panel_ids = set.intersection(
        *rejected_panel_ids_by_assignment
    )

    if len(union_rejected_panel_ids) > 0 and len(intersection_rejected_panel_ids) == 0:
        # All assignments reject at least one different panel, cannot unify
        return None

    # Check if all assignments have the same panel_ids and evidence
    # If no return None

    unified_assignments = {}

    assignments_by_panel_id = {}
    for ann_id, assignment in enumerate(assignments):
        for panel_assignment in assignment.assignments:
            if panel_assignment.panel_id not in assignments_by_panel_id:
                assignments_by_panel_id[panel_assignment.panel_id] = []

            assignments_by_panel_id[panel_assignment.panel_id].append(panel_assignment)

    # Resolve by panel id
    for panel_id, panel_assignments in assignments_by_panel_id.items():
        if len(panel_assignments) == 1:
            unified_assignments[panel_id] = {
                "panel_id": panel_id,
                "assigned_subcaption": panel_assignments[0].assigned_subcaption,
                "evidence": panel_assignments[0].evidence,
                "confidence": [panel_assignments[0].confidence],
                "reject_panel": panel_assignments[0].reject_panel,
            }
        else:
            # Check if all assignments are the same
            first_assignment = panel_assignments[0]
            if first_assignment.assigned_subcaption is None:
                all_same_subcaption = all(
                    a.assigned_subcaption is None for a in panel_assignments
                )
            else:
                all_same_subcaption = all(
                    a.assigned_subcaption == first_assignment.assigned_subcaption
                    for a in panel_assignments
                )

            if not all_same_subcaption:
                # Cannot unify
                return None

            if first_assignment.evidence is None:
                all_same_evidence_type = all(
                    a.evidence is None for a in panel_assignments
                )
            else:
                first_evidence_type = type(first_assignment.evidence[0])
                all_same_evidence_type = all(
                    a.evidence is not None
                    and len(a.evidence) > 0
                    and all(isinstance(e, first_evidence_type) for e in a.evidence)
                    for a in panel_assignments
                )

            if not all_same_evidence_type:
                # Cannot unify
                return None

            aggregated_evidence = None

            if first_assignment.evidence is not None:
                all_existing_evidence = {}
                all_positional_evidence = []
                all_new_evidence = []

                for a in panel_assignments:
                    for evidence in a.evidence:
                        if isinstance(evidence, ExistingEvidenceLabel):
                            if evidence.label_id not in all_existing_evidence:
                                all_existing_evidence[evidence.label_id] = evidence
                        elif isinstance(evidence, NewEvidenceLabel):
                            all_new_evidence.append(evidence)
                        elif isinstance(evidence, PositionalEvidenceLabel):
                            all_positional_evidence.append(evidence)
                        else:
                            raise ValueError("Unknown evidence type")

                aggregated_evidence = (
                    list(all_existing_evidence.values())
                    + all_new_evidence
                    + all_positional_evidence
                )

            unified_assignments[panel_id] = {
                "panel_id": panel_id,
                "assigned_subcaption": first_assignment.assigned_subcaption,
                "evidence": aggregated_evidence,
                "confidence": [a.confidence for a in panel_assignments],
                "reject_panel": first_assignment.reject_panel
                or first_assignment.assigned_subcaption is None,
            }

    return unified_assignments


def unify_positional_assignments(
    assignments: list[PositionalCaptionAssignments],
) -> dict | None:
    """
    Merge multiple positional assignments into a single unified result.

    Returns None if any sub-caption maps to different image sets across
    annotators, or if evidence types are inconsistent.

    Args:
        assignments (list[PositionalCaptionAssignments]): Independent assignment
            objects to unify.

    Returns:
        dict | None: Maps panel ID to a unified assignment dict, or None if
            the assignments cannot be reconciled.

    """
    unified_assignments = {}
    subcaption_to_image_ids: dict[str | None, dict[int, set[int | str]]] = {}
    subcaption_to_panel_assignments: dict[
        str | None, dict[int, CaptionAssignmentPositional]
    ] = {}

    for ann_id, assignment in enumerate(assignments):
        for panel_assignment in assignment.assignments:
            if panel_assignment.assigned_subcaption not in subcaption_to_image_ids:
                subcaption_to_image_ids[panel_assignment.assigned_subcaption] = {}
            subcaption_to_image_ids[panel_assignment.assigned_subcaption][ann_id] = set(
                panel_assignment.assigned_images
            )

            if (
                panel_assignment.assigned_subcaption
                not in subcaption_to_panel_assignments
            ):
                subcaption_to_panel_assignments[
                    panel_assignment.assigned_subcaption
                ] = {}
            subcaption_to_panel_assignments[panel_assignment.assigned_subcaption][
                ann_id
            ] = panel_assignment

    for subcaption, ann_id_to_image_ids in subcaption_to_image_ids.items():
        first_ann_id = next(iter(ann_id_to_image_ids))
        first_image_ids = ann_id_to_image_ids[first_ann_id]

        all_same_image_ids = all(
            image_ids == first_image_ids for image_ids in ann_id_to_image_ids.values()
        )

        if not all_same_image_ids:
            # Cannot unify
            return None

        ann_id_to_panel_assignments = subcaption_to_panel_assignments[subcaption]
        all_same_evidence_type = None
        first_evidence = ann_id_to_panel_assignments[first_ann_id].evidence
        if first_evidence is None:
            all_same_evidence_type = all(
                subcaption_to_panel_assignments[subcaption][ann_id].evidence is None
                for ann_id in ann_id_to_panel_assignments.keys()
            )
        else:
            first_evidence_type = type(first_evidence[0])
            all_same_evidence_type = all(
                ann_data.evidence is not None
                and len(ann_data.evidence) > 0
                and all(isinstance(e, first_evidence_type) for e in ann_data.evidence)
                for ann_data in ann_id_to_panel_assignments.values()
            )

        if not all_same_evidence_type:
            # Cannot unify
            return None

        panel_id = f"panel_{subcaption}"

        aggregated_evidence = None
        if first_evidence is not None:
            all_positional_evidence = []

            for ann_data in ann_id_to_panel_assignments.values():
                assert (
                    ann_data.evidence is not None
                )  # since all_same_evidence_type is True
                for evidence in ann_data.evidence:
                    if isinstance(evidence, PositionalEvidenceLabel):
                        all_positional_evidence.append(evidence)
                    else:
                        raise ValueError("Unknown evidence type")

            aggregated_evidence = all_positional_evidence

        unified_assignments[panel_id] = {
            "panel_id": panel_id,
            "assigned_subcaption": subcaption,
            "assigned_images": list(first_image_ids),
            "evidence": aggregated_evidence,
            "confidence": [
                ann_data.confidence for ann_data in ann_id_to_panel_assignments.values()
            ],
        }

    return unified_assignments


def is_alphanum_assignments(
    assignments: Any,
) -> TypeGuard[list[AlphaNumCaptionAssignments]]:
    """
    Check if object is a list of AlphaNumCaptionAssignments.

    Args:
        assignments (Any): Object to check.

    Returns:
        TypeGuard[list[AlphaNumCaptionAssignments]]: True if object is a list of
            AlphaNumCaptionAssignments, False otherwise.

    """
    if not isinstance(assignments, list):
        return False

    for element in assignments:
        if not isinstance(element, AlphaNumCaptionAssignments):
            return False
    return True


def is_positional_assignments(
    assignments: Any,
) -> TypeGuard[list[PositionalCaptionAssignments]]:
    """
    Check if object is a list of PositionalCaptionAssignments.

    Args:
        assignments (Any): Object to check.

    Returns:
        TypeGuard[list[PositionalCaptionAssignments]]: True if object is a list of
            PositionalCaptionAssignments, False otherwise.

    """
    if not isinstance(assignments, list):
        return False

    for element in assignments:
        if not isinstance(element, PositionalCaptionAssignments):
            return False
    return True


def unify_assignments(
    assignments: list[AlphaNumCaptionAssignments] | list[PositionalCaptionAssignments],
) -> dict | None:
    """
    Dispatch to the appropriate unification function based on assignment type.

    Args:
        assignments (list): AlphaNumCaptionAssignments or
            PositionalCaptionAssignments objects of a uniform type.

    Returns:
        dict | None: Unified assignment dict, or None if unification fails.

    """
    if is_alphanum_assignments(assignments):
        return unify_alphanum_assignments(assignments)
    elif is_positional_assignments(assignments):
        return unify_positional_assignments(assignments)
    else:
        raise ValueError("Unknown assignment type")


async def adjudicate(
    assignments: list[AlphaNumCaptionAssignments] | list[PositionalCaptionAssignments],
    client: openai.AsyncOpenAI,
    article_image_data: dict[int, dict],
    sem: asyncio.Semaphore,
    min_resolution: int = 100,
    max_resolution: int = 768,
    n_retries: int = 2,
    timeout_seconds: int = 10,
    conflict_details: str | None = None,
) -> dict:
    """
    Ask the LLM to resolve conflicts between multiple prior assignment attempts.

    Wraps the original prompt in an adjudication context containing all prior
    assignments and returns a unified assignment dict.

    Args:
        assignments (list): Prior AlphaNumCaptionAssignments or
            PositionalCaptionAssignments objects to adjudicate over.
        client (openai.AsyncOpenAI): Async OpenAI client.
        article_image_data (dict[int, dict]): Extended panel data dict.
        sem (asyncio.Semaphore): Limits concurrent in-flight requests.
        min_resolution (int): Minimum image pixel dimension for the prompt.
        max_resolution (int): Maximum image pixel dimension for the prompt.
        n_retries (int): Validation follow-up retries for the adjudication call.
        timeout_seconds (int): Seconds to wait between validation retries.
        conflict_details (str | None): Optional description of the conflict to
            append to the user instruction.

    Returns:
        dict: Maps panel ID to a unified assignment dict.

    Raises:
        RuntimeError: If a valid adjudicated assignment cannot be obtained.

    """
    messages, target_pydantic_model = prepare_llm_input(
        article_image_data,
        min_resolution=min_resolution,
        max_resolution=max_resolution,
        previous_assignments=assignments,
        conflict_details=conflict_details,
    )

    used_retries = 0

    assignment = None

    try:
        assignment = await get_and_validate_panel_assembly(
            client,
            messages,
            article_image_data,
            sem,
            target_pydantic_model,
            num_validation_follow_ups=n_retries,
            timeout_seconds=timeout_seconds,
        )
    except AssignmentValidationError as ave:
        logger.error(
            f"Failed to get valid assignment during adjudication after {used_retries} "
            f"retries. Error: {ave}"
        )

    if assignment is None:
        raise RuntimeError("Could not get successful LLM responses")

    converted_assignment = {}

    if isinstance(assignment, AlphaNumCaptionAssignments):
        for panel_assignment in assignment.assignments:
            converted_assignment[panel_assignment.panel_id] = {
                "panel_id": panel_assignment.panel_id,
                "assigned_subcaption": panel_assignment.assigned_subcaption,
                "evidence": panel_assignment.evidence,
                "confidence": [panel_assignment.confidence],
                "reject_panel": panel_assignment.reject_panel,
            }
    elif isinstance(assignment, PositionalCaptionAssignments):
        for panel_assignment in assignment.assignments:
            panel_id = f"panel_{panel_assignment.assigned_subcaption}"
            converted_assignment[panel_id] = {
                "panel_id": panel_id,
                "assigned_subcaption": panel_assignment.assigned_subcaption,
                "assigned_images": panel_assignment.assigned_images,
                "evidence": panel_assignment.evidence,
                "confidence": [panel_assignment.confidence],
            }
    else:
        raise ValueError("Unknown assignment type")

    return converted_assignment


async def run_first_stage_and_unification(
    extended_panel_data: dict,
    article_images_id: str | int,
    openai_client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    num_retries: int = 2,
) -> (
    tuple[
        list[AlphaNumCaptionAssignments] | list[PositionalCaptionAssignments] | None,
        dict | None,
        bool,
    ]
    | None
):
    """
    Run multiple assignment requests and unify.

    Args:
        extended_panel_data (dict): Panel data dict.
        article_images_id (int): ID of the article image being processed.
        openai_client (openai.AsyncOpenAI): Client for making OpenAI API calls.
        sem (asyncio.Semaphore): Semaphore for limiting concurrent API calls.
        num_retries (int, optional): Number of retries in case of validation errors.
            Defaults to 2.

    Returns:
        tuple[
            list[AlphaNumCaptionAssignments]
            | list[PositionalCaptionAssignments]
            | None,
            dict | None,
            bool,
        ]: Tuple containing:
            - List of independent assignment objects, or None if the LLM calls failed.
            - Unified assignment dict, or None if unification failed.
            - Boolean indicating whether a label conflict was found in the unified
                assignments.

    """
    try:
        assignments = await get_multi_assignments_llm(
            openai_client,
            extended_panel_data,
            sem,
            n_retries=num_retries,
        )
    except Exception as e:
        logger.error(
            f"Error getting assignments for article_images_id "
            f"{article_images_id} ({type(e).__name__}): {e}"
        )
        return None

    unified_assignments = unify_assignments(assignments)

    # Make sure each label is only assigned to one panel, if not use adjudication
    found_label_conflict = False
    if unified_assignments is not None:
        label_to_panel = {}
        for panel_id, panel_assignment in unified_assignments.items():
            if "assigned_images" in panel_assignment:
                continue
            if panel_assignment["evidence"] is not None:
                for evidence in panel_assignment["evidence"]:
                    if isinstance(evidence, ExistingEvidenceLabel):
                        if evidence.label_id in label_to_panel:
                            # Conflict, use adjudication
                            unified_assignments = None
                            found_label_conflict = True
                            break
                        else:
                            label_to_panel[evidence.label_id] = panel_id

    return assignments, unified_assignments, found_label_conflict


async def adjudicate_and_finalize(
    extended_panel_data: dict,
    assignments: list[AlphaNumCaptionAssignments] | list[PositionalCaptionAssignments],
    unified_assignments: dict | None,
    found_label_conflict: bool,
    article_images_id: str | int,
    openai_client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    num_retries: int = 2,
) -> dict | None:
    """
    Adjudicate and finalize the unified assignments.

    Args:
        extended_panel_data (dict): Panel data dict.
        assignments (list): Independent assignment objects to unify.
        unified_assignments (dict | None): Unified assignment dict, or None if
            unification failed.
        found_label_conflict (bool): True if a label conflict was found in the unified
            assignments, False otherwise.
        article_images_id (int): ID of the article image being processed.
        openai_client (openai.AsyncOpenAI): Client for making OpenAI API calls.
        sem (asyncio.Semaphore): Semaphore for limiting concurrent API calls.
        num_retries (int, optional): Number of retried in case of validation errors.
            Defaults to 2.

    Raises:
        ValueError: If a GT label is assigned to multiple panels in the unified
            assignments.

    Returns:
        dict | None: Finalized assignment dict with content boxes, or None if
            adjudication failed.
            Has the following keys:
                - unified_assignment
                - used_adjudication
                - label_box_map
                - positional_assignment
                - panel_ids_to_delete

    """
    used_adjudication = False
    if unified_assignments is None:
        try:
            adjudicated_assignment = await adjudicate(
                assignments,
                openai_client,
                extended_panel_data,
                sem,
                conflict_details=None
                if not found_label_conflict
                else "Multiple panels were assigned the same label, which is not "
                "possible. Please use the evidence provided to determine the correct "
                "assignment for each panel, ensuring that each label is only assigned "
                "to one panel.",
                n_retries=num_retries,
            )
        except Exception as e:
            logger.error(
                f"Error getting adjudicated assignments for article_images_id "
                f"{article_images_id} ({type(e).__name__}): {e}"
            )
            return None

        used_adjudication = True
        unified_assignments = adjudicated_assignment

    is_positional = any(
        "assigned_images" in panel_assignment
        for panel_assignment in unified_assignments.values()
    )

    label_box_map = {}
    for panel_id, panel_assignment in unified_assignments.items():
        # Convert confidence list to average
        panel_assignment["caption_id"] = (
            extended_panel_data["caption_name_to_id"][
                panel_assignment["assigned_subcaption"]
            ]
            if panel_assignment["assigned_subcaption"] is not None
            else None
        )

        if is_positional:
            panel_assignment["image_ids"] = panel_assignment["assigned_images"]

            # Create new content box
            panel_content_box = {
                "x0": 0,
                "y0": 0,
                "x1": 0,
                "y1": 0,
            }

            for image_id in panel_assignment["assigned_images"]:
                image_box = extended_panel_data["image_data"][image_id]["box"]
                panel_content_box["x0"] = min(panel_content_box["x0"], image_box["x0"])
                panel_content_box["y0"] = min(panel_content_box["y0"], image_box["y0"])
                panel_content_box["x1"] = max(panel_content_box["x1"], image_box["x1"])
                panel_content_box["y1"] = max(panel_content_box["y1"], image_box["y1"])

            panel_assignment["content_box"] = panel_content_box
        else:
            panel_assignment["image_ids"] = [
                panel["image_ids"]
                for current_panel_id, panel in extended_panel_data["panel_data"].items()
                if current_panel_id == panel_id
            ]

            # Use label data to add boxes to all labels
            panel_assignment["content_box"] = extended_panel_data["panel_data"][
                panel_id
            ]["panel_content_box"]
            if panel_assignment["evidence"] is not None:
                for evidence in panel_assignment["evidence"]:
                    if isinstance(evidence, ExistingEvidenceLabel):
                        current_label_data = extended_panel_data["label_data"][
                            evidence.label_id
                        ]

                        label_box_map[evidence.label_id] = {
                            "box": current_label_data["box"],
                            "text": current_label_data.get("text", None),
                        }

                        panel_assignment["content_box"]["x0"] = min(
                            panel_assignment["content_box"]["x0"],
                            current_label_data["box"]["x0"],
                        )
                        panel_assignment["content_box"]["y0"] = min(
                            panel_assignment["content_box"]["y0"],
                            current_label_data["box"]["y0"],
                        )
                        panel_assignment["content_box"]["x1"] = max(
                            panel_assignment["content_box"]["x1"],
                            current_label_data["box"]["x1"],
                        )
                        panel_assignment["content_box"]["y1"] = max(
                            panel_assignment["content_box"]["y1"],
                            current_label_data["box"]["y1"],
                        )
                    elif isinstance(evidence, NewEvidenceLabel):
                        absolute_box = convert_qwen3vl_bbox_to_original(
                            evidence.bbox_2d,
                            extended_panel_data["article_image"].width,
                            extended_panel_data["article_image"].height,
                        )
                        panel_assignment["content_box"]["x0"] = min(
                            panel_assignment["content_box"]["x0"], absolute_box["x0"]
                        )
                        panel_assignment["content_box"]["y0"] = min(
                            panel_assignment["content_box"]["y0"], absolute_box["y0"]
                        )
                        panel_assignment["content_box"]["x1"] = max(
                            panel_assignment["content_box"]["x1"], absolute_box["x1"]
                        )
                        panel_assignment["content_box"]["y1"] = max(
                            panel_assignment["content_box"]["y1"], absolute_box["y1"]
                        )

    # Ensure that all labels have exactly one panel

    assigned_labels = set()

    for panel_id, panel_data in extended_panel_data["panel_data"].items():
        if not extended_panel_data["panel_is_final"].get(panel_id, False):
            continue

        for label_id in panel_data["label_ids"]:
            new_label_id = f"a_{label_id}"
            if new_label_id in assigned_labels:
                # Conflict, use adjudication
                raise ValueError(
                    f"GT Label {label_id} is assigned to multiple panels, which is not "
                    "possible. Please check the assignments and evidence to ensure "
                    "that each label is only assigned to one panel."
                )
            assigned_labels.add(new_label_id)

    # Check assigned labels in evidence
    for panel_id, panel_assignment in unified_assignments.items():
        if panel_assignment["evidence"] is not None:
            for evidence in panel_assignment["evidence"]:
                if isinstance(evidence, ExistingEvidenceLabel):
                    if evidence.label_id in assigned_labels:
                        # Conflict, use adjudication
                        raise ValueError(
                            f"Evidence Label {evidence.label_id} is assigned to "
                            "multiple panels,  which is not possible. Please check the "
                            "assignments and evidence to ensure that each label is "
                            "only assigned to one panel."
                        )
                    assigned_labels.add(evidence.label_id)

    panel_ids_to_delete = []

    if is_positional:
        panel_ids_to_delete = list(extended_panel_data["panel_data"].keys())

    if len(unified_assignments) == 0:
        # Reject all non-final panels:
        for panel_id, panel_data in extended_panel_data["panel_data"].items():
            if not extended_panel_data["panel_is_final"].get(panel_id, False):
                panel_ids_to_delete.append(panel_id)
    else:
        # See if panel was rejected
        for panel_id, panel_data in unified_assignments.items():
            if panel_data.get("reject_panel", False):
                panel_ids_to_delete.append(panel_id)

    return {
        "unified_assignment": unified_assignments,
        "used_adjudication": used_adjudication,
        "label_box_map": label_box_map,
        "positional_assignment": is_positional,
        "panel_ids_to_delete": panel_ids_to_delete,
    }


async def refine_panel_assembly(
    extended_panel_data: dict,
    article_images_id: str | int,
    openai_client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    num_retries: int = 2,
) -> dict | None:
    """
    Run the LLM refinement and adjudication pipeline for a single article image.

    Args:
        extended_panel_data (dict): Panel data.
        article_images_id (int): ID of the article_images row being processed.
        openai_client (openai.AsyncOpenAI): Concurrent OpenAI client for making
            requests.
        sem (asyncio.Semaphore): Semaphore to limit concurrent in-flight requests.
        num_retries (int, optional): Number of retries in case of validation errors.
            Defaults to 2.

    Returns:
        dict | None: Dict containing unified assignment and metadata, or None if an
            error occurs.
            Has the following keys:
            - article_images_id (int): ID of the article_images row.
            - article_id (int): ID of the article.
            - image_cluster_id (int): ID of the image cluster.
            - unified_assignment (dict): Unified assignment mapping panel IDs to
                assigned sub-captions, evidence, confidence, and reject_panel flag.
            - used_adjudication (bool): Whether adjudication was used to resolve
                conflicts.
            - image_width (int): Width of the article image.
            - image_height (int): Height of the article image.
            - label_box_map (dict): Maps assigned label IDs to their box coordinates
                and text.
            - positional_assignment (bool): Whether the assignment is positional
                (sub-captions map to sets of images) or alphanumeric (sub-captions are
                independent of image IDs).
            - panel_ids_to_delete (list): List of panel IDs that should be deleted
                based on the assignment.

    """
    res = await run_first_stage_and_unification(
        extended_panel_data,
        article_images_id,
        openai_client,
        sem,
        num_retries=num_retries,
    )

    if res is None:
        return None

    assignments, unified_assignments, found_label_conflict = res

    if assignments is None:
        return None

    res = await adjudicate_and_finalize(
        extended_panel_data,
        assignments,
        unified_assignments,
        found_label_conflict,
        article_images_id,
        openai_client,
        sem,
        num_retries=num_retries,
    )

    if res is None:
        return None

    unified_assignments = res["unified_assignment"]
    used_adjudication = res["used_adjudication"]
    label_box_map = res["label_box_map"]
    is_positional = res["positional_assignment"]
    panel_ids_to_delete = res["panel_ids_to_delete"]

    return {
        "article_images_id": article_images_id,
        "article_id": extended_panel_data["article_id"],
        "image_cluster_id": extended_panel_data["image_cluster_id"],
        "unified_assignment": unified_assignments,
        "used_adjudication": used_adjudication,
        "image_width": extended_panel_data["article_image"].width,
        "image_height": extended_panel_data["article_image"].height,
        "label_box_map": label_box_map,
        "positional_assignment": is_positional,
        "panel_ids_to_delete": panel_ids_to_delete,
    }


async def process_article_image(
    cursor: sqlite3.Cursor,
    article_images_id: int,
    model_version: str,
    openai_client: openai.AsyncOpenAI,
    sem: asyncio.Semaphore,
    num_retries: int = 2,
) -> dict | None:
    """
    Run the full LLM assignment pipeline for a single article image.

    Loads panel data, collects multiple LLM assignments, unifies or adjudicates
    them, resolves label conflicts, and returns a result dict ready for the
    database writer.

    Args:
        cursor (sqlite3.Cursor): Active read-only SQLite cursor.
        article_images_id (int): Primary key of the article_images row.
        model_version (str): Model version string for filtering predictions.
        openai_client (openai.AsyncOpenAI): Async OpenAI client.
        sem (asyncio.Semaphore): Limits concurrent in-flight requests.
        num_retries (int): Validation follow-up retries per LLM call.

    Returns:
        dict | None: Result dict with keys article_images_id, article_id,
            image_cluster_id, unified_assignment, used_adjudication,
            image_width, image_height, label_box_map, positional_assignment,
            and panel_ids_to_delete; or None on failure.

    Raises:
        ValueError: If a GT label is assigned to multiple panels simultaneously.

    """
    extended_panel_data = get_article_images_data(
        cursor,
        article_images_id,
        model_version,
    )

    if extended_panel_data is None:
        logger.warning(
            f"Could not get panel data for article_images_id {article_images_id}"
        )
        return None

    refinement_result = await refine_panel_assembly(
        extended_panel_data,
        article_images_id,
        openai_client,
        sem,
        num_retries=num_retries,
    )

    return refinement_result


def prepare_tables(write_cur: sqlite3.Cursor) -> None:
    """
    Add LLM refinement columns to panel_assignments and label_assignments tables.

    Adds assignment_comment and used_adjudication to panel_assignments, and
    relevance to label_assignments, if those columns are not already present.

    Args:
        write_cur (sqlite3.Cursor): Writable SQLite cursor.

    """
    write_cur.execute("""
    PRAGMA table_info(panel_assignments);
    """)
    columns = [column[1] for column in write_cur.fetchall()]
    if "assignment_comment" not in columns:
        write_cur.execute("""
        ALTER TABLE panel_assignments
        ADD COLUMN assignment_comment TEXT DEFAULT '';
        """)
    if "used_adjudication" not in columns:
        write_cur.execute("""
        ALTER TABLE panel_assignments
        ADD COLUMN used_adjudication BOOLEAN DEFAULT 0;
        """)

    write_cur.execute("""
    PRAGMA table_info(label_assignments);
    """)
    columns = [column[1] for column in write_cur.fetchall()]
    if "relevance" not in columns:
        write_cur.execute("""
        ALTER TABLE label_assignments
        ADD COLUMN relevance BOOLEAN DEFAULT 1;
        """)


def write_into_database(
    cursor: sqlite3.Cursor,
    assignment_result: dict,
) -> None:
    """
    Persist a unified LLM assignment result to the database.

    Inserts new panel rows for positional assignments, inserts or updates
    label_assignments rows for each evidence entry, and updates the
    panel_assignments rows with the final sub-caption, label IDs, and metadata.

    Args:
        cursor (sqlite3.Cursor): Writable SQLite cursor.
        assignment_result (dict): Result dict from ``process_article_image``.
            Must contain the following keys — a missing key raises ``KeyError``
            at runtime with no descriptive error message:
            ``unified_assignment``, ``used_adjudication``, ``article_id``,
            ``article_images_id``, ``image_width``, ``image_height``,
            ``label_box_map``, ``positional_assignment``,
            ``panel_ids_to_delete``, and ``model_version``.

    Raises:
        ValueError: If an ExistingEvidenceLabel has a None label_id or if a
            required label box cannot be found.

    """
    is_adjudicated = assignment_result["used_adjudication"]
    unified_assignments = assignment_result["unified_assignment"]
    origin = "llm_adjudication" if is_adjudicated else "llm_assignment"
    detection_model_version = assignment_result["model_version"]

    # First handle rejected panels
    if len(unified_assignments) == 0:
        # Reject all non-final panels:
        pass

    # First collect the panels that have to be added
    old_to_new_panel_id = {}

    for panel_id, panel_assignment in unified_assignments.items():
        if panel_assignment["assigned_subcaption"] is None:
            logger.warning(
                f"Panel {panel_id} was rejected by the LLM, skipping panel and "
                "evidence assignment for this panel if it exists."
            )
            continue
        if (
            "assigned_images" in panel_assignment
            and panel_assignment["assigned_images"] is not None
            and len(panel_assignment["assigned_images"]) > 0
        ):
            cursor.execute(
                """
                INSERT INTO image_predictions (
                    article_id,
                    article_images_id,
                    predicted_boxes_x0,
                    predicted_boxes_y0,
                    predicted_boxes_x1,
                    predicted_boxes_y1,
                    predicted_labels,
                    predicted_scores,
                    secondary_predicted_labels,
                    secondary_predicted_scores,
                    model_version,
                    survives_nms,
                    prediction_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id;
            """,
                (
                    assignment_result["article_id"],
                    assignment_result["article_images_id"],
                    panel_assignment["content_box"]["x0"],
                    panel_assignment["content_box"]["y0"],
                    panel_assignment["content_box"]["x1"],
                    panel_assignment["content_box"]["y1"],
                    "Panel",
                    1.0,
                    "None",
                    -1,
                    detection_model_version,
                    True,
                    origin,
                ),
            )

            new_panel_id = cursor.fetchone()[0]
            old_to_new_panel_id[panel_id] = new_panel_id

    # Gather the labels that need to be added/updated
    label_updates = []

    inverted_added_label_to_panel = {}

    for panel_id, panel_assignment in unified_assignments.items():
        if panel_assignment["assigned_subcaption"] is None:
            continue

        assigned_subcaption_id = panel_assignment["caption_id"]

        if (
            "evidence" in panel_assignment
            and panel_assignment["evidence"] is not None
            and len(panel_assignment["evidence"]) > 0
        ):
            for ev_id, evidence in enumerate(panel_assignment["evidence"]):
                if isinstance(evidence, ExistingEvidenceLabel):
                    if evidence.label_id is None:
                        raise ValueError("ExistingEvidenceLabel must have label_id")

                    if str(evidence.label_id).startswith("u_") or str(
                        evidence.label_id
                    ).startswith("n_"):
                        # Needs to be added to the db, we can treat it as new evidence
                        absolute_box = assignment_result["label_box_map"].get(
                            evidence.label_id
                        )
                        if absolute_box is None:
                            raise ValueError(
                                f"Could not find box for label_id {evidence.label_id}"
                            )

                        origin_label_id = int(
                            str(evidence.label_id).removeprefix("u_").removeprefix("n_")
                        )
                        cursor.execute(
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
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id;
                            """,
                            (
                                assigned_subcaption_id,
                                absolute_box.get("text", ""),  # evidence.text,
                                sum(panel_assignment["confidence"])
                                / len(panel_assignment["confidence"]),
                                absolute_box["box"]["x0"],
                                absolute_box["box"]["y0"],
                                absolute_box["box"]["x1"],
                                absolute_box["box"]["y1"],
                                old_to_new_panel_id[panel_id]
                                if panel_id in old_to_new_panel_id
                                else panel_id,
                                str(origin_label_id),  # No origin
                            ),
                        )

                        new_label_id = cursor.fetchone()[0]

                        if panel_id not in inverted_added_label_to_panel:
                            inverted_added_label_to_panel[panel_id] = {}
                        inverted_added_label_to_panel[panel_id][ev_id] = new_label_id

                    else:
                        # This label already exists in the database, we just need to
                        # update its assignment
                        label_updates.append(
                            (
                                int(str(evidence.label_id).removeprefix("a_")),
                                assigned_subcaption_id,
                                # evidence.text if evidence.text is not None else "",
                                old_to_new_panel_id[panel_id]
                                if panel_id in old_to_new_panel_id
                                else panel_id,
                            )
                        )
                elif isinstance(evidence, NewEvidenceLabel):
                    # This is a new label that needs to be added and associated with
                    # the new panel assignment
                    absolute_box = convert_qwen3vl_bbox_to_original(
                        evidence.bbox_2d,
                        assignment_result["image_width"],
                        assignment_result["image_height"],
                    )

                    cursor.execute(
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
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING id;
                        """,
                        (
                            assigned_subcaption_id,
                            evidence.text,
                            sum(panel_assignment["confidence"])
                            / len(panel_assignment["confidence"]),
                            absolute_box["x0"],
                            absolute_box["y0"],
                            absolute_box["x1"],
                            absolute_box["y1"],
                            old_to_new_panel_id[panel_id]
                            if panel_id in old_to_new_panel_id
                            else panel_id,
                            "-1",  # No origin
                        ),
                    )

                    new_label_id = cursor.fetchone()[0]

                    if panel_id not in inverted_added_label_to_panel:
                        inverted_added_label_to_panel[panel_id] = {}
                    inverted_added_label_to_panel[panel_id][ev_id] = new_label_id
                elif isinstance(evidence, PositionalEvidenceLabel):
                    # Positional evidence does not correspond to a label, so we can
                    # ignore it for the purpose of database insertion
                    pass
                else:
                    raise ValueError("Unknown evidence type")

    if len(label_updates) > 0:
        # DO not update text
        cursor.executemany(
            """
            UPDATE label_assignments
            SET sub_caption_id = ?, panel_id = ?
            WHERE id = ?;
        """,
            [(update[1], update[2], update[0]) for update in label_updates],
        )

    # Lastly update the panel assignments with the new label ids
    panels_to_update = []
    for panel_id, panel_assignment in unified_assignments.items():
        if panel_assignment["assigned_subcaption"] is None:
            continue

        label_ids = []
        if "evidence" in panel_assignment and panel_assignment["evidence"] is not None:
            for ev_id, evidence in enumerate(panel_assignment["evidence"]):
                if (
                    inverted_added_label_to_panel.get(panel_id, {}).get(ev_id)
                    is not None
                ):
                    # This label was added, we should use the new label id
                    label_ids.append(inverted_added_label_to_panel[panel_id][ev_id])
                elif isinstance(evidence, ExistingEvidenceLabel):
                    if str(evidence.label_id).startswith("u_") or str(
                        evidence.label_id
                    ).startswith("n_"):
                        raise ValueError(
                            f"Could not find new label id for panel"
                            f" {panel_id} and evidence {ev_id}"
                        )
                    else:
                        label_ids.append(int(str(evidence.label_id).removeprefix("a_")))
                elif isinstance(evidence, NewEvidenceLabel):
                    new_label_id = inverted_added_label_to_panel.get(panel_id, {}).get(
                        ev_id
                    )
                    if new_label_id is None:
                        raise ValueError(
                            f"Could not find new label id for panel"
                            f" {panel_id} and evidence {ev_id}"
                        )
                    label_ids.append(new_label_id)

        image_ids = panel_assignment.get("image_ids", [])
        panel_content_box = panel_assignment["content_box"]
        assigned_subcaption = panel_assignment["caption_id"]

        update_panel_id = (
            old_to_new_panel_id[panel_id]
            if panel_id in old_to_new_panel_id
            else panel_id
        )

        assignment_comment = (
            ""
            if panel_assignment["evidence"] is None
            or len(panel_assignment["evidence"]) == 0
            else json.dumps(
                [
                    {"type": type(ev).__name__, "data": ev.model_dump()}
                    for ev in panel_assignment["evidence"]
                ]
            )
        )

        panels_to_update.append(
            (
                json.dumps(label_ids),
                json.dumps(image_ids),
                True,  # Relevance
                False,  # has_estimated_labels
                False,  # needs_review
                False,  # has_shared_labels
                panel_content_box["x0"],
                panel_content_box["y0"],
                panel_content_box["x1"],
                panel_content_box["y1"],
                assigned_subcaption,
                origin,
                assignment_comment,
                update_panel_id,
            )
        )

    cursor.executemany(
        """
        UPDATE panel_assignments
        SET label_ids = ?, image_ids = ?, relevance = ?, has_estimated_labels = ?,
        needs_review = ?, has_shared_labels = ?, panel_content_box_x0 = ?,
        panel_content_box_y0 = ?, panel_content_box_x1 = ?, panel_content_box_y1 = ?,
        assigned_subcaption = ?, origin = ?, assignment_comment = ?
        WHERE panel_id = ?;
    """,
        panels_to_update,
    )

    if (
        assignment_result["positional_assignment"]
        and len(assignment_result["panel_ids_to_delete"]) > 0
    ):
        panel_ids_to_delete_str = ",".join(
            str(pid) for pid in assignment_result["panel_ids_to_delete"]
        )

        origin = (
            "positional_assignment_rejection"
            if assignment_result["positional_assignment"]
            else "llm_assignment_rejection"
        )

        # Set panel_assignments to irrelevant instead of deleting to preserve history
        cursor.execute(
            f"""
            UPDATE panel_assignments
            SET relevance = 0, origin = '{origin}'
            WHERE panel_id IN ({panel_ids_to_delete_str});
        """,
        )

        cursor.execute(
            f"""
            UPDATE label_assignments
            SET relevance = 0
            WHERE panel_id IN ({panel_ids_to_delete_str});
        """,
        )


def batched_db_write(connection: sqlite3.Connection, batch: list[dict]) -> None:
    """
    Write a batch of LLM assignment results to the database and commit.

    Args:
        connection (sqlite3.Connection): Open SQLite connection used to obtain
            a write cursor.
        batch (list[dict]): Result dicts from ``process_article_image``.

    """
    cur = connection.cursor()
    for entry in batch:
        try:
            logger.debug(
                f"Writing article_images_id {entry['article_images_id']} into database"
            )
            write_into_database(
                cur,
                entry,
            )
        except sqlite3.InterfaceError as e:
            logger.error(
                f"Error writing into database for article_images_id "
                f"{entry['article_images_id']} ({type(e).__name__}): {e}"
            )
            continue
    connection.commit()


async def refinement_worker(
    database_path: str,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    progress_queue: multiprocessing.Queue,
    model_version: str,
    client_urls: list[str],
    num_concurrent_requests: int = 32,
    client_api_key: str | None = None,
    num_retries: int = 2,
    completion_endpoint: str = "/v1",
) -> None:
    """
    Async worker that processes article images from a queue via LLM assignment.

    Maintains a pool of concurrent async tasks, rotating across multiple
    inference server URLs in round-robin fashion. Signals progress and
    completion via progress_queue.

    Args:
        database_path (str): Path to the SQLite database file (read-only).
        task_queue (multiprocessing.Queue): Supplies article_images_id integers;
            None terminates the worker.
        result_queue (multiprocessing.Queue): Receives assignment result dicts.
        progress_queue (multiprocessing.Queue): Receives integer increments
            (1 per completed task) and a final None when the worker exits.
        model_version (str): Forwarded to ``process_article_image``.
        client_urls (list[str]): Inference server base URLs to rotate across.
        num_concurrent_requests (int): Maximum concurrent in-flight LLM requests.
        client_api_key (str | None): API key for the inference server; defaults
            to "test" if None.
        num_retries (int): Validation follow-up retries per LLM call.
        completion_endpoint (str): Endpoint suffix appended to each
            ``client_urls`` entry for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    if client_api_key is None:
        # Not required by server but by openai package
        client_api_key = "test"  # pragma: allowlist secret

    client_list = [
        openai.AsyncOpenAI(
            base_url=url + completion_endpoint,
            api_key=client_api_key,
            timeout=600,  # 600s; adjust based on expected response times
        )
        for url in client_urls
    ]
    current_client_index = 0  # Round-robin index for clients

    def get_next_client():
        nonlocal current_client_index
        client = client_list[current_client_index]
        current_client_index = (current_client_index + 1) % len(client_list)
        return client

    semaphore = asyncio.Semaphore(num_concurrent_requests)
    num_concurrent_tasks = max(num_concurrent_requests // 1.5, 1)
    try:
        with get_database_connection_context(database_path, read_only=True) as con:
            cur = con.cursor()
            tasks = set()
            received_poison_pill = False

            # Initially fill up to num_concurrent_tasks
            while len(tasks) < num_concurrent_tasks:
                task = task_queue.get()
                if task is None:
                    # Poison pill means shutdown
                    received_poison_pill = True
                    break

                tasks.add(
                    asyncio.create_task(
                        process_article_image(
                            cur,
                            task,
                            model_version,
                            get_next_client(),
                            semaphore,
                            num_retries=num_retries,
                        )
                    )
                )

            # Process tasks as they complete, requeuing immediately
            while len(tasks) > 0:
                done, pending = await asyncio.wait(
                    tasks, return_when=asyncio.FIRST_COMPLETED
                )

                for completed_task in done:
                    try:
                        result = await completed_task
                        if result is not None:
                            result["model_version"] = model_version
                            result_queue.put(result)
                    except Exception as e:
                        logger.error(
                            f"Error processing article image ({type(e).__name__}): {e}"
                        )
                    finally:
                        progress_queue.put(1)  # Indicate that one task is done

                    tasks.remove(completed_task)

                    # Try to queue a new task if poison pill not received
                    if not received_poison_pill:
                        try:
                            task = task_queue.get_nowait()
                            if task is None:
                                received_poison_pill = True
                            else:
                                tasks.add(
                                    asyncio.create_task(
                                        process_article_image(
                                            cur,
                                            task,
                                            model_version,
                                            get_next_client(),
                                            semaphore,
                                        )
                                    )
                                )
                        except queue.Empty:
                            # Queue is empty, continue with remaining tasks
                            pass

                if received_poison_pill and len(tasks) == 0:
                    break
    except Exception as e:
        logger.error(f"Error in assignment worker: {e}")
    finally:
        progress_queue.put(None)  # Indicate that this worker is done


def sync_refinement_worker(
    database_path: str,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    progress_queue: multiprocessing.Queue,
    model_version: str,
    client_urls: list[str],
    num_concurrent_requests: int = 32,
    client_api_key: str | None = None,
    num_retries: int = 2,
    completion_endpoint: str = "/v1",
) -> None:
    """
    Wrap ``refinement_worker`` in ``asyncio.run`` for use as a process target.

    Provides a synchronous entry point so the async worker can be spawned via
    ``multiprocessing.Process``.

    Args:
        database_path (str): Path to the SQLite database file (read-only).
        task_queue (multiprocessing.Queue): Supplies article_images_id integers.
        result_queue (multiprocessing.Queue): Receives assignment result dicts.
        progress_queue (multiprocessing.Queue): Receives progress signals.
        model_version (str): Forwarded to ``refinement_worker``.
        client_urls (list[str]): Inference server base URLs.
        num_concurrent_requests (int): Maximum concurrent in-flight LLM requests.
        client_api_key (str | None): API key for the inference server.
        num_retries (int): Validation follow-up retries per LLM call.
        completion_endpoint (str): Endpoint suffix appended to each
            ``client_urls`` entry. Defaults to ``"/v1"``.

    """
    asyncio.run(
        refinement_worker(
            database_path,
            task_queue,
            result_queue,
            progress_queue,
            model_version,
            client_urls,
            num_concurrent_requests,
            client_api_key,
            num_retries,
            completion_endpoint,
        )
    )


def refine_partial_panel_assembly(
    database_path: str,
    model_version: str,
    client_urls: list[str],
    num_workers: int | None = None,
    database_batch_size: int = 100,
    num_concurrent_requests_per_worker: int = 32,
    client_api_key: str | None = None,
    num_retries: int = 2,
    completion_endpoint: str = "/v1",
) -> None:
    """
    Run the LLM refinement pipeline for all partially assigned figures.

    Identifies figures that need review, spawns worker processes that query
    the LLM for refined assignments, and writes results via a dedicated
    database-writer process. Handles graceful shutdown on KeyboardInterrupt.

    Args:
        database_path (str): Path to the SQLite database file.
        model_version (str): Model version string used to filter predictions.
        client_urls (list[str]): Inference server base URLs to distribute
            across workers.
        num_workers (int | None): Number of parallel worker processes. Defaults
            to the available CPU count.
        database_batch_size (int): Number of results to accumulate before
            writing a batch to the database.
        num_concurrent_requests_per_worker (int): Maximum concurrent in-flight
            LLM requests per worker.
        client_api_key (str | None): API key for the inference server; defaults
            to "test" if None.
        num_retries (int): Validation follow-up retries per LLM call.
        completion_endpoint (str): Endpoint suffix appended to each
            ``client_urls`` entry for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    if num_workers is None:
        num_workers = get_cpu_count()

    if client_api_key is None:
        # Not required by server but by openai package
        client_api_key = "test"  # pragma: allowlist secret

    # Prepare tables

    with get_database_connection_context(database_path) as con:
        prepare_tables(con.cursor())
        con.commit()

    # Get the article_images_ids that need to be processed
    with get_database_connection_context(database_path, read_only=True) as con:
        cur = con.cursor()
        partially_assigned_article_images_ids = get_partially_assigned_article_images(
            cur, model_version
        )

    # Create queues for tasks and results
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
    for task in partially_assigned_article_images_ids:
        task_queue.put(task)

    # Start workers
    worker_processes: list[Any] = []
    for i in range(num_workers):
        p = mp_context.Process(
            target=sync_refinement_worker,
            args=(
                database_path,
                task_queue,
                result_queue,
                progress_queue,
                model_version,
                client_urls,
            ),
            kwargs={
                "num_concurrent_requests": num_concurrent_requests_per_worker,
                "client_api_key": client_api_key,
                "num_retries": num_retries,
                "completion_endpoint": completion_endpoint,
            },
            name=f"AssignmentWorker-{i}",
        )
        p.start()
        worker_processes.append(p)

    # Add poison pills to stop the workers
    for _ in range(num_workers):
        task_queue.put(None)

    # Track progress
    total_tasks_count = len(partially_assigned_article_images_ids)
    num_none = 0
    interrupted = False
    try:
        with tqdm(total=total_tasks_count, desc="Processed Entries") as pbar:
            while num_none < num_workers:
                progress = progress_queue.get()
                if progress is None:
                    num_none += 1
                else:
                    pbar.update(progress)
    except KeyboardInterrupt:
        interrupted = True
        logger.info("Shutdown signal received.")

    if interrupted:
        # Drain task queue to prevent workers from starting new tasks
        logger.info("Draining task queue...")
        try:
            while True:
                task_queue.get_nowait()
        except queue.Empty:
            pass

        # Put poison pills for all workers to stop them
        for _ in range(num_workers):
            task_queue.put(None)

        # Wait for all workers to finish
        logger.info("Waiting for workers to complete...")
        for p in worker_processes:
            p.join(timeout=10)
            if p.is_alive():
                logger.info(f"Force terminating {p.name}")
                p.terminate()
    else:
        # Wait for all workers to finish
        for p in worker_processes:
            p.join()

    # Print remaining tasks in the result queue
    remaining_results = result_queue.qsize()
    if remaining_results > 0:
        logger.info(f"Waiting for {remaining_results} remaining results to be written.")

    # Only drain queue if interrupted
    if interrupted:
        # Put poison pill for database writer to stop it
        result_queue.put(None)

        # Short wait to write results

        try:
            database_writer_process.join(timeout=30)
        except TimeoutError:
            # Drain remaining results to prevent memory leak
            logger.info("Draining remaining results from queue...")
            try:
                while True:
                    result_queue.get_nowait()
            except queue.Empty:
                pass

            # Stop the database writer
            result_queue.put(None)
            database_writer_process.join()
    else:
        # Stop the database writer
        result_queue.put(None)
        database_writer_process.join()
