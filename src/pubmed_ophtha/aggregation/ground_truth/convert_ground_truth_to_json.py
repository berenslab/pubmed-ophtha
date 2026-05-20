"""Module for converting the ground truth annotations to the target JSON format."""

import json
import logging
import os

import pandas as pd
from tqdm.auto import tqdm

from pubmed_ophtha.aggregation.page_conversion import (
    get_gt_caption_in_page_space,
    get_gt_figure_in_page_space,
)
from pubmed_ophtha.const.paths import (
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
    PUBMED_OPHTHA_DATABASE_PATH,
    PUBMED_OPHTHA_GT_JSON_FILE,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    AnnotationTypeEnum,
    ImagingTypeEnum,
    PanelTypeEnum,
)
from pubmed_ophtha.panel_assembly.automatically_assign_panels import (
    calculate_iou,
    calculate_overlap_fraction,
)
from pubmed_ophtha.util.database_interface import get_biomedica_df

from .join_with_annotations import (
    GroundTruthSample,
    get_default_gt_files,
    load_gt_annotations,
)

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.3
OVERLAP_THRESHOLD = 0.7


def process_sample(
    figure_uuid: str, samples: GroundTruthSample, biomedica_df: pd.DataFrame
) -> dict:
    """
    Convert a sample to the target format.

    Args:
        figure_uuid (str): PMC article ID + _ + Image cluster ID.
        samples (GroundTruthSample): Loaded ground truth annotations for the sample.
        biomedica_df (pd.DataFrame): BIOMEDICA metadata dataframe filtered to the
            relevant article.

    Raises:
        ValueError: Errors during conversion.

    Returns:
        dict: Dictionary containing the converted sample data.

    """
    if biomedica_df.empty:
        raise ValueError(f"No matching figure found in biomedica_df for {figure_uuid}")

    if samples["figure"] is None:
        raise ValueError(f"No figure annotation found for {figure_uuid}")

    figure_sample = samples["figure"]

    figure_position = get_gt_figure_in_page_space(figure_sample)
    caption_position = get_gt_caption_in_page_space(figure_sample)
    is_multi_page_figure = len(figure_position) > 1

    panel_data = None
    bbox_sample = samples.get("image") or samples.get("panel")
    if bbox_sample is not None:
        image_width, image_height = bbox_sample.get_image().size

        def _norm_box(box):
            return {
                "x0": box.x0 / image_width,
                "y0": box.y0 / image_height,
                "x1": box.x1 / image_width,
                "y1": box.y1 / image_height,
            }

        def _to_pred_box(d):
            return {
                "predicted_boxes_x0": d["x0"],
                "predicted_boxes_y0": d["y0"],
                "predicted_boxes_x1": d["x1"],
                "predicted_boxes_y1": d["y1"],
            }

        # Captions live in the "caption" sample
        # boxes whose IDs appear there are top-level panels
        caption_map: dict[str, dict] = {}
        caption_sample = samples.get("caption")
        if caption_sample is not None and caption_sample.finished_annotations:
            for box in caption_sample.finished_annotations[0].bounding_boxes or []:
                if box.name is not None or (box.text is not None and len(box.text) > 0):
                    caption_map[box.id] = {"name": box.name, "caption_text": box.text}

        panel_sample = samples.get("panel")
        if panel_sample is not None and panel_sample.id == 21548:
            caption_map.pop("1", None)

        if panel_sample is not None and panel_sample.id == 20429:
            # Ids changed
            entry = caption_map.pop("nOCwxVWbHl", None)
            if entry is None:
                raise ValueError(
                    f"Expected caption box ID 'nOCwxVWbHl' in {figure_uuid}."
                )
            caption_map["8-8JUwpJJI"] = entry

        bboxes = []
        if bbox_sample.finished_annotations:
            bboxes = bbox_sample.finished_annotations[0].bounding_boxes or []

        panels: dict[str, dict] = {}

        # First pass: boxes in caption_map are top-level panels;
        # without caption fall back to PanelTypeEnum.PANEL or root boxes
        # (no box_type, no parent_id)
        for box in bboxes:
            is_panel = box.box_type == PanelTypeEnum.PANEL
            if is_panel:
                cap = caption_map.get(box.id, {})
                panels[box.id] = {
                    "type": "panel",
                    **_norm_box(box),
                    "name": cap.get("name"),
                    "caption_text": cap.get("caption_text"),
                    "images": [],
                    "labels": [],
                }

        if len(panels) == 0:
            raise ValueError(f"No panel annotations found for PMC ID {figure_uuid}")

        orphan_boxes = {
            "labels": [],
            "images": [],
        }

        # Second pass: attach Image/Label children via parent_id,
        # falling back to spatial overlap
        for box in bboxes:
            if box.box_type not in (PanelTypeEnum.IMAGE, PanelTypeEnum.LABEL):
                continue

            target_id = box.parent_id
            if target_id is not None and target_id not in panels:
                logger.warning(
                    f"Parent panel {target_id!r} not found for"
                    f" {box.box_type} {box.id!r} in {figure_uuid}"
                )
                target_id = None

            child_coords = _norm_box(box)
            if target_id is None:
                for pid, pentry in panels.items():
                    panel_coords = {k: pentry[k] for k in ("x0", "y0", "x1", "y1")}
                    if (
                        calculate_iou(
                            pd.Series(_to_pred_box(child_coords)),
                            pd.Series(_to_pred_box(panel_coords)),
                        )
                        >= IOU_THRESHOLD
                        or calculate_overlap_fraction(
                            pd.Series(_to_pred_box(child_coords)),
                            pd.Series(_to_pred_box(panel_coords)),
                        )
                        >= OVERLAP_THRESHOLD
                    ):
                        target_id = pid
                        break
                if target_id is None:
                    logger.warning(
                        f"No parent found for {box.box_type} {box.id!r}"
                        f" in {figure_uuid}"
                    )

            if box.box_type == PanelTypeEnum.IMAGE:
                child_entry = {
                    **child_coords,
                    "mark_status": "Marked"
                    if AnnotationTypeEnum.ANNOTATED in box.labels
                    else "Unmarked",
                    "labels": [
                        label.value
                        for label in box.labels
                        if isinstance(label, ImagingTypeEnum)
                    ],
                }
                list_key = "images"
            else:
                child_entry = {
                    **child_coords,
                }
                list_key = "labels"
            if target_id is not None:
                panels[target_id][list_key].append(child_entry)
            else:
                orphan_boxes[list_key].append(child_entry)

        panel_data = list(panels.values())

        if len(orphan_boxes["images"]) > 0 or len(orphan_boxes["labels"]) > 0:
            panel_data.append(
                {
                    "type": "unassigned",
                    "images": orphan_boxes["images"],
                    "labels": orphan_boxes["labels"],
                }
            )
    elif samples.get("image") is not None or samples.get("caption") is not None:
        logger.warning(
            f"No panel annotation found for PMC ID {figure_uuid}, "
            "but image or caption annotation exists. Ignoring."
        )

    attribution = "'{title}', {citation}".format(
        title=biomedica_df["article_title"].iloc[0],
        citation=biomedica_df["file_list_citation"].iloc[0],
    )
    return {
        "article_id": int(figure_uuid.split("_", maxsplit=1)[0].removeprefix("PMC")),
        "image_cluster_id": figure_uuid.split("_", maxsplit=1)[1],
        "figure_locations": figure_position,
        "caption_locations": caption_position,
        "panel_data": panel_data,
        "is_multi_page_figure": is_multi_page_figure,
        "annotation": [key for key in samples.keys() if samples[key] is not None],
        "license": biomedica_df["file_list_license"].iloc[0],
        "attribution": attribution,
    }


def convert_gt_to_json(
    project_folder: str,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
):
    """
    Convert the annotated data to the target JSON format.

    Args:
        project_folder (str): Project folder containing the dataset and database.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.

    """
    output_file = os.path.join(project_folder, PUBMED_OPHTHA_GT_JSON_FILE)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Ensure existence of database
    database_path = os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH)

    if not os.path.exists(database_path):
        raise FileNotFoundError(
            f"Database not found at {database_path}. "
            "Please run the database conversion step first."
        )

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

    biomedica_df = get_biomedica_df(database_path)

    biomedica_df["figure_uuid"] = (
        biomedica_df["article_id"] + "_" + biomedica_df["image_cluster_id"]
    )
    biomedica_df = biomedica_df[biomedica_df["figure_uuid"].isin(gt_annotations.keys())]

    output_data = {}

    for figure_uuid, samples in tqdm(gt_annotations.items()):
        output_data[figure_uuid] = process_sample(
            figure_uuid,
            samples,
            biomedica_df[biomedica_df["figure_uuid"] == figure_uuid],
        )

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4, ensure_ascii=False)
