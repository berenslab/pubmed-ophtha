"""
Module for final dataset aggregation into parquet.

Aggregate the dataset from the database and GT annotations into a single parquet file
for training and analysis.
"""

import glob
import io
import json
import logging
import multiprocessing as mp
import os
import pickle
import sqlite3

import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from pubmed_ophtha.aggregation.ground_truth.join_with_annotations import (
    get_default_gt_files,
    load_gt_annotations,
)
from pubmed_ophtha.aggregation.page_conversion import (
    get_figure_in_page_space,
    get_gt_figure_in_page_space,
)
from pubmed_ophtha.const.paths import (
    ASSEMBLY_GT_FINAL_HIERARCHY_FILE,
    ASSEMBLY_GT_FOLDER,
    ASSEMBLY_GT_PARQUET_FILE,
    DATASET_FOLDER,
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
    PUBMED_OPHTHA_BATCHES_FOLDER,
    PUBMED_OPHTHA_DATASET_PATH,
    PUBMED_OPHTHA_PARQUET_FILE,
)
from pubmed_ophtha.panel_assembly.automatically_assign_panels import (
    calculate_overlap_fraction,
)
from pubmed_ophtha.panel_assembly.db_loading import get_image, perform_split_nms
from pubmed_ophtha.util.database_interface import (
    get_biomedica_df,
    get_database_connection_context,
)

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.3
OVERLAP_THRESHOLD = 0.9
_ELIGIBLE_ORIGINS = (
    "llm_assignment",
    "llm_adjudication",
    "manual_assignment",
    "manual_assignment_joining",
)
# spellchecker:ignore-next-line
_COMMERCIAL_LICENSES = {"CC0", "CC BY", "CC BY-ND", "CC BY-SA"}
_IMAGING_TYPES = {"CFP", "OCT", "Retinal Imaging", "Other"}


def convert_annotations_to_parquet(
    database_path: str,
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
):
    """
    Convert the Label Studio GT annotations to parquet.

    Args:
        database_path (str): Path to the database to load additional information from.
        gt_assembly_folder (str, optional): Folder to save the results to. Defaults to
            ASSEMBLY_GT_FOLDER.
        label_studio_root_folder (str, optional): Root folder of label studio data,
            used to load GT annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.

    """
    output_file = os.path.join(gt_assembly_folder, ASSEMBLY_GT_PARQUET_FILE)
    final_hierarchy_file = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_FINAL_HIERARCHY_FILE
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

    logger.info("Loading final hierarchy...")
    with open(final_hierarchy_file) as f:
        final_hierarchy_map = json.load(f)

    logger.info("Loading biomedica metadata...")
    bio_meta = get_biomedica_df(database_path)
    bio_meta["figure_uuid"] = (
        bio_meta["article_id"] + "_" + bio_meta["image_cluster_id"]
    )
    bio_meta = bio_meta.set_index("figure_uuid")

    rows = []
    for figure_uuid, figure_data in tqdm(final_hierarchy_map.items()):
        if (
            figure_uuid not in gt_annotations
            or gt_annotations[figure_uuid].get("panel") is None
        ):
            logger.warning(f"Warning: no GT annotation for {figure_uuid}, skipping.")
            continue

        panel_sample = gt_annotations[figure_uuid]["panel"]
        assert panel_sample is not None
        article_id = int(panel_sample.data.article_id.removeprefix("PMC"))
        image_cluster_id = panel_sample.image_id.split("_", maxsplit=1)[-1]

        figure_sample = gt_annotations[figure_uuid]["figure"]
        assert figure_sample is not None
        figure_position = get_gt_figure_in_page_space(figure_sample)

        article_image = None
        img_w = img_h = None
        try:
            article_image = panel_sample.get_image().convert("RGB")
            img_w, img_h = article_image.size
        except Exception as e:
            logger.warning(f"Warning: could not load image for {figure_uuid}: {e}")
            continue

        is_multi_page_figure = len(figure_position) > 1

        annotation_types = [
            k for k, v in gt_annotations[figure_uuid].items() if v is not None
        ]
        has_image_gt = gt_annotations[figure_uuid].get("image") is not None
        bio = bio_meta.loc[figure_uuid] if figure_uuid in bio_meta.index else {}

        for panel in figure_data["refined_hierarchy"]:
            if panel.get("rejected") or panel.get("assigned_subcaption_name") is None:
                continue

            pcb = list(panel["panel_content_box"])
            pcb = [
                max(0, min(pcb[0], img_w)),
                max(0, min(pcb[1], img_h)),
                max(0, min(pcb[2], img_w)),
                max(0, min(pcb[3], img_h)),
            ]

            image_data_list = []
            for img in panel.get("images", []):
                if not isinstance(img, dict):
                    continue
                it = img["imaging_type"]
                raw_labels = it if isinstance(it, list) else ([it] if it else [])
                labels = [
                    label_name
                    for label_name in raw_labels
                    if label_name in _IMAGING_TYPES
                ]
                b = img["box"]
                img_pos = [
                    float(b[0] / img_w),
                    float(b[1] / img_h),
                    float(b[2] / img_w),
                    float(b[3] / img_h),
                ]
                image_data_list.append(
                    {
                        "id": f"i_gt_{img['id']}",
                        "imaging_type": {
                            "label": labels,
                            "prediction_score": img["imaging_type_score"],
                        },
                        "mark_status": {
                            "label": "Marked"
                            if img.get("mark_status") == "Annotated"
                            else "Unmarked",
                            "prediction_score": img.get("mark_status_score"),
                        },
                        "is_gt": has_image_gt,
                        "included_in_content_box": True,
                        "position": json.dumps(
                            {
                                "predicted_box": img_pos,
                                "prediction_score": img["imaging_type_score"],
                            }
                        ),
                    }
                )

            label_data_list = []
            for k, v in panel.get("labels", {}).items():
                lbl_pos = None
                if "box" in v:
                    b = v["box"]
                    lbl_pos = [
                        float(b[0] / img_w),
                        float(b[1] / img_h),
                        float(b[2] / img_w),
                        float(b[3] / img_h),
                    ]
                label_data_list.append(
                    {
                        "id": f"l_gt_{k}",
                        "origin": "annotation",
                        "is_gt": True,
                        "included_in_content_box": True,
                        "position": json.dumps(
                            {
                                "predicted_box": lbl_pos,
                                "prediction_score": v.get("score"),
                            },
                            ensure_ascii=False,
                        ),
                        "origin_identifier": None,
                    }
                )

            im_buf = io.BytesIO()
            article_image.crop(
                (round(pcb[0]), round(pcb[1]), round(pcb[2]), round(pcb[3]))
            ).save(im_buf, format="PNG")

            attribution = (
                f"'{bio.get('article_title')}', {bio.get('file_list_citation')}"
            )
            rows.append(
                {
                    "panel_id": f"p_gt_{panel['panel_id']}",
                    "article_id": article_id,
                    "image_cluster_id": image_cluster_id,
                    "panel_name": panel["assigned_subcaption_name"],
                    "subcaption_text": panel["assigned_subcaption_text"],
                    "panel_image_bytes": im_buf.getvalue(),
                    "position": json.dumps(
                        {
                            "predicted_box": [
                                pcb[0] / img_w,
                                pcb[1] / img_h,
                                pcb[2] / img_w,
                                pcb[3] / img_h,
                            ],
                            "content_box": [
                                pcb[0] / img_w,
                                pcb[1] / img_h,
                                pcb[2] / img_w,
                                pcb[3] / img_h,
                            ],
                            "figure_page_coordinates": figure_position,
                            "prediction_score": None,
                        },
                        ensure_ascii=False,
                    ),
                    "is_multi_page_figure": is_multi_page_figure,
                    "in_text_mention": json.dumps(bio["image_context"].tolist()),
                    "contains_cfp": any(
                        "CFP" in img["imaging_type"]["label"] for img in image_data_list
                    ),
                    "contains_oct": any(
                        "OCT" in img["imaging_type"]["label"] for img in image_data_list
                    ),
                    "contains_retinal": any(
                        "Retinal Imaging" in img["imaging_type"]["label"]
                        for img in image_data_list
                    ),
                    "contains_other": any(
                        "Other" in img["imaging_type"]["label"]
                        for img in image_data_list
                    ),
                    "contains_marked": any(
                        img["mark_status"]["label"] == "Marked"
                        for img in image_data_list
                    ),
                    "image_data": json.dumps(image_data_list, ensure_ascii=False),
                    "identifier_data": json.dumps(label_data_list, ensure_ascii=False),
                    "assembly": figure_data["assignment_method"],
                    "annotation": json.dumps(annotation_types),
                    "license": bio.get("file_list_license"),
                    "attribution": attribution,
                    "commercial_use": bio.get("file_list_license")
                    in _COMMERCIAL_LICENSES,
                }
            )

    pd.DataFrame(rows).to_parquet(output_file, index=False)
    logger.info(f"Wrote {len(rows)} panels to {output_file}")


def load_from_database(
    cur_local: sqlite3.Cursor,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and join tables for aggregation from the database.

    Args:
        cur_local (sqlite3.Cursor): Cursor to access database.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing:
            - panel_df (pd.DataFrame): DataFrame containing joined panel assignment and
                prediction data.
            - label_assignments_df (pd.DataFrame): DataFrame containing label
                assignments, indexed by label ID.
            - predictions_df (pd.DataFrame): DataFrame containing image prediction
                data, indexed by prediction ID.

    Note:
        The returned data is a filtered subset of the database, not a full dump.
        The following hardcoded filters are applied silently:

        - ``panel_assignments``: only rows where ``needs_review = 0``,
          ``has_estimated_labels = 0``, ``has_shared_labels = 0``,
          ``relevance = 1``, ``assigned_subcaption IS NOT NULL``, and
          ``origin != 'manual_assignment_rejection'``
        - ``image_predictions``: only rows where
          ``model_version = 'detectron_figure_splitter_v1'``

        Rows excluded by these filters are silently omitted from all three
        returned DataFrames.

    """

    def _flatten_image_ids(raw):
        ids = json.loads(raw) if isinstance(raw, str) else (raw or [])
        if ids and isinstance(ids[0], list):
            return [e for sub in ids for e in sub]
        return ids

    logger.info("Loading panel assignments...")
    cur_local.execute("""SELECT * FROM panel_assignments
        WHERE needs_review = 0
            AND has_estimated_labels = 0
            AND has_shared_labels = 0
            AND relevance = 1
            AND assigned_subcaption IS NOT NULL
            AND origin != 'manual_assignment_rejection'
    """)
    columns = [desc[0] for desc in cur_local.description]
    all_panel_assignments_df = pd.DataFrame(cur_local.fetchall(), columns=columns)

    logger.info("Loading image predictions...")
    cur_local.execute(
        "SELECT * FROM image_predictions WHERE model_version = ?;",
        ("detectron_figure_splitter_v1",),
    )
    columns = [description[0] for description in cur_local.description]
    predictions_df = pd.DataFrame(cur_local.fetchall(), columns=columns)

    logger.info("Loading article_images...")
    cur_local.execute(
        """SELECT id, article_id, image_cluster_id FROM article_images;"""
    )
    columns = [description[0] for description in cur_local.description]
    article_images_df = pd.DataFrame(cur_local.fetchall(), columns=columns)

    logger.info("Loading label assignments...")
    cur_local.execute("SELECT * from label_assignments;")
    columns = [description[0] for description in cur_local.description]
    label_assignments_df = pd.DataFrame(
        cur_local.fetchall(), columns=columns
    ).set_index("id")

    logger.info("Loading biomedica file list...")
    cur_local.execute("SELECT * FROM biomedica_data_file_list;")
    columns = [description[0] for description in cur_local.description]
    biomedica_file_list_df = pd.DataFrame(cur_local.fetchall(), columns=columns)

    biomedica_file_list_df["figure_uuid"] = (
        "PMC"
        + biomedica_file_list_df["article_id"].astype(str)
        + "_"
        + biomedica_file_list_df["image_cluster_id"]
    )

    logger.info("Loading parsed split captions...")
    cur_local.execute("SELECT * FROM parsed_split_captions;")
    columns = [description[0] for description in cur_local.description]
    parsed_split_captions_df = pd.DataFrame(cur_local.fetchall(), columns=columns)

    # join information
    logger.info("Joining dataframes...")
    panel_df = all_panel_assignments_df.join(
        predictions_df.set_index("id"),
        on="panel_id",
        rsuffix="_pred",
    ).join(
        article_images_df.set_index(["id", "article_id"]),
        on=["article_images_id", "article_id"],
        rsuffix="_img",
    )

    panel_df["figure_uuid"] = (
        "PMC" + panel_df["article_id"].astype(str) + "_" + panel_df["image_cluster_id"]
    )

    panel_df = panel_df.join(
        biomedica_file_list_df.set_index(
            ["figure_uuid", "article_id", "image_cluster_id"]
        ),
        on=["figure_uuid", "article_id", "image_cluster_id"],
        rsuffix="_biomed",
    )

    panel_df["sub_caption_name"] = panel_df["assigned_subcaption"].map(
        parsed_split_captions_df.set_index("id")["sub_caption_name"]
    )

    panel_df["sub_caption_text"] = panel_df["assigned_subcaption"].map(
        parsed_split_captions_df.set_index("id")["sub_caption_text"]
    )

    panel_df["image_ids"] = panel_df["image_ids"].apply(_flatten_image_ids)
    panel_df["label_ids"] = panel_df["label_ids"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else (x or [])
    )

    panel_df.loc[
        ((panel_df["sub_caption_name"].isna()) | (panel_df["sub_caption_name"] == ""))
        & (~panel_df["sub_caption_text"].isna()),
        "sub_caption_name",
    ] = "None"

    return (panel_df, label_assignments_df, predictions_df)


def process_article_figure(
    article_images_id: int,
    sub_panel_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    result_queue: mp.Queue,
    cur: sqlite3.Cursor,
    label_assignments_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
):
    """
    Convert one figure to target format and put into queue.

    Args:
        article_images_id (int): Article images ID for the figure to process.
        sub_panel_df (pd.DataFrame): DataFrame containing the panel assignment and
            prediction data for the figure to process.
        pred_df (pd.DataFrame): Dataframe containing the image prediction data for the
            figure to process.
        result_queue (mp.Queue): Queue to put the result into after processing.
        cur (sqlite3.Cursor): Cursor to access database.
        label_assignments_df (pd.DataFrame): DataFrame containing label assignments,
            indexed by label ID.
        predictions_df (pd.DataFrame): Full DataFrame containing image prediction data,
            indexed by prediction ID, to reference for additional information as needed.

    """
    # see if nms needs to be updated
    update_images = sub_panel_df["origin"].isin(_ELIGIBLE_ORIGINS).any()
    if update_images:
        pred_df["survives_nms"] = pred_df["survives_nms"] & (
            perform_split_nms(pred_df, iou_threshold=0.3, score_threshold=0.25)
        )

    pred_df = pred_df[
        (pred_df["survives_nms"].astype(bool))
        | (pred_df["id"].isin(sub_panel_df["panel_id"]))
    ].set_index("id")

    all_image_ids = set(sub_panel_df["image_ids"].explode().dropna().unique())
    assert all_image_ids.issubset(set(pred_df.index))

    pil_img = get_image(cur, article_images_id)

    img_width, img_height = pil_img.size

    pred_df["predicted_boxes_x0"] = pred_df["predicted_boxes_x0"].clip(0, img_width)
    pred_df["predicted_boxes_y0"] = pred_df["predicted_boxes_y0"].clip(0, img_height)
    pred_df["predicted_boxes_x1"] = pred_df["predicted_boxes_x1"].clip(0, img_width)
    pred_df["predicted_boxes_y1"] = pred_df["predicted_boxes_y1"].clip(0, img_height)

    figure_position, rendering_dpi = get_figure_in_page_space(
        int(sub_panel_df["article_id"].iloc[0]),
        sub_panel_df["image_cluster_id"].iloc[0],
        cur,
    )

    is_multi_page_figure = len(figure_position) > 1

    for _, panel_data in sub_panel_df.iterrows():
        panel_id = panel_data["panel_id"]
        image_ids = panel_data["image_ids"].copy()

        panel_box = pred_df.loc[panel_id]

        new_ip_label_ids = {
            label_id
            for label_id in panel_data["label_ids"]
            if label_id in pred_df.index.tolist()
        }
        new_la_label_ids = {
            label_id
            for label_id in panel_data["label_ids"]
            if label_id not in pred_df.index.to_list()
        }

        added_image_ids = set()
        added_ip_label_ids = set()

        if update_images:
            label_preds = pred_df[pred_df["predicted_labels"] == "Label"]
            for label_id, label_row in label_preds.iterrows():
                if label_id not in new_ip_label_ids:
                    overlap_fraction = calculate_overlap_fraction(label_row, panel_box)

                    if overlap_fraction >= OVERLAP_THRESHOLD:
                        new_ip_label_ids.add(label_id)
                        added_ip_label_ids.add(label_id)
                        continue

            for image_id, image_row in pred_df[
                (pred_df["predicted_labels"] != "Label")
                & (pred_df["predicted_labels"] != "Panel")
            ].iterrows():
                if image_id not in image_ids:
                    overlap_fraction = calculate_overlap_fraction(image_row, panel_box)

                    if overlap_fraction >= OVERLAP_THRESHOLD:
                        image_ids.append(image_id)
                        added_image_ids.add(image_id)
                        continue

        recalculate_content_box = (
            label_assignments_df.loc[list(new_la_label_ids), "origin"] == "-1"
        ).any()

        if update_images or recalculate_content_box:
            current_panel_content_box = (
                pred_df.loc[[panel_id] + list(new_ip_label_ids) + image_ids]
                .agg(
                    {
                        "predicted_boxes_x0": "min",
                        "predicted_boxes_y0": "min",
                        "predicted_boxes_x1": "max",
                        "predicted_boxes_y1": "max",
                    }
                )[
                    [
                        "predicted_boxes_x0",
                        "predicted_boxes_y0",
                        "predicted_boxes_x1",
                        "predicted_boxes_y1",
                    ]
                ]
                .tolist()
            )

            for label_id in panel_data["label_ids"]:
                # Update from labels_df
                if label_id in label_assignments_df.index.tolist():
                    label_row = label_assignments_df.loc[label_id]

                    if label_row["origin"] == "-1":
                        continue

                    current_box = [
                        label_row["bbox_x0"],
                        label_row["bbox_y0"],
                        label_row["bbox_x1"],
                        label_row["bbox_y1"],
                    ]
                    current_panel_content_box[0] = min(
                        current_panel_content_box[0], current_box[0]
                    )
                    current_panel_content_box[1] = min(
                        current_panel_content_box[1], current_box[1]
                    )
                    current_panel_content_box[2] = max(
                        current_panel_content_box[2], current_box[2]
                    )
                    current_panel_content_box[3] = max(
                        current_panel_content_box[3], current_box[3]
                    )
        else:
            current_panel_content_box = panel_data[
                [
                    "panel_content_box_x0",
                    "panel_content_box_y0",
                    "panel_content_box_x1",
                    "panel_content_box_y1",
                ]
            ].tolist()

        # Clip content box to image dimensions
        current_panel_content_box[0] = max(
            0, min(current_panel_content_box[0], img_width)
        )
        current_panel_content_box[1] = max(
            0, min(current_panel_content_box[1], img_height)
        )
        current_panel_content_box[2] = max(
            0, min(current_panel_content_box[2], img_width)
        )
        current_panel_content_box[3] = max(
            0, min(current_panel_content_box[3], img_height)
        )

        # Create label and image data
        label_data = []
        image_data = []

        for label_id in new_ip_label_ids:
            label_data.append(
                {
                    "id": f"lip_{label_id}",
                    "origin": "detection",
                    "is_gt": False,
                    "included_in_content_box": True,
                    "added_content_box_via_nms": label_id in added_ip_label_ids,
                    "position": json.dumps(
                        {
                            "predicted_box": [
                                float(pred_df.loc[label_id, "predicted_boxes_x0"])  # pyright: ignore[reportArgumentType]
                                / img_width,
                                float(pred_df.loc[label_id, "predicted_boxes_y0"])  # pyright: ignore[reportArgumentType]
                                / img_height,
                                float(pred_df.loc[label_id, "predicted_boxes_x1"])  # pyright: ignore[reportArgumentType]
                                / img_width,
                                float(pred_df.loc[label_id, "predicted_boxes_y1"])  # pyright: ignore[reportArgumentType]
                                / img_height,
                            ],
                            "prediction_score": pred_df.loc[
                                label_id, "predicted_scores"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        for label_id in new_la_label_ids:
            origin_identifier = (
                int(label_assignments_df.loc[label_id, "origin"])  # pyright: ignore[reportArgumentType]
                if label_id in label_assignments_df.index
                and label_assignments_df.loc[label_id, "origin"]
                != "manual_assignment_joining"
                else None
            )

            origin_df = None
            if origin_identifier is not None and not pd.isna(origin_identifier):
                origin_df = predictions_df[predictions_df["id"] == origin_identifier]

                if origin_df.empty:
                    origin_df = None
                else:
                    origin_df = origin_df.iloc[0]

            included_in_content_box = (
                label_assignments_df.loc[label_id, "origin"] != "-1"
            )

            label_data.append(
                {
                    "id": f"la_{label_id}",
                    "origin": "assignment",
                    "is_gt": False,
                    "included_in_content_box": included_in_content_box,
                    "added_content_box_via_nms": False,
                    "position": json.dumps(
                        {
                            "predicted_box": [
                                float(label_assignments_df.loc[label_id, "bbox_x0"])  # pyright: ignore[reportArgumentType]
                                / img_width,
                                float(label_assignments_df.loc[label_id, "bbox_y0"])  # pyright: ignore[reportArgumentType]
                                / img_height,
                                float(label_assignments_df.loc[label_id, "bbox_x1"])  # pyright: ignore[reportArgumentType]
                                / img_width,
                                float(label_assignments_df.loc[label_id, "bbox_y1"])  # pyright: ignore[reportArgumentType]
                                / img_height,
                            ],
                            "prediction_score": label_assignments_df.loc[
                                label_id, "confidence"
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    "origin_identifier": None
                    if origin_df is None
                    else {
                        "id": f"lip_{origin_identifier}",
                        "origin": "detection",
                        "is_gt": False,
                        "position": json.dumps(
                            {
                                "predicted_box": [
                                    float(origin_df["predicted_boxes_x0"] / img_width),
                                    float(origin_df["predicted_boxes_y0"] / img_height),
                                    float(origin_df["predicted_boxes_x1"] / img_width),
                                    float(origin_df["predicted_boxes_y1"] / img_height),
                                ],
                                "prediction_score": origin_df["predicted_scores"],
                            },
                            ensure_ascii=False,
                        ),
                    },
                }
            )

        for image_id in image_ids:
            image_data.append(
                {
                    "id": f"i_{image_id}",
                    "imaging_type": {
                        "label": [pred_df.loc[image_id, "predicted_labels"]],
                        "prediction_score": [pred_df.loc[image_id, "predicted_scores"]],
                    },
                    "mark_status": {
                        "label": "Marked"
                        if pred_df.loc[image_id, "secondary_predicted_labels"]
                        == "Annotated"
                        else "Unmarked",
                        "prediction_score": pred_df.loc[
                            image_id, "secondary_predicted_scores"
                        ],
                    },
                    "is_gt": False,
                    "included_in_content_box": True,
                    "added_content_box_via_nms": image_id in added_image_ids,
                    "position": json.dumps(
                        {
                            "predicted_box": [
                                float(pred_df.loc[image_id, "predicted_boxes_x0"])  # pyright: ignore[reportArgumentType]
                                / img_width,
                                float(pred_df.loc[image_id, "predicted_boxes_y0"])  # pyright: ignore[reportArgumentType]
                                / img_height,
                                float(pred_df.loc[image_id, "predicted_boxes_x1"])  # pyright: ignore[reportArgumentType]
                                / img_width,
                                float(pred_df.loc[image_id, "predicted_boxes_y1"])  # pyright: ignore[reportArgumentType]
                                / img_height,
                            ],
                            "prediction_score": [
                                pred_df.loc[image_id, "predicted_scores"]
                            ],
                        },
                        ensure_ascii=False,
                    ),
                }
            )

        im_buf = io.BytesIO()
        pil_img.crop(
            (
                float(current_panel_content_box[0]),
                float(current_panel_content_box[1]),
                float(current_panel_content_box[2]),
                float(current_panel_content_box[3]),
            )
        ).save(im_buf, format="PNG")
        attribution = (
            f"'{panel_data['article_title']}', {panel_data['file_list_citation']}"
        )
        panel_output_dict = {
            "panel_id": f"p_{panel_id}",
            "article_id": panel_data["article_id"],
            "image_cluster_id": panel_data["image_cluster_id"],
            "panel_name": panel_data["sub_caption_name"],
            "subcaption_text": panel_data["sub_caption_text"],
            "panel_image_bytes": im_buf.getvalue(),
            "position": json.dumps(
                {
                    "predicted_box": [
                        panel_box["predicted_boxes_x0"] / img_width,
                        panel_box["predicted_boxes_y0"] / img_height,
                        panel_box["predicted_boxes_x1"] / img_width,
                        panel_box["predicted_boxes_y1"] / img_height,
                    ],
                    "content_box": [
                        current_panel_content_box[0] / img_width,
                        current_panel_content_box[1] / img_height,
                        current_panel_content_box[2] / img_width,
                        current_panel_content_box[3] / img_height,
                    ],
                    "figure_page_coordinates": figure_position,
                    "prediction_score": panel_data["predicted_scores"],
                }
            ),  # 0 to 1 normalized position
            "is_multi_page_figure": is_multi_page_figure,
            "in_text_mention": json.dumps(panel_data["image_context"]),
            "contains_cfp": any(
                "CFP" in img["imaging_type"]["label"] for img in image_data
            ),
            "contains_oct": any(
                "OCT" in img["imaging_type"]["label"] for img in image_data
            ),
            "contains_retinal": any(
                "Retinal Imaging" in img["imaging_type"]["label"] for img in image_data
            ),
            "contains_other": any(
                "Other" in img["imaging_type"]["label"] for img in image_data
            ),
            "contains_marked": any(
                img["mark_status"]["label"] == "Marked" for img in image_data
            ),
            "image_data": json.dumps(image_data, ensure_ascii=False),
            "identifier_data": json.dumps(label_data, ensure_ascii=False),
            "assembly": panel_data["origin"],
            "annotation": json.dumps([]),  # no GT annotation available at this stage
            "license": panel_data["file_list_license"],
            "attribution": attribution,
            "commercial_use": panel_data["file_list_license"] in _COMMERCIAL_LICENSES,
        }
        result_queue.put(panel_output_dict)


def _processing_worker(
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    progress_queue: mp.Queue,
    worker_id: int,
    db_path: str,
    panel_df_pckl,
    label_assignments_df_pckl,
    predictions_df_pckl,
):
    panel_df = pickle.loads(panel_df_pckl)
    label_assignments_df = pickle.loads(label_assignments_df_pckl)
    predictions_df = pickle.loads(predictions_df_pckl)

    with get_database_connection_context(db_path, read_only=True) as conn:
        cur = conn.cursor()
        while True:
            article_images_id = task_queue.get()
            if article_images_id is None:
                break

            sub_panel_df = panel_df[panel_df["article_images_id"] == article_images_id]
            pred_df = predictions_df[
                predictions_df["article_images_id"] == article_images_id
            ].copy()
            try:
                process_article_figure(
                    article_images_id,
                    sub_panel_df,
                    pred_df,
                    result_queue,
                    cur,
                    label_assignments_df,
                    predictions_df,
                )
            except AssertionError as e:
                logger.exception(
                    f"Worker {worker_id}: assertion error on "
                    f"article_images_id={article_images_id}: {e}"
                )
            except Exception as e:
                logger.exception(
                    f"Worker {worker_id}: error on "
                    f"article_images_id={article_images_id}: {e}"
                )

            progress_queue.put(1)

    progress_queue.put(None)  # signal that this worker is finished


def _writer_worker(
    result_queue: mp.Queue, output_dir: str, batch_size: int, start_batch_number: int
):
    os.makedirs(output_dir, exist_ok=True)
    batch = []
    batch_index = (
        start_batch_number + 1
    )  # start from the next batch number after existing files
    total_written = 0

    while True:
        item = result_queue.get()
        if item is None:
            break
        batch.append(item)
        if len(batch) >= batch_size:
            path = os.path.join(output_dir, f"batch_{batch_index:06d}.parquet")

            pd.DataFrame(batch).to_parquet(path, index=False)
            total_written += len(batch)

            logger.info(f"Wrote {total_written} panels to file.")
            batch = []
            batch_index += 1

    if len(batch) > 0:
        path = os.path.join(output_dir, f"batch_{batch_index:06d}.parquet")
        pd.DataFrame(batch).to_parquet(path, index=False)
        total_written += len(batch)
        batch_index += 1

    logger.info(f"Writer done. {total_written} panels in {batch_index} batch file(s).")


def get_remaining_article_images_ids(
    db_path: str, output_dir: str
) -> tuple[set[int], int]:
    """
    Retrieve the set of article_images_id that still need to be processed.

    Args:
        db_path (str): Path to the SQLite database containing the panel assignments and
            predictions.
        output_dir (str): Path to the directory where the Parquet files are being
            written. Used to check for already processed article_images_id.

    Returns:
        tuple[set[int], int]: Tuple containing:
            - Set of article_images_id that still need to be processed.
            - The last batch number found in the output directory, to determine where
                to start writing new batch files.

    """
    with get_database_connection_context(db_path, read_only=True) as conn:
        cur = conn.cursor()
        cur.execute("""SELECT panel_id FROM panel_assignments
            WHERE needs_review = 0
                AND has_estimated_labels = 0
                AND has_shared_labels = 0
                AND relevance = 1
                AND assigned_subcaption IS NOT NULL
                AND origin != 'manual_assignment_rejection'
        """)
        panel_ids = {row[0] for row in cur.fetchall()}
        cur.execute(
            """SELECT id, article_images_id
            FROM image_predictions
            WHERE model_version = ?;
            """,
            ("detectron_figure_splitter_v1",),
        )

        results = cur.fetchall()

        article_images = {row[1] for row in results if row[0] in panel_ids}

        cur.execute("""SELECT id, article_id, image_cluster_id FROM article_images;""")

        figure_uuid_to_article_images_id = {
            f"{row[1]}_{row[2]}": row[0] for row in cur.fetchall()
        }

    processed_articles = set()

    last_batch_number = -1

    if os.path.exists(output_dir):
        for file_path in glob.glob(os.path.join(output_dir, "batch_*.parquet")):
            try:
                table = pq.read_table(
                    file_path, columns=["article_id", "image_cluster_id"]
                )
                existing_figure_uuids = {
                    f"{a}_{b}"
                    for a, b in zip(
                        table["article_id"].to_pylist(),
                        table["image_cluster_id"].to_pylist(),
                    )
                }
                processed_articles.update(
                    [
                        figure_uuid_to_article_images_id[uuid]
                        for uuid in existing_figure_uuids
                    ]
                )

                current_batch_number = int(
                    os.path.basename(file_path).split("_")[1].split(".")[0]
                )

                last_batch_number = max(last_batch_number, current_batch_number)
            except Exception as e:
                logger.exception(f"Error reading existing file {file_path}: {e}")

    return article_images - processed_articles, last_batch_number


def convert_database_to_parquet(
    db_path: str,
    output_dir: str,
    num_workers: int = 4,
    batch_size: int = 100,
):
    """
    Convert the entries in the database to parquet files (in batches).

    Args:
        db_path (str): Path to the SQLite database.
        output_dir (str): Folder to save the Parquet batches to.
        num_workers (int, optional): Number of workers. Defaults to 4.
        batch_size (int, optional): Number of rows in a file. Defaults to 100.

    """
    logger.info("Starting conversion of database to Parquet...")
    task_queue: mp.Queue = mp.Queue()
    result_queue: mp.Queue = mp.Queue(
        maxsize=round(1.5 * batch_size)
    )  # backpressure to avoid memory bloat
    progress_queue: mp.Queue = mp.Queue()

    logger.info("Retrieving list of article_images_id to process...")
    remaining_article_ids, start_batch_number = get_remaining_article_images_ids(
        db_path, output_dir
    )

    logger.info("Starting writer process...")
    writer = mp.Process(
        target=_writer_worker,
        args=(result_queue, output_dir, batch_size, start_batch_number),
        name="WriterProcess",
    )
    writer.start()

    with get_database_connection_context(db_path, read_only=True) as conn:
        cur = conn.cursor()
        panel_df, label_assignments_df, predictions_df = load_from_database(cur)

        panel_df_pckl = pickle.dumps(panel_df)
        label_assignments_df_pckl = pickle.dumps(label_assignments_df)
        predictions_df_pckl = pickle.dumps(predictions_df)

    workers = []
    for wid in range(num_workers):
        p = mp.Process(
            target=_processing_worker,
            args=(
                task_queue,
                result_queue,
                progress_queue,
                wid,
                db_path,
                panel_df_pckl,
                label_assignments_df_pckl,
                predictions_df_pckl,
            ),
            name=f"Worker-{wid}",
        )
        p.start()
        workers.append(p)

    for article_images_id in remaining_article_ids:
        task_queue.put(article_images_id)

    for _ in workers:
        task_queue.put(None)

    num_done = 0
    with tqdm(total=len(remaining_article_ids), desc="Processing figures") as pbar:
        while num_done < num_workers:
            p = progress_queue.get()
            if p is None:
                num_done += 1
            else:
                pbar.update(p)

    for w in workers:
        w.join()

    result_queue.put(None)
    writer.join()


def create_dataset_parquet_file(
    gt_assembly_folder: str | None = ASSEMBLY_GT_FOLDER,
    database_parquet_folder: str = os.path.join(
        DATASET_FOLDER, PUBMED_OPHTHA_DATASET_PATH, PUBMED_OPHTHA_BATCHES_FOLDER
    ),
    output_path: str = os.path.join(
        DATASET_FOLDER, PUBMED_OPHTHA_DATASET_PATH, PUBMED_OPHTHA_PARQUET_FILE
    ),
    label_studio_root_path: str | None = LABEL_STUDIO_BASE_FOLDER,
):
    """
    Convert the gt annotation parquet file and the batched database files into one.

    Args:
        gt_assembly_folder (str, optional): Folder containing GT assembly parquet file.
            If None is passed, the GT assembly parquet file will not be loaded and only
            database files will be aggregated. Defaults to ASSEMBLY_GT_FOLDER.
        database_parquet_folder (str, optional): Folder containing database parquet
            batches. Defaults to "datasets/pubmed_ophtha_batches".
        output_path (str, optional): Path to save the final file to. Defaults to
            "datasets/pubmed_ophtha.parquet".
        label_studio_root_path (str, optional): Root folder of label studio data,
            used to load GT annotations. If None is passed, GT annotations will not be
            loaded. Defaults to LABEL_STUDIO_BASE_FOLDER.

    """
    if gt_assembly_folder is not None and label_studio_root_path is not None:
        gt_parquet_path = os.path.join(gt_assembly_folder, ASSEMBLY_GT_PARQUET_FILE)

        gt_df = pd.read_parquet(gt_parquet_path)
        gt_df = gt_df[
            gt_df.apply(
                lambda row: row[
                    [
                        "contains_cfp",
                        "contains_oct",
                        "contains_retinal",
                    ]
                ].any(),
                axis=1,
            )
        ]

        gt_df["dpi"] = None
        gt_df["no_buffer_render"] = True  # True for GT data, False for database data

        # Filter articles in gt
        local_image_base_path = os.path.join(
            label_studio_root_path, LABEL_STUDIO_IMAGE_PATH
        )

        ground_truth_files = get_default_gt_files(
            os.path.join(label_studio_root_path, LABEL_STUDIO_ANNOTATION_PATH)
        )

        logger.info("Loading ground truth annotations...")
        gt_annotations = load_gt_annotations(
            **ground_truth_files,
            local_image_base_path=local_image_base_path,
        )

        gt_uuids = set(list(gt_annotations.keys()))
    else:
        logger.warning(
            "GT assembly folder or label studio root path not provided, "
            "skipping loading GT data."
        )
        gt_df = pd.DataFrame()
        gt_uuids = set()

    logger.info("Loading database Parquet files...")

    for file in tqdm(
        glob.glob(os.path.join(database_parquet_folder, "batch_*.parquet"))
    ):
        try:
            db_df = pd.read_parquet(file)
            db_df = db_df[
                db_df.apply(
                    lambda row: row[
                        [
                            "contains_cfp",
                            "contains_oct",
                            "contains_retinal",
                        ]
                    ].any(),
                    axis=1,
                )
            ]
            db_df = db_df[
                ~(
                    "PMC"
                    + db_df["article_id"].astype(str)
                    + "_"
                    + db_df["image_cluster_id"].astype(str)
                ).isin(gt_uuids)
            ]

            db_df["dpi"] = None
            db_df["no_buffer_render"] = False

            gt_df = pd.concat([gt_df, db_df], ignore_index=True)
            del db_df  # free memory
        except Exception as e:
            logger.exception(f"Error reading file {file}: {e}")

    # DType checks
    gt_df["panel_id"] = gt_df["panel_id"].apply(
        lambda x: x if isinstance(x, str) else f"p_{x}"
    )

    json_columns = [
        "in_text_mention",
        "position",
        "image_data",
        "identifier_data",
        "annotation",
    ]

    for col in json_columns:
        for idx, val in gt_df[col].items():
            if val is not None and not isinstance(val, str):
                gt_df.at[idx, col] = json.dumps(val.tolist(), ensure_ascii=False)

    method_map = {
        "automatic_caption_assignment": "Automatic Caption Assignment",
        "llm_assignment": "LLM Assignment",
        "llm_adjudication": "LLM Assignment (adjudication)",
        "automatic": "Automatic Caption Assignment",
        "llm_refinement": "LLM Assignment",
        "ground_truth": "Ground Truth",
        "llm_refinement_with_adjudication": "LLM Assignment (adjudication)",
        "manual_annotation": "Manual Assignment",
        "manual_assignment": "Manual Assignment",
        "manual_assignment_joining": "Manual Assignment",
    }

    gt_df["assembly"] = gt_df["assembly"].map(method_map)

    gt_df.to_parquet(output_path, index=False)
    logger.info(f"Combined dataset written to {output_path}")
