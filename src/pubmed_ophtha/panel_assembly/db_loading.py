"""Database loading utilities for panel assembly."""

import sqlite3
from io import BytesIO

import pandas as pd
import torch
from PIL import Image
from torchvision.ops import nms as torchvision_nms


def get_base_article_data(
    cursor: sqlite3.Cursor, article_images_id: int, model_version: str
) -> dict | None:
    """
    Load the data shared by both the automatic-assignment and LLM-refinement pipelines.

    Queries article info, split captions, NMS-filtered predictions, and the article
    image.  Returns None when no full caption or no sub-captions can be found.

    Args:
        cursor: Active read-only SQLite cursor.
        article_images_id: Primary key of the article_images row.
        model_version: Model version string used to filter image_predictions rows.

    Returns:
        dict | None with keys:
            - article_id (int)
            - image_cluster_id (int)
            - split_captions_entries (list[tuple]): raw rows
              ``(id, sub_caption_name, sub_caption_text, is_full_caption)``
            - predictions_df (pd.DataFrame): all predictions with a
              ``survives_nms`` column added; id is still a plain column
            - article_image (PIL.Image.Image)

    """
    cursor.execute(
        "SELECT article_id, image_cluster_id FROM article_images WHERE id = ?;",
        (article_images_id,),
    )
    article_id, image_cluster_id = cursor.fetchone()

    cursor.execute(
        """SELECT id, sub_caption_name, sub_caption_text, is_full_caption
        FROM parsed_split_captions
        WHERE article_id = ? AND image_cluster_id = ?;""",
        (article_id, image_cluster_id),
    )
    split_captions_entries = cursor.fetchall()  # (id, name, text, is_full_caption)

    if not any(is_full for _, _, _, is_full in split_captions_entries):
        return None
    if not any(not is_full for _, _, _, is_full in split_captions_entries):
        return None

    cursor.execute(
        """SELECT *
        FROM image_predictions
        WHERE
            article_images_id = ?
            AND model_version = ?;
        """,
        (article_images_id, model_version),
    )
    predictions = cursor.fetchall()
    columns = [desc[0] for desc in cursor.description]
    predictions_df = pd.DataFrame(predictions, columns=columns)
    predictions_df["survives_nms"] = perform_split_nms(
        predictions_df, iou_threshold=0.3, score_threshold=0.25
    )

    return {
        "article_id": article_id,
        "image_cluster_id": image_cluster_id,
        "split_captions_entries": split_captions_entries,
        "predictions_df": predictions_df,
        "article_image": get_image(cursor, article_images_id),
    }


def get_image(cur: sqlite3.Cursor, article_images_id: int) -> Image.Image:
    """
    Fetch and decode the image for a given article_images record.

    Args:
        cur (sqlite3.Cursor): Active SQLite cursor.
        article_images_id (int): Primary key of the article_images row.

    Returns:
        Image.Image: RGB PIL Image decoded from the stored binary blob.

    Raises:
        ValueError: If no image is found for the given article_images_id.

    """
    cur.execute(
        """
        SELECT image FROM article_images WHERE id = ?;
        """,
        (int(article_images_id),),
    )
    result = cur.fetchone()
    if result is None:
        raise ValueError(f"No image found for article_images_id {article_images_id}")
    buf = BytesIO(result[0])
    pil_img = Image.open(buf).convert("RGB")

    return pil_img


def perform_nms(
    df: pd.DataFrame,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.25,
    class_wise_nms: bool = True,
) -> pd.Series:
    """
    Apply non-maximum suppression to a DataFrame of predicted bounding boxes.

    Args:
        df (pd.DataFrame): DataFrame with columns predicted_boxes_x0/y0/x1/y1,
            predicted_scores, and predicted_labels.
        iou_threshold (float): IoU threshold above which overlapping boxes are
            suppressed.
        score_threshold (float): Minimum confidence score for a box to be
            considered.
        class_wise_nms (bool): If True, NMS is applied per predicted class in
            addition to the global pass.

    Returns:
        pd.Series: Boolean Series indexed like df; True for boxes that survive NMS.

    """
    if df.empty:
        return pd.Series(dtype=bool)

    sub_df = [df]

    if class_wise_nms:
        classes = df["predicted_labels"].unique()
        for cls_name in classes:
            cls_df = df[df["predicted_labels"] == cls_name]
            sub_df.append(cls_df)

    keep_indices = pd.Series(dtype=bool, index=df.index, data=False)

    for sub in sub_df:
        # Apply score threshold
        filtered = sub[sub["predicted_scores"] >= score_threshold]
        if filtered.empty:
            continue

        box_tensor = torch.tensor(
            filtered[
                [
                    "predicted_boxes_x0",
                    "predicted_boxes_y0",
                    "predicted_boxes_x1",
                    "predicted_boxes_y1",
                ]
            ].values,
            dtype=torch.float32,
        )

        box_scores = torch.tensor(
            filtered["predicted_scores"].values,
            dtype=torch.float32,
        )

        selected_indices = torchvision_nms(box_tensor, box_scores, iou_threshold)

        # Mark these indices to keep
        keep_indices.loc[filtered.index[selected_indices.numpy()]] = True

    return keep_indices


def perform_split_nms(
    df: pd.DataFrame, iou_threshold: float = 0.3, score_threshold: float = 0.25
) -> pd.Series:
    """
    Apply class-wise NMS for Panel/Label predictions and global NMS for the rest.

    Args:
        df (pd.DataFrame): DataFrame with detection predictions.
        iou_threshold (float): IoU threshold for suppression.
        score_threshold (float): Minimum confidence score to keep a box.

    Returns:
        pd.Series: Boolean Series indexed like df; True for boxes that survive NMS.

    """
    # Class wise nms for Panel and Label, global nms for rest
    class_wise_nms = df["predicted_labels"].isin(["Panel", "Label"])

    class_wise_keep = perform_nms(
        df[class_wise_nms], iou_threshold, score_threshold, class_wise_nms=True
    )
    global_keep = perform_nms(
        df[~class_wise_nms], iou_threshold, score_threshold, class_wise_nms=False
    )

    return class_wise_keep.combine_first(global_keep)
