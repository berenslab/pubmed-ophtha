"""Module for converting the GT annotations into the image_predictions format."""

import logging
import os
import warnings
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pubmed_ophtha.const.models import get_default_model_args
from pubmed_ophtha.const.paths import (
    ASSEMBLY_GT_FOLDER,
    ASSEMBLY_GT_IMAGE_PREDICTIONS_FILE,
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
)
from pubmed_ophtha.figure_splitting.base_figure_splitter import BaseFigureSplitter
from pubmed_ophtha.figure_splitting.detectron_figure_splitter import (
    DetectronFigureSplitter,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    AnnotationTypeEnum,
    ImagingTypeEnum,
    JoinedBoundingBoxAnnotation,
)
from pubmed_ophtha.panel_assembly.db_loading import perform_nms

from .join_with_annotations import (
    GroundTruthSample,
    get_default_gt_files,
    load_gt_annotations,
)

logger = logging.getLogger(__name__)


def predict_sample_image_labels(
    gt_data_point: GroundTruthSample,
    model: BaseFigureSplitter,
) -> pd.DataFrame:
    """
    Get the image predictions for the given GT sample.

    If the sample has image annotations, those will be used instead of predicting.

    Args:
        gt_data_point (GroundTruthSample): Sample to convert.
        model (BaseFigureSplitter): Prediction model.

    Raises:
        ValueError: Missing image path and annotations.

    Returns:
        pd.DataFrame: DataFrame containing the predicted bounding boxes and labels for
            the sample.

    """
    has_image_annotation = "image" in gt_data_point
    panel_sample = gt_data_point["panel"]

    if panel_sample is None:
        raise ValueError("Sample does not have a panel annotation.")

    if not has_image_annotation:
        if panel_sample.local_image_base_path is None:
            raise ValueError(
                f"Sample {panel_sample.id} does not have a local image path."
            )

        with open(
            os.path.join(
                panel_sample.local_image_base_path, panel_sample.base_image_path
            ),
            "rb",
        ) as f:
            model_output = model.predict(f.read())

        # Ignore the Classes Panel and Label
        keep_indices = [
            i
            for i, label in enumerate(model_output["pred_classes"])
            if label != "Panel" and label != "Label"
        ]

        model_output_df = pd.DataFrame(
            {
                "predicted_boxes_x0": [
                    model_output["pred_boxes"][i][0] for i in keep_indices
                ],
                "predicted_boxes_y0": [
                    model_output["pred_boxes"][i][1] for i in keep_indices
                ],
                "predicted_boxes_x1": [
                    model_output["pred_boxes"][i][2] for i in keep_indices
                ],
                "predicted_boxes_y1": [
                    model_output["pred_boxes"][i][3] for i in keep_indices
                ],
                "predicted_labels": [
                    model_output["pred_classes"][i] for i in keep_indices
                ],
                "predicted_scores": [model_output["scores"][i] for i in keep_indices],
                "secondary_predicted_labels": [
                    model_output["secondary_pred_classes"][i] for i in keep_indices
                ],
                "secondary_predicted_scores": [
                    model_output["secondary_scores"][i] for i in keep_indices
                ],
                "prediction_source": ["model"] * len(keep_indices),
                "box_id": [None] * len(keep_indices),
            }
        )

        # Recalculate nms
        model_output_df["survives_nms"] = perform_nms(
            model_output_df, class_wise_nms=False
        )

        panel_annotations = panel_sample.finished_annotations
        if len(panel_annotations) == 0:
            raise ValueError(f"No finished annotations for sample {panel_sample.id}")
        panel_bounding_boxes = panel_annotations[0].bounding_boxes
        if panel_bounding_boxes is None:
            raise ValueError(
                f"No bounding boxes in annotations for sample {panel_sample.id}"
            )

        model_output_df = pd.concat(
            [
                model_output_df,
                pd.DataFrame(
                    {
                        "predicted_boxes_x0": [
                            box.x0
                            for box in panel_bounding_boxes
                            if box.box_type is not None and box.box_type != "Image"
                        ],
                        "predicted_boxes_y0": [
                            box.y0
                            for box in panel_bounding_boxes
                            if box.box_type is not None and box.box_type != "Image"
                        ],
                        "predicted_boxes_x1": [
                            box.x1
                            for box in panel_bounding_boxes
                            if box.box_type is not None and box.box_type != "Image"
                        ],
                        "predicted_boxes_y1": [
                            box.y1
                            for box in panel_bounding_boxes
                            if box.box_type is not None and box.box_type != "Image"
                        ],
                        "predicted_labels": [
                            box.box_type.value
                            for box in panel_bounding_boxes
                            if box.box_type is not None and box.box_type != "Image"
                        ],
                        "box_id": [
                            box.id
                            for box in panel_bounding_boxes
                            if box.box_type is not None and box.box_type != "Image"
                        ],
                    }
                ).assign(
                    predicted_scores=1.0,
                    survives_nms=True,
                    prediction_source="annotation",
                    secondary_predicted_labels=None,
                    secondary_predicted_scores=-1.0,
                ),
            ],
            ignore_index=True,
        )
    else:

        def get_pred_label(box: JoinedBoundingBoxAnnotation):
            if box.box_type is None:
                return None

            if box.box_type != "Image":
                return box.box_type.value

            relevant_labels = [
                label_entry.value
                for label_entry in box.labels
                if isinstance(label_entry, ImagingTypeEnum)
            ]

            if len(relevant_labels) == 0:
                raise ValueError(
                    f"Box {box} has box_type 'Image' but no imaging type label."
                )
            if len(relevant_labels) > 1:
                return relevant_labels

            return relevant_labels[0]

        def get_secondary_pred_label(box: JoinedBoundingBoxAnnotation):
            if box.box_type is None:
                return None

            if box.box_type != "Image":
                return None

            relevant_labels = [
                label_entry.value
                for label_entry in box.labels
                if isinstance(label_entry, AnnotationTypeEnum)
            ]

            if len(relevant_labels) == 0 or len(relevant_labels) > 1:
                raise ValueError(
                    f"Box {box} has box_type 'Image' but does not have exactly one "
                    "annotation type label."
                )

            return relevant_labels[0]

        image_sample = gt_data_point["image"]
        if image_sample is None:
            raise ValueError("Sample does not have an image annotation.")
        image_annotations = image_sample.finished_annotations
        if len(image_annotations) == 0:
            raise ValueError(
                f"No finished annotations for image sample {image_sample.id}"
            )
        image_bounding_boxes = image_annotations[0].bounding_boxes
        if image_bounding_boxes is None:
            raise ValueError(
                f"No bounding boxes in annotations for image sample {image_sample.id}"
            )
        model_output_df = pd.DataFrame(
            {
                "predicted_boxes_x0": [
                    box.x0 for box in image_bounding_boxes if box.box_type is not None
                ],
                "predicted_boxes_y0": [
                    box.y0 for box in image_bounding_boxes if box.box_type is not None
                ],
                "predicted_boxes_x1": [
                    box.x1 for box in image_bounding_boxes if box.box_type is not None
                ],
                "predicted_boxes_y1": [
                    box.y1 for box in image_bounding_boxes if box.box_type is not None
                ],
                "predicted_labels": [
                    get_pred_label(box)
                    for box in image_bounding_boxes
                    if box.box_type is not None
                ],
                "secondary_predicted_labels": [
                    get_secondary_pred_label(box)
                    for box in image_bounding_boxes
                    if box.box_type is not None
                ],
                "secondary_predicted_scores": [
                    1.0 if box.box_type == "Image" else -1.0
                    for box in image_bounding_boxes
                    if box.box_type is not None
                ],
                "box_id": [
                    box.id for box in image_bounding_boxes if box.box_type is not None
                ],
            }
        ).assign(
            predicted_scores=1.0,
            survives_nms=True,
            prediction_source="annotation",
        )

    model_output_df["sample_id"] = panel_sample.id

    return model_output_df


def predict_gt_image_labels(
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
    save_interval: int = 10,
    **model_args,
):
    """
    Convert the GT annotations into the image_predictions format and save as a CSV file.

    Args:
        gt_assembly_folder (str, optional): Folder to save the results to. Defaults to
            ASSEMBLY_GT_FOLDER.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.
        save_interval (int, optional): Interval in which the results will be saved.
            Defaults to 10.
        **model_args: Additional arguments for the model. Can be used to overwrite the
            default model paths and configs defined in the function.

    """
    output_file_path = os.path.join(
        gt_assembly_folder, ASSEMBLY_GT_IMAGE_PREDICTIONS_FILE
    )
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
    defaults = get_default_model_args()

    if model_args is None:
        model_args = {}

    def _deep_merge(base: dict, override: dict) -> dict:
        # Mutates base and returns it
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _deep_merge(base[k], v)
            else:
                base[k] = deepcopy(v)
        return base

    model_args = _deep_merge(deepcopy(defaults), model_args)

    model = DetectronFigureSplitter(**model_args)

    results = []
    processed_samples = set()

    if os.path.exists(output_file_path):
        # Load
        results = [pd.read_csv(output_file_path)]

        processed_samples = set(results[0]["sample_id"].unique().tolist())

    for gt_data_point in tqdm(gt_annotations.values(), desc="Processing samples"):
        panel = gt_data_point.get("panel")
        if panel is None:
            warnings.warn("Sample does not have a panel annotation. Skipping.")
            continue

        if panel.id in processed_samples:
            continue

        try:
            result_df = predict_sample_image_labels(gt_data_point, model)
            results.append(result_df)
        except Exception as e:
            warnings.warn(f"Error processing sample {panel.id}: {e}")

        if len(results) % save_interval == 0 and len(results) > 0:
            concat_results = pd.concat(results, ignore_index=True)

            concat_results.to_csv(output_file_path, index=False)
            results = [concat_results]

    concat_results = pd.concat(results, ignore_index=True)

    all_ids = np.arange(len(concat_results), dtype=np.int64)

    if "id" not in concat_results:
        concat_results["id"] = all_ids
    else:
        all_ids = np.array(
            list(
                set(all_ids)
                - set(concat_results[~pd.isna(concat_results["id"])]["id"].tolist())
            )
        )
        concat_results.loc[pd.isna(concat_results["id"]), "id"] = all_ids[
            : len(concat_results[pd.isna(concat_results["id"])])
        ]

    concat_results = concat_results[
        [
            "id",
        ]
        + concat_results.columns.drop("id").tolist()
    ]
    concat_results.to_csv(output_file_path, index=False)
