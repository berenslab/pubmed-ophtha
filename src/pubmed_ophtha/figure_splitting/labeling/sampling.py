"""Utilities for sampling images for labeling."""

import glob
import io
import json
import logging
import os
import shutil
import tarfile
from collections import defaultdict
from typing import Any
from xml.etree import ElementTree

import pandas as pd
from PIL import Image
from pmo_parser.bounding_boxes import BBox
from pmo_parser.renderer import render_page
from tqdm.auto import tqdm

from pubmed_ophtha.filtering.retrieve_original_images import (
    ExtractionError,
    get_data_from_package,
)

from ..base_figure_splitter import BaseFigureSplitter, FigureSplitterPrediction

logger = logging.getLogger(__name__)


def sample_biomedica_file(
    biomedica_file: str,
    sample_size: int,
    output_file: str,
):
    """
    Sample a Biomedica file and save it to a new file.

    Args:
        biomedica_file (str): Path to the Biomedica file.
        sample_size (int): Number of samples to take.
        output_file (str): File to save the sampled data to.

    """
    assert biomedica_file.endswith(".parquet")
    assert os.path.exists(biomedica_file)
    assert output_file.endswith(".parquet")
    biomedica_df = pd.read_parquet(biomedica_file)

    license_type = (
        biomedica_df["origin_file"]
        .apply(os.path.basename)
        .apply(
            lambda x: "other"
            if "other" in x
            else "non_commercial"
            if "noncommercial" in x
            else "commercial"
        )
    )
    sampled_df = biomedica_df[license_type == "commercial"].sample(
        n=sample_size, random_state=0, replace=False
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    sampled_df.to_parquet(output_file)


def create_labeling_csv(subset_path: str, output_file: str, package_path: str):
    """
    Create a CSV file containing the necessary information for labeling.

    Args:
        subset_path (str): Path to the parquet file containing the Biomedica subset.
        output_file (str): Path to save the output CSV file to.
        package_path (str): Path to the OA packages.

    """
    assert subset_path.endswith(".parquet")
    assert os.path.exists(subset_path)
    assert output_file.endswith(".csv")

    biomedica_df = pd.read_parquet(subset_path)

    labeling_data = {
        "image_path": [],
        "caption": [],
        "citation": [],
        "article_id": [],
        "biomedica_df_index": [],
        "keywords": [],
        "image_cluster_id": [],
        "origin": [],
    }

    for i, row in tqdm(biomedica_df.iterrows(), total=len(biomedica_df)):
        article_id = row["article_id"]

        file = row["file_list_path"] + ".tar.gz"
        file_path = os.path.join(package_path, file)
        if not os.path.exists(file_path):
            continue

        label_image_name = os.path.join(
            "images", f"{row['article_id']}_{row['image_cluster_id']}.jpg"
        )
        citation = row["file_list_citation"]
        article_citation = f"{row['article_title']} ({citation})"

        labeling_data["image_path"].append("/label-studio/" + label_image_name)
        labeling_data["caption"].append(row["image_caption"])
        labeling_data["citation"].append(article_citation)
        labeling_data["article_id"].append(article_id)
        labeling_data["biomedica_df_index"].append(i)
        labeling_data["image_cluster_id"].append(row["image_cluster_id"])

        if row.get("type") is None:
            labeling_data["origin"].append("base")
        else:
            labeling_data["origin"].append(row["type"])

        # Get article keywords
        with tarfile.open(file_path, "r:gz") as tar:
            xml_file_paths = [f for f in tar.getnames() if f.endswith(".nxml")]
            if len(xml_file_paths) == 0:
                continue
            xml_path = xml_file_paths[0]
            xml_file = tar.extractfile(xml_path)
            if xml_file is None:
                raise ValueError(f"Could not extract XML file from {file_path}")
            xml_data = ElementTree.parse(xml_file)
        keywords = [
            kwd.text
            for kwd_group in xml_data.findall(".//kwd-group")
            for kwd in kwd_group.findall("kwd")
        ]
        keywords = [kwd for kwd in keywords if kwd is not None]

        labeling_data["keywords"].append(", ".join(sorted(keywords)))

    labeling_df = pd.DataFrame(labeling_data)
    labeling_df.to_csv(output_file, index=False)


def move_retrieved_figures(
    figures_path: str,
    image_output_path: str,
    label_csv_path: str,
    pmc_package_path: str,
    updated_csv_path: str | None = None,
):
    """
    Move the retrieved images into the output folder and update the CSV file.

    This function will check if the images are in the correct format.
    Images that are part of a compound figure will be joined together.
    If the images are not in the correct format, they will be skipped.

    Args:
        figures_path (str): Path to the folder containing the retrieved images.
        image_output_path (str): Path to save the output images to.
        label_csv_path (str): Path to the labeling csv.
        pmc_package_path (str): Path to the PMC OA packages.
        updated_csv_path (str | None, optional): Path to save the updated labeling csv \
            to. If None, will overwrite the labeling csv. Defaults to None.

    """
    if updated_csv_path is None:
        updated_csv_path = label_csv_path

    assert os.path.exists(figures_path)
    assert os.path.exists(label_csv_path)
    os.makedirs(image_output_path, exist_ok=True)

    if os.path.exists(updated_csv_path):
        return  # Skip if existing

    missing_figures = []
    wrong_formatting = []

    updated_image_map = defaultdict(dict)
    for f in tqdm(
        glob.glob(os.path.join(figures_path, "**", "meta_info.json"), recursive=True)
    ):
        with open(f) as file:
            data = json.load(file)
        for k in data.keys():
            if not isinstance(data[k], list):
                wrong_formatting.append((f, k))
                continue
            figure_list = data[k] if isinstance(data[k], list) else [data[k]]
            if len(figure_list) == 0:
                missing_figures.append((f, k))
                continue
            fig_count = 0
            copy_fig = True
            has_error = False
            for figure in figure_list:
                has_type = "type" in figure
                if not has_type:
                    wrong_formatting.append((f, k))
                    copy_fig = False
                    has_error = True
                    continue

                if has_type and figure["type"] == "error":
                    copy_fig = False
                    has_error = True
                elif not has_type or figure["type"] == "figure":
                    fig_count += 1
                else:
                    wrong_formatting.append((f, k))
                    copy_fig = False
                    has_error = True

            if fig_count > 1 and not has_error:
                try:
                    _, figure_pdf = get_data_from_package(
                        f.replace(figures_path, pmc_package_path).replace(
                            "/meta_info.json", ".tar.gz"
                        )
                    )
                except ExtractionError as _:
                    missing_figures.append((f, k))
                    continue

                pmc_id = os.path.basename(os.path.dirname(f))
                dest_path = os.path.join(image_output_path, f"{pmc_id}_{k}.png")

                # join double figures
                if all(
                    [
                        figure["figure_data"]["page"]
                        == figure_list[0]["figure_data"]["page"]
                        for figure in figure_list
                    ]
                ):  # noqa: E501
                    figure_bbox = BBox.union_boxes(
                        [
                            BBox.from_dict(figure["figure_data"]["figure_bbox"])
                            for figure in figure_list
                        ]
                    )
                    figure_image = render_page(
                        figure_pdf,
                        figure_list[0]["figure_data"]["page"],
                        bbox=figure_bbox,
                    )

                    figure_bbox = figure_bbox.to_dict()
                else:
                    # Must join figures across multiple pages
                    sorted_pages = sorted(
                        list({figure["figure_data"]["page"] for figure in figure_list})
                    )

                    figure_bbox_map = {}
                    figure_map = {}

                    image_width = 0
                    image_height = 0

                    for page in sorted_pages:
                        page_figures = [
                            figure
                            for figure in figure_list
                            if figure["figure_data"]["page"] == page
                        ]
                        figure_bbox_map[page] = BBox.union_boxes(
                            [
                                BBox.from_dict(figure["figure_data"]["figure_bbox"])
                                for figure in page_figures
                            ]
                        )
                        figure_map[page] = render_page(
                            figure_pdf,
                            page,
                            bbox=figure_bbox_map[page],
                        )

                        image_width = max(image_width, figure_map[page].width)
                        image_height += figure_map[page].height

                    buffer_distance = max(
                        int(0.02 * max(image_width, image_height)), 10
                    )  # 5% of width or 10 pixels

                    figure_image = Image.new(
                        "RGB",
                        (
                            image_width,
                            image_height + buffer_distance * (len(sorted_pages) - 1),
                        ),
                        (255, 255, 255),
                    )
                    current_y = 0
                    image_ys = {}
                    for page in sorted_pages:
                        figure_image.paste(figure_map[page], (0, current_y))
                        image_ys[page] = (
                            current_y,
                            current_y + figure_map[page].height,
                        )
                        current_y += figure_map[page].height + buffer_distance

                    figure_bbox = {
                        "__image_bounds__": image_ys,
                        "base_bbox_map": {
                            page: figure_bbox_map[page].to_dict()
                            for page in sorted_pages
                        },
                    }

                figure_image.save(dest_path)

                # Add information back into the meta_info.json
                data[k] = [
                    {
                        "figure_data": {
                            "figure_bbox": figure_bbox,
                            "page": figure_list[0]["figure_data"]["page"],
                        },
                        "type": "compound_figure",
                        "figure_path": dest_path,
                        "compound_figures": data[k],
                    }
                ]
                with open(
                    f.replace("meta_info.json", "updated_meta_info.json"), "w"
                ) as file:
                    json.dump(data, file, indent=4, ensure_ascii=False)

                updated_image_map[pmc_id][k] = dest_path
            elif copy_fig:
                pmc_id = os.path.basename(os.path.dirname(f))
                source_file_name = (
                    os.path.join(os.path.dirname(f), f"{k}_hq_{0}.png")
                    if figure_list[0].get("figure_path") is None
                    else figure_list[0]["figure_path"]
                )
                dest_file_name = os.path.join(image_output_path, f"{pmc_id}_{k}.png")
                shutil.copyfile(source_file_name, dest_file_name)
                updated_image_map[pmc_id][k] = dest_file_name
            else:
                missing_figures.append((f, k))

    # Update the image paths in the CSV file
    df = pd.read_csv(label_csv_path)
    df = df[df["article_id"].isin(updated_image_map.keys())]

    to_keep = []

    for i, row in df.iterrows():
        pmc_id = row["article_id"]
        image_cluster_id = (
            os.path.basename(row["image_cluster_id"])
            .removesuffix(".png")
            .removesuffix(".jpg")
            .removesuffix(".jpeg")
        )
        if (
            pmc_id in updated_image_map
            and image_cluster_id in updated_image_map[pmc_id]
        ):
            to_keep.append(i)
            df.at[i, "image_path"] = updated_image_map[pmc_id][
                image_cluster_id
            ].replace(image_output_path, "/label-studio/images")

    df = df.loc[to_keep]
    df.to_csv(updated_csv_path, index=False)

    logger.info(
        f"Missing figures: {len(missing_figures)}\n"
        f"Wrong formatting: {len(wrong_formatting)}\n"
    )


def _process_sample(
    row: pd.Series,
    output_folder: str,
    image_folder: str,
    prediction_model: BaseFigureSplitter | None = None,
    model_name: str | None = None,
):
    """
    Convert a single sample into the label studio format.

    Args:
        row (pd.Series): Row in the labeling csv file.
        output_folder (str): Path to save the sample to
        image_folder (str): Path containing the image files.
        prediction_model (BaseFigureSplitter | None, optional): Model to use for
            predictions. If None, no predictions will be added. Defaults to None.
        model_name (str | None, optional): Name of the model to add to the predictions.
            If None, no model name will be added. Defaults to None.

    """

    def infer(image_file: str) -> FigureSplitterPrediction | None:
        if prediction_model is None:
            return None

        pil_image = Image.open(image_file).convert("RGB")
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        results = prediction_model.predict(buf.getvalue())
        return results

    output_name = os.path.join(
        output_folder,
        os.path.basename(row["image_path"]).replace(".png", ".json"),
    )

    if os.path.exists(output_name):
        return

    info_dict: dict[str, Any] = {
        "data": {
            "image_path": f"/data/local-files/?d={row['image_path']}",
            "caption": row["caption"],
            "citation": row["citation"],
            "article_id": row["article_id"],
            "biomedica_df_index": row["biomedica_df_index"],
            "keywords": row["keywords"] if not pd.isnull(row["keywords"]) else "",
        },
    }
    prediction = infer(os.path.join(image_folder, os.path.basename(row["image_path"])))

    if prediction is not None:
        output_predictions = {
            "result": [],
            "model_version": model_name,
        }
        image_height, image_width = prediction["image_dimensions"]
        min_score = None
        for j, box in enumerate(prediction["pred_boxes"]):
            if not prediction["keep_after_nms"][j]:
                continue

            x1, y1, x2, y2 = box

            score = prediction["scores"][j]

            if min_score is None or score < min_score:
                min_score = score

            secondary_score = prediction["secondary_scores"][j]

            if secondary_score < 0:
                panel_type = prediction["pred_classes"][j]
                imaging_type = None
                annotation_type = None
            else:
                panel_type = "Image"
                imaging_type = prediction["pred_classes"][j]
                annotation_type = prediction["secondary_pred_classes"][j]

            box_id = str(j)
            output_predictions["result"].append(
                {
                    "original_width": int(image_width),
                    "original_height": int(image_height),
                    "image_rotation": 0,
                    "value": {
                        "x": float((x1 / image_width) * 100),
                        "y": float((y1 / image_height) * 100),
                        "width": float((x2 - x1) / image_width * 100),
                        "height": float((y2 - y1) / image_height * 100),
                        "rotation": 0,
                        "score": float(score),
                        "rectanglelabels": [panel_type],
                    },
                    "from_name": "panel_label",
                    "to_name": "image",
                    "type": "rectanglelabels",
                    "origin": model_name,
                    "id": box_id,
                }
            )
            if imaging_type is not None:
                output_predictions["result"].append(
                    {
                        "original_width": int(image_width),
                        "original_height": int(image_height),
                        "image_rotation": 0,
                        "value": {
                            "x": float((x1 / image_width) * 100),
                            "y": float((y1 / image_height) * 100),
                            "width": float((x2 - x1) / image_width * 100),
                            "height": float((y2 - y1) / image_height * 100),
                            "rotation": 0,
                            "score": float(score),
                            "rectanglelabels": [imaging_type],
                        },
                        "from_name": "imaging_type_label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "origin": model_name,
                        "id": box_id,
                    }
                )
            if annotation_type is not None:
                output_predictions["result"].append(
                    {
                        "original_width": int(image_width),
                        "original_height": int(image_height),
                        "image_rotation": 0,
                        "value": {
                            "x": float((x1 / image_width) * 100),
                            "y": float((y1 / image_height) * 100),
                            "width": float((x2 - x1) / image_width * 100),
                            "height": float((y2 - y1) / image_height * 100),
                            "rotation": 0,
                            "score": float(score),
                            "rectanglelabels": [annotation_type],
                        },
                        "from_name": "annotation_type_label",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "origin": model_name,
                        "id": box_id,
                    }
                )
        if min_score is not None:
            output_predictions["score"] = float(min_score)
        info_dict["predictions"] = [output_predictions]

    with open(output_name, "w") as file:
        json.dump(info_dict, file, indent=4, ensure_ascii=False)


def create_label_studio_preannotated_samples(
    output_folder: str,
    labeling_csv_path: str,
    image_folder: str,
    prediction_model: BaseFigureSplitter | None = None,
    model_name: str | None = None,
):
    """
    Convert the labeling csv to label studio pre-annotated samples.

    Uses the model name if available to add the predictions to the samples.

    Args:
        output_folder (str): Folder to save the JSON files to.
        labeling_csv_path (str): Labeling csv file path.
        image_folder (str): Path to the directory containing the images.
        prediction_model (BaseFigureSplitter): Model to use for predictions. If None, no predictions will be added. Defaults to None.
        model_name (str | None, optional): Name of the model to add to the predictions. If None, no model name will be added. Defaults to None.

    """  # noqa: E501
    try:
        os.makedirs(output_folder, exist_ok=True)

        data_df = pd.read_csv(labeling_csv_path)
        data_df["keywords"] = data_df["keywords"].fillna("")

        for _, row in tqdm(
            data_df.iterrows(),
            total=len(data_df),
            desc="Creating pre-annotated samples",
        ):
            _process_sample(
                row,
                output_folder,
                image_folder,
                prediction_model=prediction_model,
                model_name=model_name,
            )
    except (Exception, KeyboardInterrupt) as e:
        logger.error("An error occurred, force stopping server....")
        raise e
