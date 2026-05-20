"""Module for retrieving figure/caption bounding boxes in page space."""

import json
import logging
import sqlite3

from pubmed_ophtha.figure_splitting.labeling.label_studio_figure_annotations import (
    CaptionLabel,
    FigureLabel,
    FigureSample,
)

logger = logging.getLogger(__name__)


def get_gt_figure_in_page_space(
    figure_sample: FigureSample,
) -> dict[int, dict[str, float]]:
    """
    Extract the figure bounding boxes in page space from the Label Studio annotations.

    Args:
        figure_sample (FigureSample): Loaded figure annotation sample containing
            bounding boxes with page space coordinates.

    Raises:
        ValueError: No page boxes found for the given figure_sample.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping page numbers to bounding box
            coordinates in page space. Uses the PyMuPDF convention of x0, y0, x1, y1
            for box coordinates (top-left and bottom-right corners).

    """
    fig_location = {}
    for bb in figure_sample.bounding_boxes or []:
        if isinstance(bb.label, FigureLabel) and bb.page_boxes:
            pb = bb.page_boxes[0]
            fig_location[pb.page_number] = {
                "x0": pb.x0,
                "y0": pb.y0,
                "x1": pb.x1,
                "y1": pb.y1,
            }

    if len(fig_location) == 0:
        raise ValueError(f"No page boxes found for image_id={figure_sample.image_id}")

    return fig_location


def get_gt_caption_in_page_space(
    figure_sample: FigureSample,
) -> dict[int, dict[str, float]]:
    """
    Extract the caption bounding boxes in page space from the Label Studio annotations.

    Args:
        figure_sample (FigureSample): Loaded figure annotation sample containing
            bounding boxes with page space coordinates.

    Raises:
        ValueError: No page boxes found for the given figure_sample.

    Returns:
        dict[int, dict[str, float]]: Dictionary mapping page numbers to bounding box
            coordinates in page space. Uses the PyMuPDF convention of x0, y0, x1, y1
            for box coordinates (top-left and bottom-right corners).

    """
    caption_location = {}
    for bb in figure_sample.bounding_boxes or []:
        if isinstance(bb.label, CaptionLabel) and bb.page_boxes:
            pb = bb.page_boxes[0]
            caption_location[pb.page_number] = {
                "x0": pb.x0,
                "y0": pb.y0,
                "x1": pb.x1,
                "y1": pb.y1,
            }

    if len(caption_location) == 0:
        raise ValueError(f"No page boxes found for image_id={figure_sample.image_id}")

    return caption_location


def get_figure_in_page_space(
    pmc_id: int, image_cluster_id: str, cur: sqlite3.Cursor
) -> tuple[dict[int, dict[str, float]], int | list[int]]:
    """
    Get the figure bounding box in PDF page space.

    Args:
        pmc_id (int): PMC ID of the article (without "PMC" prefix)
        image_cluster_id (str): Image cluster ID.
        cur (sqlite3.Cursor): Cursor for the SQLite database connection to fetch the
            metadata.

    Raises:
        ValueError: If no metadata is found for the given PMC ID, if no metadata entry
            is found for the given image cluster ID, or if no figure/multi_figure entry
            is found for the given PMC ID and image cluster ID. Also raises ValueError
            if any figure entry is missing a page number.

    Returns:
        tuple[dict[int, dict[str, float]], int | list[int]]: A tuple containing:
            - A dictionary mapping page numbers to bounding box coordinates in page
                space. Uses the PyMuPDF convention of x0, y0, x1, y1 for box coordinates
                (top-left and bottom-right corners).
            - The rendering DPI(s) specified in the metadata for the figure(s). If
                multiple figures have different DPIs or if any figure is missing a DPI,
                returns a list of the DPIs found. Otherwise, returns the single DPI
                value.

    """
    cur.execute("SELECT json FROM metadata WHERE article_id = ?;", (pmc_id,))
    row = cur.fetchone()

    if row is None:
        raise ValueError(f"No metadata for article_id=PMC{pmc_id}")

    metadata_dict = json.loads(row[0])
    metadata_entries = metadata_dict.get(image_cluster_id)
    if metadata_entries is None or len(metadata_entries) == 0:
        raise ValueError(f"No metadata entry for image_cluster_id={image_cluster_id}")

    figure_entry = next(
        (e for e in metadata_entries if e.get("type") in ("figure", "multi_figure")),
        None,
    )
    if figure_entry is None:
        raise ValueError(
            f"No figure/multi_figure entry for PMC{pmc_id}/{image_cluster_id}; "
            f"types present: {[e.get('type') for e in metadata_entries]}"
        )

    figures_per_page = {}

    if figure_entry.get("figure_data") is None:
        return {}, -1

    for entry in figure_entry["figure_data"].values():
        iterator = entry if isinstance(entry, list) else [entry]
        for figure in iterator:
            page = figure.get("page")
            if page is not None:
                figures_per_page.setdefault(page, []).append(figure)
            else:
                raise ValueError("Found figure entry without page number")

    rendering_dpis = [
        figure["dpi"] for figures in figures_per_page.values() for figure in figures
    ]

    if None in rendering_dpis or len(set(rendering_dpis)) > 1:
        logger.warning(
            f"Found entries with missing or inconsistent dpi values: {rendering_dpis}"
        )
        rendering_dpi = rendering_dpis
    else:
        rendering_dpi = rendering_dpis[0]

    page_figure_boxes = {
        page: {
            "x0": min(figure["figure_bbox"]["x0"] for figure in figures),
            "y0": min(figure["figure_bbox"]["y0"] for figure in figures),
            "x1": max(figure["figure_bbox"]["x1"] for figure in figures),
            "y1": max(figure["figure_bbox"]["y1"] for figure in figures),
        }
        for page, figures in figures_per_page.items()
    }
    return page_figure_boxes, rendering_dpi
