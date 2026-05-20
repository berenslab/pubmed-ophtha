"""Interface for PDF figure/caption annotations exported from Label Studio."""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any

from pydantic import BaseModel, computed_field, model_validator

from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    RectangleLabelEnum,
)


class FigureLabel(RectangleLabelEnum):
    """Label values for figure bounding boxes."""

    FIGURE = "Figure"
    ACTUAL_FIGURE = "Actual Figure"

    @property
    def is_actual(self) -> bool:
        """
        Check if the label is an actual figure label.

        Returns:
            bool: True if the label is ACTUAL_FIGURE, otherwise False.

        """
        return self == FigureLabel.ACTUAL_FIGURE


class CaptionLabel(RectangleLabelEnum):
    """Label values for caption bounding boxes."""

    CAPTION = "Caption"
    ACTUAL_CAPTION = "Actual Caption"

    @property
    def is_actual(self) -> bool:
        """
        Check if the label is an actual caption label.

        Returns:
            bool: True if the label is ACTUAL_CAPTION, otherwise False.

        """
        return self == CaptionLabel.ACTUAL_CAPTION


_FIGURE_FROM_NAMES = {"figure_position"}
_CAPTION_FROM_NAMES = {"caption_position"}


class PageBoundingBox(BaseModel):
    """Bounding box coordinates relative to a single PDF page image."""

    page_number: int
    page_index: int
    x0: float
    y0: float
    x1: float
    y1: float
    page_width: int
    page_height: int
    dpi: int = 72

    @property
    def normalized_coordinates(self) -> tuple[float, float, float, float]:
        """
        Return (x0, y0, x1, y1) in [0, 1] relative to the page image.

        Returns:
            tuple[float, float, float, float]: Normalized (x0, y0, x1, y1) coordinates.
                Multiply by page.rect.width / page.rect.height to get PyMuPDF points.

        """
        return (
            self.x0 / self.page_width,
            self.y0 / self.page_height,
            self.x1 / self.page_width,
            self.y1 / self.page_height,
        )


class FigureBoundingBox(BaseModel):
    """Single annotated bounding box in the combined (possibly multi-page) image."""

    id: str
    label: FigureLabel | CaptionLabel
    box_name: str | None = None
    x0: float
    y0: float
    x1: float
    y1: float
    original_width: int
    original_height: int
    page_boxes: list[PageBoundingBox] = []

    class Config:
        """Allow extra fields from Label Studio without validation errors."""

        extra = "ignore"

    @model_validator(mode="before")
    @classmethod
    def _convert_xywh(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Convert Label Studio xywh-percent coordinates to absolute xyxy.

        Args:
            values (dict[str, Any]): Input values containing Label Studio \
                bounding box data.

        Returns:
            dict[str, Any]: Dictionary with updated bounding box coordinates \
                in xyxy format.

        """
        value = values.get("value", {})
        w = values.get("original_width")
        h = values.get("original_height")
        if w is None or h is None or not value:
            return values
        x = value.get("x", 0)
        y = value.get("y", 0)
        width = value.get("width", 0)
        height = value.get("height", 0)
        values["x0"] = w * x / 100
        values["y0"] = h * y / 100
        values["x1"] = w * (x + width) / 100
        values["y1"] = h * (y + height) / 100

        from_name = values.get("from_name", "")
        labels = value.get("rectanglelabels", [])
        if labels:
            raw_label = labels[0]
            if from_name in _FIGURE_FROM_NAMES:
                values["label"] = FigureLabel(raw_label)
            else:
                values["label"] = CaptionLabel(raw_label)

        return values


class FigureAnnotation(BaseModel):
    """One annotation record for a Label Studio task."""

    id: int
    was_cancelled: bool
    pdf_pages: list[int] = []
    dpi: int = 72
    result: list[dict[str, Any]]

    class Config:
        """Allow extra fields from Label Studio without validation errors."""

        extra = "ignore"

    @computed_field
    @property
    def bounding_boxes(self) -> list[FigureBoundingBox]:
        """
        Parse result entries into FigureBoundingBox objects with per-page coords.

        Returns:
            list[FigureBoundingBox]: List of parsed bounding boxes with per-page \
                coordinates.

        """
        by_id: dict[str, dict[str, Any]] = {}
        for entry in self.result:
            entry_id = entry.get("id", "")
            if entry_id not in by_id:
                by_id[entry_id] = {}
            entry_type = entry.get("type")
            if entry_type == "rectanglelabels":
                by_id[entry_id]["rect"] = entry
            elif entry_type == "textarea":
                if entry.get("from_name") == "box_name":
                    by_id[entry_id]["box_name"] = entry

        boxes: list[FigureBoundingBox] = []
        for entry_id, parts in by_id.items():
            rect = parts.get("rect")
            if rect is None:
                continue
            box_data: dict[str, Any] = {
                "id": entry_id,
                "original_width": rect.get("original_width"),
                "original_height": rect.get("original_height"),
                "from_name": rect.get("from_name"),
                "value": rect.get("value", {}),
            }
            textarea = parts.get("box_name")
            if textarea is not None:
                text_vals = textarea.get("value", {}).get("text")
                if isinstance(text_vals, list) and text_vals:
                    box_data["box_name"] = text_vals[0]
                elif isinstance(text_vals, str):
                    box_data["box_name"] = text_vals

            box = FigureBoundingBox(**box_data)
            boxes.append(box)

        if self.pdf_pages:
            _assign_page_boxes(boxes, self.pdf_pages, self.dpi)

        return boxes


def _assign_page_boxes(
    boxes: list[FigureBoundingBox],
    pdf_pages: list[int],
    dpi: int,
) -> None:
    """
    Compute per-page coordinates for each box, mutating page_boxes in place.

    Args:
        boxes (list[FigureBoundingBox]): Bounding boxes to update.
        pdf_pages (list[int]): List of PDF page numbers for the combined image.
        dpi (int): DPI at which the PDF pages were rendered.

    """
    if not boxes:
        return
    n_pages = len(pdf_pages)
    combined_width = boxes[0].original_width
    combined_height = boxes[0].original_height
    page_width_px = combined_width / n_pages

    for box in boxes:
        page_index = int(box.x0 / page_width_px)
        page_index = min(page_index, n_pages - 1)

        right_edge = (page_index + 1) * page_width_px
        if box.x1 > right_edge + 1.0:  # 1 px tolerance for float rounding
            warnings.warn(
                f"Bounding box {box.id!r} spans a page boundary "
                f"(x1={box.x1:.1f} > page edge {right_edge:.1f}). "
                "Per-page coordinates may be inaccurate."
            )

        page_origin_x = page_index * page_width_px
        box.page_boxes = [
            PageBoundingBox(
                page_number=pdf_pages[page_index],
                page_index=page_index,
                x0=box.x0 - page_origin_x,
                y0=box.y0,
                x1=box.x1 - page_origin_x,
                y1=box.y1,
                page_width=int(page_width_px),
                page_height=combined_height,
                dpi=dpi,
            )
        ]


class FigureSampleData(BaseModel):
    """Metadata for a Label Studio task."""

    image_path: str
    base_image_path: str
    pdf_pages: list[int]
    caption: str
    citation: str
    article_id: str
    meta_annotation: str = ""

    class Config:
        """Allow extra fields from Label Studio without validation errors."""

        extra = "ignore"


class FigureSample(BaseModel):
    """Top-level Label Studio task with figure/caption annotations."""

    id: int
    data: FigureSampleData
    annotations: list[FigureAnnotation]
    dpi: int = 72

    class Config:
        """Allow extra fields from Label Studio without validation errors."""

        extra = "ignore"

    @model_validator(mode="after")
    def _inject_pdf_pages(self) -> FigureSample:
        """
        Inject pdf_pages and dpi into each annotation so they can compute page coords.

        Returns:
            FigureSample: Updated sample with pdf_pages and dpi injected into \
                annotations.

        """
        for ann in self.annotations:
            ann.pdf_pages = self.data.pdf_pages
            ann.dpi = self.dpi
        return self

    @computed_field
    @property
    def image_id(self) -> str:
        """
        Retrieve the image ID of the sample.

        Returns:
            str: Image ID derived from the base image path, without the Label Studio \
                URL prefix and .png extension.

        """
        basename = (
            os.path.basename(self.data.base_image_path)
            .removeprefix("pdf_")
            .removesuffix(".png")
            .removesuffix("_updated")
        )
        return basename

    @computed_field(repr=False)
    @property
    def image_cluster_id(self) -> str:
        """
        Retrieve the image cluster ID of the sample.

        Returns:
            str: Image cluster ID of the sample.

        """
        return self.image_id.split("_", 1)[-1]

    @computed_field(repr=False)
    @property
    def article_id(self) -> str:
        """
        Retrieve the article ID of the sample.

        Returns:
            str: Article ID of the sample.

        """
        return self.data.article_id

    @computed_field
    @property
    def was_cancelled(self) -> bool:
        """
        Check if all annotations of the sample were cancelled.

        Returns:
            bool: True if the sample was cancelled, otherwise False.

        """
        return bool(self.annotations) and all(a.was_cancelled for a in self.annotations)

    @computed_field
    @property
    def finished_annotations(self) -> list[FigureAnnotation]:
        """
        Get the finished annotations of the sample.

        Returns:
            list[FigureAnnotation]: List of annotations that were not cancelled.

        """
        return [a for a in self.annotations if not a.was_cancelled]

    @computed_field
    @property
    def bounding_boxes(self) -> list[FigureBoundingBox]:
        """
        Get bounding boxes from the first finished annotation.

        Returns:
            list[FigureBoundingBox]: List of bounding boxes from the first \
                finished annotation.

        """
        if not self.finished_annotations:
            return []
        return self.finished_annotations[0].bounding_boxes


def parse_pdf_annotations(
    path: str | Path,
    dpi: int = 72,
) -> list[FigureSample]:
    """
    Load a Label Studio JSON export and return one FigureSample per task.

    Args:
        path (str | Path): Path to the exported JSON file.
        dpi (int): DPI at which PDF pages were rendered. Used to populate \
            PageBoundingBox.dpi; does not affect coordinate values.

    Returns:
        list[FigureSample]: List of parsed FigureSample objects.

    """
    with open(path) as f:
        tasks = json.load(f)
    samples = []
    for task in tasks:
        task["dpi"] = dpi
        try:
            sample = FigureSample(**task)
            samples.append(sample)
        except Exception as e:
            warnings.warn(f"Skipping task {task.get('id')}: {e}")
    return samples
