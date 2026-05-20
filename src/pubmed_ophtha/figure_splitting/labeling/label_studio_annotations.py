"""Interface for annotations exported from label studio."""

from __future__ import annotations

import json
import os
import warnings
from enum import Enum
from functools import cached_property
from typing import Any, Literal, TypeGuard

from PIL import Image
from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from pubmed_ophtha.const.paths import (
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
)


class CaptionValues(BaseModel):
    """Values for a caption segment in Label Studio annotations."""

    start: int
    end: int
    text: str
    labels: list[str] | None = None


class LabelValues(BaseModel):
    """Base class for label values in Label Studio annotations."""

    x0: float
    y0: float
    x1: float
    y1: float
    rotation: float


class RectangleLabelEnum(str, Enum):
    """Base class for rectangle label enums in Label Studio annotations."""

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            return None

        # Case-insensitive matching
        for member in cls:
            if member.value.lower() == value.lower():
                return member
        return None


class ImagingTypeEnum(RectangleLabelEnum):
    """Enum for imaging types in Label Studio annotations."""

    CFP = "CFP"
    OCT = "OCT"
    OTHER = "Other"
    RETINAL_IMAGING = "Retinal Imaging"


class AnnotationTypeEnum(RectangleLabelEnum):
    """Enum for annotation types in Label Studio annotations."""

    ANNOTATED = "Annotated"
    PLAIN = "Plain"


class PanelTypeEnum(RectangleLabelEnum):
    """Enum for panel types in Label Studio annotations."""

    IMAGE = "Image"
    PANEL = "Panel"
    LABEL = "Label"


class BoundingBoxValues(LabelValues):
    """Values for a bounding box in Label Studio annotations."""

    rectanglelabels: (
        None | (list[ImagingTypeEnum | AnnotationTypeEnum | PanelTypeEnum])
    ) = None


class TextAreaValues(BaseModel):
    """Values for a text area in Label Studio annotations."""

    text: str | None = None

    @field_validator("text", mode="before")
    def validate_text(cls, value: list[str] | None) -> str | None:
        """
        Convert the text area value to a single string if it is a list with one element.

        Args:
            value (list[str] | None): Raw JSON value for the text area.

        Raises:
            ValueError: If the value is not a list with one string element.

        Returns:
            str | None: Updated `text` value.

        """
        if value is None:
            return None
        if isinstance(value, list):
            if len(value) == 1 and isinstance(value[0], str):
                return value[0]
        raise ValueError("Value must be a list with one string element.")


class PositionalTextAreaValues(TextAreaValues, LabelValues):
    """Values for a positional text area in Label Studio annotations."""

    pass


class LabelStudioAnnotation(BaseModel):
    """Base class for Label Studio annotations."""

    meta: dict[str, Any] | None = None

    @field_validator("meta", mode="before")
    def validate_meta(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Ensure that the meta field is either None or a non-empty dictionary.

        Args:
            value (dict[str, Any] | None): Raw JSON value for the meta field.

        Returns:
            dict[str, Any] | None: None if the value is None or an empty dictionary, \
                otherwise returns the original value.

        """
        if value is None or len(value) == 0:
            return None

        return value


class ValueLabelStudioAnnotation(LabelStudioAnnotation):
    """Base class for value-based Label Studio annotations."""

    id: str = Field(
        description="Label studio bounding box id.",
    )
    parent_id: str | None = Field(default=None, alias="parentID")
    to_name: str
    origin: str | None = None
    value: BoundingBoxValues | TextAreaValues | PositionalTextAreaValues | None = None


class BoundingBoxMixIn(BaseModel):
    """Base class for bounding box annotations in Label Studio."""

    original_width: int
    original_height: int
    image_rotation: int

    value: LabelValues

    local_image_width: int | None = None
    local_image_height: int | None = None

    @model_validator(mode="before")
    def convert_xywh_to_xyxy(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Convert the bounding box coordinates from xywh (percentage) to xyxy (absolute).

        Args:
            values (dict[str, Any]): Input values containing the bounding box \
                coordinates.

        Returns:
            dict[str, Any]: Dictionary with updated bounding box coordinates in \
                xyxy format.

        """
        box_values = values.get("value")
        if box_values is None:
            return values

        original_width = values.get("original_width")
        original_height = values.get("original_height")

        local_image_width = values.get("local_image_width")
        local_image_height = values.get("local_image_height")

        if (local_image_width is not None and local_image_height is None) or (
            local_image_width is None and local_image_height is not None
        ):
            raise ValueError(
                "Both local_image_width and local_image_height must be set or None."
            )

        if local_image_width is not None and local_image_height is not None:
            original_width = local_image_width
            original_height = local_image_height

        if original_width is None or original_height is None:
            return values

        x0 = original_width * box_values["x"] / 100.0
        y0 = original_height * box_values["y"] / 100.0
        x1 = x0 + (original_width * box_values["width"] / 100.0)
        y1 = y0 + (original_height * box_values["height"] / 100.0)

        values["value"]["x0"] = x0
        values["value"]["y0"] = y0
        values["value"]["x1"] = x1
        values["value"]["y1"] = y1
        return values


class AnnotationTypeBox(ValueLabelStudioAnnotation, BoundingBoxMixIn):  # pyright: ignore[reportIncompatibleVariableOverride]
    """Annotation type box in Label Studio annotations."""

    from_name: Literal["annotation_type_label"] = "annotation_type_label"
    type: Literal["rectanglelabels"] = "rectanglelabels"


class ImageTypeBox(ValueLabelStudioAnnotation, BoundingBoxMixIn):  # pyright: ignore[reportIncompatibleVariableOverride]
    """Image type box in Label Studio annotations."""

    from_name: Literal["imaging_type_label"] = "imaging_type_label"
    type: Literal["rectanglelabels"] = "rectanglelabels"


class PanelTypeBox(ValueLabelStudioAnnotation, BoundingBoxMixIn):  # pyright: ignore[reportIncompatibleVariableOverride]
    """Panel type box in Label Studio annotations."""

    from_name: Literal["panel_label"] = "panel_label"
    type: Literal["rectanglelabels"] = "rectanglelabels"


class CaptionSegment(LabelStudioAnnotation):
    """Caption segment in Label Studio annotations."""

    id: str
    from_name: Literal["caption_label"] = "caption_label"
    to_name: str
    type: Literal["labels"] = "labels"
    origin: str
    value: CaptionValues | None = None


class Relation(LabelStudioAnnotation):
    """Relation between bounding boxes and caption segments."""

    from_id: str
    to_id: str
    type: Literal["relation"] = "relation"
    direction: str


class CaptionText(ValueLabelStudioAnnotation):
    """Caption text in Label Studio annotations."""

    from_name: Literal["transcription"] = "transcription"
    type: Literal["textarea"] = "textarea"


class BoundingBoxName(ValueLabelStudioAnnotation):
    """Bounding box name in Label Studio annotations."""

    from_name: Literal["box_name"] = "box_name"
    type: Literal["textarea"] = "textarea"


class CaptionTextBox(CaptionText, BoundingBoxMixIn):  # pyright: ignore[reportIncompatibleVariableOverride]
    """Caption text in Label Studio annotations."""

    pass


class BoundingBoxNameBox(BoundingBoxName, BoundingBoxMixIn):  # pyright: ignore[reportIncompatibleVariableOverride]
    """Bounding box name in Label Studio annotations."""

    pass


class JoinedBoundingBoxAnnotation(BaseModel):
    """
    Joined bounding box annotation with additional properties.

    This class combines the properties of bounding boxes, labels, and caption segments
    into a single representation for easier access and manipulation.
    """

    id: str = Field(
        description="Label studio bounding box id.",
    )
    parent_id: str | None = None
    box_type: PanelTypeEnum | None = None
    x0: float
    y0: float
    x1: float
    y1: float

    original_width: int
    original_height: int

    labels: list[ImagingTypeEnum | AnnotationTypeEnum] = Field(default_factory=list)
    meta: dict[str, Any] | None = None
    name: str | None = None
    caption_text: str | None = None
    caption_segments: list[CaptionSegment] | None = None

    @property
    def text(self) -> str | None:
        """Return the name or caption text if available."""
        if self.caption_text is not None:
            return self.caption_text

        if self.caption_segments is None:
            return None

        # Join all caption segment texts if available
        sorted_captions = sorted(
            self.caption_segments,
            key=lambda caption: (
                (caption.value.start, caption.value.end) if caption.value else 0
            ),
        )
        joined_text = ""

        special_characters = [
            ",",
            ".",
            ":",
            ";",
            "(",
            "{",
            "[",
            "?",
            "!",
            ")",
            "}",
            "]",
            "\t",
            "\n",
            " ",
        ]

        space_before = [
            "(",
            "{",
            "[",
        ]

        for cap in sorted_captions:
            if cap.value is None or cap.value.text is None:
                continue
            if len(joined_text) == 0:
                joined_text += cap.value.text
            elif cap.value.text[0] not in special_characters:
                joined_text += " " + cap.value.text
            else:
                if cap.value.text[0] in space_before and joined_text[-1] != " ":
                    joined_text += " " + cap.value.text
                else:
                    joined_text += cap.value.text

        return joined_text

    @field_validator("meta", mode="before")
    def validate_meta(cls, value: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Ensure that the meta field is either None or a non-empty dictionary.

        Args:
            value (dict[str, Any] | None): Raw JSON value for the meta field.

        Returns:
            dict[str, Any] | None: None if the value is None or an empty dictionary, \
                otherwise returns the original value.

        """
        if value is None or len(value) == 0:
            return None

        return value

    @model_validator(mode="before")
    def validate_box_type(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Determine the box type based on the labels.

        Args:
            values (dict[str, Any]): Input values containing the labels.

        Returns:
            dict[str, Any]: Dictionary with updated box type.

        """
        labels = values.get("labels", [])
        panel_labels = [label for label in labels if isinstance(label, PanelTypeEnum)]
        remaining_labels = [
            label for label in labels if not isinstance(label, PanelTypeEnum)
        ]

        set_box_type = values.get("box_type", None)

        if len(panel_labels) > 1:
            raise ValueError(
                "Expected a maximum of one PanelTypeEnum but found multiple:"
                + f"\n{panel_labels}"
            )
        elif len(panel_labels) == 1:
            if set_box_type is not None and set_box_type != panel_labels[0]:
                raise ValueError(
                    f"Conflicting box_type values: {set_box_type} vs {panel_labels[0]}"
                )

            values["box_type"] = panel_labels[0]
        else:
            values["box_type"] = set_box_type  # could be None

        values["labels"] = remaining_labels

        return values

    @computed_field(repr=False)
    @cached_property
    def width(self) -> float:
        """
        Calculate the width of the bounding box.

        Returns:
            float: Width of the bounding box.

        """
        return self.x1 - self.x0

    @computed_field(repr=False)
    @cached_property
    def height(self) -> float:
        """
        Calculate the height of the bounding box.

        Returns:
            float: Height of the bounding box.

        """
        return self.y1 - self.y0

    @computed_field(repr=False)
    @cached_property
    def has_relevant_labels(self) -> bool:
        """
        Check if the bounding box has any relevant labels.

        Returns:
            bool: True if the bounding box has relevant labels, otherwise False.

        """
        return (
            self.labels is not None
            and len(self.labels) > 0
            and any(
                label
                for label in self.labels
                if isinstance(label, (ImagingTypeEnum, AnnotationTypeEnum))
            )
        )

    def check_bounds(self) -> bool:
        """
        Check if the bounding box is within the image bounds.

        Returns:
            bool: True if the bounding box is within bounds, otherwise False.

        """
        # Check x0
        if self.x0 < 0 or self.x0 > self.original_width:
            return False

        # Check y0
        if self.y0 < 0 or self.y0 > self.original_height:
            return False

        # Check width
        if self.width <= 0 or self.width > self.original_width:
            return False

        # Check height
        if self.height <= 0 or self.height > self.original_height:
            return False

        # Check x1
        if self.x1 <= 0 or self.x1 > self.original_width:
            return False

        # Check y1
        if self.y1 <= 0 or self.y1 > self.original_height:
            return False

        return True

    def clip_to_bounds(self) -> tuple[float, float, float, float] | None:
        """
        Clip the bounding box to the image bounds.

        Returns:
            tuple[float, float, float, float] | None: None if the bounding box is too
                small.
                Otherwise, returns the clipped bounding box coordinates as
                (x0, y0, x1, y1).

        """
        new_x0 = max(0, min(self.x0, self.original_width))
        new_y0 = max(0, min(self.y0, self.original_height))
        new_x1 = max(0, min(self.x1, self.original_width))
        new_y1 = max(0, min(self.y1, self.original_height))

        new_width = new_x1 - new_x0
        new_height = new_y1 - new_y0

        if (
            new_width <= 0.01 * self.original_width
            or new_height <= 0.01 * self.original_height
        ):
            return None

        return new_x0, new_y0, new_x1, new_y1


class SampleAnnotation(BaseModel):
    """Sample annotation in Label Studio annotations."""

    id: str | int = Field(
        description="Label studio annotation id.",
    )
    was_cancelled: bool
    result: (
        None
        | (
            list[
                AnnotationTypeBox
                | ImageTypeBox
                | PanelTypeBox
                | CaptionSegment
                | Relation
                | CaptionText
                | BoundingBoxName
                | BoundingBoxNameBox
                | CaptionTextBox
            ]
        )
    ) = Field(
        default_factory=list,
        description="List of results associated with the annotation.",
    )

    local_image_width: int | None = None
    local_image_height: int | None = None

    @computed_field(repr=False)
    @cached_property
    def bounding_boxes(
        self,
    ) -> list[JoinedBoundingBoxAnnotation] | None:
        """
        Joined bounding boxes from the annotation results.

        Raises:
            ValueError: When an error occurs during processing, such as duplicate IDs \
                or multiple names/captions.

        Returns:
            list[JoinedBoundingBoxAnnotation] | None: A list of joined bounding box \
                annotations if available, otherwise None.

        """
        if self.result is None:
            return None

        bounding_box_dict: dict[
            str,
            list[
                AnnotationTypeBox
                | ImageTypeBox
                | PanelTypeBox
                | CaptionText
                | BoundingBoxName
                | BoundingBoxNameBox
                | CaptionTextBox
            ],
        ] = {}
        relations: list[Relation] = []
        caption_segments: dict[str, CaptionSegment] = {}

        # Group results by their IDs
        for result in self.result:
            if isinstance(
                result,
                (
                    AnnotationTypeBox,
                    ImageTypeBox,
                    PanelTypeBox,
                    CaptionText,
                    BoundingBoxName,
                    BoundingBoxNameBox,
                    CaptionTextBox,
                ),
            ):
                if result.value is None:
                    # Ignore empty bounding boxes
                    continue
                bounding_box_dict.setdefault(result.id, []).append(result)
            elif isinstance(result, Relation):
                relations.append(result)
            elif isinstance(result, CaptionSegment):
                if result.id in caption_segments:
                    raise ValueError(f"Duplicate caption segment ID found: {result.id}")
                if result.value is None:
                    # Ignore empty caption segments
                    continue
                caption_segments[result.id] = result

        caption_segments_per_box: dict[str, list[str]] = {}

        for rel in relations:
            if rel.from_id in bounding_box_dict and rel.to_id in bounding_box_dict:
                raise ValueError(
                    f"Relation from {rel.from_id} to {rel.to_id} is not valid, "
                    "as both IDs are present in the bounding boxes."
                )
            elif rel.from_id in bounding_box_dict:
                # If from_id is a bounding box, add to its caption segments
                caption_segments_per_box.setdefault(rel.from_id, []).append(rel.to_id)
            elif rel.to_id in bounding_box_dict:
                # If to_id is a bounding box, add to its caption segments
                caption_segments_per_box.setdefault(rel.to_id, []).append(rel.from_id)

        joined_bounding_boxes = []

        for box_id, box_results in bounding_box_dict.items():
            if len(box_results) == 0:
                continue

            # This assert is only to avoid mypy errors, it should never fail
            assert box_results[0].value is not None

            # Select the first bounding box to get the coordinates
            boxes_with_positional_info = [
                box
                for box in box_results
                if isinstance(box, BoundingBoxMixIn)
                and isinstance(box.value, LabelValues)
            ]

            if len(boxes_with_positional_info) == 0:
                warnings.warn(
                    f"No bounding box with positional info found for box ID {box_id}, "
                    "skipping this box."
                )
                continue
            reference_box = boxes_with_positional_info[0]

            assert isinstance(reference_box, BoundingBoxMixIn) and isinstance(
                reference_box.value, LabelValues
            )  # for mypy
            x0 = reference_box.value.x0
            y0 = reference_box.value.y0
            x1 = reference_box.value.x1
            y1 = reference_box.value.y1
            original_width = (
                reference_box.original_width
                if reference_box.local_image_width is None
                else reference_box.local_image_width
            )
            original_height = (
                reference_box.original_height
                if reference_box.local_image_height is None
                else reference_box.local_image_height
            )

            labels = []

            for box in box_results:
                if (
                    box.value is None
                    or not isinstance(box.value, BoundingBoxValues)
                    or box.value.rectanglelabels is None
                ):
                    continue

                labels.extend(box.value.rectanglelabels)

            possible_names = [
                box.value.text
                for box in box_results
                if isinstance(box, BoundingBoxName)
                and box.value is not None
                and isinstance(box.value, TextAreaValues)
                and box.value.text is not None
            ]

            if len(possible_names) > 1:
                raise ValueError(
                    f"Multiple names found for bounding box {box_id}: {possible_names}"
                )
            name = possible_names[0] if len(possible_names) > 0 else None

            possible_captions = [
                box.value.text
                for box in box_results
                if isinstance(box, CaptionText)
                and box.value is not None
                and isinstance(box.value, TextAreaValues)
                and box.value.text is not None
            ]

            if len(possible_captions) > 1:
                raise ValueError(
                    f"Multiple captions found for bounding box {box_id}: "
                    + f"{possible_captions}"
                )
            caption_text = possible_captions[0] if len(possible_captions) > 0 else None

            caption_segments_list = sorted(
                [caption_segments[c] for c in caption_segments_per_box.get(box_id, [])],
                key=lambda x: (x.value.start, x.value.end) if x.value else 0,
            )

            parent_ids = {
                box.parent_id for box in box_results if box.parent_id is not None
            }
            if len(parent_ids) > 1:
                raise ValueError(
                    f"Conflicting parent IDs for bounding box {box_id}: {parent_ids}"
                )
            parent_id = next(iter(parent_ids)) if parent_ids else None

            meta_list = [box.meta for box in box_results if box.meta is not None]

            meta = {}
            for m in meta_list:
                for key, value in m.items():
                    if key not in meta:
                        meta[key] = []

                    meta[key].append(value)

            joined_bounding_box = JoinedBoundingBoxAnnotation(
                id=box_id,
                parent_id=parent_id,
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                original_width=original_width,
                original_height=original_height,
                labels=labels if len(labels) > 0 else [],
                name=name,
                meta=meta if len(meta) > 0 else None,
                caption_text=caption_text,
                caption_segments=caption_segments_list
                if len(caption_segments_list) > 0
                else None,
            )
            joined_bounding_boxes.append(joined_bounding_box)
        if len(joined_bounding_boxes) == 0:
            return None

        return joined_bounding_boxes

    @computed_field(repr=False)
    @cached_property
    def has_meta(self) -> bool:
        """
        Check if any bounding box has meta information.

        Returns:
            bool: True if any bounding box has meta information, otherwise False.

        """
        if any([box.meta is not None for box in self.bounding_boxes or []]):
            return True
        return False


class SampleData(BaseModel):
    """Data of a sample in Label Studio annotations."""

    image_path: str
    caption: str
    citation: str
    article_id: str
    biomedica_df_index: int
    keywords: str


class Sample(BaseModel):
    """Sample in Label Studio annotations."""

    id: str | int = Field(
        description="Label studio task id of the sample.",
    )
    base_image_path: str
    meta: dict[str, Any] | None = None
    cancelled_annotations: int
    project: int
    data: SampleData
    annotations: list[SampleAnnotation] | None = None
    local_image_base_path: str | None = None
    local_image_width: int | None = None
    local_image_height: int | None = None

    class Config:
        """Configuration for the Sample model."""

        extra = "ignore"

    @model_validator(mode="before")
    def validate_annotations(cls, values: dict[str, Any]) -> dict[str, Any]:
        """
        Convert the image path to an image ID.

        Args:
            values (dict[str, Any]): Raw JSON values for the sample.

        Raises:
            ValueError: When keys are missing in the sample data.

        Returns:
            dict[str, Any]: JSON values with the image ID extracted from the image path.

        """
        sample_data = values.get("data")

        if sample_data is None:
            raise ValueError("Missing 'data' field in sample.")

        image_path = sample_data.get("image_path")
        if image_path is None:
            raise ValueError("Missing 'image_path' in sample data.")

        values["base_image_path"] = os.path.basename(image_path)

        if "meta" in values and values["meta"] is not None and len(values["meta"]) == 0:
            values["meta"] = None

        # Set width and height if available
        if (
            "local_image_base_path" in values
            and values["local_image_base_path"] is not None
        ):
            local_image_path = os.path.join(
                values["local_image_base_path"], values["base_image_path"]
            )
            try:
                with Image.open(local_image_path) as img:
                    values["local_image_width"], values["local_image_height"] = img.size
            except Exception as e:
                warnings.warn(
                    f"Could not open image at {local_image_path} to get size: {e}"
                )

            # Give data to the next levels
            if "annotations" in values and values["annotations"] is not None:
                for annotation in values["annotations"]:
                    annotation["local_image_width"] = values["local_image_width"]
                    annotation["local_image_height"] = values["local_image_height"]
                    if "result" in annotation and annotation["result"] is not None:
                        for result in annotation["result"]:
                            if (
                                "original_width" in result
                                or "original_height" in result
                            ):
                                result["local_image_width"] = values[
                                    "local_image_width"
                                ]
                                result["local_image_height"] = values[
                                    "local_image_height"
                                ]

        return values

    @computed_field(repr=False)
    @cached_property
    def image_id(self) -> str:
        """
        Retrieve the image ID of the sample.

        Returns:
            str: Image ID of the sample.

        """
        return self.base_image_path.removesuffix(".png").removesuffix("_updated")

    @computed_field(repr=False)
    @cached_property
    def image_cluster_id(self) -> str:
        """
        Retrieve the image cluster ID of the sample.

        Returns:
            str: Image cluster ID of the sample.

        """
        return self.image_id.split("_", 1)[-1]

    @computed_field(repr=False)
    @cached_property
    def article_id(self) -> str:
        """
        Retrieve the article ID of the sample.

        Returns:
            str: Article ID of the sample.

        """
        return self.data.article_id

    @computed_field(repr=False)
    @cached_property
    def has_meta(self) -> bool:
        """
        Check if the sample has meta information.

        Returns:
            bool: True if the sample has meta information, otherwise False.

        """
        if self.meta is not None and len(self.meta) > 0:
            return True
        return any([annotation.has_meta for annotation in self.annotations or []])

    @staticmethod
    def _has_annotations(
        annotations: list[SampleAnnotation] | None,
    ) -> TypeGuard[list[SampleAnnotation]]:
        """
        Check if the sample has any annotations.

        Returns:
            bool: True if the sample has annotations, otherwise False.

        """
        return annotations is not None and len(annotations) > 0

    @computed_field(repr=False)
    @property
    def has_annotations(self) -> bool:
        """
        Check if the sample has any annotations.

        Returns:
            bool: True if the sample has annotations, otherwise False.

        """
        return Sample._has_annotations(self.annotations)

    @computed_field(repr=False)
    @property
    def was_cancelled(self) -> bool:
        """
        Check if all annotations of the sample were cancelled.

        Returns:
            bool: True if the sample was cancelled, otherwise False.

        """
        _annotations = self.annotations
        if _annotations is None or not self.has_annotations:
            return False
        return self.cancelled_annotations == len(_annotations)

    @computed_field(repr=False)
    @cached_property
    def finished_annotations(self) -> list[SampleAnnotation]:
        """
        Get the finished annotations of the sample.

        Returns:
            list[SampleAnnotation]: List of finished annotations.

        """
        _annotations = self.annotations
        if _annotations is None or not self.has_annotations:
            return []

        return [
            annotation for annotation in _annotations if not annotation.was_cancelled
        ]

    @computed_field(repr=False)
    @cached_property
    def has_captioned_annotation(self) -> bool:
        """
        Check if the sample has a caption annotation.

        Returns:
            bool: True if the sample has a caption annotation, otherwise False.

        """
        return has_captioned_annotation(self)

    @computed_field(repr=False)
    @cached_property
    def has_image_level_annotation(self) -> bool:
        """
        Check if the sample has an image-level annotation.

        Returns:
            bool: True if the sample has an image-level annotation, otherwise False.

        """
        if not self.has_annotations:
            return False

        if len(self.finished_annotations) > 1:
            warnings.warn(
                "Expected at most one finished annotation but found "
                + f"{len(self.finished_annotations)} (sample.id={self.id}). "
                "Will process first annotation only."
            )

        annotation = self.finished_annotations[0]
        if annotation.bounding_boxes is None:
            return False
        for box in annotation.bounding_boxes:
            if box.has_relevant_labels and (
                box.box_type is None or box.box_type == PanelTypeEnum.IMAGE
            ):
                return True

        return False

    def get_image(
        self,
        local_image_dir: str | None = os.path.join(
            LABEL_STUDIO_BASE_FOLDER, LABEL_STUDIO_IMAGE_PATH
        ),
    ) -> Image.Image:
        """
        Retrieve the image associated with the sample.

        Args:
            local_image_dir (str, optional): The local directory containing the \
                images. Defaults to datasets/labeling_data/images.

        Returns:
            Image.Image: The loaded PIL Image object.

        """
        if local_image_dir is None and self.local_image_base_path is not None:
            local_image_dir = self.local_image_base_path
        elif local_image_dir is None:
            raise ValueError(
                "Either local_image_dir or sample.local_image_base_path must be set."
            )
        return Image.open(os.path.join(local_image_dir, self.base_image_path))


def parse_label_studio_annotations(
    file_path: str, local_image_base_path: str | None = None
) -> list[Sample]:
    """
    Parse a list of Label Studio annotations into Sample objects.

    Args:
        file_path (str): Path to the JSON file containing Label Studio annotations.
        local_image_base_path (str | None, optional): Local base path for images.
            If given, each sample will try to load the image size from this path.
            Defaults to None.

    Returns:
        list[Sample]: Parsed list of Sample objects.

    """
    with open(file_path) as file:
        annotations = json.load(file)
    samples = []
    for annotation in annotations:
        if local_image_base_path is not None:
            annotation["local_image_base_path"] = local_image_base_path
        sample = Sample(**annotation)
        samples.append(sample)
    return samples


def has_captioned_annotation(sample: Sample) -> bool:
    """
    Test if the sample has a captioned annotation.

    Args:
        sample (Sample): Sample to check for captioned annotations.

    Raises:
        ValueError: When the sample does not have exactly one annotation.

    Returns:
        bool: True if the sample has a captioned annotation, otherwise False.

    """
    if (
        sample.has_meta
        or sample.was_cancelled
        or not sample._has_annotations(sample.annotations)
    ):
        return False

    if (len(sample.annotations) - sample.cancelled_annotations) != 1:
        raise ValueError(
            f"Expected exactly one annotation, found {len(sample.annotations)}"
            + f" (sample.id={sample.id})"
        )

    selected_annotations = sample.finished_annotations[0]

    if (
        selected_annotations.bounding_boxes is None
        or len(selected_annotations.bounding_boxes) == 0
    ):
        return False

    # Get relevant bounding boxes
    bounding_boxes = list(
        filter(
            lambda box: box.has_relevant_labels,
            selected_annotations.bounding_boxes,
        )
    )

    if any([b.text is None or len(b.text) == 0 for b in bounding_boxes]):
        return False

    if len(bounding_boxes) == 0:
        return False

    if len(bounding_boxes) == 1:
        return True

    if any([b.name is None or len(b.name) == 0 for b in bounding_boxes]):
        return False

    return True
