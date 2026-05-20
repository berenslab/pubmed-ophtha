"""Pydantic response models for LLM panel assembly outputs."""

from enum import Enum

import pydantic


class NewEvidenceLabel(pydantic.BaseModel):
    """Evidence label with a detected bounding box and text, not yet in the database."""

    bbox_2d: list[int]  # [x0, y0, x1, y1]
    text: str


class ExistingEvidenceLabel(pydantic.BaseModel):
    """Reference to an existing label in the database used as assignment evidence."""

    label_id: str | int
    # text: str   # Lost text privilege


class PositionalEvidenceType(str, Enum):
    """Categories of positional relationships used as panel assignment evidence."""

    COLUMN = "COLUMN"
    ROW = "ROW"
    POSITION_IN_IMAGE = "POSITION_IN_IMAGE"
    OTHER = "OTHER"


class PositionalEvidenceLabel(pydantic.BaseModel):
    """Evidence label describing a spatial relationship for panel assignment."""

    positional_type: PositionalEvidenceType
    short_description: str


class Assignment(pydantic.BaseModel):
    """Base model for a panel-to-subcaption assignment."""

    assigned_subcaption: str | None
    confidence: float


class CaptionAssignmentAlphaNum(Assignment):
    """Assignment for an alphanumeric-labeled panel with supporting label evidence."""

    panel_id: str | int
    assigned_subcaption: str | None
    evidence: (
        list[NewEvidenceLabel | ExistingEvidenceLabel]
        | list[PositionalEvidenceLabel]
        | None
    )
    confidence: float

    reject_panel: bool = False


class AlphaNumCaptionAssignments(pydantic.BaseModel):
    """Collection of alphanumeric panel assembly for all panels in a figure."""

    assignments: list[CaptionAssignmentAlphaNum]


class CaptionAssignmentPositional(Assignment):
    """Assignment grouping detected images into a logical panel via positional info."""

    assigned_images: list[str | int]
    assigned_subcaption: str | None
    evidence: list[PositionalEvidenceLabel] | None
    confidence: float


class PositionalCaptionAssignments(pydantic.BaseModel):
    """Collection of positional panel assembly for all image groups in a figure."""

    assignments: list[CaptionAssignmentPositional]


CaptionAssignmentResponse = pydantic.TypeAdapter(
    AlphaNumCaptionAssignments | PositionalCaptionAssignments
)
