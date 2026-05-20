"""Module defining the base class for figure splitters."""

from abc import ABC, abstractmethod
from typing import TypedDict

from pubmed_ophtha.util.registry import Registry

FIGURE_SPLITTER_REGISTRY = Registry()


class FigureSplitterPrediction(TypedDict):
    """
    Structured prediction output from a figure-splitting model.

    All fields are optional because different implementations populate different
    subsets of the keys.
    """

    pred_boxes: list[list[float]]
    pred_classes: list[str]
    scores: list[float]
    keep_after_nms: list[bool]
    secondary_pred_classes: list[str]
    secondary_scores: list[float]
    image_dimensions: tuple[int, int]


class BaseFigureSplitter(ABC):
    """Abstract base class for figure-splitting models."""

    @abstractmethod
    def predict(self, image_bytes: bytes) -> FigureSplitterPrediction:
        """
        Run inference on an image and return detected panel predictions.

        Args:
            image_bytes (bytes): Raw image bytes to run inference on.

        Returns:
            FigureSplitterPrediction: Detection results with structured fields.

        """

    @staticmethod
    @abstractmethod
    def get_estimated_size_gb() -> float:
        """
        Return the estimated GPU memory footprint of the model in gigabytes.

        Returns:
            float: Estimated model size in GB.

        """

    @staticmethod
    @abstractmethod
    def get_model_name() -> str:
        """
        Return the unique registered name of this model.

        Returns:
            str: Model name used for registry lookup.

        """
