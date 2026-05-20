"""Figure splitter model using Detectron2 and a ResNet-50."""

import os
from io import BytesIO
from typing import TypedDict

import cv2
import numpy as np
import torch
import torchvision
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from PIL import Image
from torchvision.ops import nms as torch_nms
from torchvision.transforms import v2

from pubmed_ophtha.figure_splitting.base_figure_splitter import (
    FIGURE_SPLITTER_REGISTRY,
    BaseFigureSplitter,
    FigureSplitterPrediction,
)


class DetectronPrediction(TypedDict):
    """Structured output from a Detectron2 model prediction."""

    pred_boxes: list[list[float]]
    pred_classes: list[str]
    scores: list[float]
    keep_after_nms: list[bool]


class AnnotationClassifierPrediction(TypedDict):
    """Structured output from the annotation classifier model."""

    pred_class: list[str]
    score: list[float]


class DetectronModel:
    """Wrapper around a Detectron2 model for object detection inference."""

    def __init__(self, model_path: str, config: dict, lazy_load: bool = True):
        """
        Initialize DetectronModel with weights, config, and optional lazy loading.

        Args:
            model_path (str): Path to the model weights file. A ``config.yaml``
                is expected in the same directory.
            config (dict): Prediction config with keys such as
                ``category_names``, ``score_threshold``, ``nms_threshold``,
                ``label_wise_nms``, and ``device``.
            lazy_load (bool): If True, defer model loading until first
                ``predict`` call.

        Raises:
            ValueError: If ``category_names`` is absent or empty in config.

        """
        self.model_weights = model_path
        self.model_config = os.path.join(os.path.dirname(model_path), "config.yaml")
        self.prediction_config = config

        self.category_names = config.get("category_names", [])

        if self.category_names is None or len(self.category_names) == 0:
            raise ValueError("Category names must be provided in the config.")

        if lazy_load:
            self.model = None
        else:
            self.load_model()

    def load_model(self) -> None:
        """Load the Detectron2 model weights and build the DefaultPredictor."""
        model_device = self.prediction_config.get("device", None)
        if model_device is not None:
            model_device = "cuda" if torch.cuda.is_available() else "cpu"

        score_threshold = self.prediction_config.get("score_threshold")
        cfg = get_cfg()
        cfg.set_new_allowed(True)
        cfg.merge_from_file(
            self.model_config
        )  # (os.path.join(checkpoint_folder, "config.yaml"))
        cfg.MODEL.WEIGHTS = self.model_weights

        if score_threshold is not None:
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = score_threshold
        cfg.MODEL.DEVICE = model_device

        self.model = DefaultPredictor(cfg)

    def predict(self, image_bytes: bytes) -> DetectronPrediction:
        """
        Run object detection on raw image bytes.

        Args:
            image_bytes (bytes): Raw image bytes to run inference on.

        Returns:
            DetectronPrediction: Detection results with keys
                - ``pred_boxes``
                - ``pred_classes``
                - ``scores``
                - ``keep_after_nms``.

        Raises:
            ValueError: If the image bytes cannot be decoded.

        """
        if self.model is None:
            self.load_model()
            assert self.model is not None

        # Load image_bytes in cv2
        im = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if im is None:
            raise ValueError("Could not decode image bytes.")

        outputs = self.model(im)

        if self.prediction_config.get("score_threshold", None) is not None:
            outputs["instances"] = outputs["instances"][
                outputs["instances"].scores >= self.prediction_config["score_threshold"]
            ]

        keep_after_nms = [False] * len(outputs["instances"].scores)
        if self.prediction_config.get("nms_threshold", None) is not None:
            nms_threshold = self.prediction_config["nms_threshold"]
            label_wise_nms = self.prediction_config.get("label_wise_nms", False)

            nms_boxes = outputs["instances"].pred_boxes.tensor.clone()

            if label_wise_nms:
                image_width = im.shape[1]
                offsets = outputs["instances"].pred_classes.to(torch.float32) * (
                    image_width + 1
                )
                nms_boxes += offsets[:, None]
            keep_indices = torch_nms(
                nms_boxes, outputs["instances"].scores, nms_threshold
            )

            for i in keep_indices:
                keep_after_nms[i] = True

        # Reformat output
        reformatted_output = {
            "pred_boxes": outputs["instances"].pred_boxes.tensor.cpu().numpy().tolist(),
            "pred_classes": [
                self.category_names[class_idx]
                for class_idx in outputs["instances"]
                .pred_classes.cpu()
                .numpy()
                .tolist()
            ],
            "scores": outputs["instances"].scores.cpu().numpy().tolist(),
            "keep_after_nms": keep_after_nms,
        }

        return DetectronPrediction(**reformatted_output)


class AnnotationClassifierModel:
    """ResNet-50 classifier that labels detected image regions as Plain or Annotated."""

    def __init__(self, model_path: str, config: dict, lazy_load: bool = True) -> None:
        """
        Initialize AnnotationClassifierModel with weights and optional lazy loading.

        Args:
            model_path (str): Path to the saved ResNet-50 state-dict file.
            config (dict): Config dict; supports key ``device`` to set compute
                device.
            lazy_load (bool): If True, defer model loading until first
                ``predict`` call.

        """
        self.model_weights = model_path
        self.prediction_config = config

        self.transforms = v2.Compose(
            [
                v2.Resize((224, 224)),
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.category_names = ["Plain", "Annotated"]
        if lazy_load:
            self.model = None
            self.device = None
        else:
            self.load_model()

    def load_model(self) -> None:
        """Load classifier weights and move the model to the configured device."""
        model_device = self.prediction_config.get("device", None)
        if model_device is not None:
            model_device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = model_device
        annotation_type_classifier = torchvision.models.resnet50(weights=None)
        num_ftrs = annotation_type_classifier.fc.in_features
        annotation_type_classifier.fc = torch.nn.Linear(num_ftrs, 2)

        annotation_type_classifier.load_state_dict(
            torch.load(self.model_weights, map_location=self.device)
        )
        annotation_type_classifier.eval()
        annotation_type_classifier.to(self.device)

        self.model = annotation_type_classifier

    def predict(self, pil_images: list[Image.Image]) -> AnnotationClassifierPrediction:
        """
        Classify a list of cropped panel images as Plain or Annotated.

        Args:
            pil_images (list[Image.Image]): Cropped PIL images to classify.

        Returns:
            AnnotationClassifierPrediction: Results with keys
               ``pred_class`` (list of class name strings)
                and ``score`` (list of confidence floats).

        """
        if self.model is None:
            self.load_model()
            assert self.model is not None and self.device is not None

        pred_classes = []
        pred_scores = []

        for pil_image in pil_images:
            input_image = self.transforms(pil_image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_image)
                probabilities = torch.nn.functional.softmax(outputs.data, dim=1)
                scores, pred_class_indices = torch.max(probabilities, dim=1)
            pred_classes.extend(pred_class_indices.cpu().tolist())
            pred_scores.extend(scores.cpu().tolist())
        return AnnotationClassifierPrediction(
            pred_class=[self.category_names[class_idx] for class_idx in pred_classes],
            score=pred_scores,
        )


@FIGURE_SPLITTER_REGISTRY.register("detectron_figure_splitter")
class DetectronFigureSplitter(BaseFigureSplitter):
    """Detectron2 figure splitter with panel detection and annotation classification."""

    ESTIMATED_VRAM_SIZE_GB = 2

    def __init__(
        self,
        image_detection_model_path: str | None = None,
        panel_detection_model_path: str | None = None,
        annotation_classifier_model_path: str | None = None,
        image_detection_config: dict | None = None,
        panel_detection_config: dict | None = None,
        annotation_classifier_config: dict | None = None,
        lazy_model_load: bool = True,
    ):
        """
        Initialize DetectronFigureSplitter with paths to all three sub-models.

        Args:
            image_detection_model_path (str | None): Path to the image
                detection model weights.
            panel_detection_model_path (str | None): Path to the panel
                detection model weights.
            annotation_classifier_model_path (str | None): Path to the
                annotation classifier weights.
            image_detection_config (dict | None): Config overrides for the
                image detection model.
            panel_detection_config (dict | None): Config overrides for the
                panel detection model.
            annotation_classifier_config (dict | None): Config overrides for
                the annotation classifier.
            lazy_model_load (bool): If True, defer loading all sub-models
                until first use.

        Raises:
            ValueError: If any of the three model paths is None.

        """
        if image_detection_model_path is None:
            raise ValueError("Image detection model path must be provided.")

        if panel_detection_model_path is None:
            raise ValueError("Panel detection model path must be provided.")

        if annotation_classifier_model_path is None:
            raise ValueError("Annotation classifier model path must be provided.")

        self.image_detection_config = image_detection_config
        self.panel_detection_config = panel_detection_config
        self.annotation_classifier_config = annotation_classifier_config

        if self.image_detection_config is None:
            self.image_detection_config = {
                "score_threshold": 0.5,
                "nms_threshold": 0.3,
                "label_wise_nms": False,
            }

        if self.panel_detection_config is None:
            self.panel_detection_config = {
                "score_threshold": 0.5,
                "nms_threshold": 0.3,
                "label_wise_nms": True,
            }

        if self.annotation_classifier_config is None:
            self.annotation_classifier_config = {}

        self.image_detection_model_path = image_detection_model_path
        self.panel_detection_model_path = panel_detection_model_path
        self.annotation_classifier_model_path = annotation_classifier_model_path

        self.lazy_model_load = lazy_model_load

        self.load_models()

    @staticmethod
    def get_estimated_size_gb() -> float:
        """
        Return the estimated VRAM footprint of all sub-models combined.

        Returns:
            float: Estimated model size in GB.

        """
        return DetectronFigureSplitter.ESTIMATED_VRAM_SIZE_GB

    def load_models(self) -> None:
        """Instantiate all three sub-models with their respective configs."""
        self.image_detection_model = DetectronModel(
            model_path=self.image_detection_model_path,
            config=self.image_detection_config or {},
            lazy_load=self.lazy_model_load,
        )

        self.panel_detection_model = DetectronModel(
            model_path=self.panel_detection_model_path,
            config=self.panel_detection_config or {},
            lazy_load=self.lazy_model_load,
        )

        self.annotation_classifier_model = AnnotationClassifierModel(
            model_path=self.annotation_classifier_model_path,
            config=self.annotation_classifier_config or {},
            lazy_load=self.lazy_model_load,
        )

    @staticmethod
    def get_model_name() -> str:
        """
        Return the registry name of this model.

        Returns:
            str: Model name string.

        """
        return "detectron_figure_splitter_v1"

    def predict(self, image_bytes: bytes) -> FigureSplitterPrediction:
        """
        Run panel detection and annotation classification on raw image bytes.

        Args:
            image_bytes (bytes): Raw image bytes to run inference on.

        Returns:
            FigureSplitterPrediction: Detection results with structured fields.

        """
        with Image.open(BytesIO(image_bytes)) as opened:
            pil_image = opened.convert("RGB")

        image_predictions = self.image_detection_model.predict(image_bytes)
        panel_predictions = self.panel_detection_model.predict(image_bytes)

        secondary_predictions = []
        secondary_scores = []

        cropped_images = [
            pil_image.crop(
                (
                    box[0],
                    box[1],
                    box[2],
                    box[3],
                )
            )
            for box in image_predictions["pred_boxes"]
        ]

        secondary_predictions_output = self.annotation_classifier_model.predict(
            cropped_images
        )
        secondary_predictions = secondary_predictions_output["pred_class"]
        secondary_scores = secondary_predictions_output["score"]

        output_pred_boxes = (
            panel_predictions["pred_boxes"] + image_predictions["pred_boxes"]
        )
        output_pred_classes = (
            panel_predictions["pred_classes"] + image_predictions["pred_classes"]
        )
        output_scores = panel_predictions["scores"] + image_predictions["scores"]
        output_keep_after_nms = (
            panel_predictions["keep_after_nms"] + image_predictions["keep_after_nms"]
        )
        output_secondary_pred_classes = [
            "None" for _ in range(len(panel_predictions["pred_classes"]))
        ] + secondary_predictions
        output_secondary_scores = [
            -1.0 for _ in range(len(panel_predictions["pred_classes"]))
        ] + secondary_scores

        return FigureSplitterPrediction(
            pred_boxes=output_pred_boxes,
            pred_classes=output_pred_classes,
            scores=output_scores,
            secondary_pred_classes=output_secondary_pred_classes,
            secondary_scores=output_secondary_scores,
            keep_after_nms=output_keep_after_nms,
            image_dimensions=pil_image.size,
        )
