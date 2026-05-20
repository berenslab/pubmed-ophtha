"""ML model path and configuration constants."""

import torch

DEFAULT_MODEL_REPO_ID = "pubmed-ophtha/detection-models"

DEFAULT_IMAGE_DETECTION_MODEL_PATH = (
    "models/imaging_type_detection_1515892632/model_0003909.pth"
)
DEFAULT_PANEL_DETECTION_MODEL_PATH = (
    "models/panel_detection_1020880423/model_0026865.pth"
)
DEFAULT_ANNOTATION_CLASSIFIER_MODEL_PATH = (
    "models/mark_status_classifier_482239176/model_epoch_7.pth"
)


def get_default_model_args() -> dict:
    """Return default DetectronFigureSplitter constructor arguments."""
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    return {
        "image_detection_model_path": DEFAULT_IMAGE_DETECTION_MODEL_PATH,
        "panel_detection_model_path": DEFAULT_PANEL_DETECTION_MODEL_PATH,
        "annotation_classifier_model_path": DEFAULT_ANNOTATION_CLASSIFIER_MODEL_PATH,
        "image_detection_config": {
            "score_threshold": 0.25,
            "nms_threshold": 0.3,
            "label_wise_nms": False,
            "device": device,
            "category_names": ["CFP", "OCT", "Retinal Imaging", "Other"],
        },
        "panel_detection_config": {
            "score_threshold": 0.25,
            "nms_threshold": 0.3,
            "label_wise_nms": True,
            "device": device,
            "category_names": ["Panel", "Label"],
        },
        "annotation_classifier_config": {
            "device": device,
        },
        "lazy_model_load": True,
    }
