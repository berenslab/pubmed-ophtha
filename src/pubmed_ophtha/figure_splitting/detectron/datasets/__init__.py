"""
Module for loading various datasets in Detectron2 format.

Loads the datasets in the current directory if Detectron2 is available.
If there is an error loading a dataset, it will be skipped.
After loading the datasets they are automatically registered to the DatasetCatalog
and MetadataCatalog.
"""

import logging

from pubmed_ophtha.figure_splitting.detectron.datasets.meta import (
    DETECTRON_DATASET_REGISTRY,
)

from . import (
    image_clef,  # noqa: F401
    imaging_type_detection,  # noqa: F401
    panel_detection,  # noqa: F401
)

logger = logging.getLogger(__name__)


def add_local_datasets():
    """Load and register all local datasets available in Detectron2 format."""
    for dataset_name, register_fn in DETECTRON_DATASET_REGISTRY:
        try:
            register_fn()
        except Exception as e:
            # Skip if there is an error
            logger.debug(f"Error loading {dataset_name} dataset: {e}")
