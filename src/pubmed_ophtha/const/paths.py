"""Filesystem path and file name constants."""

import os

# Root
DATASET_FOLDER = "datasets"

# Biomedica filtering intermediates
BIOMEDICA_META_DATASET_PATH = "biomedica_meta"
BIOMEDICA_FILTERED_DATASET_PATH = "biomedica_filtered"
BIOMEDICA_FILTERED_JOINED_DATASET_FILE = "biomedica_filtered_joined.parquet"
BIOMEDICA_FILTERED_WITH_FILE_LIST_DATASET_PATH = (
    "biomedica_filtered_with_file_list.parquet"
)

# Main dataset
PUBMED_OPHTHA_DATASET_PATH = "pubmed_ophtha"
PUBMED_OPHTHA_PACKAGES_PATH = "packages"
PUBMED_OPHTHA_FIGURES_PATH = "figures"
PUBMED_OPHTHA_DATABASE_PATH = "pubmed_ophtha.db"

# Aggregation / ground truth assembly
ASSEMBLY_GT_FOLDER_NAME = "aggregated_annotation_data"
ASSEMBLY_GT_FOLDER = os.path.join(DATASET_FOLDER, ASSEMBLY_GT_FOLDER_NAME)
ASSEMBLY_GT_IMAGE_PREDICTIONS_FILE = "image_label_predictions.csv"
ASSEMBLY_GT_CAPTION_SPLITTING_FILE = "caption_splitting_results.jsonl"
ASSEMBLY_GT_AUTOMATIC_ASSIGNMENT_FILE = "automatic_hierarchy_map.json"
ASSEMBLY_GT_LLM_REFINEMENT_FILE = "llm_refinement_results.json"
ASSEMBLY_GT_FINAL_HIERARCHY_FILE = "final_hierarchy.json"
ASSEMBLY_GT_PARQUET_FILE = "pubmed_ophtha_annotation.parquet"

PUBMED_OPHTHA_BATCHES_FOLDER = "pubmed_ophtha_batches"
PUBMED_OPHTHA_PARQUET_FILE = "pubmed_ophtha.parquet"
PUBMED_OPHTHA_GT_JSON_FILE = "pubmed_ophtha_annotation.json"

# Label Studio
LABEL_STUDIO_BASE_FOLDER_NAME = "labeling_data"
LABEL_STUDIO_BASE_FOLDER = os.path.join(DATASET_FOLDER, LABEL_STUDIO_BASE_FOLDER_NAME)
LABEL_STUDIO_ANNOTATION_PATH = "annotations"
LABEL_STUDIO_IMAGE_PATH = "images"

# Detectron dataset conversion
DETECTRON_CONVERTED_DATASET_PATH = "detectron"

# Ground truth annotation files
CAPTION_GT_FILE = "caption_annotations.json"
FIGURE_GT_FILE = "figure_annotations.json"
PANEL_GT_FILE = "panel_annotations.json"
IMAGE_GT_FILE = "image_annotations.json"
TRAIN_TEST_SPLIT_FILE = "train_test_split.json"


# Mark Status Classifier
MARK_STATUS_CLASSIFIER_CSV = "mark_status.csv"
