"""CLI for running the full data pipeline end-to-end."""

import json
import logging
import os

import click
from huggingface_hub import login, snapshot_download

from pubmed_ophtha.aggregation.aggregate_into_parquet import (
    convert_annotations_to_parquet,
    convert_database_to_parquet,
    create_dataset_parquet_file,
)
from pubmed_ophtha.aggregation.ground_truth.convert_ground_truth_to_json import (
    convert_gt_to_json,
)
from pubmed_ophtha.caption_splitting.split_captions_sqlite import run_caption_splitting
from pubmed_ophtha.const.models import DEFAULT_MODEL_REPO_ID
from pubmed_ophtha.const.paths import (
    ASSEMBLY_GT_FOLDER_NAME,
    BIOMEDICA_FILTERED_DATASET_PATH,
    BIOMEDICA_FILTERED_JOINED_DATASET_FILE,
    BIOMEDICA_FILTERED_WITH_FILE_LIST_DATASET_PATH,
    BIOMEDICA_META_DATASET_PATH,
    DATASET_FOLDER,
    PUBMED_OPHTHA_BATCHES_FOLDER,
    PUBMED_OPHTHA_DATABASE_PATH,
    PUBMED_OPHTHA_DATASET_PATH,
    PUBMED_OPHTHA_PACKAGES_PATH,
    PUBMED_OPHTHA_PARQUET_FILE,
)
from pubmed_ophtha.const.urls import PUBMED_CENTRAL_OA_FILE_LIST_NAME
from pubmed_ophtha.filtering.download_biomedica import save_biomedica_meta_to_parquet
from pubmed_ophtha.filtering.filter_biomedica import (
    convert_biomedica_to_table,
    filter_biomedica,
    join_filtered_files,
    join_with_file_list,
)
from pubmed_ophtha.filtering.post_processing_sqlite import postprocess_retrieved_images
from pubmed_ophtha.filtering.retrieve_original_images_sqlite import (
    extract_original_relevant_figures as extract_original_relevant_figures_sqlite,
)
from pubmed_ophtha.filtering.retrieve_original_images_sqlite import (
    fetch_relevant_pmc_articles as fetch_relevant_pmc_articles_sqlite,
)
from pubmed_ophtha.panel_assembly.automatically_assign_panels import (
    run_automatic_assignment,
)
from pubmed_ophtha.panel_assembly.llm_refinement import refine_partial_panel_assembly

try:
    from pubmed_ophtha.figure_splitting.label_prediction import (
        predict_bounding_box_labels,
    )
except ImportError:
    predict_bounding_box_labels = None

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Run the full data processing pipeline."""
    pass


@cli.command()
@click.option(
    "--project-folder",
    default=os.path.join(DATASET_FOLDER, PUBMED_OPHTHA_DATASET_PATH),
    help=(
        "Project folder containing the dataset. Default: "
        f"{os.path.join(DATASET_FOLDER, PUBMED_OPHTHA_DATASET_PATH)}."
    ),
    type=str,
)
@click.option(
    "--temp-save-interval",
    default=50000,
    help="Number of files to save in one Parquet table. Default: 50000.",
    type=int,
)
@click.option(
    "--max-files",
    default=None,
    help="Number of maximum files to download. Default: download all.",
    type=int,
)
@click.option(
    "--download-tries",
    default=5,
    help="Number of tries to download the PMC articles. Default is 5.",
    type=int,
)
@click.option(
    "--num-pdf-workers",
    default=None,
    help="Number of workers for PDF processing. Default: balance with num-workers.",
    type=int,
)
@click.option(
    "--local-dir",
    default=".",
    help="Local directory to download model weights into. Default: current directory.",
    type=click.Path(),
)
@click.option(
    "--num-workers",
    default=4,
    help=(
        "Number of worker processes for figure splitting, panel assembly, and "
        "aggregation. Default is 4."
    ),
    type=int,
)
@click.option(
    "--model-args",
    default="{}",
    help=(
        "Additional keyword arguments forwarded to DetectronFigureSplitter as a JSON "
        "string. Default is '{}'."
    ),
    type=str,
)
@click.option(
    "--server-address",
    default="http://localhost:8000",
    help=(
        "Bare base URL of the model server for caption splitting; '/v1' is "
        "appended automatically. Default: 'http://localhost:8000'."
    ),
    type=str,
)
@click.option(
    "--api-key",
    default="test",
    help=(
        "API key for the caption splitting model server. "
        "Local vLLM servers do not validate the key but require it to be non-empty. "
        "Default: 'test'."
    ),
    type=str,
)
@click.option(
    "--client-url-list",
    default="http://localhost:8000",
    help=(
        "Comma-separated list of bare base URLs for the panel assembly model "
        "server; '/v1' is appended automatically. "
        "Default: 'http://localhost:8000'."
    ),
    type=str,
)
@click.option(
    "--num-runs",
    default=5,
    help="Number of panel assembly refinement runs. Default is 5.",
    type=int,
)
@click.option(
    "--num-concurrent-requests-per-worker",
    default=16,
    help="Concurrent requests each worker sends to the model server. Default is 16.",
)
@click.option(
    "--num-retries",
    default=3,
    help="Retries for failed requests to the model server. Default is 3.",
)
@click.option(
    "--label-studio-root-folder",
    default=None,
    help="Root folder of the Label Studio annotations. Default: None.",
    type=str,
)
@click.option(
    "--parquet-batch-size",
    default=1000,
    help="Batch size when converting the database to Parquet. Default is 1000.",
    type=int,
)
def run(
    project_folder,
    temp_save_interval,
    max_files,
    download_tries,
    num_pdf_workers,
    local_dir,
    num_workers,
    model_args,
    server_address,
    api_key,
    client_url_list,
    num_runs,
    num_concurrent_requests_per_worker,
    num_retries,
    label_studio_root_folder,
    parquet_batch_size,
):
    """Run the full data processing pipeline end-to-end."""
    if predict_bounding_box_labels is None:
        raise click.ClickException(
            "Step 4 requires the optional 'detection' extra. "
            "Install it with: pip install 'pubmed-ophtha[detection]'"
        )

    model_args_str = model_args.replace("'", '"')
    try:
        model_args_dict = json.loads(model_args_str)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(
            f"--model-args must be a valid JSON object, got: {model_args!r} "
            f"({exc.msg})",
            param_hint="--model-args",
        ) from exc

    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)

    db_path = os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH)

    # Step 1: Download and filter Biomedica
    logger.info("Step 1/7: Downloading and filtering Biomedica dataset")
    biomedica_output_path = os.path.join(project_folder, BIOMEDICA_META_DATASET_PATH)
    filtered_output_path = os.path.join(project_folder, BIOMEDICA_FILTERED_DATASET_PATH)
    joined_output_file = os.path.join(
        project_folder, BIOMEDICA_FILTERED_JOINED_DATASET_FILE
    )
    merged_output_file = os.path.join(
        project_folder, BIOMEDICA_FILTERED_WITH_FILE_LIST_DATASET_PATH
    )
    file_list_path = os.path.join(
        project_folder, PUBMED_OPHTHA_PACKAGES_PATH, PUBMED_CENTRAL_OA_FILE_LIST_NAME
    )
    save_biomedica_meta_to_parquet(
        biomedica_output_path, temp_save_interval=temp_save_interval
    )
    filter_biomedica(biomedica_output_path, filtered_output_path, clean=False)
    join_filtered_files(filtered_output_path, joined_output_file)
    join_with_file_list(joined_output_file, file_list_path, merged_output_file)
    convert_biomedica_to_table(db_path, merged_output_file)

    # Step 2: Retrieve original images
    logger.info("Step 2/7: Retrieving original images from PMC articles")
    attempt_download = True
    try_number = 0
    while attempt_download and download_tries > try_number:
        logger.info(f"Attempt {try_number + 1} to download files!")
        attempt_download = False
        results = fetch_relevant_pmc_articles_sqlite(db_path, max_files=max_files)
        if not results:
            attempt_download = True
        try_number += 1
    if attempt_download:
        logger.warning("Could not download all files!")
    extract_original_relevant_figures_sqlite(
        db_path, num_pdf_workers=num_pdf_workers, num_workers=num_workers
    )
    postprocess_retrieved_images(db_path)

    # Step 3: Pull model weights
    logger.info("Step 3/7: Downloading model weights from HuggingFace")
    weights_missing = not os.path.isdir(local_dir) or not os.listdir(local_dir)
    if len(model_args_dict) == 0 and weights_missing:
        logger.info(
            "No additional model args provided and no weights found in local dir, "
            "proceeding to download model weights.",
        )
        snapshot_download(
            repo_id=DEFAULT_MODEL_REPO_ID, local_dir=local_dir, token=hf_token
        )

    # Step 4: Split figures into subpanels
    logger.info("Step 4/7: Splitting figures into subpanels")
    predict_bounding_box_labels(
        db_path,
        num_workers=num_workers,
        database_batch_size=100,
        **model_args_dict,
    )

    # Step 5: Split captions
    logger.info("Step 5/7: Splitting captions")
    run_caption_splitting(
        db_path,
        "Qwen/Qwen3-32B-AWQ",
        server_address,
        api_key=api_key,
    )

    # Step 6: Assign captions to panels
    logger.info("Step 6/7: Assigning captions to panels")
    run_automatic_assignment(
        database_path=db_path,
        model_version="detectron_figure_splitter_v1",
        num_workers=num_workers,
        easy_ocr_estimated_model_size=5,
    )
    client_urls = client_url_list.split(",")
    for i in range(num_runs):
        logger.info(f"Starting refinement run {i + 1}/{num_runs}")
        refine_partial_panel_assembly(
            database_path=db_path,
            model_version="detectron_figure_splitter_v1",
            client_urls=client_urls,
            num_workers=num_workers,
            database_batch_size=10,
            num_retries=num_retries,
            num_concurrent_requests_per_worker=num_concurrent_requests_per_worker,
        )

    # Step 7: Aggregate into final dataset
    logger.info("Step 7/7: Aggregating into final dataset")
    gt_assembly_folder = os.path.join(project_folder, ASSEMBLY_GT_FOLDER_NAME)
    pmo_batch_folder = os.path.join(project_folder, PUBMED_OPHTHA_BATCHES_FOLDER)
    output_path = os.path.join(project_folder, PUBMED_OPHTHA_PARQUET_FILE)
    convert_database_to_parquet(
        db_path,
        pmo_batch_folder,
        num_workers=num_workers,
        batch_size=parquet_batch_size,
    )
    if label_studio_root_folder is not None:
        convert_annotations_to_parquet(
            db_path,
            gt_assembly_folder=gt_assembly_folder,
            label_studio_root_folder=label_studio_root_folder,
        )
        create_dataset_parquet_file(
            gt_assembly_folder=gt_assembly_folder,
            database_parquet_folder=pmo_batch_folder,
            output_path=output_path,
            label_studio_root_path=label_studio_root_folder,
        )
        convert_gt_to_json(
            project_folder=project_folder,
            label_studio_root_folder=label_studio_root_folder,
        )
    else:
        logger.warning(
            "Label Studio root folder not provided. Skipping annotation conversion and "
            "final dataset creation with annotations."
        )

        create_dataset_parquet_file(
            gt_assembly_folder=None,
            database_parquet_folder=pmo_batch_folder,
            output_path=output_path,
            label_studio_root_path=None,
        )
