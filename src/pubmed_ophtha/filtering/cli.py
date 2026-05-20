"""Define the CLI for the scraping module."""

import logging
import os

import click
from huggingface_hub import login

from pubmed_ophtha.const.paths import (
    BIOMEDICA_FILTERED_DATASET_PATH,
    BIOMEDICA_FILTERED_JOINED_DATASET_FILE,
    BIOMEDICA_FILTERED_WITH_FILE_LIST_DATASET_PATH,
    BIOMEDICA_META_DATASET_PATH,
    DATASET_FOLDER,
    PUBMED_OPHTHA_DATABASE_PATH,
    PUBMED_OPHTHA_DATASET_PATH,
    PUBMED_OPHTHA_PACKAGES_PATH,
)
from pubmed_ophtha.const.urls import PUBMED_CENTRAL_OA_FILE_LIST_NAME
from pubmed_ophtha.filtering.download_biomedica import save_biomedica_meta_to_parquet
from pubmed_ophtha.filtering.filter_biomedica import (
    convert_biomedica_to_table,
    filter_biomedica,
    join_filtered_files,
    join_with_file_list,
)
from pubmed_ophtha.filtering.post_processing_sqlite import (
    postprocess_retrieved_images,
)
from pubmed_ophtha.filtering.retrieve_original_images_sqlite import (
    extract_original_relevant_figures as extract_original_relevant_figures_sqlite,
)
from pubmed_ophtha.filtering.retrieve_original_images_sqlite import (
    fetch_relevant_pmc_articles as fetch_relevant_pmc_articles_sqlite,
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Download and filter the Biomedica dataset and extract figures from PDFs."""
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
def download_biomedica(
    project_folder,
    temp_save_interval,
):
    """Download and filter the Biomedica dataset without figures."""
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(HF_TOKEN)

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
    convert_biomedica_to_table(
        os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH),
        merged_output_file,
    )


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
    "--max-files",
    default=None,
    help="Number of maximum files to download. Default: download all.",
    type=int,
)
@click.option(
    "--download-tries",
    default=5,
    help="Number of tries to download the pmc articles. Default is 5.",
    type=int,
)
@click.option(
    "--num-pdf-workers",
    default=None,
    help="Number of workers to use for PDF processing. "
    + "Default: Balance number of cpus with num-workers.",
    type=int,
)
@click.option(
    "--num-workers",
    default=None,
    help="Number of workers to use for figure extraction. "
    + "Default: Balance number of cpus with num-pdf-workers.",
    type=int,
)
def retrieve_original_images(
    project_folder,
    max_files,
    download_tries,
    num_pdf_workers,
    num_workers,
):
    """Extract the original figures from the article PDF files."""
    db_path = os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH)
    attempt_download = True
    try_number = 0

    while attempt_download and download_tries > try_number:
        logger.info(f"Attempt {try_number + 1} to download files!")
        attempt_download = False
        results = fetch_relevant_pmc_articles_sqlite(
            db_path,
            max_files=max_files,
        )

        if not results:
            attempt_download = True

        try_number += 1

    if attempt_download:
        logger.warning("Could not download all files!")

    extract_original_relevant_figures_sqlite(
        db_path,
        num_pdf_workers=num_pdf_workers,
        num_workers=num_workers,
    )

    postprocess_retrieved_images(db_path)
