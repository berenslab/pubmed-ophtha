"""CLI interface for the panel assembly module."""

import logging
import os

import click

from pubmed_ophtha.const.paths import (
    DATASET_FOLDER,
    PUBMED_OPHTHA_DATABASE_PATH,
    PUBMED_OPHTHA_DATASET_PATH,
)
from pubmed_ophtha.panel_assembly.automatically_assign_panels import (
    run_automatic_assignment,
)
from pubmed_ophtha.panel_assembly.llm_refinement import (
    refine_partial_panel_assembly,
)

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """Assign captions to figure panels and refine matches using a VLM."""
    pass


@click.command()
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
    "--client-url-list",
    default="http://localhost:8000",
    help=(
        "Comma-separated list of client URLs for the model server. Default is 'http://localhost:8000'."
    ),
    type=str,
)
@click.option(
    "--num-workers",
    default=4,
    help="Number of worker processes to use for panel assembly. Default is 4.",
    type=int,
)
@click.option(
    "--num-runs",
    default=5,
    help="Number of refinement runs to perform. Default is 5.",
    type=int,
)
@click.option(
    "--num-concurrent-requests-per-worker",
    default=16,
    help=(
        "Number of concurrent requests each worker should send to the model server. "
        "Default is 16."
    ),
)
@click.option(
    "--num-retries",
    default=3,
    help=("Number of retries for failed requests to the model server. Default is 3."),
)
def assign_captions(
    project_folder,
    client_url_list,
    num_workers,
    num_runs,
    num_concurrent_requests_per_worker,
    num_retries,
):
    """Assign captions to figure panels using the specified model server and refine."""
    database_path = os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH)
    run_automatic_assignment(
        database_path=database_path,
        model_version="detectron_figure_splitter_v1",
        num_workers=num_workers,
        easy_ocr_estimated_model_size=5,  # in GB
    )

    for i in range(num_runs):
        logger.info(f"Starting refinement run {i + 1}/{num_runs}")
        refine_partial_panel_assembly(
            database_path=database_path,
            model_version="detectron_figure_splitter_v1",
            client_urls=client_url_list,
            num_workers=num_workers,
            database_batch_size=10,
            num_retries=num_retries,
            num_concurrent_requests_per_worker=num_concurrent_requests_per_worker,
        )
