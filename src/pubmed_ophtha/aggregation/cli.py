"""CLI interface for the aggregation module."""

import os

import click

from pubmed_ophtha.aggregation.aggregate_into_parquet import (
    convert_annotations_to_parquet,
    convert_database_to_parquet,
    create_dataset_parquet_file,
)
from pubmed_ophtha.aggregation.ground_truth.convert_annotation_assembly import (
    run_gt_assembly,
)
from pubmed_ophtha.aggregation.ground_truth.convert_annotation_bounding_boxes import (
    predict_gt_image_labels,
)
from pubmed_ophtha.aggregation.ground_truth.convert_annotation_captions import (
    split_gt_captions,
)
from pubmed_ophtha.aggregation.ground_truth.convert_ground_truth_to_json import (
    convert_gt_to_json,
)
from pubmed_ophtha.const.paths import (
    ASSEMBLY_GT_FOLDER_NAME,
    DATASET_FOLDER,
    LABEL_STUDIO_BASE_FOLDER,
    PUBMED_OPHTHA_BATCHES_FOLDER,
    PUBMED_OPHTHA_DATABASE_PATH,
    PUBMED_OPHTHA_DATASET_PATH,
    PUBMED_OPHTHA_PARQUET_FILE,
)


@click.group()
def cli():
    """Aggregate annotations and export the dataset to Parquet."""
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
    "--num-concurrent-requests",
    default=20,
    help=("Number of concurrent requests to use for the LLM predictions. Default: 20."),
    type=int,
)
@click.option(
    "--num-retry",
    default=3,
    help=("Number of retries in case of connection issues with the LLM. Default: 3."),
    type=int,
)
@click.option(
    "--label-studio-root-folder",
    default=LABEL_STUDIO_BASE_FOLDER,
    help=(
        "Root folder of the Label Studio annotations. "
        f"Default: {LABEL_STUDIO_BASE_FOLDER}."
    ),
    type=str,
)
@click.option(
    "--caption-server-endpoint",
    default="http://localhost:8000",
    help=(
        "Bare base URL of the LLM server used for caption splitting; '/v1' is "
        "appended automatically. Default: http://localhost:8000."
    ),
    type=str,
)
@click.option(
    "--caption-server-api-key",
    default="test",
    help=("API key for the LLM client used for caption splitting. Default: 'test'."),
    type=str,
)
@click.option(
    "--panel-assembly-server-endpoint",
    default="http://localhost:8000",
    help=(
        "Bare base URL of the LLM server used for panel assembly; '/v1' is "
        "appended automatically. Default: http://localhost:8000."
    ),
    type=str,
)
@click.option(
    "--panel-assembly-server-api-key",
    default="test",
    help=("API key for the LLM client used for panel assembly. Default: 'test'."),
    type=str,
)
def convert_label_studio_annotations(
    project_folder,
    num_concurrent_requests,
    num_retry,
    label_studio_root_folder,
    caption_server_endpoint,
    caption_server_api_key,
    panel_assembly_server_endpoint,
    panel_assembly_server_api_key,
):
    """Convert the Label Studio annotations to hierarchy."""
    gt_assembly_folder = os.path.join(project_folder, ASSEMBLY_GT_FOLDER_NAME)

    # TODO could be concurrent
    predict_gt_image_labels(
        gt_assembly_folder=gt_assembly_folder,
        label_studio_root_folder=label_studio_root_folder,
    )

    # TODO could be concurrent
    split_gt_captions(
        gt_assembly_folder=gt_assembly_folder,
        num_concurrent_requests=num_concurrent_requests,
        num_retry=num_retry,
        label_studio_root_folder=label_studio_root_folder,
        server_endpoint=caption_server_endpoint,
        api_key=caption_server_api_key,
    )

    run_gt_assembly(
        server_endpoint=panel_assembly_server_endpoint,
        gt_assembly_folder=gt_assembly_folder,
        label_studio_root_folder=label_studio_root_folder,
        api_key=panel_assembly_server_api_key,
        num_concurrent_requests=num_concurrent_requests,
    )


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
    "--label-studio-root-folder",
    default=LABEL_STUDIO_BASE_FOLDER,
    help=(
        "Root folder of the Label Studio annotations. "
        f"Default: {LABEL_STUDIO_BASE_FOLDER}."
    ),
    type=str,
)
@click.option(
    "--num-workers",
    default=4,
    help=("Number of worker processes to use for panel assembly. Default is 4."),
    type=int,
)
@click.option(
    "--parquet-batch-size",
    default=1000,
    help=(
        "Batch size to use when converting the database to parquet. Default is 1000."
    ),
    type=int,
)
def aggregate_into_final_dataset(
    project_folder,
    label_studio_root_folder,
    num_workers,
    parquet_batch_size,
):
    """
    Aggregate the final dataset.

    Convert the ground truth assembly data and processed database data into parquet
    format and create the final dataset parquet file by joining the GT data with the
    database data.
    Also convert the GT data into a JSON file for easier access to the GT annotations.
    """
    database_path = os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH)
    gt_assembly_folder = os.path.join(project_folder, ASSEMBLY_GT_FOLDER_NAME)
    pmo_batch_folder = os.path.join(project_folder, PUBMED_OPHTHA_BATCHES_FOLDER)
    output_path = os.path.join(project_folder, PUBMED_OPHTHA_PARQUET_FILE)

    convert_database_to_parquet(
        database_path,
        pmo_batch_folder,
        num_workers=num_workers,
        batch_size=parquet_batch_size,
    )

    convert_annotations_to_parquet(
        database_path,
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
