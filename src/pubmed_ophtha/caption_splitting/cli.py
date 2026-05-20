"""CLI interface for the caption splitting module."""

import os

import click

from pubmed_ophtha.const.paths import (
    DATASET_FOLDER,
    PUBMED_OPHTHA_DATABASE_PATH,
    PUBMED_OPHTHA_DATASET_PATH,
)

from .split_captions_sqlite import run_caption_splitting


@click.group()
def cli():
    """Split figure captions into per-panel subcaptions."""
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
    "--server-address",
    default="http://localhost:8000",
    help=(
        "Bare base URL of the model server; '/v1' is appended automatically. "
        "Default is 'http://localhost:8000'."
    ),
    type=str,
)
@click.option(
    "--api-key",
    default="test",
    help=(
        "The API key for the model server. Local vLLM servers do not validate the key "
        "but require it to be non-empty. Default is 'test'."
    ),
    type=str,
)
def split_captions(project_folder, server_address, api_key):
    """Split captions using the specified model server."""
    run_caption_splitting(
        os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH),
        "Qwen/Qwen3-32B-AWQ",
        server_address,
        api_key=api_key,
    )
