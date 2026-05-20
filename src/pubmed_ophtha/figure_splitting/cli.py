"""CLI interface for the figure splitting module."""

import json
import os

import click
from huggingface_hub import snapshot_download

from pubmed_ophtha.const.models import DEFAULT_MODEL_REPO_ID
from pubmed_ophtha.const.paths import (
    DATASET_FOLDER,
    PUBMED_OPHTHA_DATABASE_PATH,
    PUBMED_OPHTHA_DATASET_PATH,
)

try:
    from pubmed_ophtha.figure_splitting.label_prediction import (
        predict_bounding_box_labels,
    )
except ImportError:
    predict_bounding_box_labels = None


@click.group()
def cli():
    """
    Split multi-panel figures into individual subpanels.

    Requires the optional 'detection' extra to be installed. Install it with:
    pip install 'pubmed-ophtha[detection]'
    """
    pass


@cli.command()
@click.option(
    "--local-dir",
    default=".",
    help="Local directory to download model weights into. Default: current directory.",
    type=click.Path(),
)
def pull_models(local_dir):
    """Download model weights from HuggingFace Hub into local-dir/models/."""
    token = os.environ.get("HF_TOKEN")
    snapshot_download(repo_id=DEFAULT_MODEL_REPO_ID, local_dir=local_dir, token=token)
    click.echo(f"Models downloaded to {os.path.abspath(local_dir)}")


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
    "--num-workers",
    default=4,
    help="Number of worker processes to use for figure splitting. Default is 4.",
    type=int,
)
@click.option(
    "--model-args",
    default="{}",
    help=(
        "Additional keyword arguments forwarded to DetectronFigureSplitter as a JSON "
        "string. Can be used to specify model weights, config parameters, etc. Default "
        "is '{}'."
    ),
    type=str,
)
def split_figures(project_folder, num_workers, model_args_str):
    """Split the figures into their subpanels."""
    if predict_bounding_box_labels is None:
        raise click.ClickException(
            "The 'figure_splitting' subcommand requires the optional 'detection' extra."
            " Install it with: pip install 'pubmed-ophtha[detection]'"
        )
    # Replace single quotes with double quotes for JSON parsing
    model_args_str = model_args_str.replace("'", '"')
    model_args = json.loads(model_args_str)  # Parse the JSON string into a dictionary
    predict_bounding_box_labels(
        os.path.join(project_folder, PUBMED_OPHTHA_DATABASE_PATH),
        num_workers=num_workers,
        database_batch_size=100,
        **model_args,
    )
