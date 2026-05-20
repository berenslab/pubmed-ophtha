"""Define the CLI for the filling module."""

import click

from .fill_null import fill_missing_fields


@click.group()
def cli():
    """Fill missing subcaptions and images in the dataset."""
    pass


@cli.command()
@click.argument("dataset-path")
@click.option(
    "--num-caption-workers",
    default=None,
    help=(
        "Number of worker processes to use for caption filling. If not specified, will "
        "use the number of CPU cores."
    ),
    type=int,
)
@click.option(
    "--save-folder",
    default="./tmp_packages",
    help="Folder to save temporary files to. Will be deleted after use. Default is "
    "'./tmp_packages'.",
    type=str,
)
@click.option(
    "--output-dpi",
    default=None,
    help=(
        "DPI to save the filled images with. If not specified, will keep the original "
        "DPI."
    ),
    type=int,
)
@click.option(
    "--output-dataset-path",
    default=None,
    help=(
        "Path to save the filled dataset to. If not specified, will overwrite the "
        "original dataset."
    ),
    type=str,
)
def fill(
    dataset_path,
    num_caption_workers,
    save_folder,
    output_dpi,
    output_dataset_path,
):
    """Fill missing subcaptions and images in the dataset."""
    fill_missing_fields(
        dataset_path,
        num_caption_workers=num_caption_workers,
        save_folder=save_folder,
        output_dpi=output_dpi,
        output_dataset_path=output_dataset_path,
    )
