"""CLI interface for detectron training."""

import logging
import os
from argparse import Namespace

import click
import torch
import wandb
from detectron2.engine import launch

from pubmed_ophtha.const.paths import (
    DETECTRON_CONVERTED_DATASET_PATH,
    IMAGE_GT_FILE,
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
    MARK_STATUS_CLASSIFIER_CSV,
)
from pubmed_ophtha.figure_splitting.detectron.datasets import add_local_datasets
from pubmed_ophtha.figure_splitting.detectron.mark_status_classifier import (
    create_dataloader,
    load_model,
)
from pubmed_ophtha.figure_splitting.detectron.mark_status_classifier import (
    train as train_mark_status_predictor,
)
from pubmed_ophtha.figure_splitting.detectron.train_detectron import (
    main as train_detectron_main,
)
from pubmed_ophtha.util.training import create_seed

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """
    Train Detectron2 models for figure splitting.

    Requires the optional 'detection' extra to be installed. Install it with:
    pip install 'pubmed-ophtha[detection]'
    """
    pass


@click.command()
@click.option(
    "--config-file",
    required=True,
    type=str,
    help="Path to the detectron2 config file.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Resume training from the last checkpoint.",
)
@click.option(
    "--num-gpus",
    default=1,
    type=int,
    help="Number of GPUs to use. Default: 1.",
)
@click.option(
    "--num-machines",
    default=1,
    type=int,
    help="Number of machines for distributed training. Default: 1.",
)
@click.option(
    "--machine-rank",
    default=0,
    type=int,
    help="Rank of this machine in distributed training. Default: 0.",
)
@click.option(
    "--dist-url",
    default="auto",
    type=str,
    help="URL for distributed training. Default: auto.",
)
@click.option(
    "--fold-index",
    default=-1,
    type=int,
    help="Index of the fold to use for training. Default: -1 (no folding).",
)
@click.argument("opts", nargs=-1)
def train_detectron(
    config_file,
    resume,
    num_gpus,
    num_machines,
    machine_rank,
    dist_url,
    fold_index,
    opts,
):
    """Train a Detectron2 model on the fundus labels dataset."""
    add_local_datasets()
    wandb.init(sync_tensorboard=True)

    parsed_args = Namespace(
        config_file=config_file,
        resume=resume,
        num_gpus=num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        fold_index=fold_index,
        opts=list(opts),
    )
    launch(
        train_detectron_main,
        num_gpus,
        num_machines=num_machines,
        machine_rank=machine_rank,
        dist_url=dist_url,
        args=(parsed_args,),
    )


@click.command()
@click.option(
    "--label-studio-root-folder",
    default=LABEL_STUDIO_BASE_FOLDER,
    help=(
        f"Root folder for the Label Studio data. Default: {LABEL_STUDIO_BASE_FOLDER}."
    ),
    type=str,
)
@click.option(
    "--seed",
    default=None,
    help=(
        "Seed for the dataset splitting and model training. "
        "If not provided, a random seed will be generated."
    ),
    type=int,
)
@click.option(
    "--batch-size",
    default=32,
    help="Batch size for training. Default: 32.",
    type=int,
)
@click.option(
    "--num-epochs",
    default=50,
    help="Number of epochs for training. Default: 50.",
    type=int,
)
@click.option(
    "--learning-rate",
    default=1e-4,
    help="Learning rate for training. Default: 1e-4.",
    type=float,
)
@click.option(
    "--num-workers",
    default=10,
    help="Number of workers for data loading. Default: 10.",
    type=int,
)
def train_annotation_classifier(
    label_studio_root_folder, seed, batch_size, num_epochs, learning_rate, num_workers
):
    """Train the annotation type classifier for figure splitting."""
    train_loader, test_loader, loss_weight = create_dataloader(
        annotation_file_path=os.path.join(
            label_studio_root_folder,
            LABEL_STUDIO_ANNOTATION_PATH,
            IMAGE_GT_FILE,
        ),
        image_data_csv=os.path.join(
            label_studio_root_folder,
            DETECTRON_CONVERTED_DATASET_PATH,
            MARK_STATUS_CLASSIFIER_CSV,
        ),
        batch_size=batch_size,
        image_base_path=os.path.join(label_studio_root_folder, LABEL_STUDIO_IMAGE_PATH),
        shuffle=True,
        num_workers=num_workers,
        dataset_seed=55555,
        use_oversampling=False,
        train_split_ratio=0.9,
    )
    model = load_model(num_classes=2)

    if seed is None:
        seed = create_seed()

    logger.info(f"Using seed: {seed}")

    torch.manual_seed(seed)

    train_mark_status_predictor(
        model,
        train_loader,
        test_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        output_path=f"models/mark_status_classifier_{seed}",
    )
