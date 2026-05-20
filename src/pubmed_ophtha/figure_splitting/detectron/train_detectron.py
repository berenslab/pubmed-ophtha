"""Train a Detectron2 model on the fundus labels dataset."""

import logging
from argparse import Namespace

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.engine import (
    default_setup,
)
from detectron2.utils.comm import is_main_process

from pubmed_ophtha.util.training import create_seed

from .detectron_trainer import (
    DetectronTrainer,
)

logger = logging.getLogger(__name__)


def main(parsed_args: Namespace) -> dict:
    """
    Train a Detectron2 model on the fundus labels dataset.

    Args:
        parsed_args (Namespace): Arguments from argparse.

    Returns:
        dict: Training results.

    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    cfg.merge_from_file(parsed_args.config_file)
    cfg.merge_from_list(parsed_args.opts)

    if is_main_process():
        if cfg.get("SEED", None) is None:
            cfg.SEED = create_seed()
            logger.info(f"Using random seed {cfg.SEED}.")

        # Setup k_fold

        if parsed_args.fold_index >= 0:
            fold_index = parsed_args.fold_index
        else:
            fold_index = -1
        seed = cfg.SEED
    else:
        seed = None
        fold_index = -1
    # Broadcast the seed to all processes
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        seed_list = [seed]
        fold_index_list = [fold_index]

        torch.distributed.broadcast_object_list(seed_list, src=0)
        torch.distributed.broadcast_object_list(fold_index_list, src=0)
        seed = seed_list[0]
        fold_index = fold_index_list[0]

    if fold_index >= 0:
        cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}_fold_{fold_index}"
        # Update datasets
        new_train_datasets = []
        for dataset in cfg.DATASETS.TRAIN:
            if dataset.endswith("_train"):
                new_dataset_name = (
                    f"{dataset.removesuffix('_train')}_fold_{fold_index}_train"
                )
            else:
                new_dataset_name = f"{dataset}_fold_{fold_index}"

            if new_dataset_name not in DatasetCatalog.list():
                new_dataset_name = dataset  # Fallback to original if not found
            new_train_datasets.append(new_dataset_name)
        cfg.DATASETS.TRAIN = tuple(new_train_datasets)

        # Same for test datasets
        new_test_datasets = []
        for dataset in cfg.DATASETS.TEST:
            if dataset.endswith("_test"):
                new_dataset_name = (
                    f"{dataset.removesuffix('_test')}_fold_{fold_index}_test"
                )
            else:
                new_dataset_name = f"{dataset}_fold_{fold_index}"

            if new_dataset_name not in DatasetCatalog.list():
                new_dataset_name = dataset  # Fallback to original if not found
            new_test_datasets.append(new_dataset_name)
        cfg.DATASETS.TEST = tuple(new_test_datasets)

    cfg.SEED = seed
    cfg.OUTPUT_DIR = f"{cfg.OUTPUT_DIR}_{seed}"

    cfg.DATASETS.VALIDATION = ""
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = False
    cfg.freeze()
    default_setup(cfg, parsed_args)

    # Use the custom trainer
    trainer = DetectronTrainer(cfg)
    trainer.resume_or_load(resume=parsed_args.resume)
    return trainer.train()  # pyright: ignore[reportReturnType]
