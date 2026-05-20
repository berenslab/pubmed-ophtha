"""
Module containing a custom trainer for Detectron2.

The custom trainer logs the loss on the test set and the COCO evaluation results
every `eval_period` iterations.

The code is based on https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py
"""

import datetime
import logging
import time
from collections import defaultdict
from typing import Any

import detectron2.utils.comm as comm
import numpy as np
import torch
from detectron2.config import CfgNode
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import (
    DefaultTrainer,
)
from detectron2.engine.hooks import HookBase
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator
from detectron2.utils.logger import log_every_n_seconds
from torch import nn
from torch.utils.data import DataLoader

from pubmed_ophtha.figure_splitting.detectron.transformations import (
    TransformDatasetMapper,
)


# Code from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py
class LossEvalHook(HookBase):
    """Detectron2 hook to evaluate the loss on the given dataset."""

    def __init__(self, eval_period: int, model: nn.Module, data_loader: DataLoader):
        """
        Initialize the LossEvalHook.

        Args:
            eval_period (int): Number of iterations between evaluations.
            model (nn.Module): Model to evaluate the loss on the dataset. Expected \
                behavior is that calling the model with a batch of data returns a \
                dictionary with loss values.
            data_loader (DataLoader): DataLoader to evaluate the loss on.

        """
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self) -> dict[str, list[float]]:
        """
        Evaluate the loss on the dataset.

        Returns:
            dict[str, list[float]]: Dictionary containing the loss values for each \
                loss type. The keys are the names of the losses, and the values are \
                lists of loss values for each batch in the dataset.

        """
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = defaultdict(list)
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (
                    time.perf_counter() - start_time
                ) / iters_after_start
                eta = datetime.timedelta(
                    seconds=int(total_seconds_per_img * (total - idx - 1))
                )
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            for k, v in loss_batch.items():
                losses[k].append(v)
        for k, v in losses.items():
            mean_loss = np.mean(v)
            self.trainer.storage.put_scalar(f"validation_{k}", mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data: dict[str, Any]) -> dict[str, float]:
        """
        Pass the data through the model and return the loss values.

        Args:
            data (dict[str, Any]): Input data for the model. Expected to be a batch \
                of size 1.

        Returns:
            dict[str, float]: Losses computed by the model. The keys are the \
                names of the losses, and the values are the loss values as floats.

        """
        with torch.no_grad():
            metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        metrics_dict["loss_total"] = sum(loss for loss in metrics_dict.values())
        return metrics_dict

    def after_step(self):
        """Evaluate the loss on the dataset after the backprop."""
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


# Code from https://gist.github.com/ortegatron/c0dad15e49c2b74de8bb09a5615d9f6b#file-lossevalhook-py
class DetectronTrainer(DefaultTrainer):
    """
    Custom trainer for the fundus labels dataset.

    Inherits from DefaultTrainer and overrides the build_evaluator method.
    Creates a COCOEvaluator for the fundus labels dataset.

    """

    @classmethod
    def build_evaluator(cls, cfg: CfgNode, dataset_name: str) -> DatasetEvaluator:
        """
        Build the evaluator for the given dataset.

        Args:
            cfg (CfgNode): Model configuration.
            dataset_name (str): Name of the dataset to evaluate.

        Returns:
            DatasetEvaluator: Evaluator for the dataset.

        """
        return COCOEvaluator(dataset_name, cfg, False, output_dir=cfg.OUTPUT_DIR)

    def build_hooks(self) -> list[HookBase]:
        """
        Build the hooks for the trainer.

        Returns:
            list[HookBase]: List of hooks to be used in the trainer.

        """
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,  # pyright: ignore[reportAttributeAccessIssue]
                build_detection_test_loader(  # pyright: ignore[reportOptionalCall,reportArgumentType]
                    self.cfg,
                    self.cfg.DATASETS.TEST[0],  # pyright: ignore[reportCallIssue]
                    DatasetMapper(self.cfg, True),  # pyright: ignore[reportCallIssue]
                ),
            ),
        )
        return hooks

    @classmethod
    def build_train_loader(cls, cfg: CfgNode) -> DataLoader:
        """
        Build the training data loader.

        Args:
            cfg (CfgNode): Model configuration.

        Returns:
            DataLoader: Training data loader.

        """
        mapper = None
        if cfg.INPUT.get("USE_TRANSFORMS", False):
            mapper = TransformDatasetMapper(cfg, True)  # pyright: ignore[reportCallIssue]
        return build_detection_train_loader(cfg, mapper=mapper)  # pyright: ignore[reportReturnType, reportCallIssue, reportOptionalCall]
