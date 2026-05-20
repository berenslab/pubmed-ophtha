"""Module for registering datasets in Detectron2."""

from typing import Any, Callable

from detectron2.data import DatasetCatalog, MetadataCatalog

from pubmed_ophtha.util.registry import Registry

DETECTRON_DATASET_REGISTRY = Registry()


def register_dataset(
    dataset_base_name: str,
    loading_fn: Callable[[str], list[dict[str, Any]]],
    class_list: list[str],
):
    """
    Register a dataset in Detectron2's DatasetCatalog and MetadataCatalog.

    Args:
        dataset_base_name (str): Base name of the dataset.
        loading_fn (Callable[[str], list[dict[str, Any]]]): Function to load the dataset
            split.
        class_list (list[str]): List of class names.

    """
    train_dataset_name = f"{dataset_base_name}_train"
    test_dataset_name = f"{dataset_base_name}_test"

    def get_wrapped_loading_fn(split):
        def load_dataset_wrapper():
            return loading_fn(split=split)  # pyright: ignore[reportCallIssue]

        return load_dataset_wrapper

    DatasetCatalog.register(
        train_dataset_name,
        get_wrapped_loading_fn(split="train"),
    )
    DatasetCatalog.register(
        test_dataset_name,
        get_wrapped_loading_fn(split="test"),
    )
    MetadataCatalog.get(train_dataset_name).set(thing_classes=class_list)
    MetadataCatalog.get(test_dataset_name).set(thing_classes=class_list)
