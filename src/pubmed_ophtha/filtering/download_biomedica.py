"""Module to download the BIOMEDICA dataset to parquet files."""

import gc
import logging
import math
import os
from collections import defaultdict
from collections.abc import Mapping

import polars as pl
import webdataset as wds
from tqdm.auto import tqdm

# Maximum package size for the splits
SPLIT_BOUNDS = {
    "other": (0, 115),
    "commercial": (0, 1862),
    "noncommercial": (0, 560),
}

logger = logging.getLogger(__name__)


def get_split_filename_patterns(max_files_per_package: int = 50) -> dict[str, str]:
    """
    Generate the filename patterns for the BIOMEDICA dataset splits.

    Args:
        max_files_per_package (int, optional): Number of files per split. \
            Defaults to 50.

    Returns:
        dict[str, str]: Dictionary mapping split names to their corresponding \
            file patterns.

    """
    split_file_names = {}

    for split, (start, stop) in SPLIT_BOUNDS.items():
        num_packages = math.ceil(stop / max_files_per_package)

        for package_index in range(num_packages):
            end_index = (
                min(start + (package_index + 1) * max_files_per_package, stop) - 1
            )
            split_file_names[f"{split}_{package_index}"] = (
                "{"
                f"{start + package_index * max_files_per_package:06}"
                f"..{end_index:06}"
                "}.tar"
            )  # end index is inclusive

    return split_file_names


def save_parquet(
    data: Mapping[str, str | list[str | int | list[str | int]]], path: str
):
    """
    Save the data to a parquet file using polars.

    Args:
        data (Mapping[str, str | list[str | int | list[str | int]]]): Data in a
            polars-readable format.
        path (str): Path to save the parquet file to.

    """
    df = pl.DataFrame(data)

    df.write_parquet(path)
    del df


def save_split(  # noqa: DOC201
    dataset: wds.compat.WebDataset,
    output_path: str,
    temp_save_interval: int = 1000,
    file_estimate: int | None = None,
):
    """
    Save the given BIOMEDICA split to a parquet file.

    Args:
        dataset (wds.WebDataset): Webdataset to download.
        output_path (str): Filename to save the parquet file to.
        temp_save_interval (int, optional): Interval for temporary file saves. \
            Defaults to 1000.
        file_estimate (int | None, optional): Estimated total number of files for the \
            progress bar. Defaults to None.

    Warning:
        **Silent no-op if output already exists.** If ``output_path`` is an
        existing file the function returns immediately without downloading
        anything and without notifying the caller (no return value, no
        exception, no log message). This makes re-runs safe but also means a
        caller cannot distinguish "just finished" from "did nothing because the
        file was already there".

        **In-place mutation of the resume list.** When a ``_tmp.parquet``
        checkpoint file is found, the function loads the already-processed keys
        into an internal ``parsed_lines`` list and removes each key from that
        list as it is re-encountered in the dataset stream
        (``parsed_lines.remove(file_key)``). This mutates the list throughout
        the download loop as a way of tracking which checkpoint entries have
        been seen again. The list is local to the function, but the mutation
        pattern means the in-memory skip set shrinks as the stream is consumed,
        which is intentional but non-obvious.

    """
    dataset_dict: dict[str, list[str | int | list[str | int]]] = defaultdict(list)

    parsed_lines = []
    if os.path.exists(output_path):
        return
    if os.path.exists(output_path.replace(".parquet", "_tmp.parquet")):
        parquet_df = pl.read_parquet(output_path.replace(".parquet", "_tmp.parquet"))
        parsed_lines = parquet_df["file_key"].to_list()

        dataset_dict = {k: parquet_df[k].to_list() for k in parquet_df.columns}
        del parquet_df

    for index, data in enumerate(
        tqdm(dataset, desc="Processing data", total=file_estimate)
    ):
        file_key = data["__key__"]
        if len(parsed_lines) > 0 and file_key in parsed_lines:
            parsed_lines.remove(file_key)
            continue

        dataset_dict["iteration"].append(index)
        dataset_dict["file_key"].append(file_key)
        dataset_dict["image_cluster_id"].append(data["json"]["image_cluster_id"])
        dataset_dict["image_label_id"].append(data["json"]["image_label_id"])
        dataset_dict["image_hash"].append(data["json"]["image_hash"])
        dataset_dict["image_caption"].append(data["txt"])
        dataset_dict["image_panel_type"].append(data["json"]["image_panel_type"])
        dataset_dict["image_panel_subtype"].append(data["json"]["image_panel_subtype"])
        dataset_dict["image_primary_label"].append(data["json"]["image_primary_label"])
        dataset_dict["image_secondary_label"].append(
            data["json"]["image_secondary_label"]
        )
        dataset_dict["image_context"].append(
            data["json"]["image_context"].get(dataset_dict["image_cluster_id"][-1], [])
        )
        dataset_dict["article_id"].append(data["json"]["article_accession_id"])
        dataset_dict["article_title"].append(data["json"]["article_title"])
        dataset_dict["journal_title"].append(data["json"]["article_journal"])
        dataset_dict["article_date"].append(data["json"]["article_date"])
        dataset_dict["license_type"].append(data["json"]["article_license"])

        article_subject = data["json"]["article_subject"]

        if article_subject is not None and isinstance(article_subject, list):
            # Sometimes there is a sublist -> probably error, remove
            sublist_indices = [
                i
                for i, x in enumerate(article_subject)
                if not isinstance(x, str)
                and not isinstance(x, int)
                and not isinstance(x, float)
            ][::-1]
            for i in sublist_indices:
                article_subject.pop(i)

        dataset_dict["article_subject"].append(article_subject)  # pyright: ignore[reportArgumentType]

        if index % temp_save_interval == 0:
            save_parquet(dataset_dict, output_path.replace(".parquet", "_tmp.parquet"))

    save_parquet(dataset_dict, output_path)
    os.remove(output_path.replace(".parquet", "_tmp.parquet"))


def save_biomedica_meta_to_parquet(output_path: str, temp_save_interval: int = 1000):
    """
    Download and saves the article metadata from the BIOMEDICA dataset to parquet files.

    The dataset is split up into parquet files containing a maximum of 50 webfiles.
    Requires the HF_TOKEN environment variable to be set.
    While downloading a temporary file is saved every `temp_save_interval` files.

    Args:
        output_path (str): Folder to save the parquet files to.
        temp_save_interval (int, optional):  Interval for temporary file saves. \
            Defaults to 1000.

    Raises:
        ValueError: If the HF_TOKEN environment variable is not set.

    """
    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise ValueError(
            "HF_TOKEN environment variable is not set. Please set it to your "
            "Hugging Face API token."
        )

    os.makedirs(output_path, exist_ok=True)

    split_to_filenames = get_split_filename_patterns(max_files_per_package=50)

    # TODO paralellize
    for index, (split_name, file_names) in enumerate(split_to_filenames.items()):
        gc.collect()
        base_split_name = split_name.split("_")[0]
        logger.info(
            f"Processing split ({index + 1}/{len(split_to_filenames)}): {split_name}."
        )
        dataset_url = (
            "pipe:curl -s -L "
            "https://huggingface.co/datasets/BIOMEDICA/"
            f"biomedica_webdataset_24M/resolve/main/{base_split_name}/{file_names}"
        )

        retrieve_cmd = f"{dataset_url} -H 'Authorization:Bearer {hf_token}'"

        dataset = wds.compat.WebDataset(retrieve_cmd, shardshuffle=False).decode("pil")
        save_split(
            dataset,  # pyright: ignore[reportArgumentType]
            os.path.join(output_path, f"biomedica_meta_{split_name}.parquet"),
            file_estimate=(
                int(
                    file_names.replace("{", "")
                    .replace("}", "")
                    .replace(".tar", "")
                    .split("..")[-1]
                )
                - int(
                    file_names.replace("{", "")
                    .replace("}", "")
                    .replace(".tar", "")
                    .split("..")[0]
                )
                + 1
            )
            * 10000,
            temp_save_interval=temp_save_interval,
        )
