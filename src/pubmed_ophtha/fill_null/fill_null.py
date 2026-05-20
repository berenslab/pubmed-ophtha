"""Module for filling missing images and subcaptions in the dataset."""

import glob
import json
import logging
import math
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import webdataset as wds
from PIL import Image
from pmo_parser.bounding_boxes import BBox
from pmo_parser.renderer import render_page
from tqdm.auto import tqdm

from pubmed_ophtha.filtering.download_biomedica import get_split_filename_patterns
from pubmed_ophtha.filtering.retrieve_original_images import (
    ExtractionError,
    MissingPDFError,
    MissingXMLFileError,
    get_data_from_package,
)
from pubmed_ophtha.scraping.download_files import download_packages_from_pubmed_central
from pubmed_ophtha.util.computing import get_cpu_count

logger = logging.getLogger(__name__)


def render_figure(
    article_pdf: BytesIO | str,
    figure_boxes: dict[str | int, dict[str, float]],
    rendering_dpi: int,
    output_dpi: int | None = None,
    skip_buffer: bool = False,
    align_center: bool = False,
) -> Image.Image:
    """
    Render figure from pdf using the figure boxes in page space.

    Each figure box corresponds to one page and contains the coordinates of the figure
    in page space. The figure is rendered by cropping the corresponding area from each
    page and concatenating the crops vertically.

    The figure is rendered at the output DPI if specified, otherwise it is rendered at
    the original rendering DPI. The buffer between the cropped pages can be skipped and
    the pages can be aligned by centering them using the skip_buffer and align_center
    flags respectively.

    Args:
        article_pdf (BytesIO | str): Bytes of the article PDF or path to the article
            PDF.
        figure_boxes (dict[str  |  int, dict[str, float]]): Dictionary mapping page
            numbers to figure box coordinates in page space.
        rendering_dpi (int): DPI the figure was originally rendered at. This is used to
            scale the figure to the output DPI if specified.
        output_dpi (int | None, optional): DPI the figure should be rendered at now. If
            None use rendering_dpi. Defaults to None.
        skip_buffer (bool, optional): If True skips the buffer area between pages in
            final image. Defaults to False.
        align_center (bool, optional): If True the page images are aligned by centering
            them. If False the page images left aligned. Defaults to False.

    Returns:
        Image.Image: Rendered figure as a PIL Image.

    """
    if output_dpi is None:
        output_dpi = rendering_dpi
    dpi_scale = output_dpi / rendering_dpi
    rendered_pages = {}

    image_width = 0
    image_height = 0

    for page, figure_data in figure_boxes.items():
        int_page = int(page)
        rendered_pages[page] = render_page(
            article_pdf,
            int_page,
            bbox=BBox(
                x0=figure_data["x0"],
                y0=figure_data["y0"],
                x1=figure_data["x1"],
                y1=figure_data["y1"],
            ),
            dpi=output_dpi,
        )

        image_width = max(image_width, rendered_pages[page].width)
        image_height = image_height + rendered_pages[page].height

    # Buffer distance needs to be scaled with dpi to ensure the image stays the same
    buffer_distance = max(
        int(0.02 * max(image_width, image_height)), int(round(10 * dpi_scale))
    )

    if skip_buffer:
        buffer_distance = 0

    figure_image = Image.new(
        "RGB",
        (
            image_width,
            image_height + buffer_distance * (len(figure_boxes) - 1),
        ),
        (255, 255, 255),
    )

    next_y = 0
    for i, page in enumerate(sorted(figure_boxes.keys())):
        x0 = 0
        if align_center:
            x0 = math.ceil((image_width - rendered_pages[page].width) / 2)

        figure_image.paste(rendered_pages[page], (x0, next_y))
        next_y += rendered_pages[page].height + buffer_distance

    return figure_image


def reconstruct_from_segments(
    segments: list[tuple[str] | tuple[int, int]], full_caption: str
) -> str:
    """
    Reconstructs the subcaption using the subcaption segments.

    Args:
        segments (list[tuple[str]  |  tuple[int, int]]): Ordered list of segments that
            can be either text segments (tuple[str]) or reference segments
            (tuple[int, int]) that specify the start and length of the referenced text
            in the full caption. The order is never validated — if reference
            segments are out of order the function produces silently wrong output
            (character slices into ``full_caption`` will be taken from the wrong
            positions) without raising an error.
        full_caption (str): Full caption text from which the referenced text segments
            can be extracted.

    Returns:
        str: Reconstructed subcaption.

    """
    parts = []
    for seg in segments:
        if len(seg) == 2 and isinstance(seg[0], int):
            start, length = seg
            parts.append(full_caption[start : start + length])
        else:
            parts.append(seg[0])
    return "".join(parts)


def fill_image_data(
    article_id: str,
    image_cluster_id: str,
    figure_position: dict[str | int, dict[str, float]] | None,
    original_dpi: int,
    article_package_path: str,
    output_dpi: int | None = None,
    use_centered_alignment: bool = False,
) -> Image.Image | None:
    """
    Get the rendered figure for the specified article and image cluster.

    Args:
        article_id (str): PMC article ID.
        image_cluster_id (str): Figure image cluster ID.
        figure_position (dict[str  |  int, dict[str, float]] | None): Position of the
            figure in page space. If None, the figure cannot be rendered and None is
            returned.
        original_dpi (int): DPI the figure was originally rendered at.
            This is used to scale the figure to the output DPI if specified.
        article_package_path (str): Path to the article package containing the PDF to
            render the figure from.
        output_dpi (int | None, optional): DPI the figure will be rendered at. Defaults
            to None.
        use_centered_alignment (bool, optional): If True, rendering skips the buffer
            and uses center alignment. Defaults to False.

    Returns:
        Image.Image | None: The rendered figure as a PIL Image, or None if the figure
            could not be rendered.

    """
    if figure_position is None:
        logger.warning(
            f"No figure box found for article {article_id} and image cluster "
            f"{image_cluster_id}"
        )
        return None

    try:
        _, article_pdf = get_data_from_package(article_package_path)
    except (ExtractionError, MissingPDFError, MissingXMLFileError) as e:
        logger.warning(f"Could not retrieve data for article {article_id}: {e}")
        return None
    try:
        rendered_figure = render_figure(
            article_pdf,
            figure_position,
            rendering_dpi=original_dpi,
            output_dpi=output_dpi,
            skip_buffer=use_centered_alignment,
            align_center=use_centered_alignment,
        )
        return rendered_figure
    except Exception as e:
        logger.warning(
            f"Could not render figure for article {article_id} and image cluster "
            f"{image_cluster_id}: {e}"
        )
        return None


def download_missing_articles(
    missing_figure_uuids: list[str],
    save_folder: str,
    number_of_workers: int | None = None,
) -> dict[str, str]:
    """
    Download the article packages using the FTP service.

    Args:
        missing_figure_uuids (list[str]): Figure UUIDs for which the article packages
            should be downloaded. The figure UUIDs are in the format
            "PMC{article_id}_{image_cluster_id}".
        save_folder (str): Folder to save the downloaded article packages to. Will be
            deleted after the packages have been processed.
        number_of_workers (int | None, optional): Number of worker processes to use for
            downloading the packages. If None, will use the number of CPU cores.
            Defaults to None.

    Returns:
        dict[str, str]: A mapping from article ID to the path of the downloaded article
            package.

    """
    warnings.warn("The FTP service is deprecated and will be removed in August 2026.")
    if number_of_workers is None:
        number_of_workers = get_cpu_count()

    article_ids = [
        uuid.split("_")[0].removeprefix("PMC") for uuid in missing_figure_uuids
    ]

    success = download_packages_from_pubmed_central(
        article_ids, save_folder, number_of_workers=number_of_workers
    )

    if not success:
        logger.warning(
            "Some articles could not be downloaded. Check the logs for more details."
        )

    # Get the paths of the successfully downloaded articles
    all_paths = glob.glob(os.path.join(save_folder, "**", "*.tar.gz"), recursive=True)

    # Create a mapping from article_id to package path
    article_id_to_path = {}
    for path in all_paths:
        article_id = os.path.basename(path).split(".tar.gz")[0]
        article_id_to_path[article_id] = path

    all_folders = glob.glob(os.path.join(save_folder, "**", "**"))
    # Filter out the folders that contain the downloaded packages
    for folder in all_folders:
        if os.path.isdir(folder) and len(os.listdir(folder)) == 0:
            try:
                os.rmdir(folder)
            except Exception as e:
                logger.warning(f"Could not remove folder {folder}: {e}")

    # Remove the parent folders if they are empty
    all_folders = glob.glob(os.path.join(save_folder, "**"))
    for folder in all_folders:
        if os.path.isdir(folder) and len(os.listdir(folder)) == 0:
            try:
                os.rmdir(folder)
            except Exception as e:
                logger.warning(f"Could not remove folder {folder}: {e}")

    return article_id_to_path


def get_missing_image_data(
    missing_image_data_df: pd.DataFrame,
    save_folder: str,
    number_of_workers: int | None = None,
    output_dpi: int | None = None,
) -> pd.Series:
    """
    Get the missing image data for the specified articles and image clusters.

    Args:
        missing_image_data_df (pd.DataFrame): DataFrame containing the article IDs and
            image cluster IDs for which the image data is missing. The DataFrame should
            have columns:
                - "article_id"
                - "image_cluster_id"
                - "position"
                - "dpi"
                - "no_buffer_render".
        save_folder (str): Save folder for the downloaded article packages. The folder
            will be deleted after the packages have been processed.
        number_of_workers (int | None, optional): Number of worker processes to use for
            downloading the packages and rendering the figures. If None, will use the
            number of CPU cores. Defaults to None.
        output_dpi (int | None, optional): DPI to render the filled images with. If
            None, will use the original DPI. Defaults to None.

    Returns:
        pd.Series: A pandas Series containing the filled image data as byte arrays. The
            index of the Series corresponds to the index of the input DataFrame.

    """
    # Get missing figure uuids
    missing_figure_uuids = (
        "PMC"
        + missing_image_data_df["article_id"].astype(str)
        + "_"
        + missing_image_data_df["image_cluster_id"]
    )

    missing_article_mapping = download_missing_articles(
        missing_figure_uuids.tolist(), save_folder, number_of_workers=number_of_workers
    )

    image_byte_series = pd.Series(index=missing_image_data_df.index, dtype=object)

    for index, row in missing_image_data_df.iterrows():
        article_id = row["article_id"]
        image_cluster_id = row["image_cluster_id"]

        article_package_path = missing_article_mapping.get(article_id)
        if article_package_path is None:
            logger.warning(f"No package found for article {article_id}")
            continue

        position_dict = (
            json.loads(row["position"]) if row["position"] is not None else {}
        )

        figure_position = position_dict.get("figure_page_coordinates")

        original_dpi = row["dpi"]

        figure_image = fill_image_data(
            article_id,
            image_cluster_id,
            figure_position,
            original_dpi,
            article_package_path,
            output_dpi=output_dpi,
            use_centered_alignment=row.get("no_buffer_render", False),
        )

        if figure_image is not None:
            image_byte_array = figure_image.tobytes()
            image_byte_series.at[index] = image_byte_array

    # Delete the downloaded packages to save space
    try:
        shutil.rmtree(save_folder)
    except Exception as e:
        logger.warning(f"Could not remove folder {save_folder}: {e}")

    return image_byte_series


def get_split_captions(
    retrieve_cmd: str,
    article_to_image_cluster_id: dict[str, set[str]],
) -> dict[str, str]:
    """
    Get the full captions for the specified article and image cluster IDs from the wds.

    Args:
        retrieve_cmd (str): Retrieve command to access the BIOMEDICA webdataset split.
        article_to_image_cluster_id (dict[str, set[str]]): Missing article and image
            cluster IDs for which the captions should be retrieved. The keys of the
            dictionary are the article IDs and the values are sets of image cluster IDs
            corresponding to the article IDs. The article IDs should not include the
            "PMC" prefix.

    Returns:
        dict[str, str]: A mapping from article ID and image cluster ID to the full
            caption text.

    """
    dataset = wds.WebDataset(retrieve_cmd, shardshuffle=False).decode("pil")  # pyright: ignore[reportAttributeAccessIssue]
    full_caption_map = {}
    for data in dataset:
        current_image_cluster_id = data["json"]["image_cluster_id"]
        current_article_id = data["json"]["article_accession_id"]

        if (
            current_article_id in article_to_image_cluster_id
            and current_image_cluster_id
            in article_to_image_cluster_id.get(current_article_id, set())
        ):
            full_caption = data["txt"]
            full_caption_map[f"{current_article_id}_{current_image_cluster_id}"] = (
                full_caption
            )

    return full_caption_map


def get_biomedica_meta(
    missing_figure_uuids: list[str], number_of_workers: int | None = None
) -> dict[str, str]:
    """
    Get the full captions for the specified figure UUIDs from the BIOMEDICA metadata.

    Args:
        missing_figure_uuids (list[str]): List of figure UUIDs for which the captions
            should be retrieved. The figure UUIDs are in the format
            "PMC{article_id}_{image_cluster_id}".
        number_of_workers (int | None, optional): Number of worker processes to use for
            retrieving the captions. If None, will use the number of CPU cores. Defaults
            to None.

    Raises:
        ValueError: If the HF_TOKEN environment variable is not set.

    Returns:
        dict[str, str]: A mapping from article ID and image cluster ID to the full
            caption text.

    """
    if number_of_workers is None:
        number_of_workers = get_cpu_count()

    article_to_image_cluster_id = {}

    for uuid in missing_figure_uuids:
        article_id, image_cluster_id = uuid.split("_", maxsplit=1)
        if article_id not in article_to_image_cluster_id:
            article_to_image_cluster_id[article_id] = set()
        article_to_image_cluster_id[article_id].add(image_cluster_id)

    hf_token = os.getenv("HF_TOKEN")

    if hf_token is None:
        raise ValueError(
            "HF_TOKEN environment variable is not set. "
            "Please set it to access the BIOMEDICA dataset."
        )

    split_to_filenames = get_split_filename_patterns(max_files_per_package=50)

    def fetch_split(split_name, file_names):
        base_split_name = split_name.split("_")[0]
        dataset_url = (
            "pipe:curl -s -L "
            "https://huggingface.co/datasets/BIOMEDICA/"
            f"biomedica_webdataset_24M/resolve/main/{base_split_name}/{file_names}"
        )
        retrieve_cmd = f"{dataset_url} -H 'Authorization:Bearer {hf_token}'"
        return get_split_captions(retrieve_cmd, article_to_image_cluster_id)

    full_caption_map = {}
    splits = list(split_to_filenames.items())

    with ThreadPoolExecutor(max_workers=number_of_workers) as executor:
        futures = {
            executor.submit(fetch_split, split_name, file_names): split_name
            for split_name, file_names in splits
        }
        with tqdm(total=len(futures), desc="Processing splits") as pbar:
            for future in as_completed(futures):
                full_caption_map.update(future.result())
                pbar.update(1)

    return full_caption_map


def get_missing_subcaption_texts(
    missing_caption_df: pd.DataFrame, number_of_workers: int | None = None
) -> pd.Series:
    """
    Get the missing subcaption texts for the specified articles and image clusters.

    Args:
        missing_caption_df (pd.DataFrame): DataFrame containing the article IDs and
            image cluster IDs for which the subcaption text is missing. The DataFrame
            should have columns:
            - "article_id"
            - "image_cluster_id"
            - "subcaption_segments"
        number_of_workers (int | None, optional): Number of worker processes to use for
            retrieving the captions. If None, will use the number of CPU cores.
            Defaults to None.

    Returns:
        pd.Series: A pandas Series containing the filled subcaption texts. The index of
        the Series corresponds to the index of the input DataFrame.

    """
    missing_figure_uuids = (
        "PMC"
        + missing_caption_df["article_id"].astype(str)
        + "_"
        + missing_caption_df["image_cluster_id"]
    )

    full_caption_map = get_biomedica_meta(
        missing_figure_uuids.tolist(), number_of_workers
    )

    subcaption_text_series = pd.Series(index=missing_caption_df.index, dtype=object)

    for index, row in missing_caption_df.iterrows():
        article_id = row["article_id"]
        image_cluster_id = row["image_cluster_id"]

        figure_uuid = f"{article_id}_{image_cluster_id}"
        full_caption = full_caption_map.get(figure_uuid)

        if full_caption is not None:
            subcaption_text_series.at[index] = full_caption
        else:
            logger.warning(
                f"No full caption found for article {article_id} and "
                f"image cluster {image_cluster_id}"
            )
            continue

        subcaption = reconstruct_from_segments(row["subcaption_segments"], full_caption)

        subcaption_text_series.at[index] = subcaption

    return subcaption_text_series


def fill_missing_fields(
    dataset_path: str,
    num_image_workers: int | None = None,
    num_caption_workers: int | None = None,
    save_folder: str = "./tmp_packages",
    output_dpi: int | None = None,
    output_dataset_path: str | None = None,
):
    """
    Fill missing subcaptions and images in the dataset.

    If num_image_workers or num_caption_workers is not specified, the number of CPU
    cores will be used. The filled dataset will be saved to output_dataset_path if
    specified, otherwise it will overwrite the original dataset. The article packages
    downloaded for filling the missing images will be saved to save_folder and deleted
    after processing.

    Args:
        dataset_path (str): Path to the dataset parquet file.
        num_image_workers (int | None, optional): Number of worker processes to use for
            filling the missing images. If None, will use the number of CPU cores.
            Defaults to None.
        num_caption_workers (int | None, optional): Number of worker processes to use
            for filling the missing captions. If None, will use the number of CPU cores.
            Defaults to None.
        save_folder (str, optional): Folder to save the downloaded article packages to.
            The folder will be deleted after the packages have been processed. Defaults
            to "./tmp_packages".
        output_dpi (int | None, optional): DPI to save the filled images with. If not
            specified, will keep the original DPI. Defaults to None.
        output_dataset_path (str | None, optional): Path to save the filled dataset to.
            If not specified, will overwrite the original dataset. Defaults to None.

    Warning:
        Deletion of ``save_folder`` is attempted after processing but failure is
        only logged as a warning, not raised. If the deletion fails (e.g. due to
        a permission error or a crash mid-run), the temporary folder will remain
        on disk and must be cleaned up manually.

    """
    assert os.path.exists(dataset_path), f"Dataset path {dataset_path} does not exist."
    assert dataset_path.endswith(
        ".parquet"
    ), f"Dataset path {dataset_path} is not a parquet file."

    dataset_df = pd.read_parquet(dataset_path)

    # Get rows with missing data
    missing_image_data_df = dataset_df[dataset_df["panel_image_bytes"].isnull()].drop(
        columns=["panel_image_bytes"]
    )

    missing_caption_df = dataset_df[dataset_df["subcaption_text"].isnull()].drop(
        columns=["panel_image_bytes"]
    )

    logger.info(
        f"Found {len(missing_image_data_df)} rows with missing image data and "
        f"{len(missing_caption_df)} rows with missing caption data."
    )

    # TODO caption and image fill could happen in parallel
    # TODO add option to use offline downloaded biomedica metadata for caption filling
    # Fill caption data
    logger.info("Filling missing caption data...")
    dataset_df.loc[missing_caption_df.index, "subcaption_text"] = (
        get_missing_subcaption_texts(
            missing_caption_df, number_of_workers=num_caption_workers
        )
    )

    # Fill image data
    logger.info("Filling missing image data...")
    dataset_df.loc[missing_image_data_df.index, "panel_image_bytes"] = (
        get_missing_image_data(
            missing_image_data_df,
            save_folder=os.path.join(save_folder, "image_fill_packages"),
            number_of_workers=num_image_workers,
            output_dpi=output_dpi,
        )
    )

    if len(os.listdir(save_folder)) == 0:
        try:
            os.rmdir(save_folder)
        except Exception as e:
            logger.warning(f"Could not remove folder {save_folder}: {e}")

    if output_dataset_path is None:
        output_dataset_path = dataset_path

    logger.info(f"Saving filled dataset to {output_dataset_path}...")
    dataset_df.to_parquet(output_dataset_path)
