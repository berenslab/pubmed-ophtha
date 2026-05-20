"""Retrieve higher quality images from the original articles."""

import asyncio
import json
import logging
import math
import multiprocessing
import os
import re
import tarfile
from io import BytesIO

import imagehash
import numpy as np
import numpy.typing as npt
import pandas as pd
from nltk import edit_distance as levenshtein_distance
from PIL import Image
from pmo_parser import caption_pdf
from pmo_parser.const import FIGURE_ID_REGEX_PATTERN
from pmo_parser.figure import OutputFigure
from tqdm.auto import tqdm

from pubmed_ophtha.const.labels import SIMILARITY_TESTS
from pubmed_ophtha.scraping.download_files import download_oa_package_from_pmc_ids
from pubmed_ophtha.util.database_interface import get_database_connection_context

logger = logging.getLogger(__name__)


def estimate_figure_id(image_cluster_id: str) -> tuple[int, str | None] | None:
    """
    Use regex to estimate the figure id based on the image cluster id.

    Args:
        image_cluster_id (str): Image cluster id from PMC OA package.

    Returns:
        tuple[int, int | None] | None: Estimated figure id and sub-figure id (if
            applicable) based on the image cluster id.
            Returns None if no figure id could be estimated.

    """
    split_string = (
        image_cluster_id.split("-g")[-1]  # noqa: PLC0207
        .split("-f")[-1]
        .split(".g")[-1]
        .split("-i")[-1]
        .split("fig")[-1]
    )

    try:
        figure_id = int(split_string)
        sub_figure_id = None
    except ValueError:
        try:
            figure_id = int(
                image_cluster_id.split("_", maxsplit=1)[0].replace("gr", "")
            )
            sub_figure_id = None
        except ValueError:
            try:
                figure_id = int(split_string[0:-1])
                sub_figure_id = split_string[-1]
            except ValueError:
                return None
    return figure_id, sub_figure_id


class HammingDistance:
    """
    A class to calculate and handle Hamming distances between image hashes.

    Attributes:
        delta_ahash (int): The Hamming distance for the average hash.
        delta_phash (int): The Hamming distance for the perceptual hash.
        delta_dhash (int): The Hamming distance for the difference hash.
        delta_whash (int): The Hamming distance for the wavelet hash.

    """

    def __init__(
        self, delta_ahash: int, delta_phash: int, delta_dhash: int, delta_whash: int
    ):
        """
        Initialize the class with the given hash deltas.

        Args:
            delta_ahash (int): The delta value for the average hash.
            delta_phash (int): The delta value for the perceptual hash.
            delta_dhash (int): The delta value for the difference hash.
            delta_whash (int): The delta value for the wavelet hash.

        """
        self.delta_ahash = delta_ahash
        self.delta_phash = delta_phash
        self.delta_dhash = delta_dhash
        self.delta_whash = delta_whash

    def __getitem__(self, key: int) -> int:
        """
        Retrieve the value associated with the given key.

        Args:
            key (int): The index to retrieve the value for. Valid keys are:
                   0 - delta_ahash
                   1 - delta_phash
                   2 - delta_dhash
                   3 - delta_whash

        Returns:
            int: The value associated with the given key.

        Raises:
            IndexError: If the key is not in the range 0-3.

        """
        if key == 0:
            return self.delta_ahash
        elif key == 1:
            return self.delta_phash
        elif key == 2:
            return self.delta_dhash
        elif key == 3:
            return self.delta_whash
        else:
            raise IndexError("Index out of range")

    def distance_sum(self) -> int:
        """
        Calculate the sum of distances for different hash algorithms.

        Returns:
            int: The sum of delta_ahash, delta_phash, delta_dhash, and delta_whash.

        """
        return self.delta_ahash + self.delta_phash + self.delta_dhash + self.delta_whash

    def distance_max(self) -> int:
        """
        Calculate the maximum distance among different hash deltas.

        Returns:
            int: The maximum value among delta_ahash, delta_phash, delta_dhash, and \
                delta_whash.

        """
        return max(
            self.delta_ahash, self.delta_phash, self.delta_dhash, self.delta_whash
        )

    def distance_min(self) -> int:
        """
        Calculate the minimum distance among different hash deltas.

        This method returns the smallest value among the attributes `delta_ahash`,
        `delta_phash`, `delta_dhash`, and `delta_whash`.

        Returns:
            int: The minimum value among the hash deltas.

        """
        return min(
            self.delta_ahash, self.delta_phash, self.delta_dhash, self.delta_whash
        )

    def has_delta_lte(self, threshold: int = 0) -> bool:
        """
        Check if any of the hash deltas are zero.

        Returns:
            bool: True if any of the hash deltas (ahash, phash, dhash, or whash) are \
                zero, otherwise False.

        """
        return (
            self.delta_ahash <= threshold
            or self.delta_phash <= threshold
            or self.delta_dhash <= threshold
            or self.delta_whash <= threshold
        )

    def __repr__(self) -> str:
        """
        Create string representation of the class instances.

        Returns:
            str: Representation of the class instance.

        """
        return (
            f"HammingDistance(ahash={self.delta_ahash}, phash={self.delta_phash}, "
            f"dhash={self.delta_dhash}, whash={self.delta_whash})"
        )

    @staticmethod
    def from_image_hashes(
        first_hashes: "FullImageHash", second_hashes: "FullImageHash"
    ) -> "HammingDistance":
        """
        Calculate the Hamming distance between two sets of image hashes.

        Args:
            first_hashes (FullImageHash): The first image hashes.
            second_hashes (FullImageHash): The other image hashes to compare against.

        Returns:
            HammingDistance: The Hamming distance between the provided image hashes.

        """
        return HammingDistance(
            abs(first_hashes.ahash - second_hashes.ahash),
            abs(first_hashes.phash - second_hashes.phash),
            abs(first_hashes.dhash - second_hashes.dhash),
            abs(first_hashes.whash - second_hashes.whash),
        )


class ExtractionError(Exception):
    """
    Custom exception for errors during the extraction process.

    Attributes:
        message (str): Error message.

    """

    def __init__(self, message: str):
        """
        Initialize the exception with a message.

        Args:
            message (str): Error message.

        """
        super().__init__(message)
        self.message = message


class MissingPDFError(ExtractionError):
    """
    Custom exception for missing PDF files during extraction.

    Attributes:
        message (str): Error message.
        image_data (dict[str, Image.Image] | None): Optional dictionary of image \
            names and their corresponding pillow image object.

    """

    def __init__(self, message: str, image_data: dict[str, Image.Image] | None = None):
        """
        Create the exception with a message and optional image data.

        Args:
            message (str): Message describing the error.
            image_data (dict[str, Image.Image] | None, optional): The loaded images.
                Defaults to None.

        """
        super().__init__(message)
        self.image_data = image_data


class MissingXMLFileError(ExtractionError):
    """
    Custom exception for missing XML files during extraction.

    Attributes:
        message (str): Error message.
        image_data (dict[str, Image.Image] | None): Optional dictionary of image \
            names and their corresponding pillow image object.

    """

    def __init__(self, message: str, image_data: dict[str, Image.Image] | None = None):
        """
        Create the exception with a message and optional image data.

        Args:
            message (str): Message describing the error.
            image_data (dict[str, Image.Image] | None, optional): The loaded images.
                Defaults to None.

        """
        super().__init__(message)
        self.image_data = image_data


class FullImageHash:  # noqa: PLW1641
    """
    A class to compute and store various types of image hashes for a given image.

    Attributes:
        _image (Image.Image): The original image.
        ahash (imagehash.ImageHash): The average hash of the image.
        phash (imagehash.ImageHash): The perceptual hash of the image.
        dhash (imagehash.ImageHash): The difference hash of the image.
        whash (imagehash.ImageHash): The wavelet hash of the image.

    """

    def __init__(self, image: Image.Image):
        """
        Compute image hashes based on the input.

        Args:
            image (Image.Image): The image to be processed and hashed.

        """
        self._image = image
        self.ahash = imagehash.average_hash(image)
        self.phash = imagehash.phash(image)
        self.dhash = imagehash.dhash(image)
        self.whash = imagehash.whash(image)

    def __eq__(self, value: object) -> bool:
        """
        Check if two FullImageHash objects are equal.

        Args:
            value (object): The object to compare with.

        Returns:
            bool: True if the objects are equal, False otherwise.

        """
        return (
            isinstance(value, FullImageHash)
            and self.ahash == value.ahash
            and self.phash == value.phash
            and self.dhash == value.dhash
            and self.whash == value.whash
        )

    def __sub__(self, value: object) -> int:
        """
        Calculate the difference between two FullImageHash objects.

        This method overrides the subtraction operator (-) to compute the sum of \
        absolute differences between the ahash, phash, dhash, and whash attributes \
        of the current object and another FullImageHash object.

        Args:
            value (object): The other FullImageHash object to compare with.

        Returns:
            int: The sum of absolute differences between the corresponding hash \
                attributes.

        Raises:
            AssertionError: If the provided value is not an instance of FullImageHash.

        """
        assert isinstance(value, FullImageHash)
        return (
            abs(self.ahash - value.ahash)
            + abs(self.phash - value.phash)
            + abs(self.dhash - value.dhash)
            + abs(self.whash - value.whash)
        )

    def __getitem__(self, key: int) -> imagehash.ImageHash:
        """
        Retrieve the image hash based on the provided index.

        Args:
            key (int): The index of the image hash to retrieve.
                   Valid values are:
                   0 - ahash
                   1 - phash
                   2 - dhash
                   3 - whash

        Returns:
            imagehash.ImageHash: The image hash corresponding to the provided index.

        Raises:
            IndexError: If the provided index is out of range (not between 0 and 3).

        """
        if key == 0:
            return self.ahash
        elif key == 1:
            return self.phash
        elif key == 2:
            return self.dhash
        elif key == 3:
            return self.whash
        else:
            raise IndexError("Index out of range")

    def delta(self, value: object) -> HammingDistance:
        """
        Calculate the Hamming distance between the hash object and another one.

        Args:
            value (object): An instance of FullImageHash to compare against.

        Returns:
            HammingDistance: An object containing the Hamming distances for \
                ahash, phash, dhash, and whash.

        Raises:
            AssertionError: If the provided value is not an instance of FullImageHash.

        """
        assert isinstance(value, FullImageHash)
        return HammingDistance(
            abs(self.ahash - value.ahash),
            abs(self.phash - value.phash),
            abs(self.dhash - value.dhash),
            abs(self.whash - value.whash),
        )


def retrieve_relevant_pmc_oa_packages(database_path: str) -> list[str]:
    """
    Get PMC article IDs from the BIOMEDICA dataset.

    Args:
        database_path (str): Path to the SQLite database containing the BIOMEDICA
        metadata.

    Returns:
        list[str]: List of PMC article IDs as strings.

    """
    with get_database_connection_context(database_path) as connection:
        cursor = connection.cursor()
        cursor.execute("SELECT article_id FROM biomedica_data_file_list")
        pmc_ids = list({str(row[0]) for row in cursor.fetchall()})
    return pmc_ids


def fetch_relevant_pmc_articles(
    biomedica_file: str, output_path: str, max_files: int | None = None
) -> bool:
    """
    Download the relevant PMC articles from the PMC OA dataset.

    Args:
        biomedica_file (str): Path to the joined filtered BIOMEDICA dataset file.
        output_path (str): Folder to save the PMC OA packages to.
        max_files (int | None, optional): Maximum number of files to download. \
            Defaults to None.

    Returns:
        bool: True if the download was successful.

    """
    pmc_ids = retrieve_relevant_pmc_oa_packages(biomedica_file)

    if max_files is not None:
        pmc_ids = pmc_ids[: min(max_files, len(pmc_ids))]

    logger.info(f"Downloading {len(pmc_ids)} articles!")
    results = asyncio.run(
        download_oa_package_from_pmc_ids(
            pmc_ids,
            output_path,
        )
    )

    return results


def locate_article_package(packages_path: str, oa_path: str) -> str | None:
    """
    Retrieve the path to the tar.gz package of the article.

    Args:
        packages_path (str): Path to the OA packages.
        oa_path (str): Path to the article package.

    Returns:
        str | None: Path to the article package.

    """
    article_path = os.path.join(packages_path, f"{oa_path}.tar.gz")
    if not os.path.exists(article_path):
        return None

    return article_path


def get_data_from_package(
    article_package_path: str | BytesIO,
) -> tuple[dict[str, Image.Image], BytesIO]:
    """
    Retrieve the pdf and image bytes from a PMC OA package.

    Args:
        article_package_path (str | BytesIO): Path to the article package (.tar.gz). \
            Can also be a BytesIO object containing the tar.gz data.

    Raises:
        ExtractionError: When the extraction fails.

    Returns:
        tuple[dict[str, Image.Image], BytesIO]: Tuple, where the first element is a \
            dictionary of image names and their corresponding pillow image object, and \
            the second element is a BytesIO object containing the PDF data.

    """
    article_image_bytes = {}

    # Open the tar.gz file
    if isinstance(article_package_path, BytesIO):
        tar = tarfile.open(fileobj=article_package_path, mode="r:gz")
    else:
        tar = tarfile.open(article_package_path, mode="r:gz")

    with tar:
        archive_contents = tar.getnames()

        # Get image files
        image_files = [f for f in archive_contents if f.endswith(".jpg")]
        for image_file in image_files:
            extracted_file = tar.extractfile(image_file)
            if extracted_file is None:
                continue
            article_image_bytes[image_file.split("/")[-1].removesuffix(".jpg")] = (
                extracted_file.read()
            )

        # load Pillow images
        article_images: dict[str, Image.Image] = {}
        try:
            for k, image in article_image_bytes.items():
                with Image.open(BytesIO(image)) as opened:
                    article_images[k] = opened.copy()
        except Image.DecompressionBombError as e:
            raise ExtractionError(
                f"Decompression bomb error: {e}. The image might be too large."
            ) from e

        # Get xml file_name
        xml_file_candidates = [f for f in archive_contents if f.endswith(".nxml")]
        if len(xml_file_candidates) >= 1:
            article_xml = tar.extractfile(xml_file_candidates[0])
        else:
            raise MissingXMLFileError(
                "Could not find XML file", image_data=article_images
            )

        if article_xml is None:
            raise MissingXMLFileError(
                "Could not find XML file", image_data=article_images
            )
        article_xml = article_xml.read()

        article_pdf_name = xml_file_candidates[0].replace(".nxml", ".pdf")

        if article_pdf_name not in archive_contents:
            raise MissingPDFError("Could not find PDF file")

        article_pdf = tar.extractfile(article_pdf_name)
        if article_pdf is None:
            raise MissingPDFError("Could not find PDF file")
        article_pdf = article_pdf.read()

    # load PDF
    article_pdf = BytesIO(article_pdf)

    return article_images, article_pdf


def calculate_similarity_scores(
    image_caption: str,
    current_image: Image.Image,
    estimated_image_id: int | None,
    loaded_article_figures: list[OutputFigure],
    loaded_figure_hashes: list[FullImageHash],
    loaded_figure_aspect_ratios: list[float],
) -> tuple[npt.NDArray[np.bool_], list[bool]]:
    """
    Calculate the similarity scores between the image and the figures.

    Args:
        image_caption (str): Caption of the image to compare against the figures.
        current_image (Image.Image): Image to compare against the figures.
        estimated_image_id (int | None): Estimated figure number from the image \
            cluster id.
        loaded_article_figures (list[OutputFigure]): Figures found in the article.
        loaded_figure_hashes (list[FullImageHash]): Hashes of the figures.
        loaded_figure_aspect_ratios (list[float]): Aspect ratio of the figures.

    Returns:
        tuple[npt.NDArray[np.bool_], list[bool]]: A tuple containing a boolean array of
            shape (num_figures, num_similarity_tests)
            indicating which similarity tests are passed for each figure, and a list of
            booleans indicating whether each figure is likely to be a multi-page figure.

    """
    image_hash = FullImageHash(current_image)
    aspect_ratio = current_image.size[0] / current_image.size[1]

    # Calculate image similarity
    hamming_distances = [
        hash_obj.delta(image_hash) for hash_obj in loaded_figure_hashes
    ]

    # Calculate aspect ratio similarity
    aspect_ratio_diff = [
        abs((aspect_ratio - fig_aspect_ratio) / aspect_ratio)
        for fig_aspect_ratio in loaded_figure_aspect_ratios
    ]

    # Calculate caption similarity
    text_distances = []
    caption_no_white_space = "".join(image_caption.split())

    is_multi_page = []

    for figure in loaded_article_figures:
        caption_text = figure.get_caption_texts()[0]

        # Remove Figure number from caption text since it is not included
        # in the BIOMEDICA dataset
        caption_text = re.sub(
            FIGURE_ID_REGEX_PATTERN, "", caption_text, flags=re.IGNORECASE
        )

        # Calculate distances for each string separately and combined
        # Calculate distance between words
        split_caption = "".join(caption_text.split())  # Split into words

        # Check if figure might be split across multiple pages
        # In this case the caption could be Continued, Cont. or similar
        if re.match(
            r"^(cont\.?|continued|continuation)\s*[:\-]?\s*",
            caption_text,
            re.IGNORECASE,
        ):
            # set distance to 0 if caption is only "Continued" or similar
            text_distances.append(
                {
                    "distance": 0,
                    "truncated_distance": 0,
                    "caption_length": len(split_caption),
                }
            )
            is_multi_page.append(True)
            continue

        is_multi_page.append(False)
        split_caption_distance = levenshtein_distance(
            caption_no_white_space, split_caption
        )

        min_text_length = min(len(caption_no_white_space), len(split_caption))
        truncated_caption_distance = levenshtein_distance(
            caption_no_white_space[:min_text_length],
            split_caption[:min_text_length],
        )

        # Distance is length of caption if no caption is found
        text_distances.append(
            {
                "distance": split_caption_distance,
                "truncated_distance": truncated_caption_distance,
                "caption_length": len(split_caption),
            }
        )

    figure_id_similarity = [
        fig.figure_id == estimated_image_id
        if fig.figure_id is not None and estimated_image_id is not None
        else False
        for fig in loaded_article_figures
    ]

    similarity_array = np.zeros(
        (len(loaded_article_figures), len(SIMILARITY_TESTS)), dtype=bool
    )

    # Figure ID similarity
    similarity_array[:, 0] = figure_id_similarity

    # Visual similarity
    similarity_array[:, 1] = [
        distance.has_delta_lte(2) or distance.distance_sum() <= 5
        for distance in hamming_distances
    ]

    # Caption similarity
    similarity_array[:, 2] = [
        distance["distance"] < max(5, math.ceil(distance["caption_length"] * 0.05))
        or distance["truncated_distance"]
        < max(
            2,
            math.ceil(
                min(distance["caption_length"], len(caption_no_white_space)) * 0.03
            ),
        )
        for distance in text_distances
    ]

    # Aspect ratio similarity
    similarity_array[:, 3] = [diff < 0.05 for diff in aspect_ratio_diff]

    return similarity_array, is_multi_page


def extraction_worker(
    biomedica_sub_df: pd.DataFrame,
    output_path: str,
    packages_path: str,
):
    """
    Extract the images from a single article.

    Args:
        biomedica_sub_df (pd.DataFrame): Subset of the BIOMEDICA dataset for a single \
            article.
        output_path (str): Path to save the extracted images to.
        packages_path (str): Path to the OA packages.

    """
    assert len(biomedica_sub_df) > 0
    assert len(biomedica_sub_df["article_id"].unique()) == 1

    def error(pmc_article_id, article_image_cluster_id, reason, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(
                {
                    article_image_cluster_id: [
                        {
                            "type": "error",
                            "article_id": pmc_article_id,
                            "image_cluster_id": article_image_cluster_id,
                            "reason": reason,
                        }
                    ]
                },
                f,
                ensure_ascii=False,
                indent=4,
            )

    article_id = biomedica_sub_df["article_id"].iloc[0]
    article_output_folder = os.path.join(
        output_path, biomedica_sub_df["file_list_path"].iloc[0]
    )
    meta_file_name = os.path.join(article_output_folder, "meta_info.json")
    if os.path.exists(meta_file_name):
        return

    article_package_path = locate_article_package(
        packages_path, biomedica_sub_df["file_list_path"].iloc[0]
    )
    if article_package_path is None:
        error(article_id, None, "Article not found", meta_file_name)
        return

    os.makedirs(article_output_folder, exist_ok=True)
    # open archive
    try:
        article_images, article_pdf = get_data_from_package(article_package_path)
    except ExtractionError as e:
        error(article_id, None, str(e), meta_file_name)
        return

    try:
        loaded_article_figures = caption_pdf(article_pdf)
        # Filter figures from screenshot
        loaded_article_figures = [
            fig
            for fig in loaded_article_figures
            if not (fig.used_screenshot or fig.image is None)
        ]
    except ValueError:
        error(article_id, None, "Could not extract figures", meta_file_name)
        return
    except Exception as e:
        # Probably corrupt pdf
        error(
            article_id,
            None,
            f"Could not extract figures due to an error: {e}",
            meta_file_name,
        )
        return

    if len(loaded_article_figures) == 0:
        error(article_id, None, "No figures found", meta_file_name)
        return

    # Try image hash to match images with extracted figures
    loaded_figure_hashes = []
    loaded_figure_aspect_ratios = []
    for fig in loaded_article_figures:
        img = fig.image
        assert img is not None
        loaded_figure_hashes.append(FullImageHash(img))
        loaded_figure_aspect_ratios.append(img.size[0] / img.size[1])

    estimated_article_image_ids = {
        x: estimate_figure_id(x) for x in article_images.keys()
    }

    if not all(
        [
            k in article_images.keys()
            for k in biomedica_sub_df["image_cluster_id"].values
        ]
    ):
        error(
            article_id,
            None,
            "Not all images in the article are in the BIOMEDICA dataset",
            meta_file_name,
        )
        return

    # Merge rows if necessary
    keys_with_subfigures = [
        key
        for key, value in estimated_article_image_ids.items()
        if value is not None and isinstance(value, tuple) and value[1] is not None
    ]
    if len(keys_with_subfigures) > 0:
        # There are sub-figures, merge them
        logger.warning(
            f"Warning: Found split article image ({keys_with_subfigures}). "
            "Merging is not yet implemented"
        )

    meta_info = {}
    for index, row in biomedica_sub_df.iterrows():
        est_id = estimated_article_image_ids[row["image_cluster_id"]]

        if est_id is not None:
            est_id = est_id[0]

        similarity_array, is_multi_page = calculate_similarity_scores(
            row["image_caption"],
            article_images[row["image_cluster_id"]],
            est_id,
            loaded_article_figures,
            loaded_figure_hashes,
            loaded_figure_aspect_ratios,
        )

        similarity_scores = similarity_array.sum(axis=1)

        figure_id_similarity = similarity_array[:, 0]

        score_threshold = (
            2 if estimated_article_image_ids[row["image_cluster_id"]] is not None else 2
        )

        relevant_figures: list[int] = np.where(similarity_scores >= score_threshold)[
            0
        ].tolist()  # pyright: ignore[reportAssignmentType]
        if len(relevant_figures) == 0:
            # No relevant figure found
            meta_info[row["image_cluster_id"]] = [
                {
                    "type": "error",
                    "article_id": article_id,
                    "image_cluster_id": row["image_cluster_id"],
                    "reason": "No relevant figure found",
                }
            ]
            continue

        meta_info[row["image_cluster_id"]] = []

        saved_figure_paths = []

        # Remove if the figure ID is not the same
        if estimated_article_image_ids[row["image_cluster_id"]] is not None:
            relevant_figures = [
                fig_index
                for fig_index in relevant_figures
                if figure_id_similarity[fig_index] or similarity_scores[fig_index] > 2
            ]

        # Use the one with the highest visual match if it is a multi-page figure
        if (
            len(relevant_figures) > 1
            and row["image_cluster_id"] in keys_with_subfigures
        ):
            image_hash = FullImageHash(article_images[row["image_cluster_id"]])
            hamming_distances = np.array(
                [
                    loaded_figure_hashes[fig_ind].delta(image_hash).distance_sum()
                    for fig_ind in relevant_figures
                ]
            )
            best_figure_index = hamming_distances.argmin()
            relevant_figures = [relevant_figures[best_figure_index]]

        # Save the relevant figures
        for seq_index, fig_index in enumerate(relevant_figures):
            extracted_figure_path = os.path.join(
                article_output_folder,
                f"{row['image_cluster_id']}_hq_{seq_index}.png",
            )

            saved_figure_paths.append(extracted_figure_path)
            fig_image = loaded_article_figures[fig_index].image
            assert fig_image is not None
            if fig_image.mode == "CMYK":  # Convert JPEG images to RGB first
                fig_image = fig_image.convert("RGBA")
                loaded_article_figures[fig_index].image = fig_image
            fig_image.save(extracted_figure_path)
            meta_info[row["image_cluster_id"]].append(
                {
                    "type": "figure",
                    "figure_index": fig_index,
                    "figure_id": loaded_article_figures[fig_index].figure_id,
                    "figure_path": extracted_figure_path,
                    "similarity_scores": {
                        test: similarity_array[fig_index, test_index].item()
                        for test_index, test in enumerate(SIMILARITY_TESTS)
                    },
                    "total_similarity_score": similarity_scores[fig_index].item(),
                    "figure_data": loaded_article_figures[fig_index].serialize(
                        join=False
                    )[0],
                    "is_multi_page": is_multi_page[fig_index],
                }
            )

    with open(meta_file_name, "w") as meta_file:
        json.dump(meta_info, meta_file, ensure_ascii=False, indent=4)


def extraction_worker_wrapper(
    args: tuple[pd.DataFrame, str, str],
):
    """
    Wrap the extraction worker for multiprocessing.

    Args:
        args (tuple[pd.DataFrame, str, str]): Arguments for the worker.

    """
    return extraction_worker(*args)


def extract_original_relevant_figures(
    biomedica_file: str,
    packages_path: str,
    output_path: str,
    num_workers: int = 0,
):
    """
    Extract the original figures from the article PDF files.

    Tries to extract the original figures from the PDF files of the articles and save \
        them to the output path.
    Uses the captions from the BIOMEDICA dataset to match the figures.

    Args:
        biomedica_file (str): Path to the BIOMEDICA dataset file.
        packages_path (str): Path to the OA packages.
        output_path (str): Path to save the figures to.
        num_workers (int, optional): Number of workers to use for multiprocessing. \
            0 indicates no multiprocessing. Defaults to os.cpu_count() - 1.

    """
    assert biomedica_file.endswith(".parquet")
    assert os.path.exists(biomedica_file)

    if num_workers > 0:
        logger.warning(
            "Multiprocessing is not yet implemented, running with a single worker. "
            "Set num_workers to 0 to avoid this warning."
        )
        num_workers = 0
    biomedica_df = pd.read_parquet(biomedica_file)

    os.makedirs(output_path, exist_ok=True)
    # Run the extraction worker for each chunk
    tasks = [
        (biomedica_sub_df, output_path, packages_path)
        for _, biomedica_sub_df in biomedica_df.groupby("article_id")
    ]
    pmc_ids = len(tasks)

    if num_workers == 0:
        for t in tqdm(tasks, total=pmc_ids, desc="Extracting figures"):
            extraction_worker(*t)
    else:
        # TODO fix multiprocessing
        raise NotImplementedError("Multiprocessing not implemented yet")
        with multiprocessing.Pool(num_workers) as pool:
            _ = tqdm(
                pool.imap(extraction_worker_wrapper, tasks),
                total=pmc_ids,
                desc="Extracting figures",
            )
