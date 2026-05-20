"""
Retrieve higher quality images from the original articles.

Same implementation as in `retrieve_original_images.py`, but this \
implementation saves the images to a SQLite database.
"""

import asyncio
import json
import logging
import multiprocessing
import os
import pickle
import sqlite3
import time
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from pmo_parser.algorithm import caption_pdf_parallel
from pmo_parser.renderer import render_figures
from tqdm.auto import tqdm

from pubmed_ophtha.const.labels import SIMILARITY_TESTS
from pubmed_ophtha.filtering.retrieve_original_images import (
    ExtractionError,
    FullImageHash,
    MissingPDFError,
    MissingXMLFileError,
    calculate_similarity_scores,
    estimate_figure_id,
    get_data_from_package,
    retrieve_relevant_pmc_oa_packages,
)
from pubmed_ophtha.scraping.download_files_sqlite import (
    download_oa_package_from_pmc_ids,
)
from pubmed_ophtha.util.computing import get_cpu_count
from pubmed_ophtha.util.database_interface import (
    database_writer,
    get_biomedica_df,
    get_database_connection_context,
)

logger = logging.getLogger(__name__)


def fetch_relevant_pmc_articles(
    database_path: str, max_files: int | None = None
) -> bool:
    """
    Download the relevant PMC articles from the PMC OA dataset.

    Args:
        database_path (str): Path to the SQLite database file.
        max_files (int | None, optional): Maximum number of files to download.
            Defaults to None.

    Returns:
        bool: True if the download was successful.

    """
    pmc_ids = retrieve_relevant_pmc_oa_packages(database_path)

    if max_files is not None:
        pmc_ids = pmc_ids[: min(max_files, len(pmc_ids))]

    logger.info(f"Downloading {len(pmc_ids)} articles!")
    file_list_folder = os.path.dirname(database_path)
    results = asyncio.run(
        download_oa_package_from_pmc_ids(pmc_ids, database_path, file_list_folder)
    )

    return results


def locate_article_package(
    cur: sqlite3.Cursor, oa_path: str | None = None, article_id: str | int | None = None
) -> BytesIO | None:
    """
    Retrieve the path to the tar.gz package of the article.

    Args:
        cur (sqlite3.Cursor): SQLite cursor to execute the query.
        oa_path (str): Path to the article package.
        article_id (str | int): PMC ID of the article, e.g. "PMC1234567" or 1234567.

    Returns:
        str | None: Path to the article package.

    """
    assert (oa_path is not None) != (
        article_id is not None
    ), "Either oa_path or article_id must be provided, but not both."
    if oa_path is not None:
        article_id = int(os.path.basename(oa_path).removeprefix("PMC"))
    else:
        if isinstance(article_id, str):
            article_id = int(article_id.removeprefix("PMC"))
        elif not isinstance(article_id, int):
            raise ValueError("article_id must be a str or int")

    cur.execute(
        "SELECT file_data FROM article_packages WHERE article_id = ?",
        (article_id,),
    )
    result = cur.fetchone()
    if result is None or len(result) == 0 or result[0] is None:
        return None
    article_data = result[0]

    if not isinstance(article_data, bytes):
        raise ExtractionError(
            f"Data for PMC{article_id} is not in bytes format ({type(article_data)})."
        )
    article_data = BytesIO(article_data)

    return article_data


def extraction_worker(
    biomedica_sub_df: pd.DataFrame,
    database_path: str,
    num_pdf_workers: int = 5,
) -> dict[str, Any]:
    """
    Extract the images from a single article.

    Args:
        biomedica_sub_df (pd.DataFrame): Subset of the BIOMEDICA dataset for a single
            article.
        database_path (str): Name of the database to save the figures to.
        num_pdf_workers (int, optional): Number of workers to use for PDF \
            processing. Defaults to 5.

    Returns:
        dict[str, Any]: Dictionary containing the results of the extraction. \
            Contains the following keys:
            - article_id (str): PMC ID of the article, e.g. "PMC1234567".
            - meta_info_dict (dict): Dictionary containing metadata for each image.
            - error_bit (bool): True if an error occurred during extraction.
            - image_names (list[str], optional): List of names of the extracted images.
            - image_data_list (list[bytes], optional): List of bytes of the \
                extracted images.
            - image_cluster_ids (list[str], optional): List of cluster IDs of the \
                extracted images.

    """
    assert len(biomedica_sub_df) > 0
    assert len(biomedica_sub_df["article_id"].unique()) == 1

    def _process_error(
        pmc_article_id: str, article_image_cluster_id: str | None, reason: str
    ) -> dict[str, Any]:
        """
        Create the output dictionary for an error.

        Args:
            pmc_article_id (str): Article ID.
            article_image_cluster_id (str | None): Name of the image cluster.
            reason (str): Reason for the error.

        Returns:
            dict[str, Any]: Output dictionary for the error.

        """
        return {
            "article_id": pmc_article_id,
            "meta_info_dict": {
                article_image_cluster_id: [
                    {
                        "type": "error",
                        "article_id": pmc_article_id,
                        "image_cluster_id": article_image_cluster_id,
                        "reason": reason,
                    }
                ]
            },
            "error_bit": True,
        }

    with get_database_connection_context(database_path, read_only=True) as read_con:
        read_cur = read_con.cursor()

        article_id = biomedica_sub_df["article_id"].iloc[0]

        article_data = locate_article_package(
            read_cur,
            oa_path=biomedica_sub_df["file_list_path"].iloc[0],
        )
        if article_data is None:
            return_data = _process_error(article_id, None, "Article not found")
            return return_data

        # open archive
        try:
            article_images, article_pdf = get_data_from_package(article_data)
        except (MissingPDFError, MissingXMLFileError) as e:
            # Just use images directly
            if e.image_data is not None:
                article_images = e.image_data
                meta_info = {}
                processed_image_names = []
                processed_image_bytes = []
                processed_image_cluster_ids = []
                extracted_from_pdf_list = []
                for index, row in biomedica_sub_df.iterrows():
                    # Check if image is in the article images
                    if row["image_cluster_id"] in article_images.keys():
                        extracted_figure_name = f"{row['image_cluster_id']}.png"
                        meta_info[row["image_cluster_id"]] = [
                            {
                                "type": "figure",
                                "figure_index": None,
                                "figure_id": None,
                                "figure_path": [extracted_figure_name],
                                "similarity_scores": None,
                                "total_similarity_score": None,
                                "figure_data": None,
                                "image_conversion": None,
                                "is_multi_page": None,
                            }
                        ]
                        processed_image_names.append(extracted_figure_name)
                        image_bytes = BytesIO()
                        article_images[row["image_cluster_id"]].save(
                            image_bytes, format="PNG"
                        )
                        processed_image_bytes.append(image_bytes.getvalue())
                        processed_image_cluster_ids.append(row["image_cluster_id"])
                        extracted_from_pdf_list.append(False)
                    else:
                        meta_info[row["image_cluster_id"]] = [
                            {
                                "type": "error",
                                "article_id": article_id,
                                "image_cluster_id": row["image_cluster_id"],
                                "reason": "Image not found in the article package",
                            }
                        ]
                return {
                    "article_id": article_id,
                    "meta_info_dict": meta_info,
                    "error_bit": False,
                    "image_names": processed_image_names,
                    "image_data_list": processed_image_bytes,
                    "image_cluster_ids": processed_image_cluster_ids,
                    "extracted_from_pdf": extracted_from_pdf_list,
                }
            else:
                return_data = _process_error(article_id, None, str(e))
                return return_data
        except ExtractionError as e:
            return_data = _process_error(article_id, None, str(e))
            return return_data

        try:
            loaded_article_figures = caption_pdf_parallel(
                article_pdf,
                no_render_mode=True,
                num_processes=num_pdf_workers,
            )
            render_figures(article_pdf, loaded_article_figures)
            # Filter figures from screenshot
            loaded_article_figures = [
                fig
                for fig in loaded_article_figures
                if not (fig.used_screenshot or fig.image is None)
            ]
        except ValueError:
            return_data = _process_error(article_id, None, "Could not extract figures")
            return return_data
        except Exception as e:
            # Probably corrupt pdf
            return_data = _process_error(
                article_id,
                None,
                f"Could not extract figures due to an error: {e}",
            )
            return return_data

        if len(loaded_article_figures) == 0:
            return_data = _process_error(article_id, None, "No figures found")
            return return_data

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
            return_data = _process_error(
                article_id,
                None,
                "Not all images in the article are in the BIOMEDICA dataset",
            )
            return return_data

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

        processed_image_names = []
        processed_image_bytes = []
        processed_image_cluster_ids = []
        extracted_from_pdf_list = []
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
                2
                if estimated_article_image_ids[row["image_cluster_id"]] is not None
                else 2
            )

            relevant_figures: list[int] = np.where(
                similarity_scores >= score_threshold
            )[0].tolist()  # pyright: ignore[reportAssignmentType]
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

            # Remove if the figure ID is not the same
            if estimated_article_image_ids[row["image_cluster_id"]] is not None:
                relevant_figures = [
                    fig_index
                    for fig_index in relevant_figures
                    if figure_id_similarity[fig_index]
                    or similarity_scores[fig_index] > 2
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
                extracted_figure_name = f"{row['image_cluster_id']}_hq_{seq_index}.png"

                fig_image = loaded_article_figures[fig_index].image
                assert fig_image is not None
                if fig_image.mode == "CMYK":  # Convert JPEG images to RGB first
                    fig_image = fig_image.convert("RGBA")
                    loaded_article_figures[fig_index].image = fig_image

                image_bytes = BytesIO()
                fig_image.save(image_bytes, format="PNG")

                processed_image_names.append(extracted_figure_name)
                processed_image_bytes.append(image_bytes.getvalue())
                processed_image_cluster_ids.append(row["image_cluster_id"])
                extracted_from_pdf_list.append(True)

                meta_info[row["image_cluster_id"]].append(
                    {
                        "type": "figure",
                        "figure_index": [fig_index],
                        "figure_id": [loaded_article_figures[fig_index].figure_id],
                        "figure_path": [extracted_figure_name],
                        "similarity_scores": [
                            {
                                test: similarity_array[fig_index, test_index].item()
                                for test_index, test in enumerate(SIMILARITY_TESTS)
                            }
                        ],
                        "total_similarity_score": [similarity_scores[fig_index].item()],
                        "figure_data": {
                            loaded_article_figures[
                                fig_index
                            ].page: loaded_article_figures[fig_index].serialize(
                                join=False
                            )[0]
                        },
                        "image_conversion": {
                            loaded_article_figures[fig_index].page: [
                                {
                                    "figure_y_start": loaded_article_figures[
                                        fig_index
                                    ].figure_bbox.y0,
                                    "figure_y_end": loaded_article_figures[
                                        fig_index
                                    ].figure_bbox.y1,
                                    "image_y_start": 0,
                                    "image_y_end": fig_image.size[1],
                                }
                            ]
                        },
                        "is_multi_page": is_multi_page[fig_index],
                    }
                )
        return {
            "article_id": article_id,
            "meta_info_dict": meta_info,
            "error_bit": False,
            "image_names": processed_image_names,
            "image_data_list": processed_image_bytes,
            "image_cluster_ids": processed_image_cluster_ids,
            "extracted_from_pdf": extracted_from_pdf_list,
        }


def write_into_db_batch(
    con: sqlite3.Connection, batch: list[dict[str, Any]], allow_replace: bool = False
):
    """
    Write a batch of results into the database.

    Args:
        con (sqlite3.Connection): Database connection.
        batch (list[dict[str, Any]]): Batch of results to write. Each item in the
            list should be a dictionary with the following keys:
            - article_id (str | int): PMC ID of the article, e.g. "PMC1234567" or
                1234567.
            - meta_info_dict (dict): Dictionary containing metadata for each image.
            - error_bit (bool): True if an error occurred during extraction.
            - dirty_bit (bool): True if the article is marked as dirty.
            - image_names (list[str], optional): List of names of the extracted images.
            - image_data_list (list[bytes], optional): List of bytes of the extracted
                images.
            - image_cluster_ids (list[str], optional): List of cluster IDs of the
                extracted images
        allow_replace (bool, optional): If True, existing entries will be replaced.
            Defaults to False.

    """
    article_ids = []
    meta_info_jsons = []
    error_bits = []
    dirty_bits = []

    image_article_ids = []
    image_names_list = []
    image_data_lists = []
    image_cluster_ids_list = []
    extracted_from_pdf_list = []
    for item in batch:
        article_id_int = item["article_id"]
        if isinstance(article_id_int, str):
            article_id_int = int(article_id_int.removeprefix("PMC"))
        article_ids.append(article_id_int)
        error_bits.append(item.get("error_bit", False))
        dirty_bits.append(item.get("dirty_bit", False))
        meta_info_jsons.append(json.dumps(item["meta_info_dict"], ensure_ascii=False))
        for i in range(len(item.get("image_names", []))):
            image_article_ids.append(article_id_int)
            image_names_list.append(item["image_names"][i])
            image_data_lists.append(item["image_data_list"][i])
            image_cluster_ids_list.append(item["image_cluster_ids"][i])
            extracted_from_pdf = True
            if "extracted_from_pdf" in item:
                extracted_from_pdf = item["extracted_from_pdf"][i]
            extracted_from_pdf_list.append(extracted_from_pdf)

    sql_directive = "INSERT OR REPLACE" if allow_replace else "INSERT"

    con.execute("BEGIN IMMEDIATE;")
    con.executemany(
        f"{sql_directive} INTO metadata (article_id, json, error_bit, dirty_bit) VALUES (?, ?, ?, ?)",  # noqa: E501
        zip(article_ids, meta_info_jsons, error_bits, dirty_bits),
    )

    if len(image_article_ids) > 0:
        con.executemany(
            f"""
            {sql_directive} INTO article_images (
                article_id,
                image_name,
                image,
                image_cluster_id,
                extracted_from_pdf
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            zip(
                image_article_ids,
                image_names_list,
                image_data_lists,
                image_cluster_ids_list,
                extracted_from_pdf_list,
            ),
        )
    con.commit()


def write_into_db(
    con: sqlite3.Connection,
    *,
    article_id: int | str,
    meta_info_dict: dict,
    error_bit: bool = False,
    dirty_bit: bool = False,
    image_names: list[str] | None = None,
    image_data_list: list[bytes] | None = None,
    image_cluster_ids: list[str] | None = None,
    extracted_from_pdf: list[bool] | None = None,
):
    """
    Write a result into the database.

    Args:
        con (sqlite3.Connection): Connection to the database.
        article_id (int | str): PMC ID of the article, e.g. "PMC1234567" or 1234567.
        meta_info_dict (dict): Dictionary containing metadata for each image.
        error_bit (bool, optional): True if there was an error. Defaults to False.
        dirty_bit (bool, optional): The value of the dirty bit. Defaults to False.
        image_names (list[str] | None, optional): List of image names. Defaults to None.
        image_data_list (list[bytes] | None, optional): List of image bytes. Defaults
            to None.
        image_cluster_ids (list[str] | None, optional): List of image cluster ids.
            Defaults to None.
        extracted_from_pdf (list[bool] | None, optional): List indicating whether
            the image was extracted from the PDF. If None, all images are assumed to be
            extracted from the PDF. Defaults to None.

    """
    write_into_db_batch(
        con,
        [
            {
                "article_id": article_id,
                "meta_info_dict": meta_info_dict,
                "error_bit": error_bit,
                "dirty_bit": dirty_bit,
                "image_names": image_names or [],
                "image_data_list": image_data_list or [],
                "image_cluster_ids": image_cluster_ids or [],
                "extracted_from_pdf": [True] * len(image_names or [])
                if extracted_from_pdf is None
                else extracted_from_pdf,
            }
        ],
    )


def extraction_worker_wrapper(
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    progress_queue: multiprocessing.Queue,
    worker_id: int,
):
    """
    Extract the images from articles in a worker process.

    Continuously checks the task queue for new tasks. If a task is available, it \
    processes the task using the `extraction_worker` function and puts the result \
    into the result queue. If no task is available, it waits for a short time before \
    checking again. If the task is None, the process exits.

    Args:
        task_queue (multiprocessing.Queue): Queue with tasks to process. Each task \
            should be a tuple with the following items:
            - pickled pd.DataFrame: Subset of the BIOMEDICA dataset for a single \
                article.
            - str: Name of the database to save the figures to.
            - int: Number of workers to use for PDF processing.
        result_queue (multiprocessing.Queue): Queue to put the results into.
        progress_queue (multiprocessing.Queue): Queue to track progress.
        worker_id (int): ID of the worker process.

    """
    num_times_no_task = 0

    sum_processing_time = 0.0
    num_processed = 0

    start_time = time.time()
    intermediate_start_time = start_time

    def log_performance(intermediate: bool = False):
        end_time = time.time()

        write_string = "finished processing" if not intermediate else "processed"

        print_string = (
            f"Worker {worker_id} has {write_string} {num_processed} articles.\n"
            + f"Worker {worker_id} total time: {end_time - start_time:.2f} seconds\n"
        )
        if num_processed > 0:
            tpp = (end_time - start_time) / num_processed
            print_string += (
                f"Worker {worker_id} average time per article: {tpp:.2f} seconds"
            )

        logger.info(print_string)

    while True:
        try:
            task = task_queue.get(timeout=10)  # Wait for a task for up to 10 seconds

            num_times_no_task = 0  # Reset counter on successful get

            if task is None:
                logger.info(f"Worker {worker_id} received shutdown signal. Exiting.")
                break  # Exit if a sentinel value is received

            df, db_path, num_pdf_workers = task
            try:
                processing_start_time = time.time()
                processing_result = extraction_worker(
                    pickle.loads(df),
                    db_path,
                    num_pdf_workers=num_pdf_workers,
                )
                sum_processing_time += time.time() - processing_start_time
                num_processed += 1

                if time.time() - intermediate_start_time > 300 and num_processed > 0:
                    # Log performance every 5 minutes
                    log_performance(intermediate=True)
                    intermediate_start_time = time.time()
            except Exception as e:
                # Ignore task
                logger.exception(f"Worker {worker_id} encountered an error: {e}")
                continue  # Skip this task and continue with the next

            result_queue.put(processing_result)
            progress_queue.put(1)

        except multiprocessing.queues.Empty:  # pyright: ignore[reportAttributeAccessIssue]
            logger.warning(f"Worker {worker_id} timed out waiting for tasks.")
            num_times_no_task += 1
            if num_times_no_task >= 6:
                logger.warning(f"Worker {worker_id} exiting after multiple timeouts.")
                break
            continue  # No task wait for sentinel
        except Exception as e:
            # cannot recover from this error
            logger.exception(f"Worker {worker_id} encountered an error: {e}")
            break

    progress_queue.put(None)  # Signal that this worker is done

    log_performance()


def extract_original_relevant_figures(
    database_path: str,
    num_workers: int | None = None,
    num_pdf_workers: int | None = 5,
    database_batch_size: int = 20,
):
    """
    Extract the original figures from the article PDF files.

    Tries to extract the original figures from the PDF files of the articles and save \
        them to the output path.
    Uses the captions from the BIOMEDICA dataset to match the figures.

    Args:
        database_path (str): Name of the database to save the figures to.
        num_workers (int, optional): Number of workers to use for multiprocessing. \
            0 indicates no multiprocessing. If None, uses the maximum number of \
            available cpus based on num_pdf_workers. Defaults to None.
        num_pdf_workers (int, optional): Number of workers to use for PDF \
            processing. Defaults to 5.
        database_batch_size (int, optional): Size of the batches to write to the \
            database. Defaults to 20.

    """
    biomedica_df = get_biomedica_df(database_path)

    num_db_writing_workers = 1

    if num_pdf_workers is None and num_workers is None:
        cpu_count = get_cpu_count() - 1 - num_db_writing_workers
        sqrt_count = int(np.sqrt(cpu_count))
        num_pdf_workers = max(1, sqrt_count)
        num_workers = max(1, int(cpu_count / num_pdf_workers))
    elif num_workers is None:
        assert num_pdf_workers is not None
        cpu_count = get_cpu_count()
        num_workers = max(1, cpu_count - num_pdf_workers - 1 - num_db_writing_workers)

    elif num_pdf_workers is None:
        assert num_workers is not None
        cpu_count = get_cpu_count()
        num_pdf_workers = max(1, cpu_count - num_workers - 1 - num_db_writing_workers)

    with get_database_connection_context(database_path, read_only=False) as con:
        cur = con.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS article_images(
                id INTEGER PRIMARY KEY ON CONFLICT REPLACE,
                article_id INTEGER,
                image_name TEXT,
                image BLOB,
                image_cluster_id TEXT,
                extracted_from_pdf BOOLEAN DEFAULT 1
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata(
                article_id INTEGER PRIMARY KEY,
                json TEXT,
                error_bit BOOLEAN DEFAULT 0,
                dirty_bit BOOLEAN DEFAULT 0
            )
            """
        )

        # get all article_ids in the metadata table
        cur.execute("SELECT article_id FROM metadata")
        existing_article_ids = {row[0] for row in cur.fetchall()}

        con.commit()

    # Prepare queues for multiprocessing
    result_queue = multiprocessing.Queue(maxsize=100)
    task_queue = multiprocessing.Queue()

    # Queue to track progress
    progress_queue = multiprocessing.Queue()

    # Prepare process to write into the database
    database_writer_process = multiprocessing.Process(
        target=database_writer,
        args=(database_path, result_queue, write_into_db_batch),
        kwargs={"batch_size": database_batch_size},
        name="DatabaseWriterProcess",
    )
    database_writer_process.start()

    # Put tasks into the queue
    tasks = []
    for article_id, biomedica_sub_df in biomedica_df.groupby("article_id"):
        if int(str(article_id).removeprefix("PMC")) not in existing_article_ids:
            tasks.append(
                (pickle.dumps(biomedica_sub_df), database_path, num_pdf_workers)
            )

    if num_workers < 2:
        for t in tqdm(tasks, desc="Extracting figures"):
            sub_df = pickle.loads(t[0])
            if len(sub_df) == 0:
                continue
            processing_result = extraction_worker(sub_df, t[1], t[2])
            result_queue.put(processing_result)
    else:
        # Put tasks into queue
        for t in tasks:
            task_queue.put(t)

        # Start worker processes
        workers = []
        for worker_id in range(num_workers):
            p = multiprocessing.Process(
                target=extraction_worker_wrapper,
                args=(
                    task_queue,
                    result_queue,
                    progress_queue,
                    worker_id,
                ),
                name=f"ExtractionWorker-{worker_id}",
            )
            p.start()
            workers.append(p)

        # Add sentinel values to the task queue to signal the workers to exit
        for _ in workers:
            task_queue.put(None)

        total_tasks = len(tasks)
        num_none = 0
        with tqdm(total=total_tasks, desc="Extracting figures") as pbar:
            while num_none < num_workers:
                progress = progress_queue.get()
                if progress is None:
                    num_none += 1
                else:
                    pbar.update(progress)

        # Wait for all workers to finish
        for p in workers:
            p.join()

    # Signal the database writer process to exit
    result_queue.put(None)

    # Print remaining tasks in the result queue
    remaining_results = result_queue.qsize()
    if remaining_results > 0:
        logger.info(f"Waiting for {remaining_results} remaining results to be written.")
    database_writer_process.join()
