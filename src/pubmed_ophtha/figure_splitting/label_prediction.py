"""
Module containing functions for predicting the labels for the dataset.

Contains functions to predict bounding box labels for images in the database
and predicting the captions for the detected panels.
"""

import json
import logging
import multiprocessing
import os
import sqlite3
from copy import deepcopy
from datetime import datetime
from typing import Any

import torch
from tqdm.auto import tqdm

from pubmed_ophtha.const.models import get_default_model_args
from pubmed_ophtha.figure_splitting.detectron_figure_splitter import (
    DetectronFigureSplitter,
)
from pubmed_ophtha.util.computing import get_cpu_count
from pubmed_ophtha.util.database_interface import (
    database_writer,
    get_database_connection_context,
)

logger = logging.getLogger(__name__)


def write_predictions_into_db(
    con: sqlite3.Connection, batch: list[dict[str, Any]], allow_replace: bool = False
):
    """
    Write batches of predictions into the database.

    Args:
        con (sqlite3.Connection): Connection to the database.
        batch (list[dict[str, Any]]): List of batches. Each batch is a dictionary
            containing:
            - article_id: ID of the article.
            - article_images_id: ID of the image from article_image table.
            - model_version: Version of the model used for prediction.
            - pred_boxes: List of predicted bounding boxes in [x0, y0, x1, y1].
            - pred_classes: List of predicted class names.
            - scores: List of confidence scores for each box.
            - keep_after_nms: List of booleans indicating whether the box was kept after
                NMS.
        allow_replace (bool, optional): If True, existing entries will be replaced.
            Defaults to False.

    """
    cur = con.cursor()
    cur.execute("BEGIN IMMEDIATE;")

    article_id_list = []
    article_images_id_list = []
    boxes_x0_list = []
    boxes_y0_list = []
    boxes_x1_list = []
    boxes_y1_list = []
    labels_list = []
    scores_list = []
    secondary_labels_list = []
    secondary_scores_list = []
    model_version_list = []
    survives_nms_list = []

    for result in batch:
        article_id = result["article_id"]
        article_images_id = result["article_images_id"]
        model_version = result["model_version"]
        for box, label, score, keep_after_nms, secondary_label, secondary_score in zip(
            result["pred_boxes"],
            result["pred_classes"],
            result["scores"],
            result["keep_after_nms"],
            result["secondary_pred_classes"],
            result["secondary_scores"],
        ):
            boxes_x0_list.append(box[0])
            boxes_y0_list.append(box[1])
            boxes_x1_list.append(box[2])
            boxes_y1_list.append(box[3])
            labels_list.append(label)
            scores_list.append(score)
            article_id_list.append(article_id)
            article_images_id_list.append(article_images_id)
            model_version_list.append(model_version)
            survives_nms_list.append(keep_after_nms)
            secondary_labels_list.append(secondary_label)
            secondary_scores_list.append(secondary_score)

    sql_directive = "INSERT" if not allow_replace else "INSERT OR REPLACE"
    # Build parameter tuples in the correct column order and execute once
    params = list(
        zip(
            article_id_list,
            article_images_id_list,
            boxes_x0_list,
            boxes_y0_list,
            boxes_x1_list,
            boxes_y1_list,
            labels_list,
            scores_list,
            secondary_labels_list,
            secondary_scores_list,
            model_version_list,
            survives_nms_list,
        )
    )
    cur.executemany(
        f"""
        {sql_directive} INTO image_predictions (
            article_id,
            article_images_id,
            predicted_boxes_x0,
            predicted_boxes_y0,
            predicted_boxes_x1,
            predicted_boxes_y1,
            predicted_labels,
            predicted_scores,
            secondary_predicted_labels,
            secondary_predicted_scores,
            model_version,
            survives_nms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """,
        params,
    )
    con.commit()


def prediction_worker(
    database_path: str,
    task_queue: multiprocessing.Queue,
    result_queue: multiprocessing.Queue,
    progress_queue: multiprocessing.Queue,
    model_version: str,
    device_name: str | int | None,
    **model_args,
):
    """
    Predict the labels for images in the task_queue.

    Puts the results into the result_queue and updates the progress_queue.

    Args:
        database_path (str): Path to the database file.
        task_queue (multiprocessing.Queue): Queue to get the tasks from.
        result_queue (multiprocessing.Queue): Queue to put the results into.
        progress_queue (multiprocessing.Queue): Queue to update the progress.
        model_weights_path (str): Path to the model weights file.
        model_version (str): Name of the model version.
        device_name (str | int | None): Name or ID of the device to use.
        score_threshold (float): Score threshold for the predictions.
        nms_iou_threshold (float): IoU threshold for non-maximum suppression.
        **model_args: Additional keyword arguments forwarded to DetectronFigureSplitter.

    """
    # Load model
    # Set the device
    if device_name is not None:
        if device_name == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(device_name)

    model = DetectronFigureSplitter(**model_args)

    try:
        with get_database_connection_context(database_path, read_only=True) as con:
            cur = con.cursor()
            while True:
                task = task_queue.get()
                if task is None:
                    # Poison pill means shutdown
                    break
                article_images_id = task

                # Get article_id and image bytes
                cur.execute(
                    "SELECT article_id, image FROM article_images WHERE id = ?;",
                    (article_images_id,),
                )
                row = cur.fetchone()
                if row is None:
                    logger.warning(
                        f"Article Images ID {article_images_id} not found in database."
                    )
                    progress_queue.put(1)  # Indicate that one task is done
                    continue

                article_id, image_bytes = row

                try:
                    model_prediction = model.predict(image_bytes)

                    prediction = {
                        **model_prediction,
                        "article_id": article_id,
                        "article_images_id": article_images_id,
                        "model_version": model_version,
                    }

                    result_queue.put(prediction)
                except Exception as e:
                    logger.error(
                        f"Error processing image {article_images_id} of article "
                        f"{article_id}: {e}"
                    )
                finally:
                    progress_queue.put(1)  # Indicate that one task is done
    except Exception as e:
        logger.error(f"Error in prediction worker: {e}")
    finally:
        progress_queue.put(None)  # Indicate that this worker is done


def predict_bounding_box_labels(
    database_path: str,
    num_workers: int | None = None,
    database_batch_size: int = 100,
    **model_args,
):
    """
    Predict the bounding box labels for all images in the database.

    The predictions are written into the image_predictions table in the database.
    The images are taken from the article_images table.

    Args:
        database_path (str): Path to the database file.
        num_workers (int | None, optional): Number of workers to use for prediction.
            Will be adapted to the number of possible workers based on the GPU
            capacities. If None, will be inferred using the number of CPUs and GPUs
            and the VRAM size. Defaults to None.
        database_batch_size (int, optional): Size of the batches to write into the
            database. Defaults to 100.
        **model_args: Additional keyword arguments forwarded to DetectronFigureSplitter.
            Can be used to specify model weights, config parameters, etc.

    Note:
        If every ``article_images`` row already has an entry in
        ``image_predictions``, ``tasks_to_process`` will be empty and the
        function completes without doing any prediction work. No log message is
        emitted to indicate this; the caller receives a normal return with no
        output distinguishable from a run that processed images.

    """
    if num_workers is None:
        num_workers = max(1, get_cpu_count() - 1)

    num_workers = max(num_workers, 1)

    estimated_model_size_gb = DetectronFigureSplitter.get_estimated_size_gb()

    defaults = get_default_model_args()

    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = DetectronFigureSplitter.get_model_name()

    if model_args is None:
        model_args = {}

    def _deep_merge(base: dict, override: dict) -> dict:
        # Mutates base and returns it
        for k, v in override.items():
            if k in base and isinstance(base[k], dict) and isinstance(v, dict):
                _deep_merge(base[k], v)
            else:
                base[k] = deepcopy(v)
        return base

    model_args = _deep_merge(deepcopy(defaults), model_args)

    if num_workers > 1:
        # make sure that there are enough devices per worker
        if torch.cuda.is_available():
            num_devices = torch.cuda.device_count()
            # Get VRAM on all devices
            vram_sizes = [
                torch.cuda.get_device_properties(i).total_memory
                for i in range(num_devices)
            ]
            # Sort devices by VRAM size
            sorted_devices = sorted(
                range(num_devices), key=lambda i: vram_sizes[i], reverse=True
            )

            # Select devices with enough VRAM
            selected_devices = []
            num_possible_workers = 0
            buffer_factor = 1.1
            for device in sorted_devices:
                if (
                    vram_sizes[device] > estimated_model_size_gb * buffer_factor
                ):  # 50% buffer
                    selected_devices.append(device)
                    num_possible_workers += int(
                        vram_sizes[device] // (estimated_model_size_gb * buffer_factor)
                    )

            if num_possible_workers < num_workers:
                logger.warning(
                    f"Limited VRAM: Reducing workers to {num_possible_workers}."
                )
                num_workers = num_possible_workers

            # Assign workers to devices in round-robin fashion
            worker_device_map = [
                selected_devices[i % len(selected_devices)] for i in range(num_workers)
            ]
        else:
            # All workers on CPU
            worker_device_map = ["cpu"] * num_workers
    else:
        worker_device_map = [
            torch.cuda.current_device() if torch.cuda.is_available() else "cpu"
        ]

    with get_database_connection_context(database_path, read_only=False) as con:
        cur = con.cursor()

        # Create metadata table if not exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS _label_prediction_metadata(
                model_version TEXT PRIMARY KEY,
                model_args TEXT,
                prediction_date TEXT
            )
            """
        )

        # Check if current model name is already in metadata
        cur.execute(
            """
            SELECT COUNT(*) FROM _label_prediction_metadata
            WHERE model_version = ?;
            """,
            (model_version,),
        )

        if cur.fetchone()[0] == 0:
            # Insert metadata
            cur.execute(
                """
                INSERT INTO _label_prediction_metadata (
                    model_version,
                    model_args,
                    prediction_date
                ) VALUES (?, ?, ?);
                """,
                (
                    model_version,
                    json.dumps(model_args),
                    date_str,
                ),
            )
            con.commit()
        else:
            # Retrieve existing model args
            cur.execute(
                """
                SELECT model_args FROM _label_prediction_metadata
                WHERE model_version = ?;
                """,
                (model_version,),
            )
            model_args = json.loads(cur.fetchone()[0])

        # Create table if not exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS image_predictions(
                id INTEGER PRIMARY KEY ON CONFLICT REPLACE,
                article_id INTEGER,
                article_images_id INTEGER,
                predicted_boxes_x0 INTEGER,
                predicted_boxes_y0 INTEGER,
                predicted_boxes_x1 INTEGER,
                predicted_boxes_y1 INTEGER,
                predicted_labels TEXT,
                predicted_scores REAL,
                secondary_predicted_labels TEXT,
                secondary_predicted_scores REAL,
                model_version TEXT,
                survives_nms BOOLEAN,
                FOREIGN KEY(article_images_id) REFERENCES article_images(id),
                FOREIGN KEY(article_id) REFERENCES metadata(article_id)
            )
            """
        )

        # Create index for faster lookup
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_image_predictions_id_image_article_id
            ON image_predictions (id, article_images_id, article_id);
            """
        )
        # Get all tasks
        cur.execute("SELECT id FROM article_images;")
        total_tasks = cur.fetchall()

        # Get already processed tasks
        cur.execute("SELECT DISTINCT article_images_id FROM image_predictions;")
        processed_tasks = {row[0] for row in cur.fetchall()}

        tasks_to_process = [
            row[0] for row in total_tasks if row[0] not in processed_tasks
        ]

    mp_context = multiprocessing.get_context("spawn")

    # Prepare queues for multiprocessing
    result_queue = mp_context.Queue(maxsize=1000)
    task_queue = mp_context.Queue()

    # Queue to track progress
    progress_queue = mp_context.Queue()

    # Prepare process to write into the database
    database_writer_process = mp_context.Process(
        target=database_writer,
        args=(database_path, result_queue, write_predictions_into_db),
        kwargs={
            "batch_size": database_batch_size,
        },
        name="DatabaseWriterProcess",
    )
    database_writer_process.start()

    # Put tasks into the task queue
    for task in tasks_to_process:
        task_queue.put(task)

    num_devices = len(set(worker_device_map))
    logger.info(
        f"Starting prediction with {num_workers} workers on {num_devices} devices..."
    )

    # Start workers
    worker_processes = []
    for i in range(num_workers):
        p = mp_context.Process(
            target=prediction_worker,
            args=(
                database_path,
                task_queue,
                result_queue,
                progress_queue,
                model_version,
                worker_device_map[i],
            ),
            kwargs=model_args,
            name=f"PredictionWorker-{i}",
        )
        p.start()
        worker_processes.append(p)

    # Add poison pills to stop the workers
    for _ in range(num_workers):
        task_queue.put(None)

    # Track progress
    total_tasks_count = len(tasks_to_process)
    num_none = 0
    with tqdm(total=total_tasks_count, desc="Predict Bounding Boxes") as pbar:
        while num_none < num_workers:
            progress = progress_queue.get()
            if progress is None:
                num_none += 1
            else:
                pbar.update(progress)

    # Wait for all workers to finish
    for p in worker_processes:
        p.join()

    # Stop the database writer
    result_queue.put(None)

    # Print remaining tasks in the result queue
    remaining_results = result_queue.qsize()
    if remaining_results > 0:
        logger.info(f"Waiting for {remaining_results} remaining results to be written.")
    database_writer_process.join()
