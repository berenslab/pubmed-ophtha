"""Utility functions for database interaction."""

import logging
import multiprocessing
import sqlite3
import time
from contextlib import contextmanager
from typing import Callable, Generator

import pandas as pd

logger = logging.getLogger(__name__)


def get_database_connection(
    database_path: str, read_only: bool = False
) -> sqlite3.Connection:
    """
    Create a multi-processing safe SQLite database connection.

    Args:
        database_path (str): Path to the SQLite database file.
        read_only (bool, optional): If True, establish a read-only connection. \
            Defaults to False.

    Returns:
        sqlite3.Connection: Established connection to the SQLite database.

    """
    # Following advice in https://www.reddit.com/r/golang/comments/16xswxd/concurrency_when_writing_data_into_sqlite/
    uri = f"file:{database_path}?mode=ro" if read_only else database_path
    con = sqlite3.connect(uri, uri=read_only, timeout=5.0)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")
    cur.execute("PRAGMA busy_timeout=5000;")
    return con


@contextmanager
def get_database_connection_context(
    database_path: str, read_only: bool = False
) -> Generator[sqlite3.Connection, None, None]:
    """
    Get a context manager for a database connection.

    After exiting the context, the connection is closed.

    Args:
        database_path (str): Path to the SQLite database file.
        read_only (bool, optional): If True, the database is read-only. \
            Defaults to False.

    Yields:
        sqlite3.Connection: Connection to the database.

    """
    con = get_database_connection(database_path, read_only=read_only)
    try:
        yield con
    finally:
        con.close()


def database_writer(
    database_path: str,
    result_queue: multiprocessing.Queue,
    writing_callback: Callable[[sqlite3.Connection, list], None],
    batch_size: int = 10,
):
    """
    Run the database writer process.

    Checks the result queue for new results and writes them into the database in \
    batches. If no new results are available, it waits for a short time before \
    checking again. If the result is None, the process exits.

    Args:
        database_path (str): Path to the database file.
        result_queue (multiprocessing.Queue): Queue to get the results from.
        writing_callback (Callable[[sqlite3.Connection, list], None]): Callback
            function to write a batch of results into the database. It should take two
            arguments: the database connection and a list of results.
        batch_size (int, optional): Size of the batches to gather. Defaults to 10.

    """
    logger.debug("Starting database writer process")

    batch = []
    total_written = 0
    num_write = 0
    sum_write_time = 0.0

    start_time = time.time()
    intermediate_start_time = start_time

    def log_performance(intermediate: bool = False):
        current_start_time = start_time
        end_time = time.time()
        write_text = "finished writing" if not intermediate else "written"
        print_string = (
            f"The database writer process has {write_text} {total_written} articles."
        )
        print_string += (
            f"Database total time: {end_time - current_start_time:.2f} seconds"
        )
        if num_write > 0:
            tpw = sum_write_time / num_write
            print_string += f"Database average write time per batch: {tpw:.2f} seconds"

        if total_written > 0:
            tpw = (end_time - current_start_time) / total_written
            print_string += f"Database average time per article: {tpw:.2f} seconds"

        logger.debug(print_string)

    with get_database_connection_context(database_path, read_only=False) as con:
        while True:
            result = result_queue.get()
            if result is None:
                # Sentinel value to indicate the end of processing
                break

            batch.append(result)

            if len(batch) >= batch_size:
                write_start_time = time.time()
                writing_callback(con, batch)
                sum_write_time += time.time() - write_start_time
                total_written += len(batch)
                num_write += 1
                logger.debug(f"Wrote {total_written} articles to the database")
                batch = []

            if time.time() - intermediate_start_time > 300 and total_written > 0:
                # Log performance every 5 minutes
                log_performance(intermediate=True)
                intermediate_start_time = time.time()

        # Write any remaining items in the batch
        write_start_time = time.time()
        writing_callback(con, batch)
        sum_write_time += time.time() - write_start_time
        num_write += 1
        total_written += len(batch)
        if len(batch) > 0:
            logger.debug(f"Wrote {total_written} articles to the database")

    log_performance()


def get_biomedica_df(database_path: str) -> pd.DataFrame:
    """
    Get the BIOMEDICA dataframe.

    Args:
        database_path (str): Path to the SQLite database file.

    Returns:
        pd.DataFrame: BIOMEDICA dataframe.

    """
    with get_database_connection_context(database_path, read_only=True) as con:
        cur = con.cursor()
        cur.execute("SELECT * FROM biomedica_data_file_list")
        biomedica_df = pd.DataFrame(
            cur.fetchall(), columns=[desc[0] for desc in cur.description]
        )

    return biomedica_df
