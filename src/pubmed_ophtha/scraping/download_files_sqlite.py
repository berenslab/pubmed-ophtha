"""
Module to download packages matching certain keywords from pubmed central (PMC).

Same implementation as in `download_files.py`, but with a SQLite database to store
the files.
"""

import asyncio
import datetime
import logging
import os
import tarfile
from ftplib import FTP, error_perm, error_temp
from io import BytesIO
from typing import List, Optional

from tqdm.asyncio import tqdm_asyncio

from pubmed_ophtha.const.urls import (
    PUBMED_CENTRAL_OA_PACKAGE_PATH,
    PUBMED_FTP_SERVER,
)
from pubmed_ophtha.scraping.download_files import load_oa_file_list, safe_download_file
from pubmed_ophtha.util.database_interface import get_database_connection
from pubmed_ophtha.util.file_operations import get_current_commit_id

logger = logging.getLogger(__name__)


def download_file(
    file_path: str, database_path: str, url: str, **metadata_kwargs
) -> bool:
    """
    Download a file from the PubMed Central FTP server.

    Args:
        file_path (str): Path to the file in the oa_package on the PMC FTP server.
        database_path (str): Path to the SQLite database file.
        url (str): URL to the FTP server.
        **metadata_kwargs: Additional metadata to be stored in the database. \
            the keys are the column names and the values are the values to be stored.

    Returns:
        bool: Success

    """

    def write_to_database(binary_data):
        con = get_database_connection(database_path, read_only=False)
        cur = con.cursor()

        article_int_id = int(
            os.path.basename(file_path).removeprefix("PMC").removesuffix(".tar.gz")
        )

        metadata_str = ""
        metadata_q_str = ""

        if len(metadata_kwargs) > 0:
            metadata_str = ", " + ", ".join(metadata_kwargs.keys())
            metadata_q_str = ", ?" * len(metadata_kwargs)

        data_to_store = binary_data
        if len(binary_data) > 900000000:
            # Exceeds SQL limits
            data_to_store = None

        cur.execute("BEGIN IMMEDIATE;")
        cur.execute(
            f"""
            INSERT INTO article_packages (
                article_id,
                file_path,
                file_data{metadata_str}
            )
            VALUES (?, ?, ?{metadata_q_str})
            """,
            (article_int_id, file_path, binary_data, *metadata_kwargs.values())
            if len(metadata_kwargs) > 0
            else (article_int_id, file_path, data_to_store),
        )

        if data_to_store is None:
            # Store in chunked_articles table
            chunk_size = 800000000  # Reduced chunk size to avoid SQLite limits

            saved_chunks = 0
            chunk_index = 0

            while saved_chunks < len(binary_data):
                chunk_part = chunk_index
                chunk_data = binary_data[
                    saved_chunks : min(saved_chunks + chunk_size, len(binary_data))
                ]

                cur.execute(
                    """INSERT INTO chunked_articles (
                        article_id,
                        chunk_part,
                        file_data
                    )
                    VALUES (?, ?, ?)""",
                    (article_int_id, chunk_part, chunk_data),
                )

                saved_chunks += len(chunk_data)
                chunk_index += 1
        con.commit()
        con.close()

    try:
        with FTP(url, user="anonymous") as ftp:
            binary_data = bytearray()

            def handle_binary(more_data):
                binary_data.extend(more_data)

            ftp.retrbinary(f"RETR {file_path}", handle_binary)
    except (error_temp, error_perm) as e:
        logger.error(f"Error downloading {file_path}: {e}")
        return False

    # Check if the file sanity
    try:
        with tarfile.open(fileobj=BytesIO(binary_data)) as tar:
            tar.getmembers()
    except tarfile.TarError:
        return False

    # Write to database
    write_to_database(bytes(binary_data))
    return True


async def _download_from_pmc_ids(
    pmc_ids: List[str],
    base_url: str,
    database_path: str,
    file_list_folder: str,
    override_existing: bool = False,
    number_of_workers: int = 10,
    max_files: Optional[int] = None,
) -> bool:
    """
    Download articles with the specified PMC ids to the specified path.

    Args:
        pmc_ids (List[str]): List of PMC ids to download.
        base_url (str): Base URL to the folder containing the file on the PMC \
            FTP Server.
        database_path (str): Path to the SQLite database file.
        override_existing (bool, optional): If True, will override existing files in \
            output folder. Defaults to False.
        number_of_workers (int, optional): Number of concurrent download workers. \
            Defaults to 10.
        max_files (Optional[int], optional): Number of files to download. Defaults to \
            None (= all files).
        meta_dict (Optional[Dict[str, Any]], optional): Additional metadata to include \
            in the meta info file. Defaults to None.
        file_list_folder (Optional[str], optional): Folder containing the PMC OA file \
            list. If the file is not present in the folder it will be automatically \
            downloaded. If None, the file will be saved in the output path. \
            Defaults to None.

    Returns:
        bool: Success

    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sem = asyncio.Semaphore(number_of_workers)

    con = get_database_connection(database_path, read_only=False)
    cur = con.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS article_packages(
            article_id INTEGER PRIMARY KEY,
            file_path TEXT,
            file_data BLOB,
            commit_id TEXT,
            commit_description TEXT,
            timestamp TEXT
        )
        """
    )

    # Table for articles that exceed the maximum size for a single file
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS chunked_articles(
            id INTEGER PRIMARY KEY ON CONFLICT REPLACE,
            article_id INTEGER,
            chunk_part INTEGER NOT NULL,
            file_data BLOB
        )
        """
    )
    con.commit()

    # Assure only download once
    download_pmc_ids = list(set(pmc_ids))

    # Create meta info file
    commit_id, commit_description = get_current_commit_id(
        os.path.dirname(os.path.abspath(__file__))
    )

    def download_file_with_metadata(*args):
        return download_file(
            *args,
            commit_id=commit_id,
            commit_description=commit_description,
            timestamp=timestamp,
        )

    file_df = load_oa_file_list(file_list_folder, PUBMED_FTP_SERVER)

    # Get all PMC ids from the database
    cur.execute(
        "SELECT article_id FROM article_packages",
    )
    existing_pmc_ids = {f"PMC{row[0]}" for row in cur.fetchall()}  # Convert to PMC ids
    con.close()

    # Find ids in df
    file_df = file_df[
        file_df["Accession ID"].isin(["PMC" + entry for entry in download_pmc_ids])
    ].copy()
    file_df["File"] = file_df["File"].str.replace("oa_package/", "")
    file_df["exists"] = False
    file_df["exists"] = file_df["File"].apply(
        lambda file_name: os.path.basename(file_name).removesuffix(".tar.gz")
        in existing_pmc_ids
    )
    file_df["success"] = False

    paths_to_download = file_df
    if max_files is not None:
        paths_to_download = file_df.iloc[:max_files]

    paths_to_download = (
        paths_to_download[~paths_to_download["exists"]]["File"]
        if not override_existing
        else paths_to_download["File"]
    )

    tasks = [
        asyncio.create_task(
            safe_download_file(
                f"{base_url}/{file_name}",
                database_path,
                PUBMED_FTP_SERVER,
                sem,
                download_fn=download_file_with_metadata,
            )
        )
        for file_name in paths_to_download
    ]

    # Gather results and check for missing files
    download_results = await tqdm_asyncio.gather(*tasks)
    file_df.loc[paths_to_download.index, "success"] = download_results

    if (~file_df.loc[paths_to_download.index, "success"]).any():
        logger.warning("Could not download all files!")
        return False
    return True


async def download_oa_package_from_pmc_ids(
    pmc_ids: List[str],
    database_path: str,
    file_list_folder: str,
    override_existing: bool = False,
    number_of_workers: int = 10,
    max_files: Optional[int] = None,
) -> bool:
    """
    Download article packages with the specified PMC ids to the specified path.

    Args:
        pmc_ids (List[str]): List of PMC ids to download.
        database_path (str): Path to the SQLite database file.
        override_existing (bool, optional): If True, will override existing files in \
            output folder. Defaults to False.
        number_of_workers (int, optional): Number of concurrent download workers. \
            Defaults to 10.
        max_files (Optional[int], optional): Number of files to download. Defaults to \
            None (= all files).
        meta_dict (Optional[Dict[str, Any]], optional): Additional metadata to include \
            in the meta info file. Defaults to None.
        file_list_folder (Optional[str], optional): Folder containing the PMC OA file \
            list. If the file is not present in the folder it will be automatically \
            downloaded. If None, the file will be saved in the output path. \
            Defaults to None.

    Returns:
        bool: Success

    """
    return await _download_from_pmc_ids(
        pmc_ids,
        PUBMED_CENTRAL_OA_PACKAGE_PATH,
        database_path,
        file_list_folder,
        override_existing=override_existing,
        number_of_workers=number_of_workers,
        max_files=max_files,
    )
