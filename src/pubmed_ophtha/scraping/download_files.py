"""Module to download packages matching certain keywords from pubmed central (PMC)."""

import asyncio
import datetime
import json
import logging
import os
from ftplib import FTP, error_perm, error_temp
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from tqdm.asyncio import tqdm_asyncio

from pubmed_ophtha.const.urls import (
    PUBMED_CENTRAL_OA_FILE_LIST_NAME,
    PUBMED_CENTRAL_OA_PACKAGE_PATH,
    PUBMED_CENTRAL_OA_PMC_BASE_PATH,
    PUBMED_FTP_SERVER,
)
from pubmed_ophtha.scraping.esearch import search_pmc, search_pmc_with_keywords
from pubmed_ophtha.util.file_operations import get_current_commit_id

logger = logging.getLogger(__name__)


def download_file(file_path: str, output_path: str, url: str) -> bool:
    """
    Download a file from the PubMed Central FTP server.

    Args:
        file_path (str): Path to the file in the oa_package on the PMC FTP server.
        output_path (str): Folder to save the file to
        url (str): URL to the FTP server.

    Returns:
        bool: Success

    """
    try:
        with FTP(url, user="anonymous") as ftp:
            with open(output_path, "wb") as f:
                ftp.retrbinary(f"RETR {file_path}", f.write)
    except (error_temp, error_perm) as e:
        logger.error(f"Error downloading {file_path}: {e}")
        return False

    return True


async def safe_download_file(
    file_path: str,
    output_path: str,
    url: str,
    sem: asyncio.Semaphore,
    download_fn: Callable[[str, str, str], bool] = download_file,
) -> bool:
    """
    Wrapper for :func:`download_file`, that uses a semaphore to limit concurrency.

    Args:
        file_path (str): Path to the file in the oa_package on the PMC FTP server.
        output_path (str): Folder to save the file to
        url (str): URL to the FTP server.
        sem (asyncio.Semaphore): Semaphore that limits number of concurrent download \
            workers.
        download_fn (Callable[[str, str, str], bool], optional): Function to use for \
            downloading files. Defaults to `scraping.download_files.download_file`.

    Returns:
        bool: Success

    """  # noqa: D401
    async with sem:
        # Avoid blocking event-loop
        download_result = await asyncio.to_thread(
            download_fn, file_path, output_path, url
        )
        await asyncio.sleep(0.1)  # Hand back to main thread to update progress bar

        return download_result


def load_oa_file_list(output_path: str, url: str) -> pd.DataFrame:
    """
    Load the file list from PMC open access as dataframe and save the file to the \
        output directory.

    Args:
        output_path (str): Folder to save the file to
        url (str): URL to the FTP server.

    Returns:
        pd.DataFrame: Dataframe containing the OA files. \
            See: https://pmc.ncbi.nlm.nih.gov/tools/ftp/#indart

    """  # noqa: D205
    file_list_path = os.path.join(output_path, PUBMED_CENTRAL_OA_FILE_LIST_NAME)
    if not os.path.exists(file_list_path):
        logger.info("Did not find OA file list. Starting download.")

        remote_path = (
            f"{PUBMED_CENTRAL_OA_PMC_BASE_PATH}/{PUBMED_CENTRAL_OA_FILE_LIST_NAME}"
        )
        with FTP(url, user="anonymous") as ftp:
            file_size = ftp.size(remote_path)
            with open(file_list_path, "wb") as local_file:
                with tqdm(
                    total=file_size, unit="B", unit_scale=True, desc="Downloading"
                ) as progress_bar:

                    def write_with_tqdm(chunk):
                        local_file.write(chunk)
                        progress_bar.update(len(chunk))

                    ftp.retrbinary(
                        f"RETR {remote_path}", write_with_tqdm, blocksize=32768
                    )

    return pd.read_csv(file_list_path)


async def _download_from_pmc_ids(
    pmc_ids: List[str],
    base_url: str,
    output_path: str,
    override_existing: bool = False,
    number_of_workers: int = 10,
    max_files: Optional[int] = None,
    meta_dict: Optional[Dict[str, Any]] = None,
    file_list_folder: Optional[str] = None,
) -> bool:
    """
    Download articles with the specified PMC ids to the specified path.

    Args:
        pmc_ids (List[str]): List of PMC ids to download.
        base_url (str): Base URL to the folder containing the file on the PMC \
            FTP Server.
        output_path (str): Folder to save the ids to.
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
    # Assure only download once
    download_pmc_ids = list(set(pmc_ids))

    if file_list_folder is None:
        file_list_folder = output_path

    # Set up folder structure to avoid concurrency issues
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i in range(256):
        for j in range(256):
            folder_name = os.path.join(output_path, f"{i:02x}", f"{j:02x}")

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

    # Create meta info file
    commit_id, commit_description = get_current_commit_id(
        os.path.dirname(os.path.abspath(__file__))
    )
    with open(os.path.join(output_path, f"meta_{timestamp}.json"), "w") as json_file:
        json.dump(
            {
                "timestamp": timestamp,
                "commit_id": commit_id,
                "commit_description": commit_description,
                "number_of_ids": len(pmc_ids),
                "additional": meta_dict if meta_dict is not None else {},
            },
            json_file,
            indent=2,
            ensure_ascii=False,
        )

    file_df = load_oa_file_list(file_list_folder, PUBMED_FTP_SERVER)

    # Find ids in df
    file_df = file_df[
        file_df["Accession ID"].isin(["PMC" + entry for entry in download_pmc_ids])
    ].copy()
    file_df["File"] = file_df["File"].str.replace("oa_package/", "")
    file_df["exists"] = False
    file_df["exists"] = file_df["File"].apply(
        lambda file_name: os.path.exists(
            os.path.join(output_path, *(file_name.split("/")))
        )
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
                os.path.join(output_path, *(file_name.split("/"))),
                PUBMED_FTP_SERVER,
                sem,
            )
        )
        for file_name in paths_to_download
    ]

    # Gather results and check for missing files
    download_results = await tqdm_asyncio.gather(*tasks)
    file_df.loc[paths_to_download.index, "success"] = download_results
    file_df.to_csv(os.path.join(output_path, f"file_list_{timestamp}.csv"))

    if (~file_df.loc[paths_to_download.index, "success"]).any():
        logger.warning("Could not download all files!")
        return False
    return True


async def download_oa_package_from_pmc_ids(
    pmc_ids: List[str],
    output_path: str,
    override_existing: bool = False,
    number_of_workers: int = 10,
    max_files: Optional[int] = None,
    meta_dict: Optional[Dict[str, Any]] = None,
    file_list_folder: Optional[str] = None,
) -> bool:
    """
    Download article packages with the specified PMC ids to the specified path.

    Args:
        pmc_ids (List[str]): List of PMC ids to download.
        output_path (str): Folder to save the ids to.
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
        output_path,
        override_existing=override_existing,
        number_of_workers=number_of_workers,
        max_files=max_files,
        meta_dict=meta_dict,
        file_list_folder=file_list_folder,
    )


def download_from_pubmed_central_with_keywords(
    keywords: List[str],
    output_path: str,
    max_files: Optional[int] = None,
    file_list_folder: Optional[str] = None,
) -> bool:
    """
    Download files from PubMed Central OA that match any of the keywords.

    Args:
        keywords (List[str]): List of keywords to find matching files.
        output_path (str): Folder where the downloaded files will be saved.
        max_files (Optional[int], optional): Number of files to download. Defaults to \
            None (= all files).
        file_list_folder (Optional[str], optional): Folder containing the PMC OA file \
            list. If the file is not present in the folder it will be automatically \
            downloaded. If None, the file will be saved in the output path. \
            Defaults to None.

    Returns:
        bool: Success

    """
    # Get PMC ids
    pmc_ids = search_pmc_with_keywords(
        keywords, maximum_results=max_files, step_size=100000
    )

    # Download files
    return asyncio.run(
        download_oa_package_from_pmc_ids(
            pmc_ids,
            output_path,
            max_files=max_files,
            meta_dict=dict(keywords=keywords),
            file_list_folder=file_list_folder,
        )
    )


def download_from_pubmed_central(
    search_query: str,
    output_path: str,
    max_files: Optional[int] = None,
    file_list_folder: Optional[str] = None,
) -> bool:
    """
    Download files from PubMed Central OA that match the search term.

    Args:
        search_query (str): Query that is used for searching. See https://www.ncbi.nlm.nih.gov/pmc/advanced/.
        output_path (str): Folder where the downloaded files will be saved.
        max_files (Optional[int], optional): Number of files to download. Defaults to \
            None (= all files).
        file_list_folder (Optional[str], optional): Folder containing the PMC OA file \
            list. If the file is not present in the folder it will be automatically \
            downloaded. If None, the file will be saved in the output path. \
            Defaults to None.

    Returns:
        bool: Success

    """
    # Get PMC ids
    pmc_ids = search_pmc(search_query, maximum_results=max_files, step_size=100000)

    # Download files
    return asyncio.run(
        download_oa_package_from_pmc_ids(
            pmc_ids,
            output_path,
            max_files=max_files,
            meta_dict=dict(search_query=search_query),
            file_list_folder=file_list_folder,
        )
    )


def download_packages_from_pubmed_central(
    article_ids: List[str],
    output_path: str,
    override_existing: bool = False,
    number_of_workers: int = 10,
    file_list_folder: Optional[str] = None,
) -> bool:
    """
    Download the specified article packages from PubMed Central (PMC).

    Args:
        article_ids (List[str]): List of article ids to download.
        output_path (str): Path to the folder where the packages will be saved.
        override_existing (bool, optional): If True, override existing packages. \
            Defaults to False.
        number_of_workers (int, optional): Number of download workers. Defaults to 10.
        file_list_folder (Optional[str], optional): Folder to save the PMC OA file \
            list to. Defaults to None.

    Returns:
        bool: True if all packages were downloaded successfully, False otherwise.

    """
    return asyncio.run(
        download_oa_package_from_pmc_ids(
            article_ids,
            output_path,
            override_existing=override_existing,
            number_of_workers=number_of_workers,
            file_list_folder=file_list_folder,
        )
    )


def download_fundus_dataset(
    output_path: str,
    max_files: Optional[int] = None,
    file_list_folder: Optional[str] = None,
) -> bool:
    """
    Download all articles related to fundus images from PMC.

    Args:
        output_path (str): Folder where the downloaded files will be saved.
        max_files (Optional[int], optional): Number of files to download. Defaults to \
            None (= all files).
        file_list_folder (Optional[str], optional): Folder containing the PMC OA file \
            list. If the file is not present in the folder it will be automatically \
            downloaded. If None, the file will be saved in the output path. \
            Defaults to None.

    Returns:
        bool: Success.

    """
    imaging_keywords = [
        "Color Fundus Photography",
        "CFP",
        # "fundus",
        # "OCT",
        "optical coherence tomography",
    ]

    figure_query = (
        " OR ".join([f"{entry}[Figure/Table Caption]" for entry in imaging_keywords])
        + " OR fundus[Figure/Table Caption]"
    )
    body_query = (
        " OR ".join([f"{entry}[Body - All Words]" for entry in imaging_keywords])
        + " OR retina[Body - All Words]"
    )
    key_terms_query = (
        " OR ".join([f"{entry}[Body - Key Terms]" for entry in imaging_keywords])
        + " OR retina[Body - Key Terms]"
    )

    search_query = (
        f"({figure_query} AND ({body_query} OR {key_terms_query}))"
        + " OR (OCT[Figure/Table Caption]"
        + " AND (retina[Body - All Words] OR ophthalmology[Body - All Words]))"
    )
    return download_from_pubmed_central(
        # ["Color Fundus Photography", "ophthalmology", "retina"], output_path
        search_query,
        output_path,
        max_files=max_files,
        file_list_folder=file_list_folder,
    )
