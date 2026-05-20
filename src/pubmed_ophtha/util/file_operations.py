"""Functions for restructuring directories and moving and unzipping files."""

import glob
import logging
import os
import shutil
import subprocess
import tarfile
from typing import Optional

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


def unzip(source_path: str, destination_folder: Optional[str] = None) -> None:
    """
    Unzip any archive.

    Args:
        source_path (str): Path to the archive file.
        destination_folder (Optional[str], optional): Path to extract the archive to. \
            If the path is None, will extract the archive into a subfolder in the file \
            directory with the file name. Defaults to None.

    """
    if destination_folder is None:
        destination_folder = os.path.dirname(source_path)

    shutil.unpack_archive(source_path, extract_dir=destination_folder)


def unzip_all(
    source_path: str, destination_folder: Optional[str] = None, override: bool = False
):
    """
    Unzip all archives in the source path and its subdirectories.

    Args:
        source_path (str): Path to folder that will be unzipped.
        destination_folder (Optional[str], optional): Folder to extract the archives to.
            If given, the source folder structure is kept. If not given, the files \
            will be extracted into the source directory. See :func:`unzip`. \
            Defaults to None.
        override (bool, optional): If True, will override existing files. \
            Defaults to False.

    """
    file_list = [
        f
        for f in glob.glob(os.path.join(source_path, "**"), recursive=True)
        if os.path.isfile(f)
        and (
            f.endswith(".zip")
            or f.endswith(".tar.gz")
            or f.endswith(".rar")
            or f.endswith(".tar")
        )
        and (
            override
            or (
                not override
                and not os.path.exists(
                    os.path.join(
                        (
                            os.path.dirname(f).replace(source_path, destination_folder)
                            if destination_folder is not None
                            else os.path.dirname(f)
                        ),
                        os.path.basename(f).split(".")[0],
                    )
                )
            )
        )
    ]

    for f in tqdm(file_list, desc="Extracting"):
        new_path = (
            os.path.dirname(f).replace(source_path, destination_folder)
            if destination_folder is not None
            else None
        )
        try:
            unzip(f, new_path)
        except shutil.ReadError:
            # Could not unzip since it is not an archive
            pass


def compress_and_copy_directory_structure(
    source_path: str, destination_path: str, flat: bool = False
) -> None:
    """
    Compress the structure of the given path.

    Copies the files into a condensed folder structure.
    Depending on the value of :param:`flat` the function either only keeps the folders \
    that contain files, or remove the directory structure and copies all files into \
    the same folder.

    Args:
        source_path (str): Path to the directory structure that should be compressed.
        destination_path (str): Path where the resulting directory should be saved.
        flat (bool, optional): If True, will copy all the files into the \
            :param:`destination_path`. Else, the function will move the files into the \
            same subdirectory in :param:`destination_path`. Defaults to False.

    """
    file_list = [
        f
        for f in glob.glob(os.path.join(source_path, "**"), recursive=True)
        if os.path.isfile(f)
    ]

    for f in tqdm(file_list, desc="Copying"):
        new_path = (
            f.replace(source_path, destination_path)
            if not flat
            else os.path.join(destination_path, os.path.basename(f))
        )

        if not os.path.exists(os.path.dirname(new_path)):
            os.makedirs(os.path.dirname(new_path))

        shutil.copyfile(f, new_path)


def test_and_delete_archive(archive_path: str) -> str | None:
    """
    Test whether the given archive is corrupted and delete it if it is.

    Args:
        archive_path (str): Path to the archive to test.

    Returns:
        str | None: Path to the archive if it is corrupted, else None.

    """
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.getmembers()
            return None
    except Exception:
        try:
            os.remove(archive_path)
        except FileNotFoundError:
            logger.warning(f"Could not delete {archive_path}!")
        return archive_path


def get_current_commit_id(repository_path: str | None = None) -> tuple[str, str]:
    """
    Get the current commit ID and commit description of the repository.

    Args:
        repository_path (str | None, optional): Path to the repository to check. \
            If None gets the commit id from the repository containing this file. \
            Defaults to None.

    Returns:
        tuple[str, str]: Tuple containing:
            - commit_id: The current commit ID.
            - commit_description: The current commit description.

    """
    if repository_path is None:
        repository_path = os.path.dirname(os.path.abspath(__file__))
    commit_id = (
        subprocess.check_output(
            ["git", "describe", "--always"],
            cwd=repository_path,
        )
        .strip()
        .decode()
    )
    commit_description = (
        subprocess.check_output(
            ["git", "log", "-1", "--pretty=format:%H | %D | %s"],
            cwd=repository_path,
        )
        .strip()
        .decode()
    )

    return commit_id, commit_description
