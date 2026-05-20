"""Module to filter the Biomedica dataset for images related to retinal imaging."""

import gc
import glob
import logging
import os
import shutil

import pandas as pd
from huggingface_hub import login
from tqdm.auto import tqdm

from pubmed_ophtha.filtering.download_biomedica import save_biomedica_meta_to_parquet
from pubmed_ophtha.scraping.esearch import search_pmc
from pubmed_ophtha.util.database_interface import get_database_connection_context

logger = logging.getLogger(__name__)


def filter_biomedica(biomedica_path: str, output_path: str, clean: bool = False):
    """
    Filter the Biomedica dataset for figures related to retinal imaging.

    Uses a PMC search query and the article subject from the dataset for filtering.
    Saves the filtered dataset to a new directory, with the same structure \
        as the original.
    Saved parquet files may be empty.

    Args:
        biomedica_path (str): Path to the folder containing the Biomedica dataset.
        output_path (str): Folder to save the filtered dataset to.
        clean (bool, optional): If true the output folder will be cleaned \
            before processing. Defaults to False.

    """
    HF_TOKEN = os.getenv("HF_TOKEN")
    login(HF_TOKEN)

    if (
        not os.path.exists(biomedica_path)
        or len(glob.glob(os.path.join(biomedica_path, "*"))) == 0
    ):
        save_biomedica_meta_to_parquet(biomedica_path, temp_save_interval=50000)

    # Retrieve relevant pubmed ids
    imaging_keywords = [
        "Color Fundus Photography",
        "CFP",
        # "fundus",
        "OCT",
        "optical coherence tomography",
    ]

    figure_query = (
        " OR ".join([f"{entry}[Figure/Table Caption]" for entry in imaging_keywords])
        + " OR fundus[Figure/Table Caption]"
    )
    # body_query = (
    #     " OR ".join([f"{entry}[Body - All Words]" for entry in imaging_keywords])
    #     + " OR retina[Body - All Words]"
    # )
    key_terms_query = (
        " OR ".join([f"{entry}[Body - Key Terms]" for entry in imaging_keywords])
        + " OR retina[Body - Key Terms] OR tomography, optical coherence[MeSH Terms] "
        + "OR ophthalmoscopy[MeSH Terms]"  # OR fluorescein angiography[MeSH Terms]
    )

    search_query = (
        f"(({figure_query}) AND ({key_terms_query}))"
        + " AND (retina[Body - All Words] OR ophthalmology[Body - All Words] OR "
        + "fundus oculi[MeSH Terms] OR retina[MeSH Terms])"
    )

    pmc_ids = search_pmc(search_query, maximum_results=None, step_size=100000)

    pmc_ids = [f"PMC{pmc_id}" for pmc_id in pmc_ids]

    retinal_pmc_ids = search_pmc(
        "ophthalmology[Body - Key Terms] OR retina[Body - Key Terms] OR "
        + "fundus oculi[MeSH Terms] OR retina[MeSH Terms]",
        maximum_results=None,
        step_size=100000,
    )
    retinal_pmc_ids = [f"PMC{pmc_id}" for pmc_id in retinal_pmc_ids]

    if (
        os.path.exists(output_path)
        and len(glob.glob(os.path.join(output_path, "*"))) > 0
    ):
        if clean:
            shutil.rmtree(output_path)

    os.makedirs(output_path, exist_ok=True)
    original_dataset_size = 0
    filtered_size = 0

    for file_index, file in enumerate(
        tqdm(glob.glob(os.path.join(biomedica_path, "*.parquet")))
    ):
        gc.collect()
        output_file_path = os.path.join(
            output_path, f"biomedica_filtered_{file_index}.parquet"
        )
        if "old.parquet" in file or "_tmp" in file or os.path.exists(output_file_path):
            continue

        # Load the parquet file
        dataset = pd.read_parquet(file)
        original_dataset_size += len(dataset)
        filtered = dataset.loc[
            (dataset["article_id"].isin(pmc_ids))
            | (
                dataset["image_secondary_label"].apply(
                    lambda x: "optical coherence tomography" in x
                )
                & (dataset["article_id"].isin(retinal_pmc_ids))
            )
        ].copy()
        del dataset

        filtered = filtered.assign(filtering_date=pd.Timestamp.now())
        filtered = filtered.assign(origin_file=file)
        filtered.to_parquet(output_file_path)
        filtered_size += len(filtered)
        del filtered

    logger.info(
        f"Original size: {original_dataset_size}, filtered size: {filtered_size}"
    )


def join_filtered_files(filtered_biomedica_path: str, output_file_path: str):
    """
    Join the filtered Biomedica dataset files into one.

    Args:
        filtered_biomedica_path (str): Folder containing filtered dataset files.
        output_file_path (str): File name to save the result to.

    Raises:
        ValueError: If no filtered files were found.

    """
    if not os.path.exists(os.path.dirname(output_file_path)):
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    filtered_files = glob.glob(os.path.join(filtered_biomedica_path, "*.parquet"))
    if len(filtered_files) == 0:
        raise ValueError("No filtered files found")

    dataset = pd.concat(
        [pd.read_parquet(file) for file in filtered_files], ignore_index=True
    )
    dataset = dataset.sort_values("article_id")

    dataset.to_parquet(output_file_path)


def join_with_file_list(
    joined_output_file: str, file_list_path: str, output_file_path: str
):
    """
    Join the Biomedica dataset with the OA file list.

    Args:
        joined_output_file (str): Path to the biomedica parquet file.
        file_list_path (str): Path to the OA file_list csv.
        output_file_path (str): Parquet file to save the merged dataset to.

    """
    assert joined_output_file.endswith(".parquet")
    assert file_list_path.endswith(".csv")
    assert output_file_path.endswith(".parquet")
    assert os.path.exists(joined_output_file)

    joined_df = pd.read_parquet(joined_output_file)
    file_list = pd.read_csv(file_list_path)
    file_list["File"] = file_list["File"].apply(
        lambda x: x.replace("oa_package/", "").replace(".tar", "").replace(".gz", "")
    )
    file_list["PMID"] = file_list["File"].str.split("/").str[-1]
    file_list = file_list[
        file_list["PMID"].isin(joined_df["article_id"].unique())
    ].set_index("PMID")

    joined_df = joined_df.assign(
        **{"file_list_path": "", "file_list_license": "", "file_list_citation": ""}
    )

    for i, row in tqdm(joined_df.iterrows(), total=len(joined_df)):
        file_list_entry = file_list.loc[row["article_id"]]
        joined_df.at[i, "file_list_path"] = file_list_entry["File"]
        joined_df.at[i, "file_list_license"] = file_list_entry["License"]
        joined_df.at[i, "file_list_citation"] = file_list_entry["Article Citation"]

    joined_df.to_parquet(output_file_path)


def convert_biomedica_to_table(database_path: str, file_list_path: str) -> None:
    """
    Import the Biomedica file list parquet into the database if not already present.

    Creates and populates the biomedica_data_file_list table from the parquet
    file and adds an index on (article_id, image_cluster_id). Does nothing if
    the table already exists.

    Args:
        database_path (str): Path to the database to save the table to.
        file_list_path (str): Path to the Biomedica filtered file list parquet
            file.

    """
    table_name = "biomedica_data_file_list"

    with get_database_connection_context(database_path=database_path) as write_conn:
        write_cur = write_conn.cursor()
        # If the table does not exist, create it
        # Check if table exists
        write_cur.execute(
            f"""SELECT name
            FROM sqlite_master
            WHERE type='table' AND name='{table_name}';"""
        )
        table_exists = write_cur.fetchone()

        if not table_exists:
            biomedica_df = pd.read_parquet(file_list_path)
            logger.info(f"Creating table {table_name} in database.")
            write_cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id INTEGER UNIQUE NOT NULL PRIMARY KEY,
                article_id INTEGER,
                image_cluster_id TEXT,
                iteration INTEGER,
                file_key TEXT,
                image_label_id INTEGER,
                image_hash TEXT,
                image_caption TEXT,
                image_panel_type TEXT,
                image_panel_subtype TEXT,
                image_primary_label TEXT,
                image_secondary_label TEXT,
                image_context TEXT,
                article_title TEXT,
                journal_title TEXT,
                article_date INTEGER,
                license_type TEXT,
                article_subject TEXT,
                filtering_date TEXT,
                origin_file TEXT,
                file_list_path TEXT,
                file_list_license TEXT,
                file_list_citation TEXT,
                FOREIGN KEY (article_id) REFERENCES article_packages(article_id),
                FOREIGN KEY (image_cluster_id) REFERENCES article_images(image_cluster_id)
            );
            """)  # noqa: E501

            # Convert columns where necessary
            biomedica_df["article_id"] = biomedica_df["article_id"].apply(
                lambda x: int(x.removeprefix("PMC"))
            )
            biomedica_df["image_label_id"] = pd.to_numeric(
                biomedica_df["image_label_id"], errors="raise"
            )
            biomedica_df["article_date"] = biomedica_df["article_date"].apply(
                lambda x: int(x.replace("-", "").replace(" ", "").replace("None", "-1"))
            )
            biomedica_df["article_date"] = pd.to_numeric(
                biomedica_df["article_date"], errors="raise"
            )

            # Convert filtering_date to string
            biomedica_df["filtering_date"] = biomedica_df["filtering_date"].astype(str)

            # Set index to column named id
            biomedica_df["id"] = biomedica_df.index
            biomedica_df = biomedica_df.reset_index(drop=True)

            # Enter into database
            logger.info(f"Inserting data into table {table_name}...")
            biomedica_df.to_sql(
                table_name,
                write_cur,
                if_exists="replace",
                index=False,
                chunksize=1000,
            )

            # Create index on article_id, image_cluster_id
            write_cur.execute(
                f"CREATE INDEX idx_biomedica_df_article_id_img_cluster"
                f" ON {table_name} (article_id, image_cluster_id);"
            )
