"""Module for splitting captions using a language model."""

import asyncio
import json
import logging
import multiprocessing
import sqlite3
from datetime import datetime
from typing import Any

import httpx
import openai
import pydantic
import requests
from tqdm.auto import tqdm

from pubmed_ophtha.caption_splitting.messages import (
    NAMING_FEW_SHOT_EXAMPLES_QWEN,
    NAMING_SYSTEM_MESSAGE_QWEN,
    SPLITTING_FEW_SHOT_EXAMPLES_QWEN,
    SPLITTING_SYSTEM_MESSAGE_QWEN,
    create_schema_message,
    get_messages,
)
from pubmed_ophtha.caption_splitting.response_models import (
    SplitSubCaptions,
    SubCaption,
    SubCaptionNames,
)
from pubmed_ophtha.util.database_interface import (
    database_writer,
    get_biomedica_df,
    get_database_connection_context,
)

logger = logging.getLogger(__name__)


class GenerationTimeoutError(Exception):
    """Exception raised when generation times out."""

    pass


def check_for_server(server_url: str) -> bool:
    """
    Use the /ping endpoint to check if the server is up.

    Args:
        server_url (str): URL of the server.

    Returns:
        bool: True if the server is up, False otherwise.

    """
    try:
        response = requests.get(f"{server_url}/ping")
        return response.status_code == 200
    except requests.RequestException:
        return False


def check_server_health(server_url: str) -> bool:
    """
    Use the /health endpoint to check if the server is healthy.

    Args:
        server_url (str): URL of the server.

    Returns:
        bool: True if the server is healthy, False otherwise.

    """
    try:
        response = requests.get(f"{server_url}/health")
        return response.status_code == 200
    except requests.RequestException:
        return False


def load_caption(caption_text: str) -> SplitSubCaptions | None:
    """
    Parse a JSON string into a SplitSubCaptions object.

    Attempts direct parsing first, then tries to repair common bracket
    mismatches before returning None on failure.

    Args:
        caption_text (str): Raw JSON string from the split_captions table.

    Returns:
        SplitSubCaptions | None: Parsed instance, or None if parsing fails.

    """
    try:
        split_caption_object = SplitSubCaptions.model_validate_json(caption_text)
        return split_caption_object
    except Exception:
        num_open = caption_text.count("{")
        num_close = caption_text.count("}")

        if num_open < num_close:
            if caption_text.endswith("}"):
                caption_text = caption_text[:-1]
        elif num_open > num_close:
            caption_text += "}" * (num_open - num_close)

        try:
            split_caption_object = SplitSubCaptions.model_validate_json(caption_text)
            return split_caption_object
        except Exception:
            pass
    return None


def convert_parsed_sub_captions(write_cur: sqlite3.Cursor) -> None:
    """
    Parse raw split_captions rows into the parsed_split_captions table.

    Creates the table if absent, then parses each model_response_splitting
    JSON blob and inserts both the full caption and individual sub-captions.
    Does nothing if parsed_split_captions already exists.

    Args:
        write_cur (sqlite3.Cursor): Writable SQLite cursor with access to
            split_captions and biomedica_data_file_list tables.

    Note:
        Rows are silently skipped in two cases without any log message or
        counter:

        - If ``load_caption`` returns ``None`` (JSON parsing of
          ``model_response_splitting`` failed), the row is skipped entirely and
          does not appear in ``parsed_split_captions``.
        - If no matching caption is found in ``biomedica_data_file_list`` for
          the row's ``(article_id, image_cluster_id)`` pair, the row is also
          skipped.

        In both cases the caller receives no indication of how many rows were
        dropped.

    """
    # Check if table parsed_split_captions_exists
    write_cur.execute("""
    SELECT name FROM sqlite_master WHERE type='table' AND name='parsed_split_captions';
    """)
    result = write_cur.fetchone()
    if result is not None:
        return

    # Create parsed_split_captions table
    write_cur.execute("""
    CREATE TABLE IF NOT EXISTS parsed_split_captions (
        id INTEGER PRIMARY KEY ON CONFLICT REPLACE,
        split_caption_id INTEGER,
        article_id INTEGER,
        image_cluster_id TEXT,
        sub_caption_name TEXT,
        sub_caption_text TEXT,
        is_full_caption BOOLEAN DEFAULT FALSE,
        FOREIGN KEY (split_caption_id) REFERENCES split_captions(id),
        FOREIGN KEY (article_id) REFERENCES article_packages(article_id),
        FOREIGN KEY (image_cluster_id) REFERENCES article_images(image_cluster_id)
    );
    """)

    # TODO parallelize parsing
    # Get all split captions
    write_cur.execute(
        "SELECT id, article_id, image_cluster_id, model_response_splitting"
        " FROM split_captions;"
    )
    split_captions_entries = write_cur.fetchall()

    to_write = []
    for split_caption_id, article_id, image_cluster_id, model_response in tqdm(
        split_captions_entries, desc="Parsing split captions"
    ):
        # Load as SplitSubCaptions
        split_captions = load_caption(model_response)

        if split_captions is None:
            continue

        # Get full caption from biomedica_data_file_list
        write_cur.execute(
            "SELECT image_caption FROM biomedica_data_file_list"
            " WHERE article_id = ? AND image_cluster_id = ?;",
            (
                article_id,
                image_cluster_id,
            ),
        )
        caption_entry = write_cur.fetchone()

        if caption_entry is None:
            continue

        # Insert full caption first
        to_write.append(
            (
                split_caption_id,
                article_id,
                image_cluster_id,
                "",
                caption_entry[0],
                True,
            )
        )

        for sub_caption_name, sub_caption in split_captions.sub_captions.items():
            to_write.append(
                (
                    split_caption_id,
                    article_id,
                    image_cluster_id,
                    sub_caption_name,
                    sub_caption.text,
                    False,
                )
            )

    write_cur.executemany(
        """
        INSERT INTO parsed_split_captions (
            split_caption_id,
            article_id,
            image_cluster_id,
            sub_caption_name,
            sub_caption_text,
            is_full_caption
        ) VALUES (?, ?, ?, ?, ?, ?);
    """,
        to_write,
    )


async def generate(
    openai_client: openai.AsyncOpenAI,
    input_messages: list[dict[str, str]],
    model_name: str,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int | None | openai.Omit = openai.omit,
    think_end_token: str = "</think>",
    max_retries: int = 3,
) -> tuple[str, str]:
    """
    Generate a response from the OpenAI API.

    Args:
        openai_client (openai.AsyncOpenAI): API client.
        input_messages (list[dict[str, str]]): Input messages for the chat.
        model_name (str): Name of the model to use.
        temperature (float, optional): Temperature for the completion. Defaults to 0.6.
        top_p (float, optional): Top p parameter. Defaults to 0.95.
        max_tokens (int | None | openai.Omit, optional): Maximum number of tokens to
            generate. Defaults to openai.omit (maximum possible length).
        think_end_token (str, optional): Token that indicates the end of the thinking
            output. Defaults to "</think>".
        max_retries (int, optional): Maximum number of retries on timeout. Defaults to
            3.

    Returns:
        tuple[str, str]: Tuple of (response text, thinking text).

    """
    if max_tokens is None:
        max_tokens = openai.omit

    delay = 1.0
    completion = None
    for attempt in range(max_retries):
        try:
            completion = await openai_client.chat.completions.create(
                model=model_name,
                messages=input_messages,  # pyright: ignore[reportArgumentType]
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            break
        except (openai.APITimeoutError, httpx.ReadTimeout, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                raise GenerationTimeoutError(
                    f"Max retries exceeded. Last error: {e}"
                ) from e

    assert completion is not None
    response_text = completion.choices[0].message.content or ""

    if think_end_token in response_text:
        think_text, response_text = response_text.split(think_end_token, 1)
    else:
        think_text = ""

    return response_text.strip(), think_text.strip()


async def _split_captions(
    image_id: str,
    caption_text: str,
    openai_client: openai.AsyncOpenAI,
    model_name: str,
    naming_system_prompt: str,
    splitting_system_prompt: str,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]] | None = None,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]] | None = None,
) -> dict[str, str] | None:
    naming_messages = get_messages(
        caption=caption_text,
        system_message=naming_system_prompt,
        few_shot_examples=naming_few_shot_examples,
        response_schema_message=create_schema_message(SubCaptionNames),
    )

    try:
        naming_response, naming_thinking = await generate(
            openai_client,
            naming_messages,
            model_name,
        )
    except GenerationTimeoutError:
        return None

    # Early stopping if no subfigure names are found
    try:
        parsed_names = SubCaptionNames.model_validate_json(naming_response)
        if len(parsed_names.names) == 0:
            return {
                "image_id": image_id,
                "naming_response": naming_response,
                "naming_thinking": naming_thinking,
                "splitting_response": SplitSubCaptions(
                    sub_captions={"None": SubCaption(text=caption_text)}
                ).model_dump_json(),
                "splitting_thinking": "",
            }
    except pydantic.ValidationError:
        # Ignore
        pass

    splitting_messages = get_messages(
        caption=caption_text
        + f"\n\nThe predicted subfigure names are: {naming_response}.\n",
        system_message=splitting_system_prompt,
        few_shot_examples=splitting_few_shot_examples,
        response_schema_message=create_schema_message(SplitSubCaptions),
    )

    try:
        splitting_response, splitting_thinking = await generate(
            openai_client,
            splitting_messages,
            model_name,
        )
    except GenerationTimeoutError:
        return None

    return {
        "image_id": image_id,
        "naming_response": naming_response,
        "naming_thinking": naming_thinking,
        "splitting_response": splitting_response,
        "splitting_thinking": splitting_thinking,
    }


async def process_sample(
    image_id: str,
    caption_text: str,
    semaphore: asyncio.Semaphore,
    openai_client: openai.AsyncOpenAI,
    model_name: str,
    naming_system_prompt: str,
    splitting_system_prompt: str,
    database_queue: multiprocessing.Queue,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]] | None = None,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]] | None = None,
):
    """
    Split the captions for a single sample and put the result in the database queue.

    Args:
        image_id (str): Image ID of the sample.
        caption_text (str): Full caption text of the sample.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests.
        openai_client (openai.AsyncOpenAI): Async OpenAI client.
        model_name (str): Name of the model to use.
        naming_system_prompt (str): System prompt for the naming step.
        splitting_system_prompt (str): System prompt for the splitting step.
        database_queue (multiprocessing.Queue): Queue to put the result in.
        naming_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples to use for the naming step. Defaults to None.
        splitting_few_shot_examples (list[dict[str, str  |  SplitSubCaptions]] | None):
            Few shot examples to use for the splitting step. Defaults to None.

    """
    async with semaphore:
        model_output = await _split_captions(
            image_id=image_id,
            caption_text=caption_text,
            openai_client=openai_client,
            model_name=model_name,
            naming_system_prompt=naming_system_prompt,
            splitting_system_prompt=splitting_system_prompt,
            naming_few_shot_examples=naming_few_shot_examples,
            splitting_few_shot_examples=splitting_few_shot_examples,
        )

    if model_output is None:
        logger.warning(f"Skipping sample {image_id} due to repeated timeouts.")
        return
    # Put result in database queue
    database_queue.put(model_output)


def write_into_db(
    con: sqlite3.Connection, batch: list[dict[str, Any]], allow_replace: bool = False
):
    """
    Write a batch of results into the database.

    Args:
        con (sqlite3.Connection): Connection to the database.
        batch (list[dict[str, Any]]): Batch to write. Each item in the batch is a
            dictionary with the keys:
            - image_id (str): Image ID of the sample.
            - naming_response (str): Model response for the naming step.
            - naming_thinking (str): Model reasoning text for the naming step.
            - splitting_response (str): Model response for the splitting step.
            - splitting_thinking (str): Model reasoning text for the splitting step.
        allow_replace (bool, optional): If True, allows to replace existing entries in
            the database. Defaults to False.

    Raises:
        ValueError: In case the image_id is invalid.

    """
    image_article_ids = []
    image_cluster_ids = []
    naming_responses = []
    naming_thinking_texts = []
    splitting_responses = []
    splitting_thinking_texts = []

    for result in batch:
        image_id = result["image_id"]
        if "_" not in image_id:
            raise ValueError(f"Invalid image_id: {image_id}")

        article_id = int(image_id.split("_")[0].removeprefix("PMC"))
        cluster_id = image_id.split("_", 1)[1]

        image_article_ids.append(article_id)
        image_cluster_ids.append(cluster_id)
        naming_responses.append(result["naming_response"])
        naming_thinking_texts.append(result["naming_thinking"])
        splitting_responses.append(result["splitting_response"])
        splitting_thinking_texts.append(result["splitting_thinking"])

    sql_directive = "INSERT OR REPLACE" if allow_replace else "INSERT"
    con.execute("BEGIN IMMEDIATE;")
    con.executemany(
        f"""{sql_directive} INTO split_captions (
            article_id,
            image_cluster_id,
            model_response_naming,
            model_thoughts_naming,
            model_response_splitting,
            model_thoughts_splitting
        ) VALUES (?, ?, ?, ?, ?, ?)""",
        zip(
            image_article_ids,
            image_cluster_ids,
            naming_responses,
            naming_thinking_texts,
            splitting_responses,
            splitting_thinking_texts,
        ),
    )

    con.commit()


async def run_splitting_tasks(
    tasks: list[tuple[str, str]],
    server_endpoint: str,
    model_name: str,
    naming_system_prompt: str,
    splitting_system_prompt: str,
    database_queue: multiprocessing.Queue,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]] | None = None,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]] | None = None,
    num_concurrent_requests: int = 20,
    api_key: str = "test",
):
    """
    Split all captions in the dataset asynchronously.

    Args:
        tasks (list[tuple[str, str]]): List of tasks to process. Each task is a tuple of
            (image_id, caption_text).
        server_endpoint (str): The server endpoint URL.
        model_name (str): The name of the model to use.
        naming_system_prompt (str): The system prompt for the naming step.
        splitting_system_prompt (str): The system prompt for the splitting step.
        database_queue (multiprocessing.Queue): The database queue to put results in.
        naming_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples to use for the naming step. Defaults to None.
        splitting_few_shot_examples (list[dict[str, str  |  SplitSubCaptions]] | None):
            Few shot examples to use for the splitting step. Defaults to None.
        num_concurrent_requests (int, optional): Number of concurrent requests to the
            model server. Defaults to 20.
        api_key (str, optional): API key needed to access the server. Note: For vllm
            this cannot be empty even if the server is no protected. Defaults to "test".

    """
    semaphore = asyncio.Semaphore(num_concurrent_requests)  # Limit concurrent requests

    client = openai.AsyncOpenAI(
        base_url=server_endpoint,
        api_key=api_key,  # No API key needed for local server but cannot be empty
    )

    async_io_tasks = []
    for image_id, caption_text in tasks:
        async_io_tasks.append(
            asyncio.create_task(
                process_sample(
                    image_id=image_id,
                    caption_text=caption_text,
                    semaphore=semaphore,
                    openai_client=client,
                    model_name=model_name,
                    naming_system_prompt=naming_system_prompt,
                    splitting_system_prompt=splitting_system_prompt,
                    database_queue=database_queue,
                    naming_few_shot_examples=naming_few_shot_examples,
                    splitting_few_shot_examples=splitting_few_shot_examples,
                )
            )
        )

    for f in tqdm(
        asyncio.as_completed(async_io_tasks),
        total=len(async_io_tasks),
        desc="Processing captions",
    ):
        await f


def run_splitting_tasks_sync(
    tasks: list[tuple[str, str]],
    server_endpoint: str,
    model_name: str,
    naming_system_prompt: str,
    splitting_system_prompt: str,
    database_queue: multiprocessing.Queue,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]] | None = None,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]] | None = None,
    num_concurrent_requests: int = 20,
    api_key: str = "test",
):
    """
    Split all captions in the dataset asynchronously (sync interface).

    Args:
        tasks (list[tuple[str, str]]): List of tasks to process. Each task is a tuple of
            (image_id, caption_text).
        server_endpoint (str): The server endpoint URL.
        model_name (str): The name of the model to use.
        naming_system_prompt (str): The system prompt for the naming step.
        splitting_system_prompt (str): The system prompt for the splitting step.
        database_queue (multiprocessing.Queue): The database queue to put results in.
        naming_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples to use for the naming step. Defaults to None.
        splitting_few_shot_examples (list[dict[str, str  |  SplitSubCaptions]] | None):
            Few shot examples to use for the splitting step. Defaults to None.
        num_concurrent_requests (int, optional): Number of concurrent requests to the
            model server. Defaults to 20.
        api_key (str, optional): API key needed to access the server. Note: For vllm
            this cannot be empty even if the server is no protected. Defaults to "test".

    """
    asyncio.run(
        run_splitting_tasks(
            tasks=tasks,
            server_endpoint=server_endpoint,
            model_name=model_name,
            naming_system_prompt=naming_system_prompt,
            splitting_system_prompt=splitting_system_prompt,
            database_queue=database_queue,
            naming_few_shot_examples=naming_few_shot_examples,
            splitting_few_shot_examples=splitting_few_shot_examples,
            num_concurrent_requests=num_concurrent_requests,
            api_key=api_key,
        )
    )


def run_caption_splitting(
    database_path: str,
    model_name: str,
    server_url: str,
    completion_endpoint: str = "/v1",
    num_concurrent_requests: int = 20,
    naming_system_prompt: str = NAMING_SYSTEM_MESSAGE_QWEN,
    splitting_system_prompt: str = SPLITTING_SYSTEM_MESSAGE_QWEN,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]]
    | None = NAMING_FEW_SHOT_EXAMPLES_QWEN,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]]
    | None = SPLITTING_FEW_SHOT_EXAMPLES_QWEN,
    api_key: str = "test",
    database_batch_size: int = 5,
):
    """
    Split captions in the database and store the results.

    Args:
        database_path (str): Path to the SQLite database.
        model_name (str): Name of the model to use.
        server_url (str): Bare base URL of the model server (e.g.
            ``http://localhost:8000``). Used directly for ``/ping`` and
            ``/health`` checks.
        completion_endpoint (str, optional): End-point for OpenAI formatted
            requests. Is appended to ``server_url``. Defaults to ``"/v1"``.
        num_concurrent_requests (int, optional): Number of concurrent requests to the
            model server. Defaults to 20.
        naming_system_prompt (str, optional): System prompt of the naming step.
            Defaults to
            `pubmed_ophtha.caption_splitting.messages.NAMING_SYSTEM_MESSAGE_QWEN`.
        splitting_system_prompt (str, optional): System prompt of the splitting step.
            Defaults to
            `pubmed_ophtha.caption_splitting.messages.SPLITTING_SYSTEM_MESSAGE_QWEN`.
        naming_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples for the naming step. Defaults to
            `pubmed_ophtha.caption_splitting.messages.NAMING_FEW_SHOT_EXAMPLES_QWEN`.
        splitting_few_shot_examples (list[dict[str, str  |  SplitSubCaptions]] | None):
            Few shot examples of the splitting step. Defaults to
            `pubmed_ophtha.caption_splitting.messages.SPLITTING_FEW_SHOT_EXAMPLES_QWEN`.
        api_key (str, optional): API key needed to access the server. Note: For vllm
            this cannot be empty even if the server is no protected. Defaults to "test".
        database_batch_size (int, optional): Number of entries to write into the
            database at once. Defaults to 5.

    Raises:
        RuntimeError: If the table 'split_captions' exists but no entries are found or
            the model server is not reachable/healthy.

    """
    with get_database_connection_context(database_path, read_only=False) as con:
        cur = con.cursor()

        # Check if table exists
        cur.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name='split_captions';
            """,
        )
        table_exists = cur.fetchone() is not None

        splitting_id = f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if not table_exists:
            # Create table if not exists
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS split_captions(
                    id INTEGER PRIMARY KEY ON CONFLICT REPLACE,
                    article_id INTEGER,
                    image_cluster_id TEXT,
                    model_response_naming TEXT,
                    model_thoughts_naming TEXT,
                    model_response_splitting TEXT,
                    model_thoughts_splitting TEXT,
                    model_id TEXT DEFAULT '{splitting_id}'
                        CHECK (model_id = '{splitting_id}'),
                    FOREIGN KEY(article_id) REFERENCES metadata(article_id)
                )
                """
            )
        else:
            # Get existing splitting id
            cur.execute(
                """
                SELECT model_id
                FROM _caption_splitting_metadata
                ORDER BY model_id
                DESC LIMIT 1;
                """,
            )

            row = cur.fetchone()
            if row is not None:
                splitting_id = row[0]
            else:
                raise RuntimeError(
                    "Table 'split_captions' exists but no entries found."
                )

        cur.execute(
            """CREATE INDEX IF NOT EXISTS idx_split_captions ON split_captions (
                article_id,
                image_cluster_id
            );"""
        )

        # Create metadata table if not exists
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS _caption_splitting_metadata(
                model_id TEXT PRIMARY KEY,
                model_name TEXT,
                server_url TEXT,
                completion_endpoint TEXT,
                naming_system_prompt TEXT,
                splitting_system_prompt TEXT,
                naming_few_shot_examples TEXT,
                splitting_few_shot_examples TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        # Check if metadata for this model_id already exists
        cur.execute(
            """
            SELECT
                model_id,
                naming_system_prompt,
                splitting_system_prompt,
                naming_few_shot_examples,
                splitting_few_shot_examples
            FROM _caption_splitting_metadata
            WHERE model_id = ?;""",
            (splitting_id,),
        )
        response = cur.fetchone()
        if response is None:
            # Insert metadata
            cur.execute(
                """
                INSERT INTO _caption_splitting_metadata(
                    model_id, model_name, server_url, completion_endpoint,
                    naming_system_prompt, splitting_system_prompt,
                    naming_few_shot_examples, splitting_few_shot_examples
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    splitting_id,
                    model_name,
                    server_url,
                    completion_endpoint,
                    naming_system_prompt,
                    splitting_system_prompt,
                    json.dumps(
                        [
                            {
                                "user": e["user"],
                                "assistant": e["assistant"]
                                if isinstance(e["assistant"], str)
                                else e["assistant"].model_dump_json(),
                            }
                            for e in (naming_few_shot_examples or [])
                        ]
                    ),
                    json.dumps(
                        [
                            {
                                "user": e["user"],
                                "assistant": e["assistant"]
                                if isinstance(e["assistant"], str)
                                else e["assistant"].model_dump_json(),
                            }
                            for e in (splitting_few_shot_examples or [])
                        ]
                    ),
                ),
            )
        else:
            # Get the existing system prompts and replace
            logger.info(
                f"Metadata for model_id {splitting_id} already exists. "
                "Using existing configuration."
            )
            naming_system_prompt = response[1]
            splitting_system_prompt = response[2]

            naming_few_shot_examples = [
                {
                    "user": e["user"],
                    "assistant": SubCaptionNames.model_validate(
                        json.loads(e["assistant"])
                    ),
                }
                for e in json.loads(response[3])
            ]
            splitting_few_shot_examples = [
                {
                    "user": e["user"],
                    "assistant": SplitSubCaptions.model_validate(
                        json.loads(e["assistant"])
                    ),
                }
                for e in json.loads(response[4])
            ]

        # Get all captions that have not been processed yet
        cur.execute("""SELECT article_id, image_cluster_id FROM split_captions;""")
        processed_samples = cur.fetchall()
        processed_samples = [
            f"PMC{article_id}_{image_cluster_id}"
            for article_id, image_cluster_id in processed_samples
        ]

        # Get all tasks
        cur.execute(
            """SELECT article_id, image_cluster_id FROM article_images;""",
        )

        all_samples = cur.fetchall()

        con.commit()

    # Load biomedica dataframe
    biomedica_df = get_biomedica_df(database_path)
    biomedica_df["image_id"] = (
        biomedica_df["article_id"].astype(str) + "_" + biomedica_df["image_cluster_id"]
    )
    biomedica_df.set_index("image_id", inplace=True)

    # Convert samples and filter out processed ones
    tasks = []

    for article_id, image_cluster_id in tqdm(all_samples, desc="Preparing tasks"):
        sample_id = f"PMC{article_id}_{image_cluster_id}"
        if sample_id in processed_samples:
            continue

        # Filter out samples not in biomedica data
        if sample_id not in biomedica_df.index:
            logger.warning(f"Sample {sample_id} not found in Biomedica data.")
            continue

        # Retrieve caption text
        caption_text = biomedica_df.at[sample_id, "image_caption"]
        tasks.append((sample_id, caption_text))

    if len(tasks) == 0:
        logger.info("No new samples to process.")
        return

    # Check if server is up
    if not check_for_server(server_url):
        raise RuntimeError(f"Server at {server_url} is not reachable.")

    # Check if server is healthy
    if not check_server_health(server_url):
        raise RuntimeError(f"Server at {server_url} is not healthy.")

    # Start database writer process
    result_queue = multiprocessing.Queue(maxsize=100)

    database_writer_process = multiprocessing.Process(
        target=database_writer,
        args=(database_path, result_queue, write_into_db),
        kwargs={"batch_size": database_batch_size},
        name="DatabaseWriterProcess",
    )
    database_writer_process.start()

    # Run tasks
    error = None
    try:
        run_splitting_tasks_sync(
            tasks=tasks,
            server_endpoint=server_url + completion_endpoint,
            model_name=model_name,
            naming_system_prompt=naming_system_prompt,
            splitting_system_prompt=splitting_system_prompt,
            database_queue=result_queue,
            naming_few_shot_examples=naming_few_shot_examples,
            splitting_few_shot_examples=splitting_few_shot_examples,
            num_concurrent_requests=num_concurrent_requests,
            api_key=api_key,
        )
    except Exception as e:
        error = e

    # Signal the database writer process to stop when done
    result_queue.put(None)

    # Print remaining tasks in the result queue
    remaining_results = result_queue.qsize()
    if remaining_results > 0:
        logger.info(f"Waiting for {remaining_results} remaining results to be written.")
    database_writer_process.join()

    if error is not None:
        raise error

    # Post processing: convert raw split captions into parsed format
    with get_database_connection_context(database_path, read_only=False) as con:
        cur = con.cursor()
        convert_parsed_sub_captions(cur)
        con.commit()
