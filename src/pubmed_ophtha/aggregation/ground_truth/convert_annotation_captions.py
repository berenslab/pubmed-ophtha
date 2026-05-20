"""Module for splitting the captions in the GT annotations."""

import asyncio
import json
import logging
import multiprocessing
import os
from typing import Iterable

import openai
from tqdm.auto import tqdm

from pubmed_ophtha.caption_splitting.messages import (
    NAMING_FEW_SHOT_EXAMPLES_QWEN,
    NAMING_SYSTEM_MESSAGE_QWEN,
    SPLITTING_FEW_SHOT_EXAMPLES_QWEN,
    SPLITTING_SYSTEM_MESSAGE_QWEN,
)
from pubmed_ophtha.caption_splitting.response_models import (
    SplitSubCaptions,
    SubCaption,
    SubCaptionNames,
)
from pubmed_ophtha.caption_splitting.split_captions_sqlite import (
    GenerationTimeoutError,
    _split_captions,
    load_caption,
)
from pubmed_ophtha.const.paths import (
    ASSEMBLY_GT_CAPTION_SPLITTING_FILE,
    ASSEMBLY_GT_FOLDER,
    LABEL_STUDIO_ANNOTATION_PATH,
    LABEL_STUDIO_BASE_FOLDER,
    LABEL_STUDIO_IMAGE_PATH,
)
from pubmed_ophtha.figure_splitting.labeling.label_studio_annotations import (
    Sample,
)

from .join_with_annotations import (
    GroundTruthSample,
    get_default_gt_files,
    load_gt_annotations,
)

logger = logging.getLogger(__name__)


async def split_sample_caption(
    sample: Sample,
    semaphore: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    result_queue: multiprocessing.Queue,
    model_name: str,
    naming_system_prompt: str,
    splitting_system_prompt: str,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]] | None = None,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]] | None = None,
    num_retry: int = 3,
):
    """
    Split the caption of the given sample.

    Args:
        sample (Sample): Sample whose caption is to be split.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests to the
            model.
        client (openai.AsyncOpenAI): Client to use for making requests to the model.
        result_queue (multiprocessing.Queue): Queue to put the results in after
            processing.
        model_name (str): Name of the model to use for caption splitting.
        naming_system_prompt (str): System prompt to use for the naming step of the
            caption splitting process.
        splitting_system_prompt (str): System prompt to use for the splitting step of
            the caption splitting process.
        naming_few_shot_examples (list[dict[str, str | SubCaptionNames]] | None): Few
            shot examples for the naming step. Defaults to None.
        splitting_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples for the splitting step. Defaults to None.
        num_retry (int, optional): Number of retries in case of connection issues.
            Defaults to 3.

    Raises:
        ValueError: In case of missing panel annotation or if the sample does not have
            valid annotations.

    """
    if (
        sample.has_meta
        or not sample.has_annotations
        or sample.was_cancelled
        or sample.finished_annotations[0].bounding_boxes is None
    ):
        raise ValueError(f"Sample {sample.id} does not have valid annotations.")

    # Count panels in the sample
    num_panels = sum(
        1
        for box in sample.finished_annotations[0].bounding_boxes
        if box.box_type is not None and box.box_type == "Panel"
    )

    if num_panels == 0:
        raise ValueError(
            f"Sample {sample.id} does not have any panels in its annotations."
        )

    if num_panels == 1:
        result_queue.put(
            (
                sample.id,
                SplitSubCaptions(
                    sub_captions={"None": SubCaption(text=sample.data.caption)}
                ),
                None,
            )
        )

        return

    success = False
    current_retry = 0

    while not success and current_retry < num_retry:
        async with semaphore:
            model_output = await _split_captions(
                str(sample.id),
                caption_text=sample.data.caption,
                openai_client=client,
                model_name=model_name,
                naming_system_prompt=naming_system_prompt,
                splitting_system_prompt=splitting_system_prompt,
                naming_few_shot_examples=naming_few_shot_examples,
                splitting_few_shot_examples=splitting_few_shot_examples,
            )

        if model_output is None:
            logger.warning(f"Skipping sample {sample.id} due to repeated timeouts.")
            return

        # Try validating the model output
        loaded_caption = load_caption(model_output["splitting_response"])
        if loaded_caption is not None:
            result_queue.put((sample.id, loaded_caption, model_output))
            success = True
        else:
            logger.warning(
                f"Model output for sample {sample.id} failed validation. "
                f"Retrying... (Attempt {current_retry + 1}/{num_retry})"
            )
            current_retry += 1

    if not success:
        logger.error(
            f"Failed to process sample {sample.id} after {num_retry} attempts."
        )


async def handle_caption_annotation_split(
    ground_truth_data_point: GroundTruthSample,
    semaphore: asyncio.Semaphore,
    client: openai.AsyncOpenAI,
    result_queue: multiprocessing.Queue,
    model_name: str,
    naming_system_prompt: str,
    splitting_system_prompt: str,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]] | None = None,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]] | None = None,
    num_retry: int = 3,
):
    """
    Split the captions of the sample or convert the annotation into the correct format.

    Args:
        ground_truth_data_point (GroundTruthSample): GT data of the sample to process.
        semaphore (asyncio.Semaphore): Semaphore to limit concurrent requests to the
            model.
        client (openai.AsyncOpenAI): Client to use for making requests to the model.
        result_queue (multiprocessing.Queue): Queue to put the results in after
            processing.
        model_name (str): Name of the model to use for caption splitting.
        naming_system_prompt (str): System prompt to use for the naming step of the
            caption splitting process.
        splitting_system_prompt (str): System prompt to use for the splitting step of
            the caption splitting process.
        naming_few_shot_examples (list[dict[str, str | SubCaptionNames]] | None): Few
            shot examples for the naming step. Defaults to None.
        splitting_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples for the splitting step. Defaults to None.
        num_retry (int, optional): Number of retries in case of connection issues.
            Defaults to 3.

    """
    panel_sample = ground_truth_data_point.get("panel")
    caption_sample = ground_truth_data_point.get("caption")
    if panel_sample is None:
        logger.warning("Sample does not have panel data. Skipping.")
        return

    if caption_sample is not None:
        # Load directly from the annotation, as it is already in text form
        caption_annotations = caption_sample.finished_annotations
        if len(caption_annotations) == 0:
            raise ValueError(
                f"No finished annotations for caption sample {caption_sample.id}"
            )
        caption_bounding_boxes = caption_annotations[0].bounding_boxes
        if caption_bounding_boxes is None:
            raise ValueError(
                f"No bboxes in annotations for caption sample {caption_sample.id}"
            )
        split_captions: dict[str, str] = {}
        for box in caption_bounding_boxes:
            if box.text is not None and len(box.text) > 0:
                split_captions[str(box.name)] = box.text

        result_queue.put(
            (
                panel_sample.id,
                SplitSubCaptions(
                    sub_captions={
                        key: SubCaption(text=value)
                        for key, value in split_captions.items()
                        if (key is not None and key != "None")
                        or len(split_captions) == 1
                    }
                ),
                None,
            )
        )

        return

    try:
        await split_sample_caption(
            sample=panel_sample,
            semaphore=semaphore,
            client=client,
            result_queue=result_queue,
            model_name=model_name,
            naming_system_prompt=naming_system_prompt,
            splitting_system_prompt=splitting_system_prompt,
            naming_few_shot_examples=naming_few_shot_examples,
            splitting_few_shot_examples=splitting_few_shot_examples,
            num_retry=num_retry,
        )
    except ValueError as e:
        logger.error(f"Error processing sample {panel_sample.id}: {e}")
        return
    except GenerationTimeoutError as e:
        logger.error(f"Generation timeout for sample {panel_sample.id}: {e}")
        return


async def run_ground_truth_caption_splitting(
    ground_truth_data: dict[str, GroundTruthSample],
    server_endpoint: str,
    model_name: str,
    naming_system_prompt: str,
    splitting_system_prompt: str,
    result_queue: multiprocessing.Queue,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]] | None = None,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]] | None = None,
    num_concurrent_requests: int = 20,
    api_key: str = "test",
    num_retry: int = 3,
    processed_samples: Iterable[int | str] | None = None,
    completion_endpoint: str = "/v1",
):
    """
    Run the caption splitting process.

    Args:
        ground_truth_data (dict[str, GroundTruthSample]): Loaded ground truth data to
            process.
        server_endpoint (str): Chat completion endpoint of the LLM to use for caption
            splitting.
        model_name (str): Name of the model to use for caption splitting.
        naming_system_prompt (str, optional): System prompt to use for the naming step
            of the caption splitting process. Defaults to NAMING_SYSTEM_MESSAGE_QWEN.
        splitting_system_prompt (str, optional): System prompt to use for the splitting
            step of the caption splitting process. Defaults to
            SPLITTING_SYSTEM_MESSAGE_QWEN.
        result_queue (multiprocessing.Queue): Queue to put the results in after
            processing.
        naming_few_shot_examples (list[dict[str, str | SubCaptionNames]] | None): Few
            shot examples for the naming step. Defaults to
            NAMING_FEW_SHOT_EXAMPLES_QWEN.
        splitting_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples for the splitting step. Defaults to
            SPLITTING_FEW_SHOT_EXAMPLES_QWEN.
        num_concurrent_requests (int, optional): Number of concurrent requests.
            Defaults to 20.
        api_key (str, optional): API key for the LLM client. Defaults to "test".
        num_retry (int, optional): Number of retries in case of connection issues.
            Defaults to 3.
        processed_samples (Iterable[int] | None, optional): Set of sample IDs that have
            already been processed, to avoid re-processing. Defaults to None.
        completion_endpoint (str, optional): Endpoint suffix appended to
            ``server_endpoint`` for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    semaphore = asyncio.Semaphore(num_concurrent_requests)
    client = openai.AsyncOpenAI(
        base_url=server_endpoint + completion_endpoint,
        api_key=api_key,  # No API key needed for local server but cannot be empty
    )

    if processed_samples is None:
        processed_samples = set()

    tasks = []
    for image_id, data_point in ground_truth_data.items():
        panel_data = data_point.get("panel")
        if panel_data is not None and panel_data.id in processed_samples:
            continue
        task = asyncio.create_task(
            handle_caption_annotation_split(
                data_point,
                semaphore,
                client,
                result_queue,
                model_name,
                naming_system_prompt,
                splitting_system_prompt,
                naming_few_shot_examples,
                splitting_few_shot_examples,
                num_retry,
            )
        )
        tasks.append(task)

    for f in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="Processing captions",
    ):
        await f


def _json_writer_process(
    result_queue: multiprocessing.Queue, output_file_path: str, batch_size: int = 10
):
    buffer = []
    while True:
        result = result_queue.get()
        if result is None:  # Sentinel value to indicate completion
            break
        buffer.append(result)
        if len(buffer) >= batch_size:
            with open(output_file_path, "a") as f:
                for sample_id, sub_captions, model_output in buffer:
                    output_dict = {
                        "sample_id": sample_id,
                        "sub_captions": sub_captions.model_dump_json(),
                        "naming_response": model_output["naming_response"]
                        if model_output is not None
                        else None,
                        "naming_thinking": model_output["naming_thinking"]
                        if model_output is not None
                        else None,
                        "splitting_response": model_output["splitting_response"]
                        if model_output is not None
                        else None,
                        "splitting_thinking": model_output["splitting_thinking"]
                        if model_output is not None
                        else None,
                    }
                    f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")
            buffer.clear()
    # Write any remaining results in the buffer
    with open(output_file_path, "a") as f:
        for sample_id, sub_captions, model_output in buffer:
            output_dict = {
                "sample_id": sample_id,
                "sub_captions": sub_captions.model_dump_json(),
                "naming_response": model_output["naming_response"]
                if model_output is not None
                else None,
                "naming_thinking": model_output["naming_thinking"]
                if model_output is not None
                else None,
                "splitting_response": model_output["splitting_response"]
                if model_output is not None
                else None,
                "splitting_thinking": model_output["splitting_thinking"]
                if model_output is not None
                else None,
                "is_gt": model_output
                is None,  # Indicate whether this was a GT sample or not
            }
            f.write(json.dumps(output_dict, ensure_ascii=False) + "\n")


def run_caption_splitting_sync(
    ground_truth_data: dict[str, GroundTruthSample],
    output_file_path: str,
    server_endpoint: str,
    model_name: str,
    naming_system_prompt: str = NAMING_SYSTEM_MESSAGE_QWEN,
    splitting_system_prompt: str = SPLITTING_SYSTEM_MESSAGE_QWEN,
    naming_few_shot_examples: list[dict[str, str | SubCaptionNames]]
    | None = NAMING_FEW_SHOT_EXAMPLES_QWEN,
    splitting_few_shot_examples: list[dict[str, str | SplitSubCaptions]]
    | None = SPLITTING_FEW_SHOT_EXAMPLES_QWEN,
    num_concurrent_requests: int = 20,
    api_key: str = "test",
    num_retry: int = 3,
    completion_endpoint: str = "/v1",
):
    """
    Run the caption splitting process.

    Args:
        ground_truth_data (dict[str, GroundTruthSample]): Loaded ground truth data to
            process.
        output_file_path (str): Path to the output file where the results will be
            written. Must be a .jsonl file.
        server_endpoint (str): Chat completion endpoint of the LLM to use for caption
            splitting.
        model_name (str): Name of the model to use for caption splitting.
        naming_system_prompt (str, optional): System prompt to use for the naming step
            of the caption splitting process. Defaults to NAMING_SYSTEM_MESSAGE_QWEN.
        splitting_system_prompt (str, optional): System prompt to use for the splitting
            step of the caption splitting process. Defaults to
            SPLITTING_SYSTEM_MESSAGE_QWEN.
        naming_few_shot_examples (list[dict[str, str | SubCaptionNames]] | None): Few
            shot examples for the naming step. Defaults to
            NAMING_FEW_SHOT_EXAMPLES_QWEN.
        splitting_few_shot_examples (list[dict[str, str  |  SubCaptionNames]] | None):
            Few shot examples for the splitting step. Defaults to
            SPLITTING_FEW_SHOT_EXAMPLES_QWEN.
        num_concurrent_requests (int, optional): Number of concurrent requests.
            Defaults to 20.
        api_key (str, optional): API key for the LLM client. Defaults to "test".
        num_retry (int, optional): Number of retries in case of connection issues.
            Defaults to 3.
        completion_endpoint (str, optional): Endpoint suffix appended to
            ``server_endpoint`` for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    assert output_file_path.endswith(".jsonl"), "Output file must be a .jsonl file"

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

    # Load existing results to avoid re-processing samples
    processed_samples = set()
    if os.path.exists(output_file_path):
        with open(output_file_path) as f:
            for line in f:
                try:
                    result = json.loads(line)
                    if "sample_id" in result:
                        processed_samples.add(result["sample_id"])
                except json.JSONDecodeError:
                    continue  # Skip malformed lines

    # Create result_writer and result_queue to collect results from async processing
    result_queue = multiprocessing.Queue()

    json_writer_process = multiprocessing.Process(
        target=_json_writer_process,
        args=(result_queue, output_file_path),
        kwargs={"batch_size": 10},
        name="JSONWriterProcess",
    )
    json_writer_process.start()

    raised_exception = None

    try:
        asyncio.run(
            run_ground_truth_caption_splitting(
                ground_truth_data,
                server_endpoint,
                model_name,
                naming_system_prompt,
                splitting_system_prompt,
                result_queue,  # Placeholder, as this is not used in sync mode
                naming_few_shot_examples,
                splitting_few_shot_examples,
                num_concurrent_requests,
                api_key,
                num_retry,
                processed_samples=processed_samples,
                completion_endpoint=completion_endpoint,
            )
        )
    except Exception as e:
        raised_exception = e

    finally:
        result_queue.put(None)  # Send sentinel value to indicate completion

    remaining_results = result_queue.qsize()
    if remaining_results > 0:
        logger.info(f"Waiting for {remaining_results} remaining results to be written.")
    json_writer_process.join()

    if raised_exception is not None:
        raise raised_exception


def split_gt_captions(
    gt_assembly_folder: str = ASSEMBLY_GT_FOLDER,
    num_concurrent_requests: int = 20,
    num_retry: int = 3,
    label_studio_root_folder: str = LABEL_STUDIO_BASE_FOLDER,
    server_endpoint: str = "http://localhost:8000",
    api_key: str = "test",
    completion_endpoint: str = "/v1",
):
    """
    Split the captions in the GT annotations and write the results to a JSONL file.

    Args:
        gt_assembly_folder (str, optional): Folder to save the results to. Defaults to
            ASSEMBLY_GT_FOLDER.
        num_concurrent_requests (int, optional): Number of concurrent requests to the
            model. Defaults to 20.
        num_retry (int, optional): Number of retries in case of connection issues.
            Defaults to 3.
        label_studio_root_folder (str, optional): Root folder of the Label Studio
            annotations. Defaults to LABEL_STUDIO_BASE_FOLDER.
        server_endpoint (str, optional): Bare base URL of the LLM server to use
            for caption splitting. Defaults to "http://localhost:8000".
        api_key (str, optional): API key for the LLM client. Defaults to "test".
        completion_endpoint (str, optional): Endpoint suffix appended to
            ``server_endpoint`` for OpenAI-compatible requests. Defaults to
            ``"/v1"``.

    """
    output_path = os.path.join(gt_assembly_folder, ASSEMBLY_GT_CAPTION_SPLITTING_FILE)
    local_image_base_path = os.path.join(
        label_studio_root_folder, LABEL_STUDIO_IMAGE_PATH
    )

    ground_truth_files = get_default_gt_files(
        os.path.join(label_studio_root_folder, LABEL_STUDIO_ANNOTATION_PATH)
    )

    # Load ground truth data
    gt_annotations = load_gt_annotations(
        **ground_truth_files,
        local_image_base_path=local_image_base_path,
    )

    run_caption_splitting_sync(
        ground_truth_data=gt_annotations,
        output_file_path=output_path,
        server_endpoint=server_endpoint,
        model_name="Qwen/Qwen3-32B-AWQ",
        naming_system_prompt=NAMING_SYSTEM_MESSAGE_QWEN,
        splitting_system_prompt=SPLITTING_SYSTEM_MESSAGE_QWEN,
        naming_few_shot_examples=NAMING_FEW_SHOT_EXAMPLES_QWEN,
        splitting_few_shot_examples=SPLITTING_FEW_SHOT_EXAMPLES_QWEN,
        num_concurrent_requests=num_concurrent_requests,
        api_key=api_key,
        num_retry=num_retry,
        completion_endpoint=completion_endpoint,
    )

    logger.info(f"Caption splitting completed. Results written to: {output_path}")
