"""Message templates and builders for LLM-based panel assembly."""

import base64
import json
import string
from io import BytesIO

from PIL import Image

from pubmed_ophtha.panel_assembly.response_models import (
    AlphaNumCaptionAssignments,
    PositionalCaptionAssignments,
)

SYSTEM_MESSAGE_ALPHANUM = f"""You are an expert at assigning sub-captions to panels in \
scientific figure images. Given a figure image and a set of candidate sub-captions, \
your task is to assign the most appropriate sub-caption to each panel in the figure.
You will be provided with a JSON object containing the following information:
- "subcaption_options": A dictionary mapping sub-caption names to their text.
- "full_caption": The full caption text for the figure.
- "panel_data": A list of objects, each representing a panel in the figure, containing:
  - "bbox_2d": The bounding box of the panel in normalized coordinates (0-1000).
  - "panel_id": A unique identifier for the panel.
- "unassigned_label_data": A list of objects, each representing a detected label in \
the figure that has not yet been finally assigned to a sub-caption, containing:
  - "label_id": A unique identifier for the label.
  - "bbox_2d": The bounding box of the label in normalized coordinates (0-1000).
  - "assigned_to": A list of panel_ids that this label is associated with.
  - "detected_text": The text detected within the label bounding box, if any
  - "overlaps_with": Some panels are included due to overlap with the panel but are \
    not assigned to the panel (yet). These could potentially be used as evidence.
Your response should be a JSON object mapping each "panel_id" to the name of the \
assigned sub-caption from "subcaption_options". If a panel does not have a suitable \
sub-caption, assign it the value null.
- Use the following format:
{json.dumps(AlphaNumCaptionAssignments.model_json_schema())}

Use evidence in the image to assign the panels. Due to the panel detection algorithm, \
the labels might not perfectly align with the panels. Use your judgment to determine \
the best assignments based on the provided labels and their detected text.
If no text is detected in the labels, you may still use their positions as evidence \
for your assignments.

The unassigned_label_data field does not have to be complete. Some labels may be \
missing, either because they were not detected or due to them already being assigned.
Use you judgement to decide whether to add new evidence labels to support your \
assignments.

First, locate the given labels (unassigned_label_data) in the image.
Check if any non-final panels can be assigned to a sub-caption based on the \
located labels.
In this case use ExistingEvidenceLabel to support your assignment, referring to the \
label_id of the located label.

If after this step any non-final panels remain unassigned, you can add \
NewEvidenceLabels to support your assignment. These should be based on the \
subcaption_option keys and the content of the image.
Search for the labels matching the unassigned subcaption names in the image and use \
them as evidence for your assignment.
Detect their position and the text at that position to support your assignment.
If the text in the image at the detected position and the text in the \
subcaption_option do not match, you can use the detected text as additional evidence \
for your assignment.
The detected text cannot exceed 3 words.

In the rare case that there are no labels in the image, you can use positional \
evidence to support your assignments. This can be the relative position of the \
panels to each other or to the whole image, or any other positional relationship \
that can be detected in the image and can be used as evidence to support your \
assignments.

Use the split sub-captions to guide your assignments. Use the subcaption keys to refer \
to the sub-captions in your assignments.
These keys indicate the type of evidence needed in the image. If in doubt use the \
sub-caption texts as ground truth.

Panels with final assignments are just given for context, do not change their \
assignments.
DO NOT REASSIGN PANELS THAT ARE MARKED AS FINAL.
ONLY RETURN ASSIGNMENTS FOR NON-FINAL PANELS.
DO NOT USE THE SHORT_DESCRIPTION IN POSITIONALEVIDENCE TO DESCRIBE THE PANEL CONTENT.
RATHER, USE IT TO DESCRIBE THE POSITIONAL RELATIONSHIP OF THE PANEL TO OTHER PANELS.
DO NOT USE POSITIONAL_EVIDENCE IF THERE IS A PANEL LABEL (such as A, B, C) IN THE \
IMAGE.
ONLY USE Estimated- or NewEvidenceLabel IN THESE CASES.
ONLY USE POSITIONAL_EVIDENCE WHEN THERE IS NO PANEL LABEL IN THE IMAGE.
POSITIONAL IS ONLY A RARE FALLBACK OPTION WHEN NO LABELS ARE PRESENT IN THE IMAGE.
THE PANEL IDS DO NOT INDICATE ANY SPATIAL RELATIONSHIP OR ORDER.
DO NOT USE THEM TO INFER THE PANEL LAYOUT.
Be concise in your evidence descriptions.

DO NOT DESCRIBE THE PANEL CONTENT IN THE EVIDENCE TEXT. ONLY DESCRIBE THE \
DETECTED TEXT AT THE LABEL POSITION.
THE EXISTINGEVIDENCELABELS AND NEWEVIDENCELABELS HAVE TO BE SUPPORTED BY THE \
IMAGE AND THE SPLIT CAPTION NAMES.
The label id in ExistingEvidenceLabel has to be one of the label_ids in \
unassigned_label_data and CANNOT be None.
If you assign None to the label_id in ExistingEvidenceLabel you completely and \
utterly fail your assignment.
If you want to assign None to the label_id in ExistingEvidenceLabel you might as \
well use a NewEvidenceLabel and properly locate the label position.
If that is not an option you can use a positional evidence label.

If a panel does not make any sense to be assigned to any of the sub-captions, set \
reject_panel to True in your response for that panel.
This will indicate that the panel should not be assigned to any sub-caption and should \
be rejected in the final figure assembly.
This should be used as a last resort when the panel content is completely \
irrelevant or when the panel layout is very broken and does not allow any \
reasonable assignment.
"""


SYSTEM_MESSAGE_POSITIONAL = f"""You are an expert at analyzing scientific figures.
Your task is to grouping diagrams/images in a figure into logical panels and assigning \
the correct sub-caption to each group.

You will be provided with a JSON object containing the following information:
- "subcaption_options": A dictionary mapping sub-caption names to their text.
- "full_caption": The full caption text for the figure.
- "image_data": A list of bounding boxes representing the individual images/diagrams \
detected in the figure.

Your response should serve two main purposes:
1. **Group Images:** Identify which images belong together to form a single logical \
panel.
2. **Assign Sub-captions:** For each group (panel), assign the most appropriate \
sub-caption from "subcaption_options".

Use the following JSON schema for your response:
{json.dumps(PositionalCaptionAssignments.model_json_schema())}

### Instructions:
- **Grouping:** Look at the layout of the images. Images that are close together, \
aligned, or visually separated from others often form a single panel.
- **Assignment:** Use the text within the "subcaption_options" to determine which \
sub-caption belongs to which group of images.
- **Evidence:** Use `PositionalEvidenceLabel` to explain your reasoning. Describe the \
spatial relationship (e.g., "Top-left image", "Column 1") or reference specific labels \
found near the images.
- **Unassigned Images:** If an image does not fit into any sub-caption group or is \
irrelevant, you do not need to include it in an assignment.
- **Confidence:** Provide a confidence score (0.0 to 1.0) for each assignment.

DO NOT USE THE SHORT_DESCRIPTION IN POSITIONALEVIDENCE TO DESCRIBE THE IMAGE CONTENT. \
RATHER, USE IT TO DESCRIBE THE POSITIONAL RELATIONSHIP OR THE VISUAL GROUPING \
INDICATORS.
"""


SYSTEM_MESSAGE_ADJUDICATOR = """You are an expert at analyzing scientific figures and \
their captions. Your task is to adjudicate between multiple sets of \
panel-to-subcaption assignments for a given figure, determining which set of \
assignments is more accurate based on the provided evidence and the content of the \
figure.

DO NOT ADD NEW EVIDENCE LABELS. RATHER, ONLY USE THE PROVIDED EVIDENCE TO MAKE YOUR \
DECISION.
USE THE SAME RESPONSE SCHEMA AS THE INPUT ASSIGNMENTS TO RETURN YOUR FINAL DECISION.

ONLY GIVE A JSON RESPONSE, DO NOT ADD ANY ADDITIONAL TEXT.

The system message of the previous assignment attempts are as follows:

{previous_system_message}
"""


def convert_bbox_to_qwen3vl(
    bbox: dict[str, float], image_height: int, image_width: int
) -> list[int]:
    """
    Convert an absolute bounding box to Qwen3-VL normalized coordinates (0–1000).

    Args:
        bbox (dict[str, float]): Dict with keys x0, y0, x1, y1 in absolute
            pixel coordinates.
        image_height (int): Height of the source image in pixels.
        image_width (int): Width of the source image in pixels.

    Returns:
        list[int]: [x0, y0, x1, y1] scaled to the 0–1000 range used by Qwen3-VL.

    """
    return [
        int(round(bbox["x0"] * 1000 / image_width)),
        int(round(bbox["y0"] * 1000 / image_height)),
        int(round(bbox["x1"] * 1000 / image_width)),
        int(round(bbox["y1"] * 1000 / image_height)),
    ]


def convert_qwen3vl_bbox_to_original(
    bbox: list[int], image_height: int, image_width: int
) -> dict[str, float]:
    """
    Convert a normalized Qwen3-VL bounding box (0–1000) back to absolute coordinates.

    Args:
        bbox (list[int]): [x0, y0, x1, y1] in the 0–1000 normalized range.
        image_height (int): Height of the target image in pixels.
        image_width (int): Width of the target image in pixels.

    Returns:
        dict[str, float]: Dict with keys x0, y0, x1, y1 in absolute pixel
            coordinates.

    """
    return {
        "x0": bbox[0] * image_width / 1000,
        "y0": bbox[1] * image_height / 1000,
        "x1": bbox[2] * image_width / 1000,
        "y1": bbox[3] * image_height / 1000,
    }


def convert_image_to_base64_str(image_path_or_pil: str | Image.Image) -> str:
    """
    Encode an image as a base64 PNG data URI string.

    Args:
        image_path_or_pil (str | Image.Image): File path to an image or a PIL
            Image object.

    Returns:
        str: Base64-encoded data URI string with a ``data:image/png;base64,``
            prefix.

    """
    if isinstance(image_path_or_pil, Image.Image):
        image = image_path_or_pil
    else:
        with Image.open(image_path_or_pil) as opened:
            image = opened.convert("RGB")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"


def prepare_inputs_alphanum(article_image_data: dict) -> str:
    """
    Build the JSON input string for alphanumeric panel assignment prompts.

    Args:
        article_image_data (dict): Dict containing split_captions, full_caption,
            panel_data, panel_is_final, label_data, and article_image.

    Returns:
        str: Fenced JSON code block string ready for inclusion in a chat message.

    """
    output_dict = {
        "subcaption_options": article_image_data["split_captions"],
        "full_caption": article_image_data["full_caption"],
        "panel_data": [
            {
                "bbox_2d": convert_bbox_to_qwen3vl(
                    panel_info["panel_content_box"],
                    article_image_data["article_image"].height,
                    article_image_data["article_image"].width,
                ),
                "panel_id": panel_id,
                "is_final": article_image_data["panel_is_final"].get(panel_id, False),
            }
            for panel_id, panel_info in article_image_data["panel_data"].items()
            if panel_info is not None
        ],
        "unassigned_label_data": [
            {
                "label_id": label_entry["label_id"],
                "bbox_2d": convert_bbox_to_qwen3vl(
                    label_entry["box"],
                    article_image_data["article_image"].height,
                    article_image_data["article_image"].width,
                ),
                "assigned_to": label_entry.get("assigned_to", None),
                "detected_text": label_entry["text"],
                "detection_confidence": label_entry.get("confidence", None),
                "overlaps_with": label_entry.get("overlaps_with", None),
            }
            for label_entry in article_image_data["label_data"].values()
            if not label_entry["is_final_assignment"]
        ],
    }

    return f"""```json
{json.dumps(output_dict, ensure_ascii=False, separators=(",", ":"))}
```
    """


def prepare_inputs_positional(article_image_data: dict) -> str:
    """
    Build the JSON input string for positional panel assignment prompts.

    Args:
        article_image_data (dict): Dict containing split_captions, full_caption,
            image_data, and article_image.

    Returns:
        str: Fenced JSON code block string ready for inclusion in a chat message.

    """
    output_dict = {
        "subcaption_options": article_image_data["split_captions"],
        "full_caption": article_image_data["full_caption"],
        "image_data": [
            {
                "bbox_2d": convert_bbox_to_qwen3vl(
                    image_data["box"],
                    article_image_data["article_image"].height,
                    article_image_data["article_image"].width,
                ),
                "image_id": image_data["image_id"],
            }
            for image_data in article_image_data["image_data"].values()
        ],
    }

    return f"""```json
{json.dumps(output_dict, ensure_ascii=False, separators=(",", ":"))}
```
    """


def check_string_alphanum(name: str) -> bool:
    """
    Check whether a string is an alphanumeric token with exactly one letter/digit swap.

    A valid token starts with either all letters or all digits, switches character
    type at most once, and contains both letters and digits.

    Args:
        name (str): Normalized sub-caption name to test.

    Returns:
        bool: True if the name is a valid single-swap alphanumeric token.

    """
    start_with_letter = None
    detected_swap = False

    has_letters = False
    has_digits = False

    for letter in name:
        if letter not in string.ascii_letters + string.digits:
            return False

        if letter.isalpha():
            has_letters = True
        elif letter.isdigit():
            has_digits = True

        if start_with_letter is None:
            if letter.isalpha():
                start_with_letter = True
            elif letter.isdigit():
                start_with_letter = False
            else:
                return False
        else:
            if start_with_letter and not letter.isalpha():
                if detected_swap:
                    return False
                detected_swap = True
            if not start_with_letter and not letter.isdigit():
                if detected_swap:
                    return False
                detected_swap = True
    return True if has_letters and has_digits else False


def determine_subcaption_type(sub_caption_names: list[str]) -> dict:
    """
    Classify a list of sub-caption names into a structural type category.

    Determines whether the names represent a full caption, alphanumeric labels,
    positional keywords (column/row/position), or an ambiguous/sequential set.

    Args:
        sub_caption_names (list[str]): Sub-caption name strings to classify.

    Returns:
        dict: Boolean/string flags with keys full_caption, alphanum, positional,
            ambiguous, and sequential.

    Raises:
        ValueError: If a single sub-caption name is present but is not "None".

    """

    def _is_sequential_letters(names: list[str]) -> bool:
        indices = [string.ascii_lowercase.index(n) for n in names]
        return max(indices) - min(indices) + 1 == len(names)

    def _is_sequential_digits(names: list[str]) -> bool:
        digits = [int(n) for n in names]
        return max(digits) - min(digits) + 1 == len(names)

    type_options = {
        "full_caption": False,
        "alphanum": False,
        "positional": None,
        "ambiguous": False,
        "sequential": False,
    }

    # Handle full caption case
    if len(sub_caption_names) == 1:
        if sub_caption_names[0] == "None":
            type_options["full_caption"] = True
        else:
            raise ValueError("Single sub-caption that is not 'None' is not supported")
        return type_options

    # Normalize names for comparison
    normalized_names = [
        "".join(
            c.lower()
            for c in name
            if c.lower() in string.ascii_lowercase + string.digits
        )
        for name in sub_caption_names
    ]

    # Check single character alphanumeric (letters or digits)
    if all(len(n) == 1 for n in normalized_names):
        if all(n in string.ascii_lowercase for n in normalized_names):
            type_options["alphanum"] = True
            type_options["sequential"] = _is_sequential_letters(normalized_names)
            type_options["ambiguous"] = any(
                n != name.lower()
                for n, name in zip(normalized_names, sub_caption_names)
            )
            return type_options

        if all(n in string.digits for n in normalized_names):
            type_options["alphanum"] = True
            type_options["sequential"] = _is_sequential_digits(normalized_names)
            type_options["ambiguous"] = any(
                n != name for n, name in zip(normalized_names, sub_caption_names)
            )
            return type_options

    # Check alphanumeric (letter+digit combinations)
    if all(check_string_alphanum(name) for name in normalized_names):
        type_options["alphanum"] = True
        type_options["ambiguous"] = len(set(normalized_names)) != len(normalized_names)
        return type_options

    # Check positional keywords
    positional_keywords = {
        "column": ["column"],
        "row": ["row"],
        "position_in_image": [
            "top",
            "bottom",
            "middle",
            "left",
            "right",
            "center",
            "lower",
            "upper",
            "above",
            "below",
        ],
    }

    for position_type, keywords in positional_keywords.items():
        if any(keyword in name for keyword in keywords for name in normalized_names):
            type_options["positional"] = position_type
            return type_options

    return type_options


def get_user_instruction(subcaption_type_options: dict) -> str | None:
    """
    Build a user-facing instruction string based on the detected sub-caption type.

    Args:
        subcaption_type_options (dict): Type classification dict returned by
            ``determine_subcaption_type``.

    Returns:
        str | None: Instruction string to prepend to the LLM prompt, or None
            when the figure has only a full caption and no panel labels to assign.

    Raises:
        ValueError: If the positional type in subcaption_type_options is unknown.

    """
    if subcaption_type_options["full_caption"]:
        return None

    if subcaption_type_options["positional"] is not None:
        positional_type = subcaption_type_options["positional"]
        if positional_type == "column":
            return (
                "Detected potentially column-based sub-caption names. Look for"
                " evidence of columns in the panel layout and label positions"
                " to guide your assignments."
            )

        if positional_type == "row":
            return (
                "Detected potentially row-based sub-caption names. Look for"
                " evidence of rows in the panel layout and label positions"
                " to guide your assignments."
            )
        if positional_type == "position_in_image":
            return (
                "Detected potentially position-based (e.g. top, bottom, etc.)"
                " sub-caption names. Look for evidence of panel positions in"
                " the image and label positions to guide your assignments."
            )

        raise ValueError(f"Unknown positional type: {positional_type}")

    instruction_text = ""

    if subcaption_type_options["ambiguous"]:
        instruction_text += (
            "The sub-caption names appear to be ambiguous. Use the panel content"
            " and label evidence carefully to make accurate assignments. "
        )

    if not subcaption_type_options["sequential"]:
        instruction_text += (
            "Some sub-caption names seem to be missing. Pay close attention to"
            " the content of the panels and the labels to determine the best"
            " assignments. "
        )

    return (
        "Detected alphanumeric sub-caption names. Use the panel content and"
        " label evidence to guide your assignments. " + instruction_text
    )


def prepare_llm_input(
    article_image_data: dict,
    min_resolution: int = 100,
    max_resolution: int = 768,
    previous_assignments: (
        None | list[AlphaNumCaptionAssignments] | list[PositionalCaptionAssignments]
    ) = None,
    conflict_details: str | None = None,
) -> tuple[
    list[dict],
    type[AlphaNumCaptionAssignments] | type[PositionalCaptionAssignments],
]:
    """
    Assemble the full message list and target model for an LLM assignment call.

    Selects the appropriate system message and JSON input format based on the
    sub-caption type. When previous_assignments are provided, wraps the prompt
    in an adjudication context.

    Args:
        article_image_data (dict): Dict containing split_captions, full_caption,
            panel_data, label_data, image_data, and article_image.
        min_resolution (int): Minimum pixel dimension passed to the image content
            block.
        max_resolution (int): Maximum pixel dimension passed to the image content
            block.
        previous_assignments (list | None): Optional prior assignment objects to
            adjudicate over.
        conflict_details (str | None): Optional description of conflicts found in
            previous assignments, appended to the user instruction when
            adjudicating.

    Returns:
        tuple[list[dict], type]: Tuple of (messages list, target Pydantic model
            class).

    Raises:
        ValueError: If user instructions cannot be determined from the
            sub-caption names.

    """
    sub_caption_names = list(article_image_data["split_captions"].keys())
    subcaption_type_options = determine_subcaption_type(sub_caption_names)

    user_instructions = get_user_instruction(subcaption_type_options)

    if user_instructions is None:
        raise ValueError(
            "Could not determine user instructions based on sub-caption names."
        )

    if subcaption_type_options["positional"] is None:
        data_json = prepare_inputs_alphanum(article_image_data)
        selected_system_message = SYSTEM_MESSAGE_ALPHANUM
    else:
        data_json = prepare_inputs_positional(article_image_data)
        selected_system_message = SYSTEM_MESSAGE_POSITIONAL

    previous_assignment_messages = []
    if previous_assignments is not None:
        # Wrap the system message for adjudication
        selected_system_message = SYSTEM_MESSAGE_ADJUDICATOR.format(
            previous_system_message=selected_system_message
        )

        if conflict_details is not None:
            user_instructions = (
                f"{user_instructions}\n\nConflict details from previous"
                f" assignment attempts: {conflict_details}"
            )

        for assignment_index, assignment in enumerate(previous_assignments):
            previous_assignment_messages.append(
                {
                    "type": "text",
                    "text": f"""Annotator {assignment_index + 1}:
```json
{assignment.model_dump_json(indent=None)}
```""",
                }
            )

    article_image = article_image_data["article_image"]

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": selected_system_message,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_instructions,
                },
                {
                    "type": "text",
                    "text": data_json,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": convert_image_to_base64_str(article_image)},
                    "min_pixels": min_resolution,
                    "max_pixels": max_resolution,
                },
                *previous_assignment_messages,
            ],
        },
    ]

    return messages, AlphaNumCaptionAssignments if subcaption_type_options[
        "positional"
    ] is None else PositionalCaptionAssignments
