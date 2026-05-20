"""Post-process the retrieved images from the PMC articles."""

import io
import json
from typing import Any

from PIL import Image
from pmo_parser.bounding_boxes import BBox, TextBox
from pmo_parser.renderer import render_page
from tqdm.auto import tqdm

from pubmed_ophtha.filtering.retrieve_original_images import (
    ExtractionError,
    MissingPDFError,
    MissingXMLFileError,
    get_data_from_package,
)
from pubmed_ophtha.filtering.retrieve_original_images_sqlite import (
    locate_article_package,
    write_into_db_batch,
)
from pubmed_ophtha.util.database_interface import (
    get_biomedica_df,
    get_database_connection,
)


def check_and_join_figure_boxes(
    current_page: str | int, metadata_details: list[dict[str, Any]]
) -> BBox:
    """
    Check if the figure bounding boxes are valid and join them if possible.

    Raises a ValueError if the boxes are not valid.

    Args:
        current_page (str | int): The page name in the metadata details.
        metadata_details (dict[str, Any]): Details from the figure extraction process.

    Raises:
        ValueError: If the images cannot be joined. May be one of the following reasons:
            - The figures do not refer to the same caption box
            - The a figure bounding box intersects with the caption box
            - After joining all figure bounding boxes, the caption box intersects with \
                the joined box
            - When joining two figure bounding boxes, the caption box intersects with \
                the joined box
            - The figure bounding boxes are too far from each other (more than 60 pt)

    Returns:
        BBox: The bounding box of the joined figure boxes.

    """
    caption_boxes = [
        TextBox.from_dict(figure["figure_data"][current_page]["caption"][0])
        if figure["figure_data"][current_page]["caption"] is not None
        else None
        for figure in metadata_details
        if current_page in figure["figure_data"]
    ]
    # Check difference between boxes and select the correct one
    first_caption_box = caption_boxes[0] if len(caption_boxes) > 0 else None
    if first_caption_box is not None and all(
        c is not None and first_caption_box.is_equal(c) for c in caption_boxes
    ):
        # All boxes are the same
        correct_box = first_caption_box
    else:
        raise ValueError("Caption boxes are not the same.")

    figure_bounding_boxes = [
        BBox.from_dict(figure["figure_data"][current_page]["figure_bbox"])
        for figure in metadata_details
        if current_page in figure["figure_data"]
    ]

    figure_bbox = BBox.union_boxes(figure_bounding_boxes)

    if figure_bbox.intersect(correct_box) is not None:
        raise ValueError("Figure bounding box intersects with caption box.")
    for i, figure_1 in enumerate(figure_bounding_boxes):
        min_distance = None
        for j, figure_2 in enumerate(figure_bounding_boxes):
            if j <= i:
                continue

            joined_figure = figure_1.union(figure_2)
            if joined_figure.intersect(correct_box) is not None:
                raise ValueError("Figure bounding boxes intersect with caption.")

            distance = figure_1.distance(figure_2)
            if min_distance is None or distance < min_distance:
                min_distance = distance

        if min_distance is not None and min_distance > 60.0:
            raise ValueError("Figure bounding boxes are too far from each other.")

    return figure_bbox


def postprocess_retrieved_images(database_path: str):
    """
    Join multi-figure panels and add missing images to the database.

    The missing figures are directly added from the package instead of re-extracting
    them.

    Args:
        database_path (str): The path to the biomedica sqlite database.

    Raises:
        ValueError: If the metadata is not formatted correctly.

    """
    read_conn = get_database_connection(database_path, read_only=True)
    read_cursor = read_conn.cursor()

    write_conn = get_database_connection(database_path, read_only=False)
    write_cursor = write_conn.cursor()

    def get_article_pdf(metadata_id):
        article_data = locate_article_package(read_cursor, article_id=metadata_id)
        if article_data is None:
            return None

        # open archive
        try:
            _, loaded_article_pdf = get_data_from_package(article_data)
        except ExtractionError as _:
            return None

        return loaded_article_pdf

    try:
        metadata_article_ids = read_cursor.execute(
            "SELECT article_id FROM metadata;"
        ).fetchall()
        for article_int_id in tqdm(metadata_article_ids):
            # Get row
            read_cursor.execute(
                "SELECT * FROM metadata WHERE article_id = ?", (article_int_id[0],)
            )
            row = read_cursor.fetchone()
            metadata_id, metadata_str, error_bit, dirty_bit = row

            if dirty_bit:
                # Ignore
                continue

            metadata_dict = json.loads(metadata_str)

            wrong_formatted_figures = []
            missing_figures = []

            has_updated_metadata = False

            metadata_update = {}

            for k, metadata_details in metadata_dict.items():
                is_error_figure = False if error_bit == 0 else True

                number_of_figures = 0

                if not isinstance(metadata_details, list):
                    wrong_formatted_figures.append((metadata_id, k))
                    continue

                if len(metadata_details) == 0:
                    missing_figures.append((metadata_id, k))
                    continue

                has_multi_figures = False
                for figure in metadata_details:
                    figure_type = figure["type"]

                    if figure_type == "error":
                        is_error_figure = True
                        break

                    if figure_type == "figure":
                        number_of_figures += 1
                    elif figure_type == "multi_figure":
                        # Ignore
                        # Implement joining
                        has_multi_figures = True
                        continue
                    else:
                        raise ValueError(f"Unknown figure type: {figure_type}")

                if is_error_figure:
                    missing_figures.append((metadata_id, k))
                    continue
                if has_multi_figures:
                    continue
                assert k is not None
                # Only update if there are multiple figures
                if number_of_figures > 1:
                    # Check if figure_data is None
                    if any([e["figure_data"] is None for e in metadata_details]):
                        raise ValueError(
                            "Received multiple non pdf-extracted images in "
                            + f"PMC{article_int_id}"
                        )

                    article_pdf = get_article_pdf(metadata_id)
                    if article_pdf is None:
                        missing_figures.append((metadata_id, k))
                        continue

                    pages = list(
                        {
                            key
                            for figure in metadata_details
                            for key in figure["figure_data"]
                        }
                    )

                    processing_error = False
                    figure_image = None
                    total_figure_data = None
                    figure_bbox = None
                    image_conversion = None
                    if len(pages) == 1:
                        try:
                            figure_bbox = check_and_join_figure_boxes(
                                pages[0], metadata_details
                            )
                        except ValueError as e:
                            # Delete figure update metadata
                            metadata_update[k] = [
                                {
                                    "type": "error",
                                    "article_id": metadata_id,
                                    "image_cluster_id": k,
                                    "reason": str(e),
                                }
                            ]
                            has_updated_metadata = True
                            write_cursor.execute("BEGIN IMMEDIATE;")
                            write_cursor.execute(
                                """
                                DELETE FROM article_images
                                WHERE article_id = ? AND image_cluster_id = ?;
                                """,
                                (metadata_id, k),
                            )
                            write_conn.commit()
                            processing_error = True

                        else:
                            figure_image = render_page(
                                article_pdf,
                                int(pages[0]),
                                bbox=figure_bbox,
                            )

                            image_conversion = {
                                pages[0]: [
                                    {
                                        "figure_y_start": figure_bbox.y0,
                                        "figure_y_end": figure_bbox.y1,
                                        "image_y_start": 0,
                                        "image_y_end": figure_image.size[1],
                                    }
                                ]
                            }
                            figure_bbox = {pages[0]: figure_bbox.to_dict()}
                            total_figure_data = {
                                pages[0]: [
                                    figure["figure_data"][pages[0]]
                                    for figure in metadata_details
                                ]
                            }
                    else:
                        pages = sorted(pages)
                        figure_bbox_map = {}
                        figure_map = {}

                        image_width = 0
                        image_height = 0

                        for page in pages:
                            try:
                                figure_bbox_map[page] = check_and_join_figure_boxes(
                                    page, metadata_details
                                )
                            except ValueError as e:
                                # Delete figure update metadata
                                metadata_update[k] = [
                                    {
                                        "type": "error",
                                        "article_id": metadata_id,
                                        "image_cluster_id": k,
                                        "reason": str(e),
                                    }
                                ]
                                has_updated_metadata = True
                                write_cursor.execute("BEGIN IMMEDIATE;")
                                write_cursor.execute(
                                    """
                                    DELETE FROM article_images
                                    WHERE article_id = ? AND image_cluster_id = ?;
                                    """,
                                    (metadata_id, k),
                                )
                                write_conn.commit()
                                processing_error = True
                                break
                            figure_map[page] = render_page(
                                article_pdf,
                                int(page),
                                bbox=figure_bbox_map[page],
                            )

                            image_width = max(image_width, figure_map[page].width)
                            image_height += figure_map[page].height

                        if not processing_error:
                            buffer_distance = max(
                                int(0.02 * max(image_width, image_height)), 10
                            )  # 5% of width or 10 pixels
                            figure_image = Image.new(
                                "RGB",
                                (
                                    image_width,
                                    image_height + buffer_distance * (len(pages) - 1),
                                ),
                                (255, 255, 255),
                            )

                            current_y = 0
                            image_conversion = {}
                            for page in pages:
                                figure_image.paste(figure_map[page], (0, current_y))
                                image_conversion[page] = [
                                    {
                                        "figure_y_start": figure_bbox_map[page].y0,
                                        "figure_y_end": figure_bbox_map[page].y1,
                                        "image_y_start": current_y,
                                        "image_y_end": current_y
                                        + figure_map[page].height,
                                    }
                                ]
                                current_y += figure_map[page].height + buffer_distance

                            figure_bbox = {
                                page: figure_bbox_map[page].to_dict() for page in pages
                            }

                            total_figure_data = {
                                page: [
                                    figure["figure_data"][page]
                                    for figure in metadata_details
                                    if page in figure["figure_data"]
                                ]
                                for page in pages
                            }

                    if not processing_error:
                        assert figure_image is not None
                        # Save figure image and delete other images
                        # First delete old images
                        write_cursor.execute("BEGIN IMMEDIATE;")
                        write_cursor.execute(
                            """
                            DELETE FROM article_images
                            WHERE article_id = ? AND image_cluster_id = ?;
                            """,
                            (metadata_id, k),
                        )

                        # Save new image
                        image_bytes = io.BytesIO()
                        figure_image.save(image_bytes, format="PNG")
                        image_bytes = image_bytes.getvalue()

                        write_cursor.execute(
                            """
                            INSERT INTO article_images (
                                article_id,
                                image_name,
                                image,
                                image_cluster_id,
                                extracted_from_pdf
                            )
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            (metadata_id, f"{k}_hq_0.png", image_bytes, k, True),
                        )
                        write_conn.commit()

                        metadata_update[k] = [
                            {
                                "type": "multi_figure",
                                "figure_index": [
                                    entry["figure_index"] for entry in metadata_details
                                ],
                                "figure_id": [
                                    entry["figure_id"] for entry in metadata_details
                                ],
                                "figure_path": f"{k}_hq_0.png",
                                "similarity_scores": [
                                    entry["similarity_scores"]
                                    for entry in metadata_details
                                ],
                                "total_similarity_score": [
                                    entry["total_similarity_score"]
                                    for entry in metadata_details
                                ],
                                "figure_data": total_figure_data,
                                "joined_figure_bbox": figure_bbox,
                                "image_conversion": image_conversion,
                                "is_multi_page": len(pages) > 1,
                            }
                        ]

                        # Set dirty bit
                        has_updated_metadata = True

            if has_updated_metadata:
                metadata_dict = {
                    k: metadata_update.get(k, metadata_details)
                    for k, metadata_details in metadata_dict.items()
                }
                meta_info_json = json.dumps(metadata_dict, ensure_ascii=False)
                write_cursor.execute("BEGIN IMMEDIATE;")
                write_cursor.execute(
                    """INSERT OR REPLACE INTO metadata (
                        article_id,
                        json,
                        error_bit,
                        dirty_bit
                    )
                    VALUES (?, ?, ?, ?)""",
                    (metadata_id, meta_info_json, error_bit, True),
                )
                write_conn.commit()
    except Exception as e:
        # Close connections on error and reraise
        read_conn.rollback()
        write_conn.rollback()
        read_conn.close()
        write_conn.close()

        raise e

    # Get biomedica file
    biomedica_df = get_biomedica_df(database_path)

    # Create new table that maps image_cluster_id to article and image_id
    write_cursor.execute("BEGIN IMMEDIATE;")
    write_cursor.execute(
        """

        CREATE TABLE IF NOT EXISTS image_id_map (
            image_cluster_id TEXT PRIMARY KEY ON CONFLICT REPLACE,
            article_id INTEGER,
            image_ids TEXT,
            FOREIGN KEY(article_id) REFERENCES metadata(article_id)
        )
        """
    )
    write_conn.commit()

    written_image_cluster_ids = read_cursor.execute(
        "SELECT image_cluster_id, article_id FROM image_id_map;"
    ).fetchall()

    written_article_map = {}

    for article_id, image_cluster_id in written_image_cluster_ids:
        if article_id not in written_article_map:
            written_article_map[article_id] = set()
        written_article_map[article_id].add(image_cluster_id)

    missing_image_cluster_ids = []

    # unique article id image cluster id combinations
    unique_article_image_clusters = biomedica_df[
        ["article_id", "image_cluster_id"]
    ].drop_duplicates()

    for _, article_data in tqdm(
        unique_article_image_clusters.iterrows(),
        desc="Checking existing image_cluster_ids",
        total=len(unique_article_image_clusters),
    ):
        article_id = int(article_data["article_id"].removeprefix("PMC"))
        image_cluster_id = article_data["image_cluster_id"]

        if (
            article_id in written_article_map
            and image_cluster_id in written_article_map[article_id]
        ):
            continue

        image_ids = read_cursor.execute(
            """
            SELECT id FROM article_images
            WHERE article_id = ? AND image_cluster_id = ?;
            """,
            (
                article_id,
                image_cluster_id,
            ),
        ).fetchall()
        image_ids = [row[0] for row in image_ids]

        if len(image_ids) == 0:
            # Get directly from biomedica_df
            missing_image_cluster_ids.append(image_cluster_id)
            continue

        write_cursor.execute("BEGIN IMMEDIATE;")
        write_cursor.execute(
            """INSERT INTO image_id_map (image_cluster_id, article_id, image_ids)
            VALUES (?, ?, ?);""",
            (
                image_cluster_id,
                article_id,
                json.dumps(image_ids),
            ),
        )
        write_conn.commit()

    # Filter biomedica_df to only include entries with image_cluster_id
    # not in existing_image_clusters
    filtered_biomedica_df = biomedica_df[
        biomedica_df["image_cluster_id"].isin(missing_image_cluster_ids)
    ]

    # Search for images and add them to the database
    for _, sub_df in tqdm(
        filtered_biomedica_df.groupby("article_id"),
        total=filtered_biomedica_df["article_id"].nunique(),
        desc="Processing missing image_cluster_ids",
    ):
        article_id = sub_df["article_id"].iloc[0]
        int_article_id = int(str(article_id).removeprefix("PMC"))

        # Get metadata entry
        read_cursor.execute(
            "SELECT json FROM metadata WHERE article_id = ?;", (int_article_id,)
        )
        metadata_entry = read_cursor.fetchone()

        if metadata_entry is not None:
            # load dict
            metadata_dict = json.loads(metadata_entry[0])
            if "null" in metadata_dict.keys():
                if len(metadata_dict) == 1:
                    metadata_dict = {}
                else:
                    raise ValueError(
                        "Metadata contains 'null' key along with other keys."
                    )
        else:
            metadata_dict = {}

        article_package_data = locate_article_package(
            read_cursor,
            oa_path=sub_df["file_list_path"].iloc[0],
        )
        if article_package_data is None:
            # Ignore
            continue

        # open archive
        image_data = {}
        try:
            image_data, _ = get_data_from_package(article_package_data)
        except (MissingPDFError, MissingXMLFileError) as e:
            # Just use images directly
            if e.image_data is not None:
                image_data = e.image_data
            else:
                # Ignore
                continue
        except ExtractionError as _:
            # Ignore
            continue

        processed_image_names = []
        processed_image_bytes = []
        processed_image_cluster_ids = []
        extracted_from_pdf_list = []
        old_reasons = {
            k: [
                entry["reason"]
                for entry in metadata_dict[k]
                if entry["type"] == "error" and "reason" in entry
            ]
            for k in metadata_dict
            if k in sub_df["image_cluster_id"].values
            and any(entry["type"] == "error" for entry in metadata_dict[k])
        }

        for _, row in sub_df.iterrows():
            image_cluster_id = row["image_cluster_id"]
            figure_name = f"{image_cluster_id}.png"
            extracted_from_pdf = False

            if image_cluster_id in metadata_dict:
                if not all(
                    entry["type"] == "error"
                    for entry in metadata_dict[image_cluster_id]
                ):
                    raise ValueError(
                        f"Image cluster ID {image_cluster_id} already exists."
                    )

            if image_cluster_id not in image_data:
                metadata_dict[image_cluster_id] = [
                    {
                        "type": "error",
                        "article_id": article_id,
                        "image_cluster_id": image_cluster_id,
                        "reason": "Figure not found in package.",
                        "old_reasons": old_reasons.get(image_cluster_id, []),
                    }
                ]
                continue

            pil_image = image_data[image_cluster_id]
            image_bytes = io.BytesIO()
            pil_image.save(image_bytes, format="PNG")
            image_bytes = image_bytes.getvalue()

            processed_image_names.append(figure_name)
            processed_image_bytes.append(image_bytes)
            processed_image_cluster_ids.append(image_cluster_id)
            extracted_from_pdf_list.append(extracted_from_pdf)

            metadata_dict[image_cluster_id] = [
                {
                    "type": "figure",
                    "figure_index": None,
                    "figure_id": None,
                    "figure_path": [figure_name],
                    "similarity_scores": None,
                    "total_similarity_score": None,
                    "figure_data": None,
                    "image_conversion": None,
                    "is_multi_page": None,
                    "old_reasons": old_reasons.get(image_cluster_id, []),
                }
            ]

        # Write to database
        write_into_db_batch(
            write_conn,
            [
                {
                    "article_id": article_id,
                    "meta_info_dict": metadata_dict,
                    "error_bit": False,
                    "dirty_bit": True,
                    "image_names": processed_image_names,
                    "image_data_list": processed_image_bytes,
                    "image_cluster_ids": processed_image_cluster_ids,
                    "extracted_from_pdf": extracted_from_pdf_list,
                }
            ],
            allow_replace=True,
        )

        # Write into image_id_map
        # First get image_ids
        image_ids = read_cursor.execute(
            """
            SELECT id
            FROM article_images
            WHERE article_id = ? AND image_cluster_id IN ({seq});
            """.format(seq=",".join(["?"] * len(processed_image_cluster_ids))),
            [article_id] + processed_image_cluster_ids,
        ).fetchall()

        image_ids = [row[0] for row in image_ids]
        if len(image_ids) != len(processed_image_cluster_ids):
            raise ValueError(
                "Number of image IDs does not match number of cluster IDs."
            )

        # Insert into image_id_map
        write_cursor.execute("BEGIN IMMEDIATE;")
        for image_cluster_id, image_id in zip(processed_image_cluster_ids, image_ids):
            write_cursor.execute(
                """
                INSERT INTO image_id_map (
                    image_cluster_id,
                    article_id,
                    image_ids
                ) VALUES (?, ?, ?);
                """,
                (
                    image_cluster_id,
                    article_id,
                    json.dumps([image_id]),
                ),
            )

        write_conn.commit()

    read_conn.close()
    write_conn.close()
