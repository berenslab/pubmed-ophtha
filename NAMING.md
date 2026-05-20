# Naming differences between paper and code

The [paper](https://arxiv.org/abs/2605.02720) and the published dataset (`pubmed_ophtha.parquet` and `pubmed_ophtha_annotation.json`) use the final terminology at publication time. Three other artifacts still use earlier names that could not be changed without breaking compatibility:

- the **SQLite database** (`pubmed_ophtha.db`) — table and column names produced by the pipeline,
- the **Label Studio interface** and its exported annotations,
- the **trained detection models** — class labels are baked into the Detectron2 checkpoints released on Hugging Face.

This file maps the old names to the paper terminology so that readers can find the corresponding code, database column, or model class.

## Paper ↔ code mapping

### Panel identifier → "Label"

The paper uses **panel identifier** for the letter, number, or positional descriptor (`A`, `B`, `top`, `left`, …) that distinguishes panels within a figure. In code, Label Studio, and the released detection model, this is consistently called **`Label`**.

| Paper term | Old name | Where it appears |
|---|---|---|
| panel identifier (class) | `Label` | Detectron2 panel-detection model — class index `1` in [`config_panel_detection.yaml`](src/pubmed_ophtha/figure_splitting/detectron/model_config/config_panel_detection.yaml); category list `["Panel", "Label"]` in [src/pubmed_ophtha/const/models.py](src/pubmed_ophtha/const/models.py) |
| panel identifier (Label Studio) | `Label` / `PanelTypeEnum.LABEL` | `Panel Type` widget in [label_studio_interface.xml](src/pubmed_ophtha/figure_splitting/labeling/label_studio_interface.xml); `PanelTypeEnum` in [label_studio_annotations.py](src/pubmed_ophtha/figure_splitting/labeling/label_studio_annotations.py) |
| panel identifier (SQLite) | `label_assignments` table | Pipeline output table in `pubmed_ophtha.db` (see [automatically_assign_panels.py](src/pubmed_ophtha/panel_assembly/automatically_assign_panels.py)) |
| panel identifier IDs | `label_ids` column | `panel_assignments` table in `pubmed_ophtha.db` |

### Mark status → "Annotation Type" with values `Annotated` / `Plain`

The paper uses **mark status** with values **marked** / **unmarked**. The Label Studio interface and the trained mark-status classifier use **`Annotation Type`** with values **`Annotated`** / **`Plain`**.

| Paper term | Old name | Where it appears |
|---|---|---|
| mark status (field) | `Annotation Type` / `annotation_type_label` | Label Studio widget in [label_studio_interface.xml](src/pubmed_ophtha/figure_splitting/labeling/label_studio_interface.xml) |
| mark status (enum) | `AnnotationTypeEnum` | [label_studio_annotations.py](src/pubmed_ophtha/figure_splitting/labeling/label_studio_annotations.py) |
| marked | `Annotated` | Label Studio label; class index `1` of the mark-status classifier ([mark_status_classifier.py](src/pubmed_ophtha/figure_splitting/detectron/mark_status_classifier.py)) |
| unmarked | `Plain` | Label Studio label; class index `0` of the mark-status classifier |

### Panel → "subpanel" / "subfigure"

The paper uses **panel** for a labeled component of a figure (the unit each row of `pubmed_ophtha.parquet` represents). Earlier versions of the code, the caption-splitting LLM prompts, and some pmo-parser variable names instead call these **subpanels** or **subfigures**. All three terms refer to the same thing.

| Paper term | Old name | Where it appears |
|---|---|---|
| panel | `subpanel` | Docstring of `split-figures` in [figure_splitting/cli.py](src/pubmed_ophtha/figure_splitting/cli.py) and incidental references |
| panel | `subfigure` | Throughout the caption-splitting prompts and few-shot examples in [caption_splitting/messages.py](src/pubmed_ophtha/caption_splitting/messages.py) (visible to the LLM) |
| panel ID (from figure filename) | `sub_figure_id` | Parsed in [filtering/retrieve_original_images.py](src/pubmed_ophtha/filtering/retrieve_original_images.py) |
| figures containing multiple panels | `keys_with_subfigures` | Variable in [retrieve_original_images.py](src/pubmed_ophtha/filtering/retrieve_original_images.py) and [retrieve_original_images_sqlite.py](src/pubmed_ophtha/filtering/retrieve_original_images_sqlite.py) |

### Figure → SQLite `article_images` / `image_predictions`

The paper distinguishes a **figure** (the complete visual element from an article, possibly spanning multiple panels) from an **image** (a single bitmap inside a panel). The SQLite schema predates this distinction and uses `article_images` and `image_predictions` for what the paper calls *figure-level* tables.

| Paper term | Old name | Where it appears |
|---|---|---|
| figure (table) | `article_images` | Created in [retrieve_original_images_sqlite.py](src/pubmed_ophtha/filtering/retrieve_original_images_sqlite.py); one row per extracted figure |
| figure bitmap (column) | `article_images.image` | BLOB column storing the full figure PNG |
| figure ID (foreign key) | `article_images_id` / `article_images(id)` | Referenced by `image_predictions`, `label_assignments`, etc. |
| figure-level detections (table) | `image_predictions` | Created in [label_prediction.py](src/pubmed_ophtha/figure_splitting/label_prediction.py); stores all panel/identifier/image detections grouped by source figure |

## Conventions

- **Paper term**: the wording used in the published manuscript.
- **Name in code / data**: the identifier as it appears in the SQLite database, Label Studio interface/exports, or in the released Detectron2 model class labels.
- **Where it appears**: a pointer (database column, Label Studio label, class index, etc.) so the mapping is verifiable.
- **Notes**: why the old name is retained (e.g. baked into trained model weights, exported annotation schema).

The published Parquet and JSON files already use the paper terminology and do not need to appear here.
