# PubMed-Ophtha

**An open resource for training ophthalmology vision-language models on scientific literature**

[![arXiv](https://img.shields.io/badge/arXiv-2605.02720-b31b1b.svg)](https://arxiv.org/abs/2605.02720)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-pubmed--ophtha-yellow)](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-detection--models-yellow)](https://huggingface.co/pubmed-ophtha/detection-models)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

PubMed-Ophtha is a hierarchical dataset of **102,023 ophthalmological image-caption pairs** extracted from **15,842 open-access articles** in PubMed Central. Figures are extracted directly from article PDFs at full resolution and decomposed into their constituent panels, panel identifiers, and individual images. Each image is annotated with its imaging modality (CFP, OCT, Retinal Imaging, or Other) and a mark status indicating the presence of annotations such as arrows.

This repository contains the **full dataset generation pipeline** described in the paper, the **trained detection models**, and (soon) **usage examples**.

> **Note on naming:** The published Parquet/JSON dataset uses the paper terminology. The SQLite database, Label Studio interface, and trained model class labels still use earlier names that couldn't be changed without breaking compatibility. See [NAMING.md](NAMING.md) for the mapping.

## Contents

- [Dataset](#dataset)
- [Prerequisites](#prerequisites)
- [Quick Start: Filling Missing Panels and Subcaptions](#quick-start-filling-missing-panels-and-subcaptions)
- [Models](#models)
- [Installation](#installation)
- [Pipeline Overview](#pipeline-overview)
- [Command-Line Interface](#command-line-interface)
- [Repository Structure](#repository-structure)
- [Development](#development)
- [Examples](#examples)
- [Related Repositories](#related-repositories)
- [Citation](#citation)
- [License](#license)

## Dataset

The dataset is hosted on Hugging Face:

[**pubmed-ophtha/PubMed-Ophtha**](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha)

It contains two files:

| File | Description |
|---|---|
| `pubmed_ophtha.parquet` | Panel-centric dataset for VLM training (102,023 panels) |
| `pubmed_ophtha_annotation.json` | Human-annotated ground-truth dataset (PubMed-Ophtha-Annotation) |

Each row of `pubmed_ophtha.parquet` represents a single panel and contains, among other fields, the panel image (as PNG bytes), the subcaption text, in-text mentions, imaging-type indicators (`contains_cfp`, `contains_oct`, `contains_retinal`, `contains_other`), `contains_marked`, panel/identifier bounding boxes, the assembly method used, and license/attribution metadata. See **Table 2** in the [paper](https://arxiv.org/abs/2605.02720) for the full column description.

## Prerequisites

### BIOMEDICA access

`dataset fill-null` and **every stage of the main pipeline** read full article captions from the [**BIOMEDICA**](https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M) dataset on Hugging Face. BIOMEDICA is **gated** — before running anything you must:

1. Request access on the dataset page: <https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M>.
2. Create a Hugging Face access token at <https://huggingface.co/settings/tokens> and export it so the BIOMEDICA download can authenticate:

   ```bash
   export HF_TOKEN=hf_...
   ```

   (Equivalently, `huggingface-cli login` once.)

### Python and the package

The project requires **Python 3.12+** and the `pubmed-ophtha` package installed locally — see [Installation](#installation). The Quick Start below assumes you've already run `uv sync` (or the pip equivalent) from a clone of this repository.

### LLM endpoint (main pipeline only — *not* needed for the Quick Start)

The main dataset-generation pipeline (`pipeline stages caption-splitting`, `pipeline stages panel-assembly`, `pipeline stages aggregation convert-label-studio-annotations`) requires you to **host two LLMs yourself** and expose them through an OpenAI-compatible chat-completion endpoint. We use [vLLM](https://github.com/vllm-project/vllm). The required models are:

- **Qwen3-32B-AWQ** — for caption splitting.
- **Qwen3-VL-30B-A3B** — for panel assembly.

The prompts, few-shot examples, and structured-output schemas are tuned to these specific models and the pipeline is **not expected to work with other models**.

The endpoint URL and API key are passed per-stage via `--server-address` / `--client-url-list` / `--caption-server-endpoint` / `--panel-assembly-server-endpoint` / `--api-key`. Pass the **bare base URL** of the server (e.g. `http://localhost:8000`); the pipeline appends `/v1` itself for OpenAI-compatible calls and uses the bare host for health checks.

The `--api-key` defaults to `"test"`, which works for a local vLLM server (vLLM requires the key to be non-empty but does not validate it). For hosted endpoints, pass your actual key.

`dataset fill-null` (the [Quick Start](#quick-start-filling-missing-panels-and-subcaptions)) does **not** require this — skip this section if you're only restoring missing fields in the published dataset.

## Quick Start: Filling Missing Panels and Subcaptions

> **Read this if you've just downloaded the dataset from Hugging Face.**

A subset of rows in the published `pubmed_ophtha.parquet` have **`panel_image_bytes` and/or `subcaption_text` set to null**. These fields are derived from articles whose license does not permit redistribution of the derived images or text, so they are stripped from the released file. The dataset is **not directly usable for training** until these fields are restored locally by re-extracting them from the source PubMed Central PDFs.

The `dataset fill-null` command reads the Parquet file, identifies rows with missing panels or subcaptions, re-downloads the underlying PMC article packages, re-renders the affected figure panels from the source PDF, and reconstructs subcaptions from the stored character-index segments plus the full article caption retrieved from BIOMEDICA. **No LLM endpoint is needed for `fill_null`** — it only relies on the data already stored in the Parquet file plus the public PMC and BIOMEDICA sources.

```bash
# 1. Install the package (see Installation below)
uv sync

# 2. Download the dataset from Hugging Face
huggingface-cli download pubmed-ophtha/PubMed-Ophtha \
    --repo-type dataset --local-dir ./pubmed-ophtha

# 3. Fill in the missing fields (overwrites the input file in place)
pubmed-ophtha dataset fill-null ./pubmed-ophtha/pubmed_ophtha.parquet
```

Useful options:

| Option | Description |
|---|---|
| `--output-dataset-path PATH` | Write to a new file instead of overwriting the input. |
| `--output-dpi INT` | Re-render missing panels at a fixed DPI (default: keep original). |
| `--num-caption-workers INT` | Parallelism for the caption-splitting step. Defaults to `os.cpu_count()`. |
| `--save-folder PATH` | Scratch folder for downloaded PMC packages (default `./tmp_packages`, removed after use). |

## Models

Detection models used in the pipeline are released at [**pubmed-ophtha/detection-models**](https://huggingface.co/pubmed-ophtha/detection-models) on Hugging Face. The repository contains three trained models:

| Model | Architecture | Task |
|---|---|---|
| Panel detector | RetinaNet (ResNet-50 backbone) | Detects panel and panel-identifier bounding boxes |
| Image detector | RetinaNet (ResNet-50 backbone) | Detects individual images and classifies their imaging type (CFP / OCT / Retinal Imaging / Other) |
| Mark status classifier | ResNet-50 | Predicts whether an image is annotated with a mark (arrow, dot, etc.) |

Performance numbers are reported in Section 5 of the [paper](https://arxiv.org/abs/2605.02720). Default model paths and inference configurations are defined in [src/pubmed_ophtha/const/models.py](src/pubmed_ophtha/const/models.py).

Download the weights with:

```bash
pubmed-ophtha dataset pull-models
```

This places the model checkpoints under `./models/` in the current directory. Pass `--local-dir PATH` to place them elsewhere.

## Installation

The project requires **Python 3.12+**. We recommend [uv](https://docs.astral.sh/uv/), which resolves the git dependencies automatically:

```bash
uv pip install setuptools  # setuptools is required for installation
uv pip install --no-build-isolation "git+https://github.com/berenslab/pubmed-ophtha.git@v.1.0.0" # [detection,figures,examples]
```

This installs the package as `pubmed-ophtha`, pulling in PyTorch 2.8 and [pmo-parser](https://github.com/berenslab/pmo-parser) (pinned at `v1.0.0`) for PDF figure and caption extraction. Detectron2 is an optional dependency installed via the `[detection]` extra — required only for the `pipeline stages figure-splitting` and `train` subcommands. Stages like `dataset fill-null`, `pipeline stages filtering`, `pipeline stages aggregation`, `pipeline stages caption-splitting`, and `pipeline stages panel-assembly` work without it.

### Installing with pip

`detectron2` (optional, in the `[detection]` extra) and `pmo-parser` are declared under `[tool.uv.sources]` in `pyproject.toml`, which pip does not read. With pip, install them from git first and then install the package itself:

```bash
pip install torch==2.8.0 torchvision==0.23.0 setuptools
pip install --no-build-isolation "git+https://github.com/berenslab/pubmed-ophtha.git@v.1.0.0" # [detection,figures,examples]
```

If you only need stages that don't depend on `detectron2` (e.g. `fill_null`), omit the `[detection]` extra — `torch` is then the only heavy dependency you need.

Detectron2's `setup.py` imports `torch` at build time but does not declare it as a build dependency, so its build fails inside pip's default isolated build environment. Installing `torch` first and passing `--no-build-isolation` lets the build step see the torch you just installed.

### Optional dependencies

`detection`, `figures`, and `examples` are extras (part of the package metadata, so users who installed the package via pip/uv can also opt into them). `dev` is a dependency group (project-local only — for contributors working from a clone).

| Name | Kind | Purpose | Install command |
|---|---|---|---|
| `detection` | extra | Detectron2 (from source) — required for the `pipeline stages figure-splitting` and `train` subcommands | `uv pip install --no-build-isolation "pubmed-ophtha[detection]"` (or the pip equivalent) |
| `figures` | extra | Plotting utilities (matplotlib, seaborn) | `uv pip install "pubmed-ophtha[figures]"` (or `pip install "pubmed-ophtha[figures]"`) |
| `examples` | extra | Run example training/inference notebooks (transformers, peft, accelerate, datasets) | `uv pip install "pubmed-ophtha[examples]"` (or `pip install "pubmed-ophtha[examples]"`) |
| `dev` | group | Development tooling (ruff, pyright, pre-commit, etc.) | `uv sync --group dev` (local clone only) |

### Environment variables

- `HF_TOKEN` — Hugging Face token. Required by **every pipeline stage that reads from BIOMEDICA** (`pipeline stages filtering download-biomedica`, `dataset fill-null`, and any aggregation step that joins with BIOMEDICA captions), as well as by `dataset pull-models`. BIOMEDICA is gated — see [Prerequisites → BIOMEDICA access](#biomedica-access).

For the LLM endpoint used by `pipeline stages caption-splitting`, `pipeline stages panel-assembly`, and `pipeline stages aggregation convert-label-studio-annotations`, see [Prerequisites → LLM endpoint](#llm-endpoint-main-pipeline-only--not-needed-for-the-quick-start).

## Pipeline Overview

The dataset is constructed in a multi-stage pipeline. Each stage is exposed as a CLI subcommand:

```
                ┌──────────────────────────────────────────────────────────┐
                │  1. filtering          Filter PMC + extract PDF figures  │
                └──────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │  2. figure_splitting   Detect panels / images / marks    │
                └──────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │  3. caption_splitting  Split captions into subcaptions   │
                └──────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │  4. panel_assembly     Assign subcaptions to panels      │
                └──────────────────────────────────────────────────────────┘
                                            │
                                            ▼
                ┌──────────────────────────────────────────────────────────┐
                │  5. aggregation        Aggregate into final Parquet      │
                └──────────────────────────────────────────────────────────┘
```

Intermediate state is stored in a SQLite database (`pubmed_ophtha.db`) inside the project folder so that each stage can be re-run independently. The `--project-folder` option (defaulting to `datasets/pubmed_ophtha`) controls where the database, downloaded PDFs, and intermediate Parquet tables live.

The stages that talk to an LLM endpoint (`caption_splitting`, `panel_assembly`, and `aggregation convert-label-studio-annotations`) use `asyncio` to issue concurrent requests since they are I/O-bound; the other stages are synchronous CPU/IO pipelines.

Two further command groups support the pipeline:

- **`train`** — trains the panel/image detection and mark-status classifier models from scratch on the human-annotated subset.
- **`dataset fill-null`** — restores panels and subcaptions that were nulled in the published Parquet file for license reasons. **This is the entry point for most users of the released dataset** — see [Quick Start](#quick-start-filling-missing-panels-and-subcaptions).

## Command-Line Interface

The package installs a single console script:

```bash
pubmed-ophtha --help
```

which dispatches to three top-level command groups: `dataset`, `pipeline`, and `train`. All commands support `--help` for a full listing of options.

### `dataset` — Dataset utilities

#### `pull-models` — Download detection model weights

```bash
pubmed-ophtha dataset pull-models [OPTIONS]
```

Downloads the panel detector, image detector, and mark-status classifier weights from [`pubmed-ophtha/detection-models`](https://huggingface.co/pubmed-ophtha/detection-models) into `--local-dir/models/`. Required before running `pipeline stages figure-splitting split-figures`.

#### `fill-null` — Restore missing data in a published copy

```bash
pubmed-ophtha dataset fill-null DATASET_PATH [OPTIONS]
```

Some downloaded Parquet rows have null panel images or subcaptions (e.g. due to license restrictions on redistribution). This command re-downloads the source PMC packages, re-renders the affected figures at a configurable DPI (`--output-dpi`), and re-runs caption splitting to fill missing fields in place — or writes a new file with `--output-dataset-path`.

### `pipeline` — Run the data pipeline

#### `run` — End-to-end pipeline

```bash
pubmed-ophtha pipeline run [OPTIONS]
```

Executes all pipeline stages sequentially with a single command. See individual stage options below for details on each option.

#### `stages` — Individual pipeline stages

##### `filtering` — PMC filtering and figure extraction

```bash
pubmed-ophtha pipeline stages filtering download-biomedica       [OPTIONS]
pubmed-ophtha pipeline stages filtering retrieve-original-images [OPTIONS]
```

| Command | Description |
|---|---|
| `download-biomedica` | Downloads the [BIOMEDICA](https://huggingface.co/datasets/BIOMEDICA/biomedica_webdataset_24M) metadata, filters it for ophthalmology articles (using image-type labels, MeSH headings, and full-text keywords), joins with the PMC Open Access file list, and loads the result into the SQLite database. Requires `HF_TOKEN`. |
| `retrieve-original-images` | Downloads the matching PMC article packages over FTP and re-extracts figures from the source PDFs at full resolution using `pmo-parser`. Supports multiple workers for PDF processing (`--num-pdf-workers`) and figure extraction (`--num-workers`), and retries flaky downloads (`--download-tries`). |

Key options (shared): `--project-folder`, `--max-files`, `--temp-save-interval`.

##### `figure-splitting` — Panel / image / mark detection

```bash
pubmed-ophtha pipeline stages figure-splitting split-figures [OPTIONS]
```

Runs the three detection models over every extracted figure in the database, writing predicted bounding boxes and labels back into SQLite. Accepts a JSON-encoded `--model-args` string to override the default `DetectronFigureSplitter` configuration. Requires model weights downloaded via `dataset pull-models`.

##### `caption-splitting` — Decompose figure captions

```bash
pubmed-ophtha pipeline stages caption-splitting split-captions [OPTIONS]
```

Splits the full figure caption of each article into panel-level subcaptions using a two-step LLM prompt (panel-identifier extraction + subcaption assignment). Requires an OpenAI-compatible chat-completion endpoint:

- `--server-address` — bare base URL of the model server, `/v1` is appended automatically (default `http://localhost:8000`).
- `--api-key` — API key for the endpoint (default `test`; vLLM requires a non-empty value but does not validate it).

The prompts and few-shot examples are defined in [src/pubmed_ophtha/caption_splitting/messages.py](src/pubmed_ophtha/caption_splitting/messages.py).

##### `panel-assembly` — Assign subcaptions to panels

```bash
pubmed-ophtha pipeline stages panel-assembly assign-captions [OPTIONS]
```

Matches detected panel bounding boxes to subcaptions in three stages:

1. **Geometric matching** of detected images to detected panels (IoU and overlap thresholds).
2. **OCR-based assignment** of subcaptions using [EasyOCR](https://github.com/JaidedAI/EasyOCR) on panel-identifier locations (or the full figure as fallback).
3. **LLM-based refinement** for panels where automatic assignment failed, prompted independently twice per figure for stability and adjudicated when answers conflict.

Key options: `--client-url-list` (comma-separated LLM endpoints), `--num-workers`, `--num-runs`, `--num-concurrent-requests-per-worker`, `--num-retries`. Prompts live in [src/pubmed_ophtha/panel_assembly/messages.py](src/pubmed_ophtha/panel_assembly/messages.py).

##### `aggregation` — Build the final dataset

```bash
pubmed-ophtha pipeline stages aggregation convert-label-studio-annotations [OPTIONS]
pubmed-ophtha pipeline stages aggregation aggregate-into-final-dataset     [OPTIONS]
```

| Command | Description |
|---|---|
| `convert-label-studio-annotations` | Converts Label Studio exports of the human-annotated subset (PubMed-Ophtha-Annotation) into a panel hierarchy by running the same image-labeling, caption-splitting and panel-assembly steps on the ground-truth boxes. |
| `aggregate-into-final-dataset` | Joins the database state, the ground-truth hierarchy, and (optionally) the Label Studio data into the published `pubmed_ophtha.parquet` and `pubmed_ophtha_annotation.json` files. |

### `train` — Train the detection models

```bash
pubmed-ophtha train detectron              --config-file CONFIG [OPTIONS] [OPTS...]
pubmed-ophtha train mark-status-classifier [OPTIONS]
```

| Command | Description |
|---|---|
| `detectron` | Fine-tunes a RetinaNet (ResNet-50 backbone) with [Detectron2](https://github.com/facebookresearch/detectron2). Supports multi-GPU training (`--num-gpus`), distributed training across machines (`--num-machines`, `--machine-rank`, `--dist-url`), and k-fold splits (`--fold-index`). Logs to Weights & Biases via `wandb`. Use it with [`config_panel_detection.yaml`](src/pubmed_ophtha/figure_splitting/detectron/model_config/config_panel_detection.yaml) and [`config_imaging_type_detection.yaml`](src/pubmed_ophtha/figure_splitting/detectron/model_config/config_imaging_type_detection.yaml). |
| `mark-status-classifier` | Trains the ResNet-50 mark-status classifier on Label-Studio-exported annotations. Options: `--batch-size`, `--num-epochs`, `--learning-rate`, `--num-workers`, `--seed`, `--label-studio-root-folder`. |

## Repository Structure

```
src/pubmed_ophtha/
├── main.py                       # CLI entry point (Click group)
├── logging_config.py             # Shared logging setup
├── const/                        # Cross-module constants
│   ├── paths.py                  # File/folder layout
│   ├── urls.py                   # PubMed Central FTP / Entrez URLs
│   ├── models.py                 # Default model paths + DetectronFigureSplitter args
│   ├── labels.py                 # Similarity test identifiers
│   └── thresholds.py             # Detection overlap thresholds
├── scraping/                     # Low-level PMC FTP download utilities
│   ├── download_files.py
│   ├── download_files_sqlite.py
│   └── esearch.py                # NCBI Entrez search wrapper
├── filtering/                    # Pipeline stage 1
│   ├── cli.py
│   ├── download_biomedica.py
│   ├── filter_biomedica.py
│   ├── retrieve_original_images.py
│   ├── retrieve_original_images_sqlite.py
│   └── post_processing_sqlite.py
├── figure_splitting/             # Pipeline stage 2
│   ├── cli.py
│   ├── base_figure_splitter.py
│   ├── detectron_figure_splitter.py   # Wraps the three Detectron2 models
│   ├── label_prediction.py
│   ├── dataset_preprocessing/         # Convert annotations to COCO format
│   ├── labeling/                      # Label Studio interface + Pydantic models
│   └── detectron/                     # Detectron2 training code
│       ├── cli.py
│       ├── train_detectron.py
│       ├── detectron_trainer.py
│       ├── mark_status_classifier.py
│       ├── transformations.py
│       ├── model_config/              # YAML configs for RetinaNet training
│       └── datasets/                  # Dataset registrations for Detectron2
├── caption_splitting/            # Pipeline stage 3
│   ├── cli.py
│   ├── split_captions_sqlite.py
│   ├── messages.py               # System / few-shot prompts
│   └── response_models.py        # Pydantic schemas for structured LLM output
├── panel_assembly/               # Pipeline stage 4
│   ├── cli.py
│   ├── automatically_assign_panels.py
│   ├── llm_refinement.py
│   ├── db_loading.py
│   ├── messages.py
│   └── response_models.py
├── aggregation/                  # Pipeline stage 5
│   ├── cli.py
│   ├── aggregate_into_parquet.py
│   ├── page_conversion.py
│   └── ground_truth/             # PubMed-Ophtha-Annotation construction
├── fill_null/                    # Re-derive missing data in a published Parquet
│   ├── cli.py
│   └── fill_null.py
└── util/
    ├── database_interface.py     # Multi-process-safe SQLite helpers
    ├── computing.py              # CPU count / worker balancing
    ├── file_operations.py
    ├── registry.py
    └── training.py               # Seed utilities
examples/                         # Usage notebooks (upcoming)
```

## Development

Contributor tooling lives in the `dev` dependency group:

```bash
uv sync --group dev
uv run pre-commit install
```

This installs ruff, pyright, pre-commit, detect-secrets, pydoclint, pandas stubs, and the rest of the lint/type/test stack. After `pre-commit install`, every commit is checked locally with the hooks in [.pre-commit-config.yaml](.pre-commit-config.yaml) (ruff-format, ruff, pyupgrade, detect-secrets, whitespace/EOF/YAML hygiene, and a guard against committing directly to `main`).

The same checks run in CI via four GitHub Actions workflows under [.github/workflows/](.github/workflows/):

| Workflow | File | Triggers | What it runs |
|---|---|---|---|
| Linting | [linting.yml](.github/workflows/linting.yml) | push, PR | Ruff on `src/` (changed files only) via `astral-sh/ruff-action` |
| Pre-commit | [pre-commit.yml](.github/workflows/pre-commit.yml) | PR | The full `.pre-commit-config.yaml` hook suite via `pre-commit/action` |
| Spell Checking | [spell_checking.yml](.github/workflows/spell_checking.yml) | push, PR | `crate-ci/typos` against the project, configured by [_typos.toml](_typos.toml) (see below for silencing false positives) |
| Static Type Checking | [static_type_checking.yml](.github/workflows/static_type_checking.yml) | push, PR (on `*.py`) | `uvx pyright src/` after `uv sync --dev --all-groups --all-extras` |

To reproduce a specific CI check locally:

```bash
uv run ruff check src/           # Linting
uv run pre-commit run --all-files # Pre-commit (all hooks)
uvx typos                         # Spell Checking
uvx pyright src/                  # Static Type Checking
```

### Silencing spell-check false positives

`typos` is configured in [_typos.toml](_typos.toml) with three inline directives:

- **Disable one line** — end the line with `# spellchecker:disable-line`:

  ```python
  result = somevar  # spellchecker:disable-line
  ```

- **Disable the next line** — put `# spellchecker:ignore-next-line` on its own line:

  ```python
  # spellchecker:ignore-next-line
  flagged_identifier = ...
  ```

- **Disable a block** — wrap it with `# spellchecker:off` / `# spellchecker:on`:

  ```python
  # spellchecker:off
  WEIRD_TERMS = ["foo", "bar", "baz"]
  # spellchecker:on
  ```

For larger files dominated by domain vocabulary (currently the caption-splitting and panel-assembly LLM prompt files), add the path to `[files].extend-exclude` in `_typos.toml` instead.

## Examples

The [`examples/`](examples/) folder will host runnable notebooks demonstrating dataset loading, panel-level VLM training, and inference on the released detection models.

> **Note:** The folder is currently empty. Example notebooks will be uploaded in an upcoming release.

## Related Repositories

| Repository | Purpose |
|---|---|
| [berenslab/pubmed-ophtha](https://github.com/berenslab/pubmed-ophtha) | This repository — dataset pipeline, trained models, and examples |
| [berenslab/pmo-parser](https://github.com/berenslab/pmo-parser) | Standalone library for extracting figures and captions from PDFs (used by the `filtering` stage) |
| [pubmed-ophtha/PubMed-Ophtha](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha) | The published dataset on Hugging Face |
| [pubmed-ophtha/detection-models](https://huggingface.co/pubmed-ophtha/detection-models) | Trained panel, image, and mark-status detection models |

## Citation

If you use PubMed-Ophtha, the code in this repository, or the released detection models, please cite:

```bibtex
@article{hallitschke2026pubmed,
  title={{PubMed-Ophtha}: An open resource for training ophthalmology vision-language models on scientific literature},
  author={Hallitschke, Verena Jasmin and Eickhoff, Carsten and Berens, Philipp},
  journal={arXiv preprint arXiv:2605.02720},
  year={2026}
}
```

## License

This repository is released under the [MIT License](LICENSE). The dataset itself inherits the licenses of the underlying PubMed Central Open Access articles; per-article license and attribution information is included in the `license` and `attribution` columns of `pubmed_ophtha.parquet`, and the `commercial_use` flag indicates whether commercial reuse is permitted.
