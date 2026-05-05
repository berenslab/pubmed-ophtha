# PubMed-Ophtha

**An open resource for training ophthalmology vision-language models on scientific literature**

[![arXiv](https://img.shields.io/badge/arXiv-2605.02720-b31b1b.svg)](https://arxiv.org/abs/2605.02720)
[![Hugging Face](https://img.shields.io/badge/🤗%20Dataset-pubmed--ophtha-yellow)](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha)

---

PubMed-Ophtha is a hierarchical dataset of **102,023 ophthalmological image-caption pairs** extracted from 15,842 open-access articles in PubMed Central. Figures are extracted directly from article PDFs at full resolution and decomposed into their constituent panels, panel identifiers, and individual images. Each image is annotated with its imaging modality (CFP, OCT, Retinal Imaging, or Other) and a mark status indicating the presence of annotations such as arrows.

## Dataset

The dataset is available on Hugging Face:

🤗 **[pubmed-ophtha/PubMed-Ophtha](https://huggingface.co/datasets/pubmed-ophtha/PubMed-Ophtha)**

It includes:
- `pubmed_ophtha.parquet` — the main panel-centric dataset for VLM training
- `pubmed_ophtha_annotation.json` — human-annotated ground-truth data (PubMed-Ophtha-Annotation)

## Code

The dataset generation pipeline is split across two repositories:

- **[berenslab/pubmed-ophtha](https://github.com/berenslab/pubmed-ophtha)** — main pipeline, trained models, and usage examples
- **[berenslab/pmo-parser](https://github.com/berenslab/pmo-parser)** — PDF figure and caption extraction

> **Note:** Code will be uploaded soon.

## Citation

```bibtex
@article{hallitschke2026pubmedophtha,
  title   = {PubMed-Ophtha: An open resource for training ophthalmology vision-language models on scientific literature},
  author  = {Hallitschke, Verena Jasmin and Eickhoff, Carsten and Berens, Philipp},
  journal = {arXiv preprint arXiv:2605.02720},
  year    = {2026}
}
```
