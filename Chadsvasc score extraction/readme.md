# CHADSVASc Score Extraction

This folder contains implementations and experiments for extracting CHADSVASc components and scores from clinical notes. The repository includes multiple strategies (rule-based, expert/weak supervision, ML, and LLM-assisted) so results can be compared.

## Folder overview

```
Chadsvasc score extraction/
├── Data-based/        # ML models trained on labelled datasets
├── Expert-based/      # Weak supervision (Snorkel) + soft-training pipelines
├── LLM-based/         # Retrieval-augmented generation and LLM experiments
├── Regex-based/       # Pattern matching rules / regex prototypes
└── [Supporting files]
```

## What each folder contains

- **Data-based/**
  - Implements supervised ML approaches using labeled CHADSVASc data.
  - Key files:
    - `medRoberta_training.py`: Train MedRoBERTa on labelled Chadsvasc data, without considering the components  (note: executed via VS Code/HPC in our setup).
    - `datasets/`: Train/val/test splits

- **Expert-based/**
  - Weak supervision pipeline based on hand-crafted labeling functions (LFs) and downstream soft training.
  - Main entry point: `snorkel.ipynb` — contains the full set of labelling functions for each CHADSVASc component and code to train/evaluate a Snorkel `LabelModel` and a `Majority Voting` baseline.
  - Other files:
    - `soft_training_seeds.py`, `medRoberta_soft_training.py`, `medRoberta_soft_training_sliding_window.py` — soft-training scripts (executed on VSC).
    - `jessa_QI.ipynb` — Quality indicators (considering only the reconstructed scores) and sub-analyses (e.g., medication effects).
    - `feather_files/`, `models/`, `results/`, `trainingsets/` — data artifacts, model checkpoints, and outputs. Feather files are kept privately to ensure data protection.

- **LLM-based/**
  - RAG and LLM experiments for extraction and augmentation of clinical context. All these experiments were conducted on VSC.
  - Key files:
    - `clinical_rag.py` — Retrieval-augmented generation pipeline.
    - `configuration_fietje.py`, `configuration_geitje.py`, `configuration_gemma.py` — model-specific configs.
    - `Fietje/`, `Geitje/`, `Gemma/` — per-experiment outputs.

- **Regex-based/**
  - Pattern-matching prototypes for direct extraction.
  - Notebooks: `jessa.ipynb`, `mimic.ipynb` with dataset-specific regex patterns.

## Quality Indicators
- `fusion.ipynb` — Merges Expert-based and LLM-based outputs into an intersection/fusion model. The fusion model produces consensus component labels by combining or intersecting predictions from the Snorkel-based labeling functions and LLM outputs; these fused labels are used as the reference for downstream quality analyses.

- `jessa_QI.ipynb` — Evaluates the defined quality indicators using the intersection/fusion model produced by `fusion.ipynb`. The notebook reports metrics on reconstructed CHADSVASc scores and provides sub-analyses based on the fused labels.

## Execution environments and path differences

- Notebooks (`.ipynb`) are generally runnable locally, but several scripts and long-running experiments were executed on VSC (Vlaams Supercomputer Centrum). Those files often contain absolute or environment-specific paths.

## Data sensitivity

- Feather files and some intermediate datasets contain protected health information and are not included in the repository.

## Recommended starting points

1. Review `Expert-based/snorkel.ipynb` to understand how labelling functions were designed and how the LabelModel / MajorityVoter are trained and evaluated.
2. Run `prep_datasets.ipynb` to recreate the preprocessing steps if you need to re-generate local test data. This notebook contains the steps taken to split the dataset in train/val/test.
