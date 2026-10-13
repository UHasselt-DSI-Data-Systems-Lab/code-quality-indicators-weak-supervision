# code-quality-indicators-weak-supervision

This repo contains the source code, used for the master thesis: Extracting quality indicators from cardiology notes using modern NLP techniques.

## Structure

- `Preprocessing/`
- `Chadsvasc score extraction/`: Main implementation folder for CHADSVASc extraction.
  - `Data-based/`: Supervised machine learning methods.
  - `Expert-based/`: Weak supervision and Snorkel-based label modeling.
  - `LLM-based/`: Large language model experiments and RAG pipelines.
  - `Regex-based/`: Rule-based regex prototypes for extraction.

## Execution steps

1. Install dependencies and set up a Python environment.
   - Recommended: use a virtual environment or conda environment.
   - Install the code folder requirements with:
     ```bash
     pip install -r requirements.txt
     ```
2. Inspect the `Chadsvasc score extraction/readme.md` for folder-specific details.
3. Run preprocessing notebooks first if you need to recreate datasets:
   - `Preprocessing/preprocess_notes_jessa.ipynb`
   - `Preprocessing/preprocess_notes_mimiciv.ipynb`
4. When running scripts executed on VSC, adapt paths and environment settings where needed.
   - Notebooks are local-run friendly.
   - Scripts were executed on VSC, and may include environment-specific paths.


## Data sensitivity

The datasets contain protected health information and are not included in the repository.
