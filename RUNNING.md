# Running the Classification Experiments

This document provides instructions for running the main classification evaluation supporting the revised manuscript, *Thematic Classification and Topic Modeling of Health- and Nature-Related Hadith Translations*.

## 1. Requirements

- Python 3
- Dependencies listed in `requirements.txt`
- Curated dataset v1.1 containing 1,323 hadith records

## 2. Dataset

The required dataset is located at:

`data/v1.1/dataset_rilis_tanpa_enam.csv`

The evaluation scripts expect the columns `text`, `label`, `url`, and `collection`.

## 3. Installation

Clone the repository and open its root directory.

```bash
git clone https://github.com/Sintia-Afriyani/hadith-theme-classification-extended-evaluation.git
cd hadith-theme-classification-extended-evaluation
```

Create and activate a Python virtual environment, then install the required packages.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment using `.venv\Scripts\activate` in Command Prompt.

## 4. Run the Main Evaluation

For a preliminary test with one repetition:

```bash
python scripts/reviewer_revision/master_cv.py 1 results/test_cv
```

For the complete evaluation with ten repetitions:

```bash
python scripts/reviewer_revision/master_cv.py 10 results/main_cv
```

The full experiment uses five outer folds, three inner folds, and near-duplicate-aware grouping.

## 5. Generated Outputs

The script produces:

- `dataset_ids_v1.1.csv`
- `cv_per_fold.csv`
- `oof_all.csv`
- `cv_best_params_per_fold.csv`
- `fold_assignments.csv`

These files contain dataset identifiers, per-fold metrics, predictions, selected hyperparameters, and fold assignments.

## 6. Reproducibility Notes

The instructions above describe the main classical classification evaluation only. Other experiments, including transformer fine-tuning, topic modeling, sensitivity analysis, and retrieval evaluation, may require additional dependencies and separate execution instructions.

The repository has not yet been fully verified for reproduction across computing environments.
