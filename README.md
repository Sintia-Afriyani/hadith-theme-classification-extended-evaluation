# Hadith Theme Classification — Extended Evaluation and Reproducibility

Supplementary code, experimental outputs, and documentation for the revised manuscript **“Thematic Classification and Topic Modeling of Health- and Nature-Related Hadith Translations”** (Nur Kholis, Sugiyarto Surono, and Sintia Afriyani).

> **Scope:** This is a companion repository for the revised study and its additional reviewer-response experiments. It does not replace the earlier archived dataset/release. The availability of individual scripts, outputs, and data should be verified before a public release is created.

## Research overview

The study evaluates thematic classification of English hadith translations using **operational labels derived from selected source-book titles**: Purification, Food, Medicine, and Natural Phenomena. These labels describe corpus organization, not a definitive semantic interpretation of each hadith.

The source dataset contains 39,318 records; 29,796 belong to six selected canonical collections, and 2,674 records fall within 30 mapped books. After cleaning and deduplication, 2,611 records remained. The **historical v1.0** analytical snapshot contains 1,329 hadith. Six records were excluded from the **curated v1.1** benchmark on the basis of *source-provided* grading metadata, leaving **1,323 hadith** (Purification 399, Food 398, Medicine 343, Natural Phenomena 183). The exclusions are documented corpus decisions and are not independent scholarly judgments of hadith authenticity.

## Main evaluation protocols

| Component | Protocol / scope |
| --- | --- |
| Classical text classification | TF-IDF, SVM, naive Bayes, random forest, soft-voting ensemble; **10 repetitions × 5 outer group-aware folds**, with nested inner cross-validation |
| Near-duplicate leakage control | TF-IDF cosine similarity and connected-component grouping; primary threshold 0.80, with threshold sensitivity analyses |
| Cross-collection generalization | Leave-one-collection-out (LOCO) evaluation |
| Class-prevalence sensitivity | Comparison of Purification undersampling, full-data class weighting, and unweighted full-data training using a separate group-aware protocol |
| Transformer comparison | Fine-tuned BERT, RoBERTa, mBERT, XLM-RoBERTa against matched classical baselines on **3 × 5** group-aware folds |
| Topic modeling | LDA (primary K = 3), NMF, and K-Means; coherence and preprocessing sensitivity analyses |
| Expert validation | Historical error-enriched review of 237 records and independent double-annotation of 60 hadith |
| Retrieval pilot | Eight information needs; BM25, hard-theme filtering, and soft-theme reranking; 63 unique pooled query–hadith pairs judged by two experts |

**Important protocol distinction:** The historical v1.0 fixed/default Run B had 117 errors and provided the 237-record expert-review sample; an earlier tuned run had 120 errors. The final v1.1 tuned ensemble's first complete out-of-fold repetition had 123 errors. These are different experiments and must not be pooled as if they were one result.

## Reported highlights

- Final v1.1 classical tuned ensemble: accuracy **0.9058**, macro-F1 **0.9129** (mean across 50 outer folds).
- Matched transformer comparison: RoBERTa macro-F1 **0.9304**, compared with the matched tuned ensemble **0.9109**; the corrected test did not establish a significant difference (*p* = 0.0832).
- Primary LDA configuration: K = 3, mean coherence **0.5896**; alignment with operational labels was limited (ARI **0.2475**).
- Shared expert annotation: agreement on **53/60** primary themes (88.3%; Cohen's κ **0.853** before adjudication).
- Retrieval pilot: BM25 Precision@5 **0.750** and nDCG@5 **0.802**; theme-assisted methods did not improve overall retrieval results.

These results describe the specific archived protocols and should not be taken as full semantic validation of hadith meaning.

## Repository organization

```text
.
├── README.md
├── CHANGELOG.md
├── docs/
│   ├── EXPERIMENTS.md
│   └── DATA_PROVENANCE.md
├── data/
│   └── README.md
├── scripts/
│   └── README.md
└── results/
    └── README.md
```

Add the **actual** scripts, configuration files, split assignments, environment specification, and evaluation outputs to the corresponding directories; record each executable command and input/output file in `docs/EXPERIMENTS.md`. Do not infer that a script was executed or a result reproduced merely because it appears in this directory structure.

## Previous research materials

- Earlier repository: https://github.com/Sintia-Afriyani/hadith-theme-classification-UAD
- Earlier Zenodo curated dataset (v1.1): https://doi.org/10.5281/zenodo.23475488
- Earlier Zenodo concept DOI (all versions of that record): https://doi.org/10.5281/zenodo.23439205

This repository is **related to, but separate from,** that prior dataset archive. The v1.0/v1.1 labels above describe **dataset versions**, not GitHub software release numbers.

## Data and ethical/legal considerations

The underlying public source is the `freococo/sunnah_dataset` dataset on Hugging Face, reported as licensed **CC BY-NC-SA 4.0**. Check attribution and redistribution requirements before republishing adapted records. Review the rights and consent associated with expert annotations before making them public. Do not commit private, identifying, or restricted data or any credentials.

## Reproducibility and citation

Before publishing a release, provide complete dependency versions, executable instructions, random seeds, dataset/version checksums, and the saved fold assignments needed to connect tables and figures to source outputs. Add the new GitHub/Zenodo DOI here **only after it has been issued**. Cite the corresponding dataset version and the eventual revised article separately when appropriate.

## Authors

Nur Kholis; Sugiyarto Surono; Sintia Afriyani.
