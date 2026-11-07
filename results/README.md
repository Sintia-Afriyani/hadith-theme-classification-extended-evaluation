# Experimental Results

This directory contains numerical summaries and figures supporting the revised manuscript **"Thematic Classification and Topic Modeling of Health- and Nature-Related Hadith Translations."**

## Available Files

| File | Description |
|---|---|
| Figure_2_Confusion_Matrix.png | Confusion matrix heatmap of the tuned ensemble |
| Figure_3_Model_Comparison.png | Comparison of classical and fine-tuned transformer models |
| Figure_4_Retrieval_Evaluation.png | Comparison of BM25 and theme-assisted retrieval |
| confusion_matrix_v11.csv | Confusion matrix values for the final v1.1 benchmark |
| model_comparison_matched_folds.csv | Summary of model performance on matched evaluation folds |
| retrieval_evaluation_summary.csv | Summary of retrieval evaluation metrics |

## Experimental Details

**Classification evaluation:** The final v1.1 corpus contains 1,323 hadith. The confusion matrix represents the first complete out-of-fold repetition of the tuned ensemble, with 1,200 correct predictions and 123 errors.

**Model comparison:** Classical baselines and fine-tuned transformers were evaluated using 3 × 5 matched group-aware folds. The CSV contains aggregated performance results rather than individual fold predictions.

**Retrieval evaluation:** Eight information needs were evaluated using expert relevance judgments on 63 unique pooled query–hadith pairs. The CSV summarizes Precision@5, strict Precision@5, and nDCG@5.

## Reproducibility Notes

These files are result summaries and visualizations. They do not independently provide all materials required to reproduce the experiments.

Full reproducibility additionally requires the relevant dataset versions, preprocessing procedures, fold assignments, experimental scripts, dependencies, and applicable record-level outputs.

The results and supporting materials should be verified against the final manuscript before public release.
