# KinyaNews AI: completed notebook results

Run date: 2026-10-18. All executable notebook cells completed without errors.

## Data

The KINNEWS cleaned corpus contains 21,268 rows. Cleaning retained 10,924 unique articles across 14 categories: 7,646 training, 1,639 validation, and 1,639 test rows.
Removed 8,891 duplicate rows after removing contradictory text groups (159 distinct conflicting texts). Exact cross-split overlap: zero.

Original labels were converted from one-based numeric IDs to official English names and verified against the original raw English labels. Source URLs and SHA-256 hashes are in `data/source.json`.

## Validation comparison

| Model | Accuracy | Macro F1 | Training seconds |
| --- | ---: | ---: | ---: |
| tfidf_logreg | 79.38% | 0.71840 | 13.329 |
| tfidf_linear_svm | 80.29% | 0.71710 | 8.596 |
| tfidf_char_svm | 79.38% | 0.70543 | 41.345 |
| majority_dummy | 19.04% | 0.02285 | 0.008 |

Logistic Regression was selected using validation macro F1, although word Linear SVM had higher validation accuracy. The test set was not used for selection.

## Held-out test result

- Accuracy: 77.49%.
- Macro F1: 0.72509.
- Test articles: 1,639.
- Sport F1: 0.93886. History F1: 0.57143. Education F1: 0.59016.
- Most frequent confusion: politics predicted as health (40 articles).

## Model verification

All three candidates were saved, reloaded, and reproduced their recorded validation macro F1. Each has 14 classes, 60,000 features, and finite coefficients. The deployed classifier produces the same validation predictions as the selected Logistic Regression artifact.

Artifacts: `models/tfidf_logreg.joblib`, `models/tfidf_linear_svm.joblib`, `models/tfidf_char_svm.joblib`, and selected `models/classifier.joblib` (with `models/metadata.json`). Hashes are in `results/model_verification.json`; package versions are in `results/environment.txt`.

The homepage and health endpoint returned HTTP 200. A held-out article was sent through `/api/predict`; API output matched direct model inference. That smoke-test article was misclassified (economy → health), illustrating that operational correctness does not imply perfect classification.

## Scope and limitations

This notebook pools the original train/test files before cleaning and making a new stratified 70/15/15 split (seed 42). These measurements are not directly comparable to the paper’s original benchmark. Semantic near-duplicate and source overlap have not been audited. Minority-class estimates use small test samples. Logistic Regression scores are uncalibrated. The optional GPU transformer experiment was not run.

Source: [Hugging Face dataset card](https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews), [research repository](https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus).

## Extended post-selection analysis

Four validation-only ablations and uncertainty/quality analyses are now in `docs/ANALYSIS.md` and their corresponding result files. The selected model and original test predictions remain unchanged. A lexical cosine screen flags 51 test articles at training similarity ≥0.90 (42 at ≥0.95), so zero exact overlap does not establish absence of leakage. At least one inspected test article is English. These limitations must accompany the reported results. The later unigram ablation slightly exceeds the original winner on validation macro F1 but is not promoted after test inspection.
