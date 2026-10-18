# Measured research evidence

- `validation_experiments.csv`: original baseline and three candidate models; winner selected by validation macro F1.
- `test_metrics.json`, `classification_report.csv`, `confusion_matrix.png`: frozen winner on held-out test.
- `validation_ablations.csv`: later exploratory validation-only ablations; never replace the frozen winner.
- `uncertainty.json`: seeded bootstrap intervals conditional on the current split and trained models.
- `quality_analysis.json`, `lexical_overlap.csv`: length and lexical overlap screens.
- `error_confusions.csv`, `explained_examples.json`: confusion counts and exact linear contributions, without full articles.
- `analysis_manifest.json`: hashes of splits and frozen inference artifacts used for analysis.
- `deployment_verification.json`: clean runtime installation, dependency check, extracted package test, and reproduced held-out scores.
- `model_verification.json`, `readiness.json`: artifact/inference checks, not proof of public deployment.
- `per_example_predictions.csv`: complete local article predictions; excluded from future Git additions.

Reproduce additional analysis: `OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m src.analyze_results`. It fits only validation ablations and does not rewrite the selected model or original test outputs. See `docs/ANALYSIS.md` for interpretation.
