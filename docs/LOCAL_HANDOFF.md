# Local handoff: deployment, video, and final report

## Completed locally

The data preparation, original three-model experiment, held-out evaluation, saved inference artifacts, Flask API and interface are present. Additional validation-only ablations, uncertainty intervals, lexical leakage screening, error interpretation, invalid-label regression coverage, pinned local dependencies, and an isolated deployment ZIP builder have been added. No model retraining is needed to run the saved app. No public deployment has been performed.

Start with `docs/ANALYSIS.md`, `docs/DATA_AND_ATTRIBUTION.md`, `results/run_summary.md`, and aggregate CSV/JSON evidence for the final report. Read the actual code and inspect the cited test rows yourself before presenting interpretations.

## Local commands (Python 3.13)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements-training.txt
python -m src.check_setup
python -m pytest -q
python app.py
```

The existing `.venv` already works. `requirements.txt` pins runtime packages from that environment; `requirements-training.txt` adds the locally verified classical-training/notebook packages. `requirements-transformer.txt` is optional, unexecuted, and not a deployment dependency. Full original environment inventory is retained in `results/environment.txt`.

Reproduce the original training pipeline from the local corpus:

```bash
python -m src.prepare_data --local-csv data/official_kinnews.csv
python -m src.train_baseline
python -m src.evaluate
python -m src.run_notebook
```

The notebook itself repeats preparation/training/evaluation; running it alone also performs that workflow. Rerunning training may require regenerating artifact-verification records through the notebook before `check_setup`. After the original run is finalized:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m src.analyze_results
python -m src.package_deployment
```

The analysis preserves the selected classifier and original test outputs. It generates measured results, while `docs/ANALYSIS.md` is a reviewed interpretation of the current run; update the narrative if changing data or experiments.

## Deployment package

`dist/kinyanews-deployment.zip` contains the saved selected model and metadata, minimal Python source, pinned runtime requirements, Python 3.13 selection, Vercel configuration, templates, and static assets. `dist/manifest.json` lists contents, archive checksum, and isolated smoke-test outcome. Extract it into a separate deployment folder/repository if desired. ZIP size excludes installed dependencies and does not establish the hosting platform's final bundle size.

Alternatively deploy the full research repository; `.vercelignore` and function exclusions omit training data, notebook/results, local environment, and unused candidate models. The original selected `models/classifier.joblib` and `models/metadata.json` must remain included. Install only `requirements.txt` in production. No external AI API key is needed.

Vercel documents Flask detection, an `app.py` entrypoint exposing `app`, and `.python-version` selection: https://vercel.com/docs/functions/runtimes/python . Import the deployment repository using Flask detection, review build logs, and test:

- `/`: interface loads with an enabled classify button.
- `/health`: HTTP 200 and `model_loaded: true`.
- `POST /api/predict`: JSON text returns the expected category and top model scores.
- Invalid inputs return 400. Inputs must have 30–10,000 characters after whitespace normalization.

A separate clean Python 3.13 environment was created in `/tmp`, the pinned runtime wheels were downloaded and installed, `pip check` passed, and the extracted ZIP passed its smoke test using that environment. Held-out metrics also matched in the clean environment; see `results/deployment_verification.json`. This verifies local installation and inference, not a cloud build. Public runtime/build compatibility and access must still be verified during deployment. Never claim public readiness based only on `results/readiness.json`.

## What you still provide

1. Push the research source and aggregate evidence to your GitHub repository; confirm actual tracked files and article reuse conditions. This project directory has no usable project-specific Git metadata; Git discovers a parent repository spanning unrelated files. Use a dedicated project repository or the extracted deployment folder, and check the repository root before staging files. No Git metadata or unrelated files were changed.
2. Deploy, verify from an external browser, and record the direct live-system URL.
3. Record your own approximately 10-minute demo. The assignment has conflicting duration ranges whose shared boundary is 10 minutes. Explain code and decisions in your own words, including the leakage and language-quality findings.
4. Write the final research paper in the specified 11-section format, with consistent citations. Report the original winner and later exploratory ablations distinctly.
5. Add repository, video and live URLs to the README and final paper. Do not submit pending-link labels as completed links.

## Suggested live examples and defense preparation

Use held-out rows 1 (success), 0 (health-related content with economy label), 4 (environment/payment overlap), and 9 (celebrity/relationship overlap). Retrieve text from `data/test.csv` and respect the character limit; do not silently truncate a long test example while quoting its original metric. Mention the English article at row 10 as a corpus limitation, not a typical Kinyarwanda success.

Be able to explain: TF-IDF and train-only vocabulary fitting; class weighting; Logistic Regression versus SVM; C and regularization; macro versus weighted F1; validation selection versus held-out test evaluation; why the narrow original model difference is uncertain; why near-duplicates threaten estimates; and how the Flask request loads the saved pipeline without retraining. Show the exact code in `src/train_baseline.py`, `src/prepare_data.py`, `src/evaluate.py`, and `app.py`.
