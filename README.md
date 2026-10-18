# KinyaNews AI — Kinyarwanda News Topic Classification

KinyaNews classifies Kinyarwanda news articles into 14 topics using TF-IDF features and supervised classifiers. The repository contains data preparation, model training, evaluation, and a Flask application that loads the selected trained pipeline. Experiments use the KINNEWS corpus; measured results are recorded in [results/run_summary.md](results/run_summary.md).

The final research report, demonstration video, and public deployment are pending. Installation and deployment preparation are documented in [docs/LOCAL_HANDOFF.md](docs/LOCAL_HANDOFF.md).

## Submission resources

- GitHub repository: [alusamuel/KinyaNews-AI](https://github.com/alusamuel/KinyaNews-AI).
- Live system: deployment pending.
- Demo video: recording pending.
- Local handoff and deployment instructions: [docs/LOCAL_HANDOFF.md](docs/LOCAL_HANDOFF.md).
- Measured analysis and error discussion: [docs/ANALYSIS.md](docs/ANALYSIS.md).
- Data, attribution, intended users and limitations: [docs/DATA_AND_ATTRIBUTION.md](docs/DATA_AND_ATTRIBUTION.md).

Original test accuracy is **77.49%**, macro F1 **0.72509**. A lexical screen flags **51 test articles** with training similarity ≥0.90; this limits interpretation of the scores. The original classifier remains frozen after additional exploratory validation experiments.

## Source and research

KINNEWS/KIRNEWS: Niyongabo et al. (2020), https://arxiv.org/abs/2010.12174 . Dataset https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews and research implementation https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus . The corpus includes 14 categories. Attribution and the upstream repository license are discussed in `docs/DATA_AND_ATTRIBUTION.md`.

## Run on Python 3.13 (verified locally)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements-training.txt
python -m src.prepare_data
python -m src.train_baseline
python -m src.evaluate
python app.py
```

The public Hugging Face dataset uses an old Python dataset loader that may be incompatible with some newer library releases. `src.prepare_data` also accepts the downloaded original CSV with columns `label,title,content`:

```bash
python -m src.prepare_data --local-csv /path/to/official_kinnews.csv
```

Local CSV labels must be English category names or zero-based Hugging Face IDs. The original corpus CSVs have **one-based** labels; convert them before passing them to this loader. For the `KINNEWS.zip` archive referenced by the Hugging Face dataset loader, use:

```bash
curl -fL --retry 3 https://raw.githubusercontent.com/saradhix/kinnews_kirnews/master/KINNEWS.zip -o data/KINNEWS.zip
python -m src.import_corpus
python -m src.run_notebook
```

The import checks the label order against the original English labels and records source hashes in `data/source.json`. The notebook uses `data/official_kinnews.csv` when present, executes every analysis cell, and saves outputs back to `notebooks/research_walkthrough.ipynb`. It pools the original train/test files and creates a new deduplicated split; these scores are not directly comparable to the published benchmark split. `results/model_verification.json` records artifact reload checks and API inference checks.

If Hugging Face loading fails, obtain data from the official corpus, convert to the required CSV columns without relabeling rows, then supply `--local-csv`. Reported training and evaluation results use the imported KINNEWS records. Small software-test fixtures are separate from the research dataset and are not used to calculate reported performance.

## Research process

1. Inspect `data/audit.json` for original vs. retained records and label balance.
2. Run the Jupyter notebook: `notebooks/research_walkthrough.ipynb`, opening it from the project root.
3. `src.train_baseline` trains a majority-class dummy comparator and three trainable pipelines: TF-IDF word n-grams + Logistic Regression, word n-grams + Linear SVM, character n-grams + Linear SVM. Winner is selected **only on validation macro F1**.
   All three trained candidates are saved as named `.joblib` files in `models/`; `classifier.joblib` is the selected candidate used by the web app.
4. `src.evaluate` evaluates the selected model on untouched `data/test.csv`, saving metrics, per-class report, confusion matrix, and prediction-level errors.
5. Optional neural experiment (first install `requirements-transformer.txt`): `python -m src.train_transformer --model xlm-roberta-base --epochs 2`. Compare reported validation metrics; this large model is **not automatically deployed on Vercel**.
6. Read `docs/ANALYSIS.md` and inspect the identified held-out rows and feature contributions locally. The documented examples retain their original corpus labels.

The 70/15/15 stratified split is created **after text normalization, removal of exact duplicates and contradictory identical articles**. Exact overlap is checked. A lexical similarity screen in `results/lexical_overlap.csv` flags high-similarity cross-split articles; semantic/source equivalence remains unaudited.

## Web application / Vercel

For the prepared local workspace, run `./start.sh` (or `PORT=5001 ./start.sh` if port 5000 is occupied). This checks corpus and model checksums, data splits, all trained candidate models, static assets, and API inference before starting the app. Run `.venv/bin/python -m src.check_setup` to check readiness without starting a server; its report is saved to `results/readiness.json`.

The app uses `models/classifier.joblib` and `models/metadata.json` for predictions. It does not retrain on requests or need the training CSVs for inference. The corpus, train/validation/test CSVs, all three candidate models, and experiment outputs remain available locally for further research tasks. Keep the `.venv` environment with its recorded package versions (`results/environment.txt`) when loading these models.

Once training succeeds, `models/classifier.joblib` and `models/metadata.json` are required inference artifacts. They are available in `models/` and are not excluded by `.gitignore`; include them when packaging the app. Keep the whole Vercel function (including dependencies) within the platform limits. The Flask entrypoint is `app.py` and the API route is `POST /api/predict`, with `{ "text": "..." }`. Health route: `GET /health`.

1. Test the trained system locally at `http://localhost:5000`.
2. Run `pytest -q` and check inference on several real held-out Kinyarwanda articles.
3. Push the source code and appropriately versioned model artifact to a repository you own. No credentials, secret keys or private news records.
4. Import repository into Vercel as a Python/Flask project. Ensure required model weights are packaged with the deployment, then visit `/health` (expected HTTP 200 and `model_loaded:true`).
5. Verify public URL from an external browser, and demonstrate examples including real errors.

The application intentionally responds with HTTP 503 when the model is missing; it **does not fake predictions**. Linear SVM doesn't output probabilities; the interface explicitly says so. Logistic Regression outputs uncalibrated model scores.

## File map

- `src/prepare_data.py` — original data loader, cleaning, labels, stratified splits
- `src/labels.py` — corpus's official numeric-label order and display names
- `src/train_baseline.py` — controlled classical ML experiments, model selection
- `src/train_transformer.py` — optional GPU fine-tuning
- `src/evaluate.py` — held-out evaluation and error table
- `notebooks/research_walkthrough.ipynb` — step-by-step research workflow
- `app.py`, `templates`, `static` — web interface and inference API
- `tests` — validation/unit/API tests (API tests require Flask)
- `vercel.json`, `requirements.txt` — deployment configuration

## Known limitations

- News classification, not fact-checking, translation or an open-domain chatbot.
- Category errors are possible, especially on ambiguous topics and very short text.
- A model trained on news data may misclassify non-news text.
- No claims about generalization, quality or accuracy should be made without executing real experiments.
- A fresh checkout without the saved artifacts needs the official KINNEWS data and a full training/evaluation run. The local executed run is documented in `results/run_summary.md`.

## Additional reproducible analysis and packaging

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -m src.analyze_results
python -m src.package_deployment
```

The first command writes validation-only ablations, bootstrap intervals, lexical-overlap flags, and model explanations without changing the frozen classifier. The second creates `dist/kinyanews-deployment.zip` and tests the extracted package in a fresh interpreter using the installed dependencies. Use Python 3.13 and pinned runtime requirements for deployment; public verification remains required. See the handoff for details.

## Attribution and assistance

Dataset, software, and implementation contributions are distinguished in [docs/DATA_AND_ATTRIBUTION.md](docs/DATA_AND_ATTRIBUTION.md), including the use of AI assistance. The repository reports measured outputs and distinguishes completed experiments from optional or pending work.
