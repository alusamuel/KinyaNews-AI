# Local KINNEWS data

This workspace retains the downloaded archive, converted corpus, and research splits. `.gitignore` excludes ZIP/CSV data from future Git additions; it does not untrack previously committed files. Provenance and SHA-256 hashes are in `source.json`; cleaning counts and class distribution are in `audit.json`.

Recreate the exact route using the archive URL and hash in `source.json`, then run `python -m src.import_corpus` and `python -m src.prepare_data --local-csv data/official_kinnews.csv`. The alternative Hugging Face loader is `python -m src.prepare_data`; upstream revisions are not guaranteed byte-identical.

See `docs/DATA_AND_ATTRIBUTION.md` for sources, label conventions, preparation, research limitations, and publication considerations. These data are not required in the deployment package.
