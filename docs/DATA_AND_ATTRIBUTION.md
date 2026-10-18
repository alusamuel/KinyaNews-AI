# Data, attribution, and scope

## Problem and intended users

KinyaNews assigns one of 14 news topics to a Kinyarwanda article. Intended users are news editors organizing an incoming queue, archivists indexing a collection, and readers filtering topics. These are proposed uses, not user-study findings. A person should review decisions before publication or consequential use. The project studies low-resource topic classification, not translation or fact-checking.

## Existing resources and implemented work

The existing resource is KINNEWS, introduced by Niyongabo, Qu, Kreutzer, and Huang (2020), *KINNEWS and KIRNEWS: Benchmarking Cross-Lingual Text Classification for Kinyarwanda and Kirundi*, COLING, https://aclanthology.org/2020.coling-main.480/ . The research repository describes collection from news websites and newspapers: https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus . The dataset card is https://huggingface.co/datasets/andreniyongabo/kinnews_kirnews . This project uses the existing cleaned KINNEWS corpus, not newly scraped articles, generated training examples, or newly annotated labels.

Implemented here: archive import with numeric-label conversion checks, extra duplicate/conflict removal, a new stratified split, TF-IDF/classifier training, validation comparison, held-out evaluation, post-selection analysis, model packaging, and Flask inference. scikit-learn supplies the algorithms; this project configures and trains them rather than implementing their optimizers from scratch.

## Corpus and preparation

The imported cleaned corpus has 21,268 rows, originally split 17,014/4,254. Labels follow the corpus's 14-category one-based convention in its archive; the importer converts to English names and verifies original English labels. The Hugging Face loader uses zero-based labels. Do not pass original one-based numbers directly to `prepare_data`.

The existing corpus has upstream cleaning. Locally, title and content are concatenated, Unicode-casefolded, and whitespace normalized; rows shorter than 30 characters are excluded. Every group of identical normalized text with conflicting labels is removed, then remaining identical texts are deduplicated. There are 159 distinct conflicting texts, 1,453 rows in those conflicting groups (derived from recorded counts), and 8,891 additional duplicates removed. The retained corpus has 10,924 articles. A stratified 70/15/15 split with seed 42 produces 7,646/1,639/1,639 articles. The largest class has 2,082 articles and the smallest 109. See `data/audit.json` for all counts.

Pooling and deduplicating changes the published evaluation protocol: do not compare our score directly with the paper's benchmark leaderboard. The TF-IDF vocabulary and document frequencies are fitted only on training text. Validation selects the original winner; held-out test measures its final performance. Later ablations are explicitly exploratory validation analyses and do not replace that model.

## Quality limitations

Normalized exact overlap is zero, but article rewrites and shared source templates can cross splits. `results/lexical_overlap.csv` screens for high lexical similarity, not semantic equivalence. Source-disjoint or time-disjoint evaluation has not been implemented because the modeling CSVs do not retain source/time fields. Some inspected articles mix topics, and at least one inspected test article is English despite the nominal Kinyarwanda corpus. No automatic language-based deletion or test-label correction was performed. Thus results measure agreement with supplied labels on this corpus, not verified linguistic purity or an independently adjudicated gold standard.

## License and publication

The original research repository displays an MIT license. Preserve the upstream license and attribution for reused material: https://github.com/Andrews2017/KINNEWS-and-KIRNEWS-Corpus/blob/main/LICENSE . That repository license alone is not a separate determination of rights in each third-party news article. The archive used locally came from the mirror recorded in `data/source.json`; hashes identify the exact local bytes.

Corpus ZIP/CSVs and the full per-article prediction CSV are excluded by `.gitignore` for future additions. This does not remove files already tracked in a repository; check tracking before publishing. Local files are retained for research. Aggregate metrics and the text-free example identifiers are included as evidence. The executed notebook contains article excerpts: review those before public distribution. Deployment packages omit the corpus, research notebook, and per-article outputs entirely.

## Method references

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html
- scikit-learn, text feature extraction: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- scikit-learn, Logistic Regression: https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
- scikit-learn, classification metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics

These notes document provenance and implementation scope. The final research paper is a separate submission.

## AI assistance and verification

AI assistance was used to generate and revise parts of the code, tests, analysis scripts, and documentation. The quantitative results cited here were produced by executing the implemented workflows on KINNEWS; no generative model was used to create the primary training dataset. Automated verification includes model reloads, agreement between API and direct inference, held-out metric reproduction, and a clean runtime installation.

The qualitative error explanations are AI-assisted interpretations supported by article inspection and linear feature contributions. They remain interpretations rather than independently adjudicated annotations. These automated checks do not establish the author’s individual understanding; that is assessed through explanation of the implementation and results.
