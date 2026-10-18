# Analysis of measured results

## Questions and original model comparison

1. Does learning lexical features beat a majority-class sanity baseline? Validation macro F1 rises from 0.02285 to 0.71840 with word TF-IDF plus balanced Logistic Regression. The majority baseline's 19.04% accuracy illustrates class imbalance; it is not a useful classifier for minority topics.
2. Does changing the classifier help with the same word features? Linear SVM achieves 80.29% accuracy versus Logistic Regression's 79.38%, but slightly lower macro F1 (0.71710 versus 0.71840). Macro F1 gives each category equal weight, so it was the prespecified selection criterion.
3. Do character features help? Character 3–5 grams with SVM yield 0.70543 macro F1. Character features can represent recurring subword forms and spelling patterns without a morphological analyzer. On this fixed feature budget and split they did not beat word features. This result does not prove they are generally unsuitable for Kinyarwanda.

The original Logistic Regression winner stays frozen. Scores are from the pooled, deduplicated split, not the published benchmark. No Transformer results are claimed.

## Additional validation-only ablations

All use the same training and validation data, seed 42, min_df=2, max_features=60,000, sublinear TF and maximum 1,000 optimizer iterations. Only the listed factor changes from the selected balanced word (1,2)-gram Logistic Regression with C=1.

| Configuration | Accuracy | Macro F1 | Interpretation |
| --- | ---: | ---: | --- |
| Original frozen configuration | 0.79378 | 0.71840 | Original model-selection winner |
| Unigrams only | 0.79073 | 0.71942 | Slightly higher macro F1; bigrams are not clearly beneficial here |
| No class weighting | 0.76693 | 0.55462 | Large macro-F1 loss supports balancing classes in this setting |
| C=0.1 | 0.73703 | 0.66774 | Stronger regularization loses useful discrimination |
| C=10 | 0.80232 | 0.70728 | Higher accuracy but lower macro F1; accuracy alone obscures minority performance |

All four fits finished below the iteration limit. Unigrams narrowly exceed the original winner in this later exploration. The deployed model is correctly described as the winner of the **original three-candidate comparison**, not the best of every configuration ever tested. Because test results were already inspected before this extension, we do not switch models or evaluate new ablations on that test set. A future prespecified comparison should use a fresh untouched holdout or nested cross-validation. This prevents exploratory analysis from becoming hidden test-driven model selection.

## Evaluation and uncertainty

The frozen model achieves 77.486% test accuracy, macro precision 0.71435, macro recall 0.74172 and macro F1 0.72509. Weighted F1 is 0.77620. Accuracy is the fraction of correct labels; precision measures how often a predicted category is right; recall measures how many true members of a category were found. F1 is the harmonic mean of precision and recall, computed per category before macro averaging. Macro F1 is useful here because class counts are highly unequal.

A seeded 1,000-replicate class-stratified bootstrap gives test macro F1 a percentile 95% interval of [0.68869, 0.75660]. The paired validation macro-F1 difference (Logistic Regression minus word SVM) has interval [-0.02073, 0.02644]. The interval crosses zero, so the small observed advantage is not persuasive evidence of stable superiority. These intervals condition on this model and split, preserve observed class counts, and do not account for training-seed or source variation.

Sport has F1 0.93886 on 257 examples. History and education have F1 0.57143 and 0.59016 on only 34 and 28 examples. Small support limits precision of minority-class estimates; the observed imbalance is a plausible contributor, not proof of the cause of every error.

## Representative examples and model explanations

Rows are zero-based positions in `data/test.csv`. Retrieve their full text locally; `results/explained_examples.json` records SHA-256 text identifiers and feature contributions. Examples are deliberately selected for interpretation, not a random sample of errors. Contributions are feature value multiplied by the coefficient for the predicted class; they explain part of the linear score, not calibrated confidence, causality, or the entire comparison with other classes. Intercepts and competing-class coefficients also affect the final decision.

| Test row | Supplied label → prediction | Observation and plausible explanation |
| --- | --- | --- |
| 1 | entertainment → entertainment | A Miss Rwanda contestant article. `miss`, `nyampinga`, and `miss rwanda` provide strong positive evidence, matching the entertainment topic. |
| 0 | economy → health | The article discusses doctors, treatment and hospitals. `abaganga`, `abarwayi`, and `bitaro` contribute strongly toward health. The supplied economy label may reflect broad editorial taxonomy or an annotation issue; independent adjudication is needed before calling it wrong. |
| 2 | economy → politics | An article on gorilla conservation prominently names the president. `perezida`, `kagame`, and `perezida kagame` contribute toward politics. Entity shortcuts can override the economic/conservation framing. |
| 4 | environment → economy | Tree-planting workers discuss unpaid wages. `amafaranga`, `rwiyemezamirimo`, and payment vocabulary favor economy despite the environmental activity. A single label compresses a multi-topic story. |
| 9 | entertainment → relationship | A celebrity discusses partner preferences. Romance vocabulary such as `urukundo`, `umugabo`, and `gukundana` favors relationship; celebrity context supports the supplied entertainment label. |
| 10 | environment → entertainment | The inspected article is English and concerns forest-related compensation. `the` is the largest positive entertainment contribution. This is a language-quality/domain-mismatch example and a lexical shortcut, not evidence that the environmental content is entertainment. |
| 13 | history → relationship | A Gandhi article mixes historical identity with advice about personal conduct. General advice vocabulary contributes toward relationship; bag-of-ngrams features do not explicitly represent historical framing. |

Across all 369 errors, the largest confusions are politics→health (40), politics→economy (25), economy→politics (20), health→economy (14), and health→politics (14). These counts show overlapping editorial topics; they do not establish one universal cause. No test labels were edited, and no individual English article was removed to improve reported scores.

## Data quality and leakage screen

Exact normalized text overlap is zero. A new nearest-training-article screen using frozen word TF-IDF cosine similarity flags 36 validation articles and 51 test articles at similarity ≥0.90; 22 validation and 42 test articles reach ≥0.95. Maximum similarities are 0.99794 and 0.99918. Thus exact deduplication alone leaves highly similar lexical content across splits. These are candidates for duplicated/reworked articles or shared templates, requiring manual review; they are not all confirmed duplicate events. Report this threat to validity prominently. A stronger future evaluation would group related articles and sources before splitting, then train and select models again with a fresh holdout.

Training article length has median 1,779 characters and maximum 43,898. Test accuracy is 0.76667 for 210 articles below 1,000 characters, 0.76844 for 1,166 at 1,000–2,999 characters, and 0.80769 for 260 at 3,000–10,000. These descriptive groups cannot establish that length causes correctness. Three articles exceed the API's 10,000-character input limit; all were classified correctly offline. Whole-corpus offline evaluation and the deployed input contract differ slightly and the distinction affects the interpretation of deployment coverage.

## Practical limitations and next steps

Scores measure label agreement on one corpus and split. There is no user study, time-based drift evaluation, probability calibration, automatic language verification, or out-of-domain rejection. A news-trained model always chooses a known category for accepted inputs. First priorities for further research are auditing high-similarity pairs and language quality, independently reviewing ambiguous annotations, and testing grouped splits; then compare embeddings or neural models using a fresh evaluation design. More model complexity is not a substitute for trustworthy data evaluation.

The dataset paper and research code include established TF-IDF baselines, so the contribution of this implementation is a reproducible local study and practical classifier. Sources and implementation ownership are documented in `docs/DATA_AND_ATTRIBUTION.md`.
