"""Reproduce post-selection analysis without tuning on the test set.

Uses validation only for ablations. Never replaces the frozen deployed model.
Full article texts stay in the existing local prediction CSV, not this summary.
"""
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

SEED = 42


def bootstrap(y, pred, other=None, repeats=1000):
    """Paired, class-stratified bootstrap; conditional on this split/model."""
    rng = np.random.default_rng(SEED)
    groups = [np.flatnonzero(y == label) for label in np.unique(y)]
    values = []
    for _ in range(repeats):
        ix = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        score = f1_score(y[ix], pred[ix], average='macro', zero_division=0)
        if other is not None:
            score -= f1_score(y[ix], other[ix], average='macro', zero_division=0)
        values.append(score)
    return np.quantile(values, [.025, .975]).tolist()


def analyze():
    dest = Path('results')
    train, val, test = [pd.read_csv(f'data/{s}.csv') for s in ('train', 'val', 'test')]
    model = joblib.load('models/classifier.joblib')
    predicted = model.predict(test.text)
    y = test.label.to_numpy()
    saved = pd.read_csv(dest / 'per_example_predictions.csv')
    if not (saved.text.equals(test.text) and saved.label.equals(test.label)
            and np.array_equal(saved.prediction, predicted)):
        raise ValueError('Saved test predictions do not match frozen model and split')
    # Controlled validation-only ablations of the selected configuration.
    specs = [
        ('word_unigram_balanced_C1', (1, 1), 'balanced', 1.0, 'Remove bigrams'),
        ('word_bigram_unweighted_C1', (1, 2), None, 1.0, 'Remove class weighting'),
        ('word_bigram_balanced_C0.1', (1, 2), 'balanced', .1, 'Stronger regularization'),
        ('word_bigram_balanced_C10', (1, 2), 'balanced', 10., 'Weaker regularization'),
    ]
    rows = []
    for label, ngrams, weighting, c, question in specs:
        candidate = Pipeline([
            ('tfidf', TfidfVectorizer(ngram_range=ngrams, min_df=2,
                                    max_features=60000, sublinear_tf=True)),
            ('clf', LogisticRegression(C=c, class_weight=weighting,
                                      max_iter=1000, random_state=SEED))])
        print('Validation ablation:', label, flush=True)
        candidate.fit(train.text, train.label)
        p = candidate.predict(val.text)
        rows.append(dict(experiment=label, change=question, C=c,
                         ngram_range=str(ngrams), class_weight=str(weighting),
                         validation_accuracy=accuracy_score(val.label, p),
                         validation_macro_f1=f1_score(val.label, p, average='macro'),
                         convergence_iterations=int(candidate['clf'].n_iter_.max()),
                         iteration_limit=1000))
    pd.DataFrame(rows).to_csv(dest / 'validation_ablations.csv', index=False)
    svm = joblib.load('models/tfidf_linear_svm.joblib')
    print('Bootstrap intervals...', flush=True)
    intervals = {
        'method': '1000 class-stratified paired bootstrap replicates; percentile 95% intervals; seed 42',
        'limitations': 'Conditional on one split and fixed trained models; not repeated-training uncertainty or a proof of superiority.',
        'test_macro_f1_95_interval': bootstrap(y, predicted),
        'validation_logreg_minus_word_svm_macro_f1_95_interval': bootstrap(
            val.label.to_numpy(), model.predict(val.text), svm.predict(val.text))}
    (dest / 'uncertainty.json').write_text(json.dumps(intervals, indent=2) + '\n')
    # Lexical similarity screen, not a semantic/source leakage guarantee.
    vectorizer = model['tfidf']
    xtrain = vectorizer.transform(train.text)
    nearest = NearestNeighbors(n_neighbors=1, metric='cosine', algorithm='brute', n_jobs=2).fit(xtrain)
    similarity_rows = []
    for split, frame in [('val', val), ('test', test)]:
        print('Lexical overlap screen:', split, flush=True)
        x = vectorizer.transform(frame.text)
        for start in range(0, len(frame), 128):
            distance, indices = nearest.kneighbors(x[start:start+128])
            for offset, (d, ix) in enumerate(zip(distance[:, 0], indices[:, 0])):
                similarity_rows.append(dict(split=split, row=start+offset, nearest_train_row=int(ix),
                                            cosine_similarity=float(1-d),
                                            same_label=bool(frame.label.iloc[start+offset] == train.label.iloc[ix])))
    sim = pd.DataFrame(similarity_rows)
    sim.to_csv(dest / 'lexical_overlap.csv', index=False)
    quality = {
        'method': 'Nearest training article by cosine similarity in frozen word TF-IDF space; flags need manual review. Not semantic deduplication.',
        'splits': {s: {'at_least_0.90': int((g.cosine_similarity >= .9).sum()),
                       'at_least_0.95': int((g.cosine_similarity >= .95).sum()),
                       'maximum': float(g.cosine_similarity.max())} for s, g in sim.groupby('split')},
        'test_over_api_limit': int((test.text.str.len() > 10000).sum()),
        'train_characters': train.text.str.len().describe().to_dict(),
        'test_short_long_accuracy': {},
    }
    lengths = test.text.str.len()
    for label, mask in [('30_to_999', lengths < 1000), ('1000_to_2999', lengths.between(1000, 2999)),
                        ('3000_to_10000', lengths.between(3000, 10000)), ('over_10000', lengths > 10000)]:
        quality['test_short_long_accuracy'][label] = {'n': int(mask.sum()), 'accuracy': float(accuracy_score(y[mask], predicted[mask])) if mask.any() else None}
    (dest / 'quality_analysis.json').write_text(json.dumps(quality, indent=2) + '\n')
    errors = saved[~saved.correct].groupby(['label', 'prediction']).size().sort_values(ascending=False)
    errors.rename('count').to_csv(dest / 'error_confusions.csv')
    # Explain individual decisions using exact additive linear contributions.
    chosen = [1, 0, 2, 4, 9, 10, 13]
    examples = []
    for ix in chosen:
        row = test.iloc[ix]
        v = vectorizer.transform([row.text])
        cls = str(predicted[ix]); ci = list(model.classes_).index(cls)
        contributions = v.multiply(model['clf'].coef_[ci]).tocoo()
        top = sorted(zip(contributions.col, contributions.data), key=lambda item: -item[1])[:10]
        examples.append({'test_row': ix, 'text_sha256': hashlib.sha256(row.text.encode()).hexdigest(),
                         'true_label': row.label, 'predicted_label': cls,
                         'top_positive_contributions': [{'feature': vectorizer.get_feature_names_out()[j], 'contribution_to_logit': float(value)} for j, value in top]})
    (dest / 'explained_examples.json').write_text(json.dumps(examples, ensure_ascii=False, indent=2) + '\n')
    checksums = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                 [Path(f'data/{s}.csv') for s in ('train', 'val', 'test')] + [Path('models/classifier.joblib'), Path('models/metadata.json')]}
    (dest / 'analysis_manifest.json').write_text(json.dumps({'sha256': checksums, 'seed': SEED, 'selected_model_changed': False}, indent=2) + '\n')
    print('Analysis complete; selected model unchanged.', flush=True)


if __name__ == '__main__':
    analyze()
