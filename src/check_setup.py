"""Verify local data, trained pipelines, and application inference without retraining."""
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from src.labels import LABELS

ROOT = Path(__file__).resolve().parent.parent


def check_setup():
    os.chdir(ROOT)
    audit = json.loads(Path('data/audit.json').read_text())
    source = json.loads(Path('data/source.json').read_text())
    metadata = json.loads(Path('models/metadata.json').read_text())
    for filename, key in [('official_kinnews.csv', 'csv_sha256'), ('KINNEWS.zip', 'archive_sha256')]:
        actual = hashlib.sha256((Path('data') / filename).read_bytes()).hexdigest()
        if actual != source[key]:
            raise ValueError(f'Data checksum mismatch: {filename}')
    splits = {}
    for split in ('train', 'val', 'test'):
        frame = pd.read_csv(Path('data') / (split + '.csv'))
        if frame[['text', 'label']].isna().any().any():
            raise ValueError(f'Missing text or labels in {split}')
        if len(frame) != audit['splits'][split] or frame.text.duplicated().any():
            raise ValueError(f'Row count or duplicate failure in {split}')
        if set(frame.label) != set(LABELS):
            raise ValueError(f'Category coverage failure in {split}')
        splits[split] = frame
    for first, second in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if set(splits[first].text) & set(splits[second].text):
            raise ValueError(f'Exact overlap between {first} and {second}')
    if metadata['training_rows'] != len(splits['train']) or metadata['validation_rows'] != len(splits['val']):
        raise ValueError('Model metadata does not match local data')
    verification = json.loads(Path('results/model_verification.json').read_text())
    models = {}
    for record in verification['artifacts']:
        path = Path('models') / (record['model'] + '.joblib')
        if hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            raise ValueError(f'Model checksum mismatch: {path}')
        model = joblib.load(path)
        check_is_fitted(model.named_steps['tfidf'])
        check_is_fitted(model.named_steps['clf'])
        if set(model.classes_) != set(LABELS) or not np.isfinite(model.named_steps['clf'].coef_).all():
            raise ValueError(f'Invalid trained coefficients or classes: {path}')
        models[record['model']] = model
    deployed = joblib.load('models/classifier.joblib')
    if not np.array_equal(deployed.predict(splits['val'].text), models[metadata['selected_model']].predict(splits['val'].text)):
        raise ValueError('Application model does not match the selected candidate')
    from app import app, get_model, get_metadata
    get_model.cache_clear()
    get_metadata.cache_clear()
    client = app.test_client()
    for route in ('/', '/static/app.js', '/static/style.css', '/health'):
        if client.get(route).status_code != 200:
            raise ValueError(f'Application route unavailable: {route}')
    samples = splits['test'].loc[splits['test'].text.str.len().between(30, 10000)].groupby('label').head(1)
    for text in samples.text:
        response = client.post('/api/predict', json={'text': text})
        if response.status_code != 200 or response.get_json()['label'] != deployed.predict([text])[0]:
            raise ValueError('Application prediction differs from direct inference')
    if len(samples) != len(LABELS):
        raise ValueError('Missing an API test example for a category')
    result = {'status': 'ready', 'selected_model': metadata['selected_model'],
              'classes': len(deployed.classes_), 'data_rows': {key: len(value) for key, value in splits.items()},
              'saved_candidates': list(models), 'api_examples_checked': len(samples),
              'data_checksums_verified': True, 'model_checksums_verified': True,
              'exact_split_overlap': 0, 'application_model_matches_selected': True,
              'scope': 'local data, model artifacts and Flask test-client inference',
              'public_deployment_verified': False}
    Path('results/readiness.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    print(json.dumps(check_setup(), indent=2))
