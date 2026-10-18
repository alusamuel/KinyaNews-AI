"""Regression coverage for corpus labels and real deployed-pipeline behavior."""
from pathlib import Path
import pandas as pd
import pytest

from app import app, get_model
from src.labels import name
from src.prepare_data import clean_frame


@pytest.mark.parametrize('value', [-1, 14, 'unknown'])
def test_invalid_corpus_labels_rejected(value):
    frame = pd.DataFrame({'title': ['Amakuru mashya yerekeye uburezi mu Rwanda'],
                          'content': [''], 'label': [value]})
    with pytest.raises(ValueError, match='Unrecognized label'):
        clean_frame(frame)


def test_negative_id_does_not_alias_last_category():
    assert name(-1) == '-1'


@pytest.mark.skipif(not Path('models/classifier.joblib').exists(), reason='Train model first')
@pytest.mark.parametrize('payload', [None, [], {'text': 123}, {'text': ' ' * 40}, {'text': 'a' * 10001}])
def test_real_pipeline_rejects_invalid_requests(payload):
    assert get_model() is not None
    assert app.test_client().post('/api/predict', json=payload).status_code == 400


@pytest.mark.skipif(not Path('models/classifier.joblib').exists() or not Path('data/test.csv').exists(), reason='Prepare data and train first')
def test_real_pipeline_matches_held_out_inference():
    frame = pd.read_csv('data/test.csv')
    frame = frame[frame.text.str.len().between(30, 10000)].groupby('label').head(1)
    client = app.test_client()
    model = get_model()
    for text, expected in zip(frame.text, model.predict(frame.text)):
        response = client.post('/api/predict', json={'text': text})
        assert response.status_code == 200
        assert response.json['label'] == expected
        scores = response.json['top_categories']
        assert scores[0]['label'] == expected
        assert all(0 <= row['score'] <= 1 for row in scores)
        assert scores == sorted(scores, key=lambda row: -row['score'])
