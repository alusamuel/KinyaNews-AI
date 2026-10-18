import app as module
from sklearn.dummy import DummyClassifier

def test_missing_model(monkeypatch):
    monkeypatch.setattr(module,'get_model',lambda:None)
    client=module.app.test_client()
    assert client.get('/health').status_code==503
    assert client.post('/api/predict',json={'text':'amakuru menshi yanditse mu kinyarwanda'}).status_code==503

def test_classifier_inference(monkeypatch):
    m=DummyClassifier(strategy='most_frequent').fit(['a','b','c'],['sport','sport','health'])
    monkeypatch.setattr(module,'get_model',lambda:m)
    monkeypatch.setattr(module,'get_metadata',lambda:{'selected_model':'test-only'})
    client=module.app.test_client()
    assert client.get('/health').status_code==200
    reply=client.post('/api/predict',json={'text':'Ikipe yatsinze umukino wabereye mu mujyi wa Kigali'}).get_json()
    assert reply['label']=='sport'
    assert client.post('/api/predict',json={'text':'short'}).status_code==400
