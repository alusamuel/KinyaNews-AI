from app import app

def test_home():
    response = app.test_client().get("/")
    assert response.status_code == 200
    assert b"KinyaNews" in response.data

def test_health_reports_training_state():
    response = app.test_client().get("/health")
    assert response.status_code in (200, 503)
    assert "model_loaded" in response.get_json()

def test_predict_without_artifact_is_honest():
    from app import get_model
    if get_model() is None:
        response=app.test_client().post("/api/predict",json={"text":"This is a long enough example news article for the classifier to process."})
        assert response.status_code == 503
