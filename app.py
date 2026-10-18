"""KinyaNews classifier web interface / API (Flask for Vercel)."""
import json
import os
from functools import lru_cache
from pathlib import Path
import joblib
from flask import Flask,jsonify,render_template,request
from src.labels import name,kin_name

ROOT=Path(__file__).resolve().parent
MODEL_PATH=ROOT/'models'/'classifier.joblib'
META_PATH=ROOT/'models'/'metadata.json'
app=Flask(__name__)
app.config['MAX_CONTENT_LENGTH']=64*1024

@lru_cache(maxsize=1)
def get_model():
    return joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@lru_cache(maxsize=1)
def get_metadata():
    return json.loads(META_PATH.read_text(encoding='utf8')) if META_PATH.exists() else {}

@app.get('/')
def home():
    return render_template('index.html',model_ready=get_model() is not None,metadata=get_metadata())

@app.get('/health')
def health():
    ready=get_model() is not None
    return jsonify(status='ready' if ready else 'model_missing',model_loaded=ready,model=get_metadata().get('selected_model')),200 if ready else 503

@app.post('/api/predict')
def predict():
    model=get_model()
    if model is None:
        return jsonify(error='Model unavailable: train first, then deploy models/classifier.joblib.'),503
    data=request.get_json(silent=True)
    if not isinstance(data,dict) or not isinstance(data.get('text'),str):
        return jsonify(error='Send a JSON object with a text string.'),400
    sentence=' '.join(data['text'].split()).strip()
    if not 30<=len(sentence)<=10000:
        return jsonify(error='Enter between 30 and 10,000 characters.'),400
    predicted=model.predict([sentence])[0]
    result={'label':name(predicted),'label_kin':kin_name(predicted),'model':get_metadata().get('selected_model','trained-classifier'),'warning':'Category prediction is not fact-checking. Confidence, if shown, is not guaranteed calibration.'}
    if hasattr(model,'predict_proba'):
        scores=model.predict_proba([sentence])[0]
        ranked=sorted(zip(model.classes_,scores),key=lambda p:-p[1])[:3]
        result['top_categories']=[{'label':name(k),'label_kin':kin_name(k),'score':round(float(v),4)} for k,v in ranked]
    return jsonify(result)

if __name__=='__main__':
    app.run(host='127.0.0.1',port=int(os.environ.get('PORT','5000')))
