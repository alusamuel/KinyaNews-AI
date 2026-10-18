"""Controlled validation experiments; save the best non-dummy classifier."""
import argparse
import json
import time
import platform
from pathlib import Path
import joblib
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

SEED=42

def experiments(data_dir='data',model_dir='models',results_dir='results'):
    train=pd.read_csv(Path(data_dir)/'train.csv').dropna()
    val=pd.read_csv(Path(data_dir)/'val.csv').dropna()
    assert set(train.text).isdisjoint(val.text),'Train-validation overlap'
    Path(model_dir).mkdir(parents=True,exist_ok=True)
    Path(results_dir).mkdir(parents=True,exist_ok=True)
    configurations={
      'majority_dummy':DummyClassifier(strategy='most_frequent'),
      'tfidf_logreg':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True)),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=SEED))]),
      'tfidf_linear_svm':Pipeline([('tfidf',TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=60000,sublinear_tf=True)),('clf',LinearSVC(class_weight='balanced',random_state=SEED))]),
      'tfidf_char_svm':Pipeline([('tfidf',TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,max_features=60000,sublinear_tf=True)),('clf',LinearSVC(class_weight='balanced',random_state=SEED))])
    }
    rows=[];models={}
    for label,model in configurations.items():
        print(f'Training {label}...', flush=True)
        tic=time.perf_counter();model.fit(train.text,train.label)
        pred=model.predict(val.text)
        rows.append({'experiment':label,'validation_accuracy':round(float(accuracy_score(val.label,pred)),5),'validation_macro_f1':round(float(f1_score(val.label,pred,average='macro',zero_division=0)),5),'train_seconds':round(time.perf_counter()-tic,3),'train_size':len(train),'val_size':len(val),'seed':SEED})
        models[label]=model
        if label != 'majority_dummy':
            joblib.dump(model,Path(model_dir)/(label+'.joblib'),compress=3)
        print(f"{label}: validation macro F1={rows[-1]['validation_macro_f1']:.5f}", flush=True)
    rows.sort(key=lambda x:(x['validation_macro_f1'],x['validation_accuracy']),reverse=True)
    candidate=[r for r in rows if r['experiment']!='majority_dummy']
    winner=candidate[0]['experiment']
    joblib.dump(models[winner],Path(model_dir)/'classifier.joblib',compress=3)
    metadata={'selected_model':winner,'selection':'highest validation macro F1 among non-dummy methods','labels':sorted(train.label.unique().tolist()),'seed':SEED,'training_rows':len(train),'validation_rows':len(val),'python_version':platform.python_version(),'sklearn_version':sklearn.__version__,'candidate_artifacts':[r['experiment']+'.joblib' for r in candidate]}
    (Path(model_dir)/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    pd.DataFrame(rows).to_csv(Path(results_dir)/'validation_experiments.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False));return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default='data');p.add_argument('--model-dir',default='models');p.add_argument('--results-dir',default='results');a=p.parse_args()
    experiments(a.data_dir,a.model_dir,a.results_dir)
