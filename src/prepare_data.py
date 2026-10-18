"""Reproducible KINNEWS preparation with exact duplicate and conflict removal."""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from src.labels import name, LABELS

SEED = 42
DATASET_ID = 'andreniyongabo/kinnews_kirnews'

def canonical(text):
    return re.sub(r'\s+', ' ', str(text).casefold()).strip()

def clean_frame(frame):
    expected = {'label','title','content'}
    if not expected.issubset(frame.columns):
        raise ValueError(f'Missing required columns: {sorted(expected-set(frame.columns))}')
    df = frame.copy()
    counts = {'raw_rows':len(df)}
    df['title'] = df.title.fillna('').astype(str)
    df['content'] = df.content.fillna('').astype(str)
    df['text'] = (df.title+' '+df.content).map(canonical)
    df = df[df.text.str.len()>=30].copy()
    counts['valid_length_rows'] = len(df)
    df['label'] = df.label.map(name)
    if df.label.isna().any() or not df.label.isin(LABELS).all():
        raise ValueError('Unrecognized label')
    conflict = df.groupby('text').label.nunique()
    ambiguous = set(conflict[conflict>1].index)
    counts['conflicting_unique_texts'] = len(ambiguous)
    df = df[~df.text.isin(ambiguous)]
    before = len(df)
    df = df.drop_duplicates('text').reset_index(drop=True)
    counts['duplicate_rows_removed'] = before-len(df)
    counts['unique_rows'] = len(df)
    if df.empty or df.label.value_counts().min()<7:
        raise ValueError('Too few rows per class for stratified 70/15/15 split')
    return df,counts

def fetch_dataset():
    try:
        from datasets import load_dataset
        ds = load_dataset(DATASET_ID,'kinnews_cleaned',trust_remote_code=True)
        return pd.concat([split.to_pandas() for split in ds.values()],ignore_index=True)
    except Exception as e:
        raise RuntimeError('KINNEWS download failed. Download the official corpus yourself and use --local-csv with columns label,title,content. Original error: '+str(e)) from e

def prepare(out_dir='data',local_csv=None):
    source = str(local_csv) if local_csv else DATASET_ID+'/kinnews_cleaned'
    raw = pd.read_csv(local_csv) if local_csv else fetch_dataset()
    clean,audit = clean_frame(raw)
    train,hold = train_test_split(clean,test_size=.30,stratify=clean.label,random_state=SEED)
    valid,test = train_test_split(hold,test_size=.5,stratify=hold.label,random_state=SEED)
    folder = Path(out_dir)
    folder.mkdir(parents=True,exist_ok=True)
    splits = {'train':train,'val':valid,'test':test}
    for split,frame in splits.items():
        frame[['text','label']].to_csv(folder/(split+'.csv'),index=False)
    indexes = {k:set(v.text) for k,v in splits.items()}
    assert not any(indexes[a]&indexes[b] for a,b in [('train','val'),('train','test'),('val','test')])
    audit.update({'source':source,'seed':SEED,'splits':{k:len(v) for k,v in splits.items()},'label_counts':clean.label.value_counts().to_dict(),'exact_cross_split_overlap':0,'warning':'Exact duplicates checked; semantic near-duplicates and sources may still cause leakage.'})
    (folder/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    return audit

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--local-csv')
    p.add_argument('--out-dir',default='data')
    args=p.parse_args()
    print(json.dumps(prepare(args.out_dir,args.local_csv),ensure_ascii=False,indent=2))
