"""Frozen MiniLM embeddings + balanced logistic regression, as Chapter 3 specifies."""
import json
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from src.data import ROOT, training_data
from src.models import embedder, EMBEDDING
from src.preprocessing import clean_text
MODEL_PATH=ROOT/'models/stress_classifier.joblib'


def train():
    train_df,test_df,audit=training_data()
    prepared=ROOT/'data/processed'; prepared.mkdir(exist_ok=True)
    pd.concat([train_df.assign(split='train'),test_df.assign(split='test')]).assign(source='supplied_dreaddit')[['id','text','label','split','source']].to_parquet(prepared/'dreaddit_labeled.parquet',index=False)
    encoder=embedder()
    x=encoder.encode(train_df.text.tolist(),batch_size=32,show_progress_bar=True)
    xt=encoder.encode(test_df.text.tolist(),batch_size=32,show_progress_bar=True)
    clf=LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42).fit(x,train_df.label)
    y=test_df.label.to_numpy(); p=clf.predict(xt); scores=clf.predict_proba(xt)[:,1]
    metrics={'accuracy':accuracy_score(y,p),'precision':precision_score(y,p,zero_division=0),
        'recall':recall_score(y,p,zero_division=0),'f1':f1_score(y,p,zero_division=0),
        'roc_auc':roc_auc_score(y,scores),'baseline_accuracy':float(np.mean(y==train_df.label.mode()[0])),
        'confusion_matrix':confusion_matrix(y,p,labels=[0,1]).tolist()}
    import sklearn
    bundle={'classifier':clf,'metrics':metrics,'metadata':{'embedding_model':EMBEDDING,
        'trained_at_utc':datetime.now(timezone.utc).isoformat(),'sklearn_version':sklearn.__version__,
        'method':'Frozen MiniLM + balanced logistic regression; C=1; threshold=0.5',
        'data_audit':audit,'seed':42,'clinical_validation':False}}
    MODEL_PATH.parent.mkdir(exist_ok=True); (ROOT/'reports').mkdir(exist_ok=True)
    temporary=MODEL_PATH.with_suffix('.tmp'); joblib.dump(bundle,temporary); temporary.replace(MODEL_PATH)
    (ROOT/'reports/evaluation.json').write_text(json.dumps({'metrics':metrics,'metadata':bundle['metadata']},indent=2))
    test_df[['id','label']].assign(prediction=p,probability=scores).to_csv(ROOT/'reports/test_predictions.csv',index=False)
    return bundle


def load():
    if not MODEL_PATH.exists(): raise FileNotFoundError('Run SETUP_WINDOWS.bat once, or python setup_project.py.')
    bundle=joblib.load(MODEL_PATH)
    if bundle['metadata']['embedding_model']!=EMBEDDING: raise ValueError('Embedding model mismatch. Retrain the classifier.')
    return bundle


def predict(texts,bundle=None):
    if not texts: return np.array([])
    bundle=load() if bundle is None else bundle
    cleaned=[clean_text(t) for t in texts]
    if any(len(t)<=5 for t in cleaned): raise ValueError('Please enter meaningful text longer than five characters.')
    vectors=embedder().encode(cleaned,batch_size=32,show_progress_bar=False)
    return bundle['classifier'].predict_proba(vectors)[:,1]

if __name__=='__main__':
    print(json.dumps(train()['metrics'],indent=2))
