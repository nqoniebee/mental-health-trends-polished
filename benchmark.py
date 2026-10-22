"""Optional Chapter 4-style TF-IDF comparison, separate from the deployed MiniLM model.
Uses cleaned, group-separated supplied benchmark data, never the repetitive sample.
"""
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
from src.data import ROOT,training_data

def main():
    train,test,audit=training_data();rows=[]
    estimators={'Logistic Regression':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),
        'Multinomial Naive Bayes':MultinomialNB(),
        'Linear SVM':LinearSVC(class_weight='balanced',random_state=42),
        'Random Forest':RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=42,n_jobs=-1)}
    for name,estimator in estimators.items():
        pipe=make_pipeline(TfidfVectorizer(ngram_range=(1,2),max_features=10000,min_df=2),estimator)
        pipe.fit(train.text,train.label);prediction=pipe.predict(test.text)
        score=pipe.predict_proba(test.text)[:,1] if hasattr(pipe,'predict_proba') else pipe.decision_function(test.text)
        rows.append(dict(model=name,accuracy=accuracy_score(test.label,prediction),precision=precision_score(test.label,prediction,zero_division=0),recall=recall_score(test.label,prediction,zero_division=0),f1=f1_score(test.label,prediction),roc_auc=roc_auc_score(test.label,score)))
    out=ROOT/'reports';out.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'tfidf_comparison.csv',index=False)
    (out/'tfidf_audit.json').write_text(json.dumps(audit,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__': main()
