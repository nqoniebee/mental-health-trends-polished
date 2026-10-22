"""Report models. Imports are lazy; no silent replacement with keyword heuristics."""
from functools import lru_cache
import numpy as np
import pandas as pd
from src.preprocessing import clean_text
EMBEDDING = 'sentence-transformers/all-MiniLM-L6-v2'
SENTIMENT = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
EMOTION = 'j-hartmann/emotion-english-distilroberta-base'

@lru_cache(maxsize=1)
def embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING, device='cpu')

@lru_cache(maxsize=2)
def pipeline_for(kind):
    from transformers import pipeline
    name = SENTIMENT if kind=='sentiment' else EMOTION
    return pipeline('text-classification', model=name, tokenizer=name, device=-1)


def classify(texts, kind):
    # Score repeated texts once and retain the input order.
    unique = list(dict.fromkeys(texts))
    if not unique:
        return pd.DataFrame(columns=[kind+'_label',kind+'_score'])
    result = pipeline_for(kind)(unique, truncation=True, max_length=512, batch_size=16)
    values = {text: (row['label'].lower(), float(row['score'])) for text,row in zip(unique,result)}
    return pd.DataFrame([values[t] for t in texts], columns=[kind+'_label',kind+'_score'])


def annotate(frame, bundle):
    from src.classifier import predict
    out = frame.reset_index(drop=True).copy()
    texts = out.text.tolist()
    for kind in ('sentiment','emotion'):
        scored = classify(texts, kind)
        out[scored.columns] = scored
    out['stress_probability'] = predict(texts,bundle)
    out['stress_label'] = (out.stress_probability>=0.5).astype(int)
    return out


def topics(frame, minimum=5):
    from bertopic import BERTopic
    from umap import UMAP
    from hdbscan import HDBSCAN
    if len(frame)<max(10,minimum*2):
        raise ValueError('Use at least 10 unique posts and twice the minimum topic size.')
    model = BERTopic(embedding_model=embedder(),
        umap_model=UMAP(n_neighbors=min(15,len(frame)-1), n_components=5, metric='cosine', random_state=42),
        hdbscan_model=HDBSCAN(min_cluster_size=minimum, prediction_data=True),
        calculate_probabilities=False, verbose=False)
    assigned,_ = model.fit_transform(frame.text.tolist())
    result=frame.copy(); result['topic']=assigned
    return model, result


def keywords(texts, top_n=15):
    from keybert import KeyBERT
    model=KeyBERT(model=embedder())
    # Per-document extraction avoids truncating a giant concatenated corpus.
    results=model.extract_keywords(list(texts),keyphrase_ngram_range=(1,2),stop_words='english',top_n=5)
    if len(texts)==1 and results and isinstance(results[0],tuple): results=[results]
    rows=[{'keyword':term,'score':float(score)} for doc in results for term,score in doc]
    if not rows: return pd.DataFrame(columns=['keyword','score','documents'])
    return pd.DataFrame(rows).groupby('keyword').agg(score=('score','mean'),documents=('score','size')).sort_values(['documents','score'],ascending=False).head(top_n).reset_index()


def tfidf_change(before, after):
    from sklearn.feature_extraction.text import TfidfVectorizer
    if not before or not after: raise ValueError('Both time windows must contain posts.')
    vectorizer=TfidfVectorizer(stop_words='english',ngram_range=(1,2),max_features=3000)
    matrix=vectorizer.fit_transform(before+after)
    delta=np.asarray(matrix[len(before):].mean(axis=0)-matrix[:len(before)].mean(axis=0)).ravel()
    return pd.DataFrame({'term':vectorizer.get_feature_names_out(),'change':delta}).sort_values('change',ascending=False)
