"""Run real model smoke checks after setup; writes evidence without raw text."""
import json
import pandas as pd
from src.data import ROOT,normalize_posts
from src import classifier,models
from src.trends import aggregate

def main():
    frame,_=normalize_posts(pd.read_csv(ROOT/'data/raw/sample_reddit_mental_health.csv'))
    bundle=classifier.load()
    result=models.annotate(frame,bundle)
    assert len(result)==len(frame)
    assert result.stress_probability.between(0,1).all()
    assert set(result.sentiment_label).issubset({'negative','neutral','positive'})
    assert result.emotion_label.notna().all()
    keywords=models.keywords(tuple(frame.head(5).text))
    assert not keywords.empty
    topic_model,assigned=models.topics(frame,2)
    assert len(assigned)==len(frame)
    trends=aggregate(result,window=2,minimum=1)
    report={'real_inference_rows':len(result),'sentiment_labels':sorted(result.sentiment_label.unique()),
            'emotion_labels':sorted(result.emotion_label.unique()),'keywords_returned':len(keywords),
            'topic_assignments':len(assigned),'topic_count_excluding_outliers':int(assigned.loc[assigned.topic.ge(0),'topic'].nunique()),
            'trend_periods':len(trends),'classifier_metrics':bundle['metrics']}
    (ROOT/'reports/model_smoke_test.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__': main()
