import unittest
import numpy as np
import pandas as pd
from src.data import normalize_posts,training_data
from src.trends import aggregate
from src.models import tfidf_change

class PipelineTests(unittest.TestCase):
    def test_cleaning(self):
        raw=pd.DataFrame({'text':['hello @person https://example.com world','hello world',None,'tiny','another meaningful sentence'],
                          'date':['2024-01-01']*4+['bad'],'author':['secret']*5})
        clean,audit=normalize_posts(raw)
        self.assertEqual(len(clean),1)
        self.assertNotIn('author',clean)
        self.assertEqual(clean.text.iloc[0],'hello world')
        self.assertEqual(audit['duplicate_text_rows_removed'],1)
    def test_heldout_integrity(self):
        train,test,_=training_data()
        self.assertFalse(set(train._key)&set(test._key))
        self.assertFalse(set(train.post_id)&set(test.post_id))
        self.assertEqual(train.label.nunique(),2)
    def test_current_period_not_in_baseline(self):
        frame=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=9,tz='UTC'),
                            'sentiment_label':['positive']*7+['negative','positive']})
        result=aggregate(frame,window=7,minimum=1)
        self.assertEqual(result.baseline.iloc[7],0)
        self.assertTrue(result.alert.iloc[7])
        self.assertTrue(np.isinf(result.z_score.iloc[7]))
    def test_no_posts_is_not_zero_signal(self):
        frame=pd.DataFrame({'date':pd.to_datetime(['2024-01-01','2024-01-03'],utc=True),'sentiment_label':['negative','positive']})
        result=aggregate(frame,window=2,minimum=1)
        self.assertTrue(pd.isna(result.share.iloc[1]));self.assertFalse(result.alert.iloc[1])
    def test_short_history_and_stable_signal(self):
        frame=pd.DataFrame({'date':pd.date_range('2024-01-01',periods=9,tz='UTC'),'sentiment_label':['positive']*9})
        result=aggregate(frame,window=7,minimum=1)
        self.assertFalse(result.alert.any());self.assertEqual(result.z_score.iloc[-1],0)
    def test_empty_keyword_window(self):
        with self.assertRaises(ValueError): tfidf_change([],['some words'])
    def test_epoch_dates(self):
        frame,_=normalize_posts(pd.DataFrame({'body':['a meaningful sentence'],'created_utc':[1704067200]}))
        self.assertEqual(frame.date.iloc[0].year,2024)

if __name__=='__main__': unittest.main()
