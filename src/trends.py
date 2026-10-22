"""Calendar-bin trends; baseline excludes the current observation."""
import numpy as np
import pandas as pd
FREQUENCIES={'Hourly':'h','Daily':'D','Weekly':'W-MON','Monthly':'MS','Yearly':'YS'}


def aggregate(frame,frequency='Daily',signal='negative',window=7,minimum=3,threshold=2.0):
    d=frame.copy()
    d['date']=pd.to_datetime(d.date,utc=True,errors='coerce')
    if d.date.isna().any(): raise ValueError('Every post needs a valid date.')
    if signal=='negative': d['value']=d.sentiment_label.eq('negative').astype(float)
    else: d['value']=d.stress_label.astype(float)
    result=d.set_index('date').value.resample(FREQUENCIES[frequency]).agg(['mean','count']).rename(columns={'mean':'share','count':'posts'})
    result['share']=result['share'].where(result.posts>=minimum)
    past=result['share'].shift(1)
    result['baseline']=past.rolling(window,min_periods=window).mean()
    result['std']=past.rolling(window,min_periods=window).std()
    delta=result['share']-result.baseline
    result['z_score']=delta/result['std'].replace(0,np.nan)
    flat=result['std'].eq(0)&delta.notna()
    result.loc[flat&delta.eq(0),'z_score']=0.0
    result.loc[flat&delta.ne(0),'z_score']=np.sign(delta[flat&delta.ne(0)])*np.inf
    result['alert']=result.z_score.abs().ge(threshold)&result.posts.ge(minimum)
    result['status']=np.select([result.posts.lt(minimum),result.baseline.isna(),result.alert&delta.gt(0),result.alert&delta.lt(0)],['Too few posts','Need more history','Unusual increase','Unusual decrease'],default='Within baseline')
    return result.reset_index()
