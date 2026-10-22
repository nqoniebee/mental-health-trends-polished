"""One cleaning path for preparation, training and interactive inference."""
from pathlib import Path
import hashlib
import pandas as pd
from src.preprocessing import clean_text, filter_english
ROOT = Path(__file__).resolve().parents[1]


def normalize_posts(frame, english=False):
    aliases = {'text': ['text','body','selftext','content','post'],
               'date': ['date','timestamp','created_utc','social_timestamp','created'],
               'subreddit': ['subreddit','category','community']}
    cols = {str(c).lower(): c for c in frame.columns}
    mapping = {target: next((cols[a] for a in options if a in cols), None)
               for target, options in aliases.items()}
    if not mapping['text'] or not mapping['date']:
        raise ValueError('CSV needs text and date columns (timestamp or created_utc also accepted).')
    out = pd.DataFrame({'text': frame[mapping['text']].map(clean_text)})
    dates = frame[mapping['date']]
    out['date'] = pd.to_datetime(dates, unit='s', utc=True, errors='coerce') if pd.api.types.is_numeric_dtype(dates) else pd.to_datetime(dates, utc=True, errors='coerce', format='mixed')
    out['subreddit'] = frame[mapping['subreddit']].fillna('unknown').astype(str) if mapping['subreddit'] else 'unknown'
    out['source'] = 'uploaded'  # Only approved fields are retained; author and identifiers are excluded.
    audit = {'input_rows': len(out)}
    valid = out.text.str.len().gt(5) & out.date.notna()
    audit['invalid_rows_removed'] = int((~valid).sum())
    out = out.loc[valid].copy()
    out['_key'] = out.text.str.casefold()
    before = len(out)
    out = out.drop_duplicates('_key').drop(columns='_key')
    audit['duplicate_text_rows_removed'] = before-len(out)
    before = len(out)
    if english:
        out = filter_english(out)
    audit['language_rows_removed'] = before-len(out)
    out = out.sort_values('date').reset_index(drop=True)
    out['id'] = [f'post_{i:06}' for i in range(len(out))]
    audit['usable_rows'] = len(out)
    return out, audit


def training_data():
    """Keep supplied partitions; remove train/test text AND post-group overlap."""
    train = pd.read_csv(ROOT/'data/raw/dreaddit-train.csv')
    test = pd.read_csv(ROOT/'data/raw/dreaddit-test.csv')
    audit = {'raw_train': len(train), 'raw_test': len(test)}
    for frame in (train, test):
        if not frame.label.isin([0,1]).all():
            raise ValueError('Labels must be 0 or 1.')
        frame['text'] = frame.text.map(clean_text)
        frame['_key'] = frame.text.str.casefold()
    cleaned = []
    for name, frame in [('train',train),('test',test)]:
        frame = frame[frame.text.str.len().gt(5)].copy()
        conflict = frame.groupby('_key').label.nunique()
        bad = set(conflict[conflict.gt(1)].index)
        audit[name+'_conflicting_texts'] = len(bad)
        before = len(frame)
        frame = frame[~frame._key.isin(bad)].drop_duplicates('_key')
        audit[name+'_invalid_or_duplicate_rows_removed'] = before-len(frame)
        cleaned.append(frame)
    train,test = cleaned
    before = len(train)
    train = train[~train._key.isin(test._key) & ~train.post_id.isin(test.post_id)].copy()
    audit['train_overlap_rows_removed'] = before-len(train)
    if train.label.nunique()!=2 or test.label.nunique()!=2:
        raise ValueError('Both classes are needed in training and test data.')
    audit.update(clean_train=len(train),clean_test=len(test))
    audit['sha256'] = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data/raw').glob('dreaddit-*.csv')}
    return train.reset_index(drop=True), test.reset_index(drop=True), audit
