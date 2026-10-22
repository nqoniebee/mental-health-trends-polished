"""One-time model download, preparation and training. Re-run safely after a failed download."""
import argparse
import json
from src.data import ROOT, normalize_posts
from src import models, classifier
import pandas as pd


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--retrain',action='store_true');args=parser.parse_args()
    print('1/4 Downloading/loading MiniLM sentence encoder...',flush=True);models.embedder()
    print('2/4 Downloading/loading sentiment and emotion models...',flush=True)
    models.pipeline_for('sentiment');models.pipeline_for('emotion')
    print('3/4 Preparing demonstration data...',flush=True)
    frame,audit=normalize_posts(pd.read_csv(ROOT/'data/raw/sample_reddit_mental_health.csv'))
    frame['source']='synthetic_demo'
    (ROOT/'data/processed').mkdir(exist_ok=True)
    frame.to_parquet(ROOT/'data/processed/posts.parquet',index=False)
    (ROOT/'reports/preparation.json').write_text(json.dumps(audit,indent=2))
    print('4/4 Training/loading report stress classifier...',flush=True)
    bundle=classifier.train() if args.retrain or not classifier.MODEL_PATH.exists() else classifier.load()
    print(json.dumps(bundle['metrics'],indent=2))
    print('Setup complete. Run RUN_WINDOWS.bat or python -m streamlit run app.py.',flush=True)

if __name__=='__main__':
    main()
