# Early Detection of Mental Health Trends

A simplified Streamlit research dashboard aligned with the supplied report's Chapters 1–3. It analyses English social-media **language**, not a person's clinical condition.

## Start on Windows

1. Extract the entire ZIP into a normal folder, for example `Documents\Mental_Health_Trends`. Do not run from inside the ZIP.
2. Install Python **3.11 or 3.12**, including the Windows Python launcher.
3. Double-click **SETUP_WINDOWS.bat** once. Keep an internet connection available. It creates `.venv`, installs packages, downloads three pretrained models, prepares data and trains the classifier if it is missing. Initial downloads are substantial and can take several minutes. Let the window finish.
4. Double-click **RUN_WINDOWS.bat**. Open the localhost address it displays if the browser does not open automatically.
5. For later sessions, use **RUN_WINDOWS.bat** again. You do not need to recreate the virtual environment. Close the console or press Ctrl+C to stop.

The application computes locally. The first setup downloads models from Hugging Face; posts are not submitted to a remote inference API. Cached models can be reused; do not delete the model cache if you want to work offline. The ZIP does not contain the large pretrained transformer weights.

## Manual setup / PyCharm / Linux / macOS

Open this folder as the PyCharm project and select `.venv` as its interpreter. In a terminal in this folder:

```text
python -m venv .venv
```

Windows activation: `.venv\Scripts\activate`. Linux/macOS activation: `source .venv/bin/activate`.

```text
python -m pip install -r requirements.txt
python setup_project.py
python -m streamlit run app.py
```

If your terminal uses `python3`, substitute it when creating the environment. After activating, use `python`. Use `python setup_project.py --retrain` after replacing the labelled training files or changing the embedding model. Never load untrusted Joblib files.

## A simple demonstration

1. **Data overview:** inspect the synthetic sample, or choose **Supplied Dreaddit benchmark** and click **Load and clean data**. Only 28 unique texts remain in the synthetic sample after cleaning. It is not a meaningful temporal research dataset.
2. **Text analysis:** choose communities and dates, then **Run analysis**. Start with 100–600 posts for responsiveness. Use **One text example** to demonstrate the three model outputs.
3. **Topics & keywords:** discover BERTopic themes, extract KeyBERT phrases, compare TF-IDF vocabulary between two time windows, or make a word cloud.
4. **Trends & alerts:** select daily or weekly aggregation. A period needs enough posts and a complete history window before an alert can be calculated. Missing history does not mean absence of distress.
5. **Model evaluation:** review held-out metrics, the majority baseline, confusion matrix, ROC/precision–recall curves and the training audit. You can upload human-labelled text to assess sentiment/emotion agreement.

## CSV input

Minimum columns: `text,date`. Optional: `subreddit`. Recognised aliases include `body`, `selftext`, `timestamp`, `created_utc` and `social_timestamp`. Numeric dates mean Unix seconds; text dates should use ISO format. Times are normalized to UTC.

```csv
text,date,subreddit
"Work has been demanding and I am finding it hard to rest",2026-10-01T09:00:00Z,work
"The weekend walk helped me feel refreshed",2026-10-02T10:00:00Z,wellbeing
```

Do not include names or unnecessary identifiers. Cleaning removes URLs and user mentions, but cannot guarantee anonymity. No automatic scraping or account access is included. Use only data you are authorised to analyse. Uploads and analytical outputs remain in the running session unless you download them.

## Methods

| Component | Implementation |
|---|---|
| Stress classification | Frozen `all-MiniLM-L6-v2` embeddings + balanced logistic regression, C=1, max_iter=1000, threshold=0.5 |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` |
| Emotion | `j-hartmann/emotion-english-distilroberta-base` |
| Topics | BERTopic with seeded UMAP and HDBSCAN |
| Keywords | KeyBERT per document, aggregated by document count and mean similarity |
| Vocabulary change | Mean TF-IDF difference between the two date windows |
| Alerts | Absolute z-score using only previous calendar periods; minimum post count required |
| Evaluation | Held-out accuracy, precision, recall, F1, ROC-AUC, baseline and confusion matrix |

The optional `python benchmark.py` runs the four TF-IDF classifiers discussed in Chapter 4 on the supplied benchmark data. It does not replace the report-methodology classifier. Its separate measured output is `reports/tfidf_comparison.csv`.

## Project layout

- `app.py`: the single application entry point and five simple sections.
- `src/data.py`: cleaning, dates, schema checks and leakage controls.
- `src/models.py`: sentiment, emotion, topics and keywords.
- `src/classifier.py`: stress training, prediction and persistence.
- `src/trends.py`: aggregation and alerts.
- `assets/background.png`: your original background, displayed behind the interface.
- `data/raw/`: the supplied benchmark and synthetic examples.
- `models/`: trained stress-classifier bundle, when training has completed.
- `reports/`: measured outputs and validation evidence.
- `tests/`: data, alert and navigation checks.
- `REPORT_ALIGNMENT.md`: implementation mapping and contradictions in the submitted report.

## Troubleshooting

- **Missing module:** run SETUP_WINDOWS.bat, then use RUN_WINDOWS.bat so both use the same interpreter.
- **Download error:** check the connection and retry setup. Completed cached downloads are reused.
- **No model file:** setup must finish; alternatively run `python train.py` after dependencies and MiniLM are available.
- **Incompatible saved model after package changes:** run `python setup_project.py --retrain` in the current environment.
- **Too few posts or history:** analyse more timestamped data, select a wider interval or reduce the minimum count/window for exploration. Do not manufacture dates or counts.
- **No topics:** some corpora form only one cluster or all outliers. Try more varied texts and a smaller minimum topic size.
- **Memory pressure:** close other applications and reduce the selected post limit. CPU operation is supported but slower than a suitable GPU.
- **Offline operation:** after successful online setup, models can use cached weights. Set `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in the terminal before launch to prevent update checks. Missing cached weights still require online setup.

## Validation

Run `python -m unittest discover -s tests -v`. Read `VALIDATION.md` for what was actually checked in the delivery environment. The models have not been clinically or locally validated. Sampling, language, platform bias and duplicate-removal choices affect findings. A stress-language probability is not a calibrated health-risk probability.
