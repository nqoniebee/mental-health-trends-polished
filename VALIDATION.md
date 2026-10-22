# Delivery validation — 21 October 2026

## Executed successfully

- Installed the declared runtime dependencies in Python 3.12; exact direct versions are listed in requirements-tested.txt. The main requirements file pins scikit-learn to the version that wrote the included classifier.
- Downloaded and loaded all three actual pretrained models: MiniLM, CardiffNLP sentiment and Hartmann emotion.
- Trained the report-methodology classifier on **2,814 cleaned training records** and evaluated on **715 held-out records**. Training/test text and post-ID overlap were excluded. The classifier is included in models/stress_classifier.joblib.
- Accuracy **0.7622**, precision **0.7653**, recall **0.7778**, F1 **0.7715**, ROC-AUC **0.8499**. Majority baseline accuracy **0.5161**. Confusion matrix: TN=258, FP=88, FN=82, TP=287.
- Ran actual sentiment, emotion and stress inference on all **28 unique synthetic example texts**; checked output cardinality and probability range.
- Ran actual KeyBERT extraction: 15 aggregate phrases returned from a five-document smoke sample.
- Ran actual BERTopic on 28 unique examples: 28 assignments and two non-outlier clusters at minimum size 2. This smoke result is not a validated thematic research finding.
- Ran the temporal aggregation/alert path on the inferred outputs.
- Ran the optional four-model TF-IDF comparison on the cleaned benchmark. See reports/tfidf_comparison.csv.
- **Nine automated checks passed:** cleaning and duplicate handling; timestamp conversion; train/test independence; current-period exclusion from the baseline; missing-calendar-period handling; insufficient-history and constant-signal handling; empty keyword windows; all five Streamlit sections; loading the full benchmark through the UI.
- Python files passed compilation. Streamlit server startup succeeded.

Evidence is in reports/evaluation.json, reports/model_smoke_test.json, reports/test_results.txt and reports/tfidf_comparison.csv. No measured scores were invented or copied from the report's historical sample results.

## Limits

- Windows batch scripts were reviewed but cannot be executed on this Linux validation host.
- A browser screenshot could not be obtained because the browser binary download failed. Navigation was exercised with Streamlit AppTest; this does not establish pixel-level layout verification.
- The ZIP contains the trained logistic-regression head, data and source code. Large pretrained weights stay in the model cache and are **not** included. First setup on another computer needs internet to download them.
- This is benchmark and functional validation, not clinical validation or external validation on Zimbabwean data.
- Human-label agreement tools are implemented, but no independent sentiment/emotion validation labels were supplied. No such agreement scores are claimed.
- BERTopic silhouette diagnostics depend on the corpus and require two or more suitable clusters. Tiny repetitive samples do not support substantive trend conclusions.
