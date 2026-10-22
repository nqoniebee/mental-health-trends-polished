# Alignment with the supplied Final Project Doc.docx

The report contains incompatible descriptions. This delivery uses **Chapters 1–3 as the main implementation specification**, while providing a separate TF-IDF comparison for the models in Chapter 4. It does not attempt to force the new evaluation to match historical sample-data results.

| Report requirement | Delivered implementation |
|---|---|
| Multi-page Streamlit workflow | Five sidebar sections served by one app.py, reducing duplicate pages and entry points |
| Common cleaned post schema | data.py retains id, text, date, subreddit and source; optional English filtering |
| Parquet storage | setup_project.py writes data/processed/posts.parquet |
| Sentence-BERT + logistic regression | src/classifier.py, the main classifier; no SVM substituted silently |
| Sentiment and emotion transformers | src/models.py uses the report's named pretrained models |
| BERTopic, UMAP and HDBSCAN | Topics section with seed 42 and adjustable cluster minimum |
| KeyBERT and TF-IDF delta | Keyword extraction and before/after date comparison |
| Word clouds | Word-cloud action in Topics & keywords |
| Daily/weekly trend monitoring | Trends section; hourly/monthly/yearly also available |
| Negative-share rolling anomaly detection | Default negative-sentiment signal, prior-period baseline, absolute z-score |
| Stress performance and majority baseline | Model evaluation page and reports/evaluation.json |
| Pretrained model spot-checks | CSV upload with sentiment_gold/emotion_gold, accuracy and Cohen's kappa |
| Topic separation diagnostic | Cosine silhouette on non-outlier sentence embeddings when at least two clusters exist |
| Background image | Original assets/background.png embedded directly into app CSS |
| Four Chapter 4 classifiers | Optional benchmark.py: TF-IDF LR, Naive Bayes, Linear SVM and Random Forest |

## Report text that should be reconciled before submission

1. **Method mismatch:** the abstract and Chapter 3 describe MiniLM embeddings + logistic regression, whereas Chapter 4 describes TF-IDF classifiers. Present the latter as supplementary comparisons, or rewrite the results chapter around the measured main-model outputs.
2. **Perfect scores:** Chapter 4's synthetic samples contain only 28 distinct texts. The new pipeline does not reuse its 100% scores or treat them as evidence of real-world accuracy. Populate the results from fresh `reports/evaluation.json` and the separate comparison CSV.
3. **Dataset status:** the uploaded archive actually contains `dreaddit-train.csv` and `dreaddit-test.csv` as well as synthetic samples. The report's statements that the archive only supports future real-data acquisition are outdated. Dataset source, permitted use and provenance still need to be documented by the researchers.
4. **Statistical inference:** Chapter 3 says chi-square testing is not implemented; Chapter 4 contains a chi-square result. The core app remains descriptive and anomaly-oriented. Historical weak-flag associations must not be attributed to transformer inference or causal findings.
5. **Deduplication:** deleting repeated text collapses the synthetic corpus to 28 posts and destroys its intended repeated-template temporal pattern. The app reports this directly. Use an appropriately sourced, genuinely varied temporal corpus for substantive trend findings.
6. **Objective numbering:** some parts refer to five objectives and others to eight. Reconcile this in the report rather than adding unrelated modules.
7. **Scope:** the interface reports aggregate language signals and stress/non-stress text classification, not individual Low/Moderate/High clinical risk categories, diagnoses or forecasts.

## Deliberate implementation improvements

- The labelled-data loader removes within-split duplicate text and conflicting labels, then excludes training records sharing cleaned text or post IDs with the test partition. Model training sees no test labels for tuning. Test data are used only for fixed evaluation and overlap exclusion.
- All stress-model inputs use identical cleaning at train and prediction time.
- Current-period observations are excluded from their rolling baseline to avoid diluting spikes. Low-count periods are gaps rather than zeros.
- Default text previews are hidden. No author columns are retained in uploaded analytical data.
- Neural models are loaded only when requested and failures produce setup guidance. There is no silent fallback to heuristics or synthetic predictions.
- KeyBERT is applied per document so a giant joined corpus is not silently truncated to one encoder input.
- No user-written report paragraphs have been changed. This file records the decisions needed to make the code and report defensible together.
