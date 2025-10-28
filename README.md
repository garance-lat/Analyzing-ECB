# Analyzing-ECB
# ECB Text Analytics (TF-IDF + FinBERT + Market Reactions)

# Purpose / Goal
# Extend a text-analysis methodology applied to the Fed to the ECB, using press releases, press conferences, Q&A, interviews.
# Build a hybrid indicator combining FinBERT (financial tone) and TF-IDF (lexical weighting) and evaluate its link with market reactions (EUR/USD, equities, rates).

ANalyzing-ECB/

  ├─ecb_scraper/
  │  ├─ 
  │  ├─ 
  │  ├─ 
  │  ├─ 
  ├─ src/
  │  ├─ run_pipeline.py         # End-to-end CLI
  │  ├─ data_merge.py           # CSV merge (speeches + pressers)
  │  ├─ text_preprocess.py      # Text cleaning
  │  ├─ tfidf_analysis.py       # TF-IDF + top terms
  │  ├─ finbert_sentiment.py    # FinBERT sentiment (lightweight fallback included)
  │  ├─ hybrid_indicator.py     # Combine FinBERT + TF-IDF (hawk vs dove)
  │  ├─ market_analysis.py      # Event study + regressions
  │  ├─ report.py               # Figures (matplotlib) + HTML report
  │  └─ config.py               # Paths and constants
  ├─ tests/
  │  ├─ test_data_merge.py
  │  ├─ test_tfidf.py
  │  ├─ test_finbert.py
  │  └─ test_market_analysis.py
  ├─ data/
  │  └─ hawk_dove_lexicon.json  # Minimal hawkish/dovish lexicon
  ├─ sample_data/
  │  ├─ speeches_sample.csv
  │  └─ pressers_sample.csv
  ├─ requirements.txt
  ├─ pyproject.toml
  └─ README.md
