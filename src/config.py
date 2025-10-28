from dataclasses import dataclass
from pathlib import Path

@dataclass
class Paths:
    speeches_csv: Path
    pressers_csv: Path
    outdir: Path

DEFAULT_TFIDF_MAX_FEATURES = 20000
DEFAULT_TFIDF_MIN_DF = 2
DEFAULT_TFIDF_NGRAM = (1, 2)
DEFAULT_FINBERT_MODEL = "yiyanghkust/finbert-tone"  # labels: Positive/Neutral/Negative
