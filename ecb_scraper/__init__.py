
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_ecb"

def load_all_speeches_csv():
    path = DATA_DIR / "all_ECB_conferences.csv"
    return pd.read_csv(path, sep="//", engine="python", encoding="utf-8-sig")
