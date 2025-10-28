import pandas as pd
from pathlib import Path
from src.data_merge import merge_sources

def test_merge_sources(tmp_path: Path):
    sp = tmp_path / "all_ECB_speeches.csv"
    pr = tmp_path / "ecb_conferences_1998_2025.csv"
    sp.write_text("date,title,link,text\n2020-01-10,a,la,hello\n")
    pr.write_text("date,title,link,text\n2020-01-11,b,lb,world\n")
    df = merge_sources(sp, pr)
    assert set(df.columns) >= {"date","title","link","text","source","doc_id"}
    assert len(df) == 2
    assert set(df["source"]) == {"speech","presser"}
