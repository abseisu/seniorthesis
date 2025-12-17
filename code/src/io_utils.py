from __future__ import annotations
from pathlib import Path
import pandas as pd

def save_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path

def load_parquet(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)
