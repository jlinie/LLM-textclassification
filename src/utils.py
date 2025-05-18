import pandas as pd
from pathlib import Path

def load_split(path: Path):
    df = pd.read_csv(path, sep="\t")
    return df["Text"].tolist(), df["Category"].tolist()
