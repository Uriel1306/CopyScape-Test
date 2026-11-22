"""Safe CSV loading shared by training and inference (never drops rows)."""
import csv
import sys

import pandas as pd

TEXT_COLS = ("source_text", "candidate_text")


def load_pairs(path: str) -> pd.DataFrame:
    """Read a pairs CSV, keeping every row and order; missing text becomes ''."""
    csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
    df = pd.read_csv(path, keep_default_na=False, dtype={"pair_id": str, "source_id": str},
                     encoding="utf-8", encoding_errors="replace")
    for col in TEXT_COLS:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    return df
