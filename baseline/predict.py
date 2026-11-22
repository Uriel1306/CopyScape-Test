"""Score (source, candidate) pairs with the trained baseline.

Usage:
    python baseline/predict.py --input data/sample_input.csv --output preds.csv --model baseline/model.pkl
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from features import clean, containment, jaccard, shingles


def build_features(df: pd.DataFrame, model: dict):
    """Rebuild the same feature matrix that train.py used."""
    common = model["common"]
    vectorizer = model["vectorizer"]
    rows = []
    for source_text, candidate_text in zip(df.source_text, df.candidate_text):
        s = shingles(source_text) - common
        c = shingles(candidate_text) - common
        len_s = len(clean(source_text).split())
        len_c = len(clean(candidate_text).split())
        rows.append([
            jaccard(s, c),
            containment(s, c),
            min(len_s, len_c) / max(len_s, len_c, 1),
            float(np.log1p(abs(len_s - len_c))),
        ])
    tfidf_s = vectorizer.transform(df.source_text)
    tfidf_c = vectorizer.transform(df.candidate_text)
    cosine = np.asarray(tfidf_s.multiply(tfidf_c).sum(axis=1))
    return hstack([csr_matrix(np.array(rows)), csr_matrix(cosine), tfidf_s, tfidf_c]).tocsr()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="baseline/model.pkl")
    args = parser.parse_args()

    with open(args.model, "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv(args.input)
    df = df.dropna(subset=["source_text", "candidate_text"]).reset_index(drop=True)

    scores = model["clf"].predict_proba(build_features(df, model))[:, 1]
    out = pd.DataFrame({
        "pair_id": df.pair_id,
        "score": np.round(scores, 4),
        "label": (scores >= model["threshold"]).astype(int),
    })
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
