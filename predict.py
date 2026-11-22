"""Score (source, candidate) pairs: one output row per input row, in order, never crashing.

Usage: python predict.py --input X.csv --output Y.csv [--model model.joblib]
"""
import argparse
import logging
import os
import sys
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import detector as D  # noqa: E402
from data_io import load_pairs  # noqa: E402

log = logging.getLogger("predict")


def row_features(src, cand, res: D.Resources, groups) -> Optional[np.ndarray]:
    """Features for one pair, or None if the row cannot be processed (-> score 0)."""
    try:
        return D.extract_features(src, cand, res, groups)
    except Exception as exc:  # untrusted text must never take the run down
        log.warning("row failed (%s: %s); scoring 0.0", type(exc).__name__, exc)
        return None


def positive_scores(model: dict, X: np.ndarray) -> np.ndarray:
    return np.mean([clf.predict_proba(X)[:, 1] for clf in model["classifiers"]], axis=0)


def predict_frame(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    """Return DataFrame(pair_id, score, label) with exactly len(df) rows in input order."""
    res = D.Resources(**model["resources"])
    groups = model["config"]["groups"]
    feats = [row_features(s, c, res, groups) for s, c in zip(df.source_text, df.candidate_text)]
    ok = [i for i, f in enumerate(feats) if f is not None and f.any()]
    scores = np.zeros(len(df))
    if ok:
        try:
            scores[ok] = positive_scores(model, np.vstack([feats[i] for i in ok]))
        except Exception as exc:
            log.warning("batch scoring failed (%s); falling back to per-row", exc)
            for i in ok:
                scores[i] = _safe_score(model, feats[i])
    scores = np.clip(np.round(np.nan_to_num(scores), 6), 0.0, 1.0)
    return pd.DataFrame({"pair_id": df.pair_id.values, "score": scores,
                         "label": (scores >= model["threshold"]).astype(int)})


def _safe_score(model: dict, x: np.ndarray) -> float:
    try:
        return float(positive_scores(model, x.reshape(1, -1))[0])
    except Exception as exc:
        log.warning("row scoring failed (%s); scoring 0.0", exc)
        return 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default=os.path.join(ROOT, "model.joblib"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    df = load_pairs(args.input)
    if "pair_id" not in df.columns:
        df["pair_id"] = [str(i) for i in range(len(df))]
    out = predict_frame(df, joblib.load(args.model))
    out.to_csv(args.output, index=False, float_format="%.6f", lineterminator="\n")
    print(f"Wrote {len(out)} predictions to {args.output}")


if __name__ == "__main__":
    main()
