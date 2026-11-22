"""Train the reuse detector (grouped CV, threshold on out-of-fold probabilities).

Usage: python train.py [--data data/train.csv] [--model model.joblib]
                       [--groups overlap,idf,hits,length[,masked,char]] [--min-sources 10]
"""
import argparse
import os
import sys
from typing import Callable, Dict, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
import detector as D  # noqa: E402
from data_io import load_pairs  # noqa: E402

N_SPLITS = 5
SEEDS = (0, 1, 2)
MIN_SOURCES = 10      # train.csv: no 3-gram occurs in 10..17 articles, so 5..17 are equivalent
THRESHOLDS = np.round(np.arange(0.05, 0.9501, 0.01), 2)
SMOOTH = 5            # moving-average width (in threshold steps) used to prefer flat F1 regions
TOLERANCE = 0.002     # prefer the simpler model unless a richer one is better by more than this

MODELS: Dict[str, Callable[[], object]] = {
    "lr": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
    "hgb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                  l2_regularization=1.0, random_state=0),
}
COMPLEXITY = ["lr", "hgb", "ens"]


def source_groups(df: pd.DataFrame, idx: np.ndarray) -> Dict[str, List[str]]:
    """source_id -> distinct source texts of that article (candidates are never used)."""
    groups: Dict[str, List[str]] = {}
    sub = df.iloc[idx]
    for sid, text in zip(sub.source_id, sub.source_text):
        texts = groups.setdefault(sid, [])
        if text not in texts:
            texts.append(text)
    return groups


def feature_matrix(df: pd.DataFrame, res: D.Resources, groups: Sequence[str]) -> np.ndarray:
    return np.vstack([D.extract_features(s, c, res, groups)
                      for s, c in zip(df.source_text, df.candidate_text)])


def f1_curve(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    return np.array([f1_score(y, prob >= t, pos_label=1, zero_division=0) for t in THRESHOLDS])


def pick_threshold(y: np.ndarray, prob: np.ndarray) -> Tuple[float, float]:
    """Threshold at the peak of the smoothed F1 curve (prefers flat regions); returns (t, F1 at t)."""
    curve = f1_curve(y, prob)
    smooth = np.convolve(np.pad(curve, SMOOTH // 2, mode="edge"), np.ones(SMOOTH) / SMOOTH, mode="valid")
    i = int(np.argmax(smooth))
    return float(THRESHOLDS[i]), float(curve[i])


def fit_predict(models: Sequence[str], X_tr, y_tr, X_va) -> np.ndarray:
    probs = [MODELS[m]().fit(X_tr, y_tr).predict_proba(X_va)[:, 1] for m in models]
    return np.mean(probs, axis=0)


def oof_for_seed(df, groups, min_sources, seed) -> Dict[str, np.ndarray]:
    """Out-of-fold probabilities per model name for one StratifiedGroupKFold seed."""
    y = df.label.values
    oof = {name: np.zeros(len(df)) for name in COMPLEXITY}
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for tr, va in cv.split(df, y, df.source_id):
        res = D.learn_resources(source_groups(df, tr), min_sources)   # train articles only
        X = feature_matrix(df, res, groups)
        for name in MODELS:
            oof[name][va] = fit_predict([name], X[tr], y[tr], X[va])
        oof["ens"][va] = fit_predict(list(MODELS), X[tr], y[tr], X[va])
    return oof


def evaluate(df, groups, min_sources, seeds=SEEDS) -> Dict[str, dict]:
    y = df.label.values
    per_seed = [oof_for_seed(df, groups, min_sources, s) for s in seeds]
    out = {}
    for name in COMPLEXITY:
        pooled_t, _ = pick_threshold(np.tile(y, len(seeds)), np.concatenate([o[name] for o in per_seed]))
        f1s = [f1_score(y, o[name] >= pooled_t, pos_label=1) for o in per_seed]
        out[name] = {"threshold": pooled_t, "f1_mean": float(np.mean(f1s)), "f1_std": float(np.std(f1s)),
                     "oof": per_seed[0][name]}
    return out


def choose_model(results: Dict[str, dict]) -> str:
    best = max(r["f1_mean"] for r in results.values())
    return next(n for n in COMPLEXITY if results[n]["f1_mean"] >= best - TOLERANCE)


def fit_final(df, groups, min_sources, name: str, threshold: float) -> dict:
    res = D.learn_resources(source_groups(df, np.arange(len(df))), min_sources)
    X, y = feature_matrix(df, res, groups), df.label.values
    members = list(MODELS) if name == "ens" else [name]
    return {"classifiers": [MODELS[m]().fit(X, y) for m in members], "threshold": threshold,
            "resources": res._asdict(), "feature_names": D.feature_names(groups),
            "config": {"model": name, "groups": list(groups), "min_sources": min_sources}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "train.csv"))
    ap.add_argument("--model", default=os.path.join(ROOT, "model.joblib"))
    ap.add_argument("--groups", default=",".join(D.BASE_GROUPS))
    ap.add_argument("--min-sources", type=int, default=MIN_SOURCES)
    ap.add_argument("--oof-out", default=os.path.join(ROOT, "notes", "oof_predictions.csv"))
    args = ap.parse_args()
    groups = tuple(args.groups.split(","))
    df = load_pairs(args.data)
    print(f"{len(df)} rows, {df.label.mean():.1%} positive, {df.source_id.nunique()} articles; groups={groups}")
    results = evaluate(df, groups, args.min_sources)
    for n, r in results.items():
        print(f"  {n:4s} grouped-CV F1 {r['f1_mean']:.4f} +- {r['f1_std']:.4f}  threshold {r['threshold']:.2f}")
    name = choose_model(results)
    print(f"Selected: {name}")
    pd.DataFrame({"pair_id": df.pair_id, "score": np.round(results[name]["oof"], 6),
                  "label": (results[name]["oof"] >= results[name]["threshold"]).astype(int)}
                 ).to_csv(args.oof_out, index=False)
    joblib.dump(fit_final(df, groups, args.min_sources, name, results[name]["threshold"]), args.model)
    print(f"Saved {args.model}; out-of-fold predictions (seed 0) in {args.oof_out}")


if __name__ == "__main__":
    main()
