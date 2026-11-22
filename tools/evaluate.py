#!/usr/bin/env python3
"""Score a predictions file against a labelled file.

    python tools/evaluate.py --pred preds.csv --truth data/train.csv

The metric is F1 of the positive class (label == 1, "reused").
`--truth` needs columns pair_id,label (data/train.csv works).  Remember that a
score on data your model was fitted on tells you very little.
"""
import argparse
import csv
import sys

csv.field_size_limit(10**9)


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--truth", required=True)
    a = ap.parse_args()

    truth = read(a.truth)
    pred_rows = read(a.pred)
    problems = []
    if pred_rows and not {"pair_id", "score", "label"} <= set(pred_rows[0]):
        sys.exit(f"FORMAT ERROR: {a.pred} needs columns pair_id,score,label (found {list(pred_rows[0])})")
    pred = {}
    for r in pred_rows:
        try:
            score, label = float(r["score"]), int(float(r["label"]))
            assert 0.0 <= score <= 1.0 and label in (0, 1)
        except Exception:
            problems.append(f"bad row for {r.get('pair_id')!r}: score={r.get('score')!r} label={r.get('label')!r}")
            continue
        if r["pair_id"] in pred:
            problems.append(f"duplicate pair_id {r['pair_id']}")
        pred[r["pair_id"]] = label
    ids = {r["pair_id"] for r in truth}
    missing = [i for i in ids if i not in pred]
    extra = [i for i in pred if i not in ids]
    if missing:
        problems.append(f"{len(missing)} pair_ids have no valid prediction (counted as label 0), e.g. {sorted(missing)[:3]}")
    if extra:
        problems.append(f"{len(extra)} unknown pair_ids in predictions (ignored)")

    tp = fp = fn = tn = 0
    for r in truth:
        y, yhat = int(float(r["label"])), pred.get(r["pair_id"], 0)
        tp += y and yhat; fp += (not y) and yhat; fn += y and (not yhat); tn += (not y) and (not yhat)
    p, r_, f1 = prf(tp, fp, fn)
    print(f"pairs scored: {tp + fp + fn + tn}   positives: {tp + fn}")
    print(f"precision {p:.3f}   recall {r_:.3f}   F1 {f1:.3f}      (TP {tp}  FP {fp}  FN {fn}  TN {tn})")
    if problems:
        print("\nPROBLEMS:")
        for m in problems[:15]:
            print("  -", m)
    return f1


if __name__ == "__main__":
    main()
