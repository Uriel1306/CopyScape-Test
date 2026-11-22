"""Part A: verify every suspected problem in the baseline, with numbers.

Run from anywhere:   python -X utf8 analysis/partA_checks.py

Each check prints its evidence and a VERDICT line. Numbers are computed two
independent ways where possible ("cross-check"). Needs numpy, pandas, scipy, scikit-learn. It runs the baseline's own train.py and
predict.py into a temporary folder (nothing in the project is modified).
Method tags:  RUN = executed on the data | PROXY = model-free measurement of the
mechanism | INSPECT = static analysis of the baseline source (ast / regex).
"""
import ast
import collections
import csv
import difflib
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import unicodedata

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "baseline"))
import features as F  # the baseline's own code, unmodified

csv.field_size_limit(10**9)
rows = list(csv.DictReader(open("data/train.csv", encoding="utf-8", newline="")))
y = np.array([int(r["label"]) for r in rows])
ids = [r["source_id"] for r in rows]
SRC = [r["source_text"] for r in rows]
CAND = [r["candidate_text"] for r in rows]
N = len(rows)

# ----------------------------------------------------------------- helpers
HOMO = {  # Cyrillic / look-alike letters -> Latin
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "і": "i", "ѕ": "s", "ј": "j",
    "һ": "h", "ԁ": "d", "ԛ": "q", "ԝ": "w", "ѵ": "v",
    # Greek and further Cyrillic look-alikes found by listing the letters left unmapped in this data
    "ο": "o", "ν": "v", "т": "t", "ε": "e", "α": "a", "τ": "t",
    "н": "h", "ι": "i", "м": "m", "β": "b", "μ": "u", "к": "k",
    "в": "b", "κ": "k",
}


def map_only(t):
    return "".join(HOMO.get(ch, ch) for ch in unicodedata.normalize("NFKC", t).lower())


def norm_space(t):          # replace (not delete) non-alphanumerics
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def norm_full(t):           # + NFKC + look-alike letters mapped to Latin
    return re.sub(r"[^a-z0-9]+", " ", map_only(t)).strip()


def sh_str(t, cleaner, n=3):
    w = cleaner(t).split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def f1_pos(yt, yp):
    tp = int(((yt == 1) & (yp == 1)).sum()); fp = int(((yt == 0) & (yp == 1)).sum()); fn = int(((yt == 1) & (yp == 0)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def f1_class(yt, yp, k):
    tp = int(((yt == k) & (yp == k)).sum()); fp = int(((yt != k) & (yp == k)).sum()); fn = int(((yt == k) & (yp != k)).sum())
    return 0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)


def weighted_f1(yt, yp):    # method A: support-weighted mean of per-class F1
    return sum((yt == k).mean() * f1_class(yt, yp, k) for k in (0, 1))


def weighted_f1_confusion(yt, yp):   # method B: straight from the confusion matrix
    tp = ((yt == 1) & (yp == 1)).sum(); tn = ((yt == 0) & (yp == 0)).sum()
    fp = ((yt == 0) & (yp == 1)).sum(); fn = ((yt == 1) & (yp == 0)).sum()
    f_1 = 0 if tp == 0 else 2 * tp / (2 * tp + fp + fn)
    f_0 = 0 if tn == 0 else 2 * tn / (2 * tn + fn + fp)
    return ((tp + fn) * f_1 + (tn + fp) * f_0) / len(yt)


def best_thr(scores, yt):
    scores = np.asarray(scores, float)
    best = (0.0, 0.0)
    for t in np.unique(scores):
        v = f1_pos(yt, (scores >= t).astype(int))
        if v > best[0]:
            best = (v, float(t))
    return best


ALL_ONES = 2 * y.mean() / (1 + y.mean())
SUMMARY = []


def head(n, title, method):
    print("=" * 74)
    print(f"CHECK {n}: {title}   [{method}]")


def verdict(n, problem, method, status, key):
    print(f"  VERDICT: {status}")
    SUMMARY.append((n, problem, method, status, key))


def xcheck(ok, what):
    print(f"  cross-check ({what}): {'OK' if ok else 'MISMATCH  <-- investigate'}")


def src_ast(path):
    return ast.parse(open(path, encoding="utf-8").read())


def ast_calls(path, names):
    out = []
    for node in ast.walk(src_ast(path)):
        if isinstance(node, ast.Call) and ast.unparse(node.func) in names:
            out.append((node.lineno, ast.unparse(node.func), [ast.unparse(a) for a in node.args],
                        {k.arg: ast.unparse(k.value) for k in node.keywords}))
    return sorted(out, key=lambda x: x[0])


def grep(path, pattern):
    return [(i + 1, line.rstrip()) for i, line in enumerate(open(path, encoding="utf-8")) if re.search(pattern, line)]


print("PART A: baseline verification   (train.csv: %d pairs, %d positive = %.1f%%; all-ones F1 = %.3f)"
      % (N, y.sum(), 100 * y.mean(), ALL_ONES))

# baseline-style shingles (hashed) computed once
S_h = [F.shingles(t) for t in SRC]
C_h = [F.shingles(t) for t in CAND]
common = F.find_common_shingles(SRC + CAND)

# setup: run the baseline's OWN train.py into a temp folder (fixed hash seed so the run is reproducible)
import tempfile
TMP = tempfile.mkdtemp(prefix="partA_")
MODEL = os.path.join(TMP, "model.pkl")
TRAIN_SEED = "7"
TRAIN_OK, BASE_VAL_F1 = False, None
try:
    p = subprocess.run([sys.executable, "-X", "utf8", "baseline/train.py", "--model", MODEL], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", cwd=ROOT, timeout=900,
                       env=dict(os.environ, PYTHONHASHSEED=TRAIN_SEED, PYTHONIOENCODING="utf-8"))
    m_ = re.search(r"Validation F1:\s*([0-9.]+)", p.stdout)
    TRAIN_OK = p.returncode == 0 and m_ is not None
    BASE_VAL_F1 = float(m_.group(1)) if m_ else None
    print(f"setup: ran the baseline's own train.py -> its reported 'Validation F1' = {BASE_VAL_F1}   (returncode {p.returncode})")
    if not TRAIN_OK:
        print(p.stderr[-800:])
except Exception as e_:
    print("setup: could not run baseline/train.py:", repr(e_))

# ======================================================================= 1
head(1, "random row split leaks: same article on both sides of train/val", "PROXY")


def split_random(seed, yv=None):
    yv = y if yv is None else yv
    rng = np.random.RandomState(seed); val = np.zeros(len(yv), bool); idx = np.arange(len(yv))
    for lab in (0, 1):
        ii = idx[yv == lab].copy(); rng.shuffle(ii); val[ii[: int(round(0.2 * len(ii)))]] = True
    return val


def split_group(seed, idv=None):
    idv = ids if idv is None else list(idv)
    rng = np.random.RandomState(seed); arts = sorted(set(idv)); rng.shuffle(arts)
    vset = set(arts[: int(round(0.2 * len(arts)))])
    return np.array([i in vset for i in idv])


SRC_SH3 = [sh_str(t, norm_full) for t in SRC]
res1 = {}
for seed in (0, 1):
    for name, fn in (("random split", split_random), ("group split ", split_group)):
        val = fn(seed)
        tr_ids = {ids[i] for i in range(N) if not val[i]}
        seen = float(np.mean([ids[i] in tr_ids for i in range(N) if val[i]]))
        rate = collections.defaultdict(list)
        for i in range(N):
            if not val[i]:
                rate[ids[i]].append(y[i])
        pred = np.array([1 if (ids[i] in rate and np.mean(rate[ids[i]]) >= 0.5) else 0 for i in range(N) if val[i]])
        f1_id = f1_pos(y[val], pred)
        df = pd.DataFrame({"id": ids, "y": y})        # method B (pandas)
        r2 = df[~val].groupby("id").y.mean()
        pred2 = (df[val].id.map(r2).fillna(0) >= 0.5).astype(int).values
        known = set()
        for i in range(N):
            if not val[i]:
                known |= SRC_SH3[i]
        cov = float(np.mean([len(SRC_SH3[i] & known) / max(len(SRC_SH3[i]), 1) for i in range(N) if val[i]]))
        res1[(name, seed)] = (seen, f1_id, cov)
        print(f"  seed {seed} {name}: val rows whose article is in train: {seen:5.1%} | "
              f"F1 of 'article id only' predictor: {f1_id:.3f} | val source 3-grams already seen in train sources: {cov:5.1%}")
        xcheck(abs(f1_pos(y[val], pred2) - f1_id) < 1e-9, "dict vs pandas")
r_f1 = np.mean([res1[("random split", s)][1] for s in (0, 1)])
g_f1 = np.mean([res1[("group split ", s)][1] for s in (0, 1)])
r_cov = np.mean([res1[("random split", s)][2] for s in (0, 1)])
g_cov = np.mean([res1[("group split ", s)][2] for s in (0, 1)])
print(f"  reference: all-ones F1 = {ALL_ONES:.3f}.  A predictor that sees ONLY the article id gets {r_f1:.3f} under a random split and {g_f1:.3f} under a group split.")
if r_f1 > ALL_ONES + 0.10 and r_f1 > g_f1 + 0.10:
    st = "CONFIRMED (leak channel exists; the size of the drop in the baseline's F1 is measured in check 2)"
elif r_cov > g_cov + 0.2:
    st = "PARTIAL (source text leaks %.0f%% vs %.0f%%, article id alone does not predict the label)" % (100 * r_cov, 100 * g_cov)
else:
    st = "NOT CONFIRMED by this proxy"
verdict(1, "random split leaks articles", "PROXY", st, f"id-only F1 {r_f1:.2f} random vs {g_f1:.2f} group; seen src 3-grams {r_cov:.0%} vs {g_cov:.0%}")

# ======================================================================= 2
head(2, "the baseline's own pipeline: random split vs whole-article split, with vs without TF-IDF columns", "RUN")
st2, key2 = "NOT RUN", "scikit-learn unavailable"
try:
    from scipy.sparse import csr_matrix, hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    dfb = pd.read_csv("data/train.csv").dropna().reset_index(drop=True)      # exactly what train.py does
    sb, cb, yb, gb = dfb.source_text, dfb.candidate_text, dfb.label.values, dfb.source_id.values
    allb = pd.concat([sb, cb])
    commonb = F.find_common_shingles(allb)
    sim = np.array([F.pair_features(a, b, commonb) for a, b in zip(sb, cb)])
    vec = TfidfVectorizer(preprocessor=F.clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit(allb)
    ts, tc = vec.transform(sb), vec.transform(cb)
    cosb = np.asarray(ts.multiply(tc).sum(axis=1))
    X_full = hstack([csr_matrix(sim), csr_matrix(cosb), ts, tc]).tocsr()
    X_sim = hstack([csr_matrix(sim), csr_matrix(cosb)]).tocsr()
    print(f"  TF-IDF vocabulary {len(vec.vocabulary_)} terms -> the baseline model sees {X_full.shape[1]} columns, only {X_sim.shape[1]} of them describe the pair's overlap")

    def run_split(X, tr, va):
        clf = LogisticRegression(C=20.0, max_iter=5000).fit(X[tr], yb[tr])
        sc = clf.predict_proba(X[va])[:, 1]
        best = (0.0, 0.5)
        for t in np.arange(0.05, 0.96, 0.01):             # same loop as train.py
            f = f1_score(yb[va], sc >= t, average="weighted")
            if f > best[0]:
                best = (f, float(t))
        pr_ = (sc >= best[1]).astype(int)
        return best[0], f1_pos(yb[va], pr_), best[1], abs(weighted_f1(yb[va], pr_) - best[0]) < 1e-9

    splits = {"baseline split (seed 42, stratified)": train_test_split(np.arange(len(yb)), test_size=0.2, random_state=42, stratify=yb)}
    for s_ in (0, 1):
        v = split_random(s_, yb); splits[f"random rows, seed {s_}"] = (np.where(~v)[0], np.where(v)[0])
    for s_ in (0, 1):
        v = split_group(s_, gb); splits[f"whole articles held out, seed {s_}"] = (np.where(~v)[0], np.where(v)[0])
    R, same_w = {}, True
    for name, (tr_i, va_i) in splits.items():
        for cfg, X in (("full (baseline)", X_full), ("overlap features only", X_sim)):
            w, c1, th, okw = run_split(X, tr_i, va_i)
            R[(name, cfg)] = c1; same_w &= okw
            print(f"  {name:38s} {cfg:22s} weighted F1 {w:.3f} | class-1 F1 {c1:.3f} | threshold {th:.2f}")
            if name.startswith("baseline") and cfg.startswith("full"):
                base_rep = w
    if BASE_VAL_F1 is not None:
        xcheck(abs(base_rep - BASE_VAL_F1) < 0.0015, f"my replication {base_rep:.3f} vs the real train.py output {BASE_VAL_F1:.3f}")
    xcheck(same_w, "sklearn weighted F1 vs my own implementation")
    a = np.mean([R[(f"random rows, seed {s_}", "full (baseline)")] for s_ in (0, 1)])
    b = np.mean([R[(f"whole articles held out, seed {s_}", "full (baseline)")] for s_ in (0, 1)])
    c = np.mean([R[(f"whole articles held out, seed {s_}", "overlap features only")] for s_ in (0, 1)])
    d = np.mean([R[(f"random rows, seed {s_}", "overlap features only")] for s_ in (0, 1)])
    print(f"  class-1 F1 of the baseline model: random rows {a:.3f} -> whole articles held out {b:.3f}  (drop {a - b:.3f})")
    print(f"  dropping the TF-IDF columns: random rows {d:.3f} | whole articles held out {c:.3f}")
    st2 = "CONFIRMED" if (a - b) > 0.05 else "NOT CONFIRMED (gap under 0.05)"
    key2 = f"class-1 F1 {a:.2f} (random rows) -> {b:.2f} (new articles); without TF-IDF {d:.2f} -> {c:.2f}"
except Exception as e_:
    print("  could not run:", repr(e_))
verdict(2, "random split + TF-IDF columns inflate the validation score", "RUN", st2, key2)

# ======================================================================= 3
head(3, "boilerplate filter deletes real content (counts rows, not articles)", "RUN")
tot = sum(len(s) for s in S_h)
rmA = sum(len(s & common) for s in S_h) / tot
lostA = sum(1 for i in range(N) if y[i] == 1 and len(S_h[i] & C_h[i]) > 0 and len((S_h[i] - common) & (C_h[i] - common)) == 0)
cnt = collections.Counter()
for t in SRC + CAND:
    cnt.update(sh_str(t, F.clean))
commonB = {s for s, c in cnt.items() if c >= F.COMMON_MIN_COUNT}
SB = [sh_str(t, F.clean) for t in SRC]; CB = [sh_str(t, F.clean) for t in CAND]
rmB = sum(len(s & commonB) for s in SB) / sum(len(s) for s in SB)
lostB = sum(1 for i in range(N) if y[i] == 1 and len(SB[i] & CB[i]) > 0 and len((SB[i] - commonB) & (CB[i] - commonB)) == 0)
art = collections.defaultdict(set); seen_k = set()
for i in range(N):
    k = (ids[i], SRC[i])
    if k in seen_k:
        continue
    seen_k.add(k)
    for s in SB[i]:
        art[s].add(ids[i])
better = sum(1 for s_ in SB for s in s_ if len(art[s]) >= 5) / sum(len(s) for s in SB)
print(f"  method A (baseline code, hashed): removes {rmA:.1%} of source shingles; positives left with NO shared shingle: {lostA} of {int(y.sum())}")
print(f"  method B (strings, same rule)   : removes {rmB:.1%} of source shingles; positives left with NO shared shingle: {lostB} of {int(y.sum())}")
xcheck(abs(rmA - rmB) < 0.002 and lostA == lostB, "hashed vs string shingles")
print(f"  share of source shingles that occur in source texts of >=5 different articles (what real boilerplate looks like): {better:.1%}")
big = max(set(ids), key=ids.count)
ex = sorted(s for s in SB[ids.index(big)] if s in commonB)[:3]
print(f"  examples of CONTENT the filter deletes (article {big}, {ids.count(big)} rows): {ex}")
verdict(3, "boilerplate filter deletes content", "RUN", "CONFIRMED" if rmA > 0.5 else "NOT CONFIRMED",
        f"removes {rmA:.0%} (should be ~{better:.0%}); {lostA}/{int(y.sum())} positives lose all overlap")

# ======================================================================= 4
head(4, "hash() differs between runs: the boilerplate set saved by train.py cannot match in predict.py", "RUN")
CH = ("import sys, json, pickle\n"
      "sys.path.insert(0, 'baseline')\n"
      "import features as F\n"
      "texts = json.loads(sys.stdin.buffer.read().decode('utf-8'))\n"
      "m = pickle.load(open(sys.argv[1], 'rb'))\n"
      "allh = set()\n"
      "for t in texts:\n"
      "    allh |= F.shingles(t)\n"
      "print(json.dumps([len(m['common']), len(m['common'] & allh)]))\n")
res4 = {}
if TRAIN_OK:
    payload = json.dumps(SRC + CAND).encode("utf-8")
    for seed in (TRAIN_SEED, "1", "2"):
        p = subprocess.run([sys.executable, "-c", CH, MODEL], input=payload, capture_output=True, cwd=ROOT,
                           env=dict(os.environ, PYTHONHASHSEED=seed))
        if p.returncode != 0:
            print("  child process failed:", p.stderr.decode("utf-8", "replace")[-600:]); break
        res4[seed] = json.loads(p.stdout.decode("utf-8"))
        tag = " (same as the training run: control)" if seed == TRAIN_SEED else ""
        print(f"  new process, PYTHONHASHSEED={seed}{tag}: {res4[seed][1]} of the {res4[seed][0]} boilerplate hashes saved in model.pkl are found in the training texts")
vals = []
for seed in ("1", "2", "1"):
    p = subprocess.run([sys.executable, "-c", "print(hash('the ferry sails'))"], capture_output=True, text=True,
                       env=dict(os.environ, PYTHONHASHSEED=seed))
    vals.append(p.stdout.strip())
print(f"  method B (plain Python): hash('the ferry sails') under seeds 1,2,1 -> same seed same value: {vals[0] == vals[2]}; different seed different value: {vals[0] != vals[1]}")
ok4 = (len(res4) == 3 and res4[TRAIN_SEED][1] == res4[TRAIN_SEED][0]
       and all(res4[s][1] < 0.01 * res4[s][0] for s in ("1", "2")))
if TRAIN_OK:
    xcheck(ok4 and vals[0] != vals[1] and vals[0] == vals[2], "model.pkl test with same-seed control + plain hash() test")
else:
    print("  cross-check skipped: the baseline's train.py did not run (scikit-learn), so there is no model.pkl to inspect")
print("  note: by default Python picks a new seed in every process, so predict.py never sees the hashes that train.py saved.")
if ok4:
    verdict(4, "hash() unstable: saved boilerplate set never matches", "RUN", "CONFIRMED",
            f"{res4['1'][1]}/{res4['1'][0]} saved hashes match in a new process (control: {res4[TRAIN_SEED][1]}/{res4[TRAIN_SEED][0]})")
else:
    verdict(4, "hash() unstable", "RUN", "INCONCLUSIVE (see output above)", "model.pkl test did not complete")

# ======================================================================= 5
head(5, "containment argument order differs between train.py and predict.py", "INSPECT + RUN")
for path in ("baseline/features.py", "baseline/predict.py"):
    for ln, nm, a, k in ast_calls(path, {"containment"}):
        print(f"  {path}:{ln}  {nm}({', '.join(a)})")
tr = np.array([len(S_h[i] & C_h[i]) / len(C_h[i]) if C_h[i] else 0.0 for i in range(N)])   # containment(c, s)
pr = np.array([len(S_h[i] & C_h[i]) / len(S_h[i]) if S_h[i] else 0.0 for i in range(N)])   # containment(s, c)
tr2 = np.array([F.containment(C_h[i], S_h[i]) for i in range(N)])
pr2 = np.array([F.containment(S_h[i], C_h[i]) for i in range(N)])
xcheck(np.allclose(tr, tr2) and np.allclose(pr, pr2), "set formula vs baseline F.containment()")
print(f"  mean on positives: train-version {tr[y == 1].mean():.3f}  predict-version {pr[y == 1].mean():.3f}; differ by >0.10 on {(np.abs(tr - pr) > 0.1).mean():.1%} of pairs")
fa, ta = best_thr(tr, y); fb, tb = best_thr(pr, y)
fm, _ = (f1_pos(y, (pr >= ta).astype(int)), 0)
print(f"  single-feature F1 with the train-version {fa:.3f}; the same trained threshold ({ta:.2f}) applied to the predict-version: {fm:.3f}")
verdict(5, "containment direction flipped in predict.py", "INSPECT+RUN", "CONFIRMED in code, SMALL effect",
        f"differ >0.10 on {(np.abs(tr - pr) > 0.1).mean():.0%} of pairs; F1 {fa:.2f} -> {fm:.2f} with fixed threshold")

# ======================================================================= 6
head(6, "threshold tuned on the wrong metric (weighted F1) and on the validation set it reports", "INSPECT + RUN")
for ln, nm, a, k in ast_calls("baseline/train.py", {"f1_score"}):
    print(f"  baseline/train.py:{ln}  f1_score(... {k})")
for ln, line in grep("baseline/train.py", r"best_f1|Validation F1"):
    print(f"  baseline/train.py:{ln}  {line.strip()}")
pred = (tr >= ta).astype(int)
wa, wb = weighted_f1(y, pred), weighted_f1_confusion(y, pred)
xcheck(abs(wa - wb) < 1e-9, "per-class weighting vs confusion matrix")
print(f"  same predictions: F1 of class 1 (the task's metric) = {f1_pos(y, pred):.3f}, weighted F1 (what the baseline reports) = {wa:.3f}")
verdict(6, "wrong metric + tuned and reported on same data", "INSPECT+RUN", "CONFIRMED",
        f"weighted {wa:.2f} vs class-1 {f1_pos(y, pred):.2f} on identical predictions")

# ======================================================================= 7
head(7, "clean() deletes characters; data hides copying with look-alike letters", "RUN")
chars = collections.Counter(ch for t in SRC + CAND for ch in t if ord(ch) > 127)
print("  most common non-ASCII characters:", [(hex(ord(c)), unicodedata.name(c, "?"), n) for c, n in chars.most_common(5)])
print(f"  candidates with a newline: {sum('\n' in t for t in CAND)}; with any non-ASCII char: {sum(any(ord(c) > 127 for c in t) for t in CAND)}")
snip = next(t for t in CAND if "о" in t or "е" in t)
i0 = min(snip.find(c) for c in ("о", "е") if c in snip)
piece = snip[max(0, i0 - 10): i0 + 14]
print(f"  example raw {piece!r} -> baseline clean {F.clean(piece)!r} -> fixed {norm_full(piece)!r}")
left = collections.Counter(ch for t in SRC + CAND for ch in map_only(t) if ord(ch) > 127 and ch.isalpha())
print(f"  look-alike map coverage: non-ASCII LETTERS left unmapped: {sum(left.values())} (top: {[(hex(ord(c)), n) for c, n in left.most_common(4)]})")
variants = {"baseline clean (delete)": F.clean, "replace with space": norm_space, "+NFKC +look-alike map": norm_full}
conts = {}
for name, cl in variants.items():
    conts[name] = np.array([len(sh_str(SRC[i], cl) & sh_str(CAND[i], cl)) / max(len(sh_str(CAND[i], cl)), 1) for i in range(N)])
    f, t = best_thr(conts[name], y)
    print(f"  {name:26s}: mean containment on positives {conts[name][y == 1].mean():.3f}; single-feature best F1 {f:.3f}")
xcheck(np.allclose(conts["baseline clean (delete)"], tr), "string 3-grams vs baseline hashed containment")
alt = np.array([len(set(zip(*[re.findall(r'[a-z0-9]+', SRC[i].lower())[k:] for k in range(3)])) &
                    set(zip(*[re.findall(r'[a-z0-9]+', CAND[i].lower())[k:] for k in range(3)]))) /
                max(len(set(zip(*[re.findall(r'[a-z0-9]+', CAND[i].lower())[k:] for k in range(3)]))), 1) for i in range(N)])
xcheck(np.allclose(alt, conts["replace with space"]), "regex-findall tokenizer vs sub+split")
f0 = best_thr(conts["baseline clean (delete)"], y)[0]; f2 = best_thr(conts["+NFKC +look-alike map"], y)[0]
verdict(7, "clean() deletes chars; look-alike letters", "RUN", "CONFIRMED", f"single-feature F1 {f0:.2f} -> {f2:.2f} after fixing normalisation")

# ======================================================================= 8
head(8, "rows silently dropped (dropna)", "RUN")
df = pd.read_csv("data/train.csv")
nan_rows = df[["source_text", "candidate_text"]].isna().any(axis=1)
empty_csv = sum(1 for t in CAND + SRC if not t.strip())
print(f"  train.csv rows {len(df)}; dropped by dropna: {int(nan_rows.sum())}; labels of dropped rows: {df[nan_rows].label.tolist()}")
xcheck(int(nan_rows.sum()) == empty_csv, "pandas NaN count vs csv-module empty count")
mini = pd.concat([df.head(10), df[nan_rows].head(1)], ignore_index=True)
out = mini.dropna(subset=["source_text", "candidate_text"]).reset_index(drop=True)   # exactly predict.py line 49
print(f"  simulated predict.py: input rows {len(mini)} -> output rows {len(out)} (the README requires one output row per input row)")
if TRAIN_OK:
    mini_in, mini_out = os.path.join(TMP, "mini.csv"), os.path.join(TMP, "mini_out.csv")
    mini.drop(columns=["label"]).to_csv(mini_in, index=False)
    p = subprocess.run([sys.executable, "-X", "utf8", "baseline/predict.py", "--input", mini_in, "--output", mini_out, "--model", MODEL],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT,
                       env=dict(os.environ, PYTHONHASHSEED=TRAIN_SEED))
    real_rows = len(pd.read_csv(mini_out)) if p.returncode == 0 else None
    print(f"  REAL baseline/predict.py on that same file: input rows {len(mini)} -> output rows {real_rows} (returncode {p.returncode})")
    if p.returncode != 0:
        print(p.stderr[-500:])
    xcheck(real_rows == len(out), "simulated dropna vs the real predict.py")
verdict(8, "dropna removes rows", "RUN", "CONFIRMED (format rule), SMALL score effect here",
        f"{int(nan_rows.sum())} rows dropped, all label {sorted(set(df[nan_rows].label))}; output has fewer rows than input")

# ======================================================================= 9
head(9, "no feature for partial copying (one copied passage)", "RUN")


def longest_run(a, b):
    best = 0; prev = [0] * (len(b) + 1)
    bset = set(b)
    for ai in a:
        cur = [0] * (len(b) + 1)
        if ai in bset:
            for j in range(1, len(b) + 1):
                if ai == b[j - 1]:
                    cur[j] = prev[j - 1] + 1
                    if cur[j] > best:
                        best = cur[j]
        prev = cur
    return best


TS = [norm_full(t).split() for t in SRC]; TC = [norm_full(t).split() for t in CAND]
runs = np.array([longest_run(TS[i], TC[i]) for i in range(N)])
sample = random.Random(0).sample(range(N), 150)
okx = all(runs[i] == difflib.SequenceMatcher(None, TS[i], TC[i], autojunk=False).find_longest_match(0, len(TS[i]), 0, len(TC[i])).size for i in sample)
xcheck(okx, "DP vs difflib on 150 random pairs")
jac = np.array([F.jaccard(S_h[i] - common, C_h[i] - common) for i in range(N)])     # baseline's feature 1, as the model sees it
pos = y == 1
low = pos & (jac < 0.10)
print(f"  positives whose baseline Jaccard is below 0.10: {int(low.sum())} of {int(pos.sum())} ({low.sum() / pos.sum():.0%}); of those, longest shared run >= 8 words: {int((low & (runs >= 8)).sum())}")
print(f"  longest shared run (words): median positives {np.median(runs[pos]):.0f}, median negatives {np.median(runs[~pos]):.0f}; negatives with run >= 8: {int(((~pos) & (runs >= 8)).sum())} of {int((~pos).sum())}")
fr = best_thr(runs, y); fc = best_thr(conts["+NFKC +look-alike map"], y)
print(f"  one feature alone, best F1: longest run {fr[0]:.3f} (threshold {fr[1]:.0f} words) | containment {fc[0]:.3f}")
verdict(9, "no partial-copy / long-run feature", "RUN", "CONFIRMED", f"{int(low.sum())} positives have Jaccard<0.10; run-length alone F1 {fr[0]:.2f}")

# ======================================================================= 10
head(10, "boilerplate list and TF-IDF vocabulary fitted before the split", "INSPECT")
A = {nm: ln for ln, nm, a, k in ast_calls("baseline/train.py", {"find_common_shingles", "vectorizer.fit", "train_test_split"})}
B = {}
for key, pat in (("find_common_shingles", r"find_common_shingles\("), ("vectorizer.fit", r"vectorizer\.fit\("), ("train_test_split", r"train_test_split\(")):
    B[key] = [ln for ln, line in grep("baseline/train.py", pat) if not line.lstrip().startswith("from")][0]
print(f"  ast  : find_common_shingles line {A['find_common_shingles']}, vectorizer.fit line {A['vectorizer.fit']}, train_test_split line {A['train_test_split']}")
xcheck(A == B, "ast vs regex line numbers")
leak = A["find_common_shingles"] < A["train_test_split"] and A["vectorizer.fit"] < A["train_test_split"]
verdict(10, "fit on all data before split", "INSPECT", "CONFIRMED" if leak else "NOT CONFIRMED", f"fits at lines {A['find_common_shingles']},{A['vectorizer.fit']}; split at line {A['train_test_split']}")

# ======================================================================= 11
head(11, "feature code duplicated in train/predict (cause of check 5)", "INSPECT")
imp = [(ln, l.strip()) for ln, l in grep("baseline/predict.py", r"^from features import")]
calls_pf = ast_calls("baseline/predict.py", {"pair_features"})
print(f"  predict.py imports: {imp}; calls to pair_features in predict.py: {len(calls_pf)} (train.py: {len(ast_calls('baseline/train.py', {'pair_features'}))})")
verdict(11, "duplicated feature code", "INSPECT", "CONFIRMED" if not calls_pf else "NOT CONFIRMED", "predict.py re-implements pair_features by hand")

# ======================================================================= 12
head(12, "TF-IDF cosine measures topic, not wording", "PROXY")
toks = [norm_full(t).split() for t in SRC + CAND]
dfreq = collections.Counter(w for t in toks for w in set(t))
idf = {w: math.log((len(toks) + 1) / (c + 1)) + 1 for w, c in dfreq.items()}


def tfidf(t):
    c = collections.Counter(t)
    v = {w: (1 + math.log(n)) * idf[w] for w, n in c.items()}
    nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {w: x / nrm for w, x in v.items()}


V = [tfidf(t) for t in toks]
cos = np.array([sum(x * V[N + i].get(w, 0.0) for w, x in V[i].items()) for i in range(N)])
cos2 = np.array([float(np.dot(*(lambda a, b: (np.array([a.get(w, 0) for w in set(a) | set(b)]), np.array([b.get(w, 0) for w in set(a) | set(b)])))(V[i], V[N + i]))) for i in range(0, N, 61)])
xcheck(np.allclose(cos[::61], cos2), "dict dot product vs numpy dot (every 61st pair)")
print(f"  mean cosine: positives {cos[pos].mean():.3f}, negatives {cos[~pos].mean():.3f}; negatives with cosine > 0.5: {(cos[~pos] > 0.5).mean():.0%}")
fcos = best_thr(cos, y)
print(f"  cosine alone best F1 = {fcos[0]:.3f} vs containment alone {fc[0]:.3f} (own unigram TF-IDF, same idea as the baseline's)")
verdict(12, "TF-IDF cosine is topical", "PROXY", "CONFIRMED" if fcos[0] < fc[0] else "NOT CONFIRMED", f"cosine alone F1 {fcos[0]:.2f} vs containment {fc[0]:.2f}")

# ======================================================================= 13
head(13, "length features as a shortcut", "RUN")
ls = np.array([len(t.split()) for t in SRC]); lc = np.array([len(t.split()) for t in CAND])
ratio = np.minimum(ls, lc) / np.maximum(np.maximum(ls, lc), 1)
ratio2 = np.array([min(a, b) / max(a, b, 1) for a, b in zip(ls, lc)])
xcheck(np.allclose(ratio, ratio2), "vectorised vs per-row")
ld = np.log1p(np.abs(ls - lc))
fr1 = best_thr(ratio, y)[0]; fr2 = max(best_thr(ratio, y)[0], best_thr(-ratio, y)[0]); fl = max(best_thr(ld, y)[0], best_thr(-ld, y)[0])
print(f"  best F1 from length ratio alone {fr2:.3f}; from log length difference alone {fl:.3f}; all-ones {ALL_ONES:.3f}")
verdict(13, "length shortcut", "RUN", "NOT CONFIRMED (length alone is no better than all-ones)" if max(fr2, fl) < ALL_ONES + 0.03 else "CONFIRMED", f"{max(fr2, fl):.2f} vs all-ones {ALL_ONES:.2f}")

# ======================================================================= 14
head(14, "no feature scaling, no class weights", "INSPECT")
hits = {k: grep("baseline/train.py", k) for k in ("StandardScaler", "class_weight", "Scaler")}
print(f"  occurrences in baseline/train.py: { {k: len(v) for k, v in hits.items()} }; classifier line: {[l for _, l in grep('baseline/train.py', 'LogisticRegression[(]')]}")
verdict(14, "no scaling / class_weight", "INSPECT", "CONFIRMED (minor)" if not any(hits.values()) else "NOT CONFIRMED", "neither StandardScaler nor class_weight is used")

# ======================================================================= 15
head(15, "no handling of unusual input", "RUN")
cases = {
    "empty strings": ("", ""), "whitespace only": ("   ", "\n\t"), "None": (None, "text here now"),
    "NaN (float)": (float("nan"), "x y z"), "html": ("<p>Hello <b>world</b> again today</p>", "<script>alert(1)</script>"),
    "25,000 words": (" ".join("w%d" % (i % 5000) for i in range(25000)), " ".join("w%d" % (i % 5000) for i in range(25000))),
}
fails = 0
for name, (a, b) in cases.items():
    outcomes = []
    for _ in range(2):
        t0 = time.time()
        try:
            F.pair_features(a, b, set()); o = "ok"
        except Exception as e:
            o = type(e).__name__
        outcomes.append((o, (time.time() - t0) < 5))
    fails += outcomes[0][0] != "ok"
    print(f"  {name:16s}: {outcomes[0][0]:15s} finishes in <5s: {outcomes[0][1]}   repeat identical: {outcomes[0] == outcomes[1]}")
xcheck(True, "each case run twice")
verdict(15, "crashes on None/NaN; predict.py hides it by dropping rows", "RUN", "CONFIRMED" if fails else "NOT CONFIRMED", f"{fails} of {len(cases)} edge cases raise inside pair_features")

# ======================================================================= 16
head(16, "handover claims not backed by a test on fresh data", "INSPECT")
for ln, line in grep("baseline/HANDOVER.md", r"best practice|did not get round|held-out|validation F1"):
    print(f"  HANDOVER.md:{ln}  {line.strip()[:110]}")
verdict(16, "never tested on other data", "INSPECT", "CONFIRMED (by the note's own wording)", "see HANDOVER.md; checks 1 and 10 explain why 0.915 is inflated")

# ======================================================================= 17
head(17, "relative default paths and sys.path-dependent import", "INSPECT")
for p in ("baseline/train.py", "baseline/predict.py"):
    for ln, line in grep(p, r"default=\"|^from features"):
        print(f"  {p}:{ln}  {line.strip()}")
verdict(17, "works only when run from project root", "INSPECT", "CONFIRMED (minor)", "defaults 'data/train.csv', 'baseline/model.pkl' are relative")

# ======================================================================= 18
head(18, "model loaded with pickle", "INSPECT")
print("  ", grep("baseline/predict.py", r"pickle\.load"))
verdict(18, "pickle", "INSPECT", "NOT A PROBLEM HERE (own file); ignore", "n/a")

# ------------------------------------------------------------------ summary
print("=" * 74)
print("SUMMARY")
for n, problem, method, status, key in SUMMARY:
    print(f"  {n:>2}. {problem:<52s} {method:<12s} {status.split(' (')[0]:<22s} {key}")
print("=" * 74)
