"""Shared detection logic: cleaning, shingles, boilerplate, pair features.

`extract_features` is the ONE feature function used by train.py and predict.py.
No Python ``hash()`` anywhere: shingles are crc32 values (deterministic).
All per-pair work is linear in document length (numpy set ops, no O(N*M) DP).
"""
import html
import re
import unicodedata
import zlib
from functools import lru_cache
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

NS: Tuple[int, ...] = (2, 3, 5, 8)        # global overlap sizes
HIT_NS: Tuple[int, ...] = (3, 5)          # sizes used for run / window signals
WINDOW_WORDS = 30                         # ~ one long sentence
WINDOW_STRIDE = 5
MIN_RUN_WORDS = 8                         # a "copied passage" is >= 8 consecutive words
CHAR_N, CHAR_WINDOW, CHAR_STRIDE = 6, 160, 80
MAX_TOKENS = 200_000                      # hard cap for pathological documents
NUM, ENT = "<NUM>", "<ENT>"
BASE_GROUPS: Tuple[str, ...] = ("overlap", "idf", "hits", "length")
OPTIONAL_GROUPS: Tuple[str, ...] = ("masked", "char")

_TAGS = re.compile(r"<!--.*?-->|</?[A-Za-z][^<>]*>", re.S)
_INVISIBLE = re.compile("[​-‏‪-‮⁠-⁤­﻿]")
_TOKEN = re.compile(r"([^\W\d_]+)|(\d+)|([.!?])")
# Look-alike letters (measured on data/train.csv) -> Latin. Applied after NFKC + lower().
_HOMOGLYPHS = str.maketrans("аеорсухіѕјтнмкв" "ονεατικβμ", "aeopcyxisjthmkb" "ovea" "tikbm")


class Base(NamedTuple):
    """Resource-independent view of one text (cached)."""
    tokens: Tuple[str, ...]
    masked: Tuple[str, ...]              # digits -> NUM, mid-sentence capitals -> ENT
    hashes: Dict[int, np.ndarray]        # n -> crc32 of each word n-gram (int64)


class Resources(NamedTuple):
    """Learned from training sources only; stored in model.joblib."""
    common: np.ndarray                   # sorted 3-gram hashes seen in >= min_sources articles
    idf_keys: np.ndarray                 # sorted 3-gram hashes
    idf_vals: np.ndarray                 # idf per key
    idf_default: float                   # idf of an unseen 3-gram (== idf at df=1)


# ------------------------------------------------------------------ cleaning
def normalize_text(text) -> str:
    """html.unescape -> tags to space -> strip invisibles -> NFKC -> lower -> look-alikes."""
    if not isinstance(text, str):
        return ""
    text = _TAGS.sub(" ", html.unescape(text))
    text = _INVISIBLE.sub("", text)
    return unicodedata.normalize("NFKC", text).lower().translate(_HOMOGLYPHS)


def _tokenize_pass(text) -> Tuple[List[str], List[str]]:
    """Aligned (tokens, masked). Capitalisation is read from the un-lowered NFKC text."""
    if not isinstance(text, str):
        return [], []
    raw = unicodedata.normalize("NFKC", _INVISIBLE.sub("", _TAGS.sub(" ", html.unescape(text))))
    low = raw.lower().translate(_HOMOGLYPHS)
    tokens: List[str] = []
    masked: List[str] = []
    sentence_start = True
    for m_raw, m_low in zip(_TOKEN.finditer(raw), _TOKEN.finditer(low)):
        word, digits, end = m_low.groups()
        if end:
            sentence_start = True
            continue
        tokens.append(word or digits)
        capital = bool(word) and m_raw.group()[0].isupper() and not sentence_start
        masked.append(NUM if digits else (ENT if capital else word))
        sentence_start = False
        if len(tokens) >= MAX_TOKENS:
            break
    return tokens, masked


def tokenize(text) -> List[str]:
    """Words and digit runs of the normalised text ('the1680s' -> the 1680 s)."""
    return _tokenize_pass(text)[0]


def clean_text(text) -> str:
    return " ".join(tokenize(text))


def shingle_hashes(tokens: Sequence[str], n: int) -> np.ndarray:
    """crc32 of every word n-gram in position order (empty array if too short)."""
    k = len(tokens) - n + 1
    return np.array([zlib.crc32(" ".join(tokens[i:i + n]).encode("utf-8")) for i in range(max(k, 0))],
                    dtype=np.int64)


@lru_cache(maxsize=4096)
def _base_cached(text: str) -> Base:
    tokens, masked = _tokenize_pass(text)
    return Base(tuple(tokens), tuple(masked), {n: shingle_hashes(tokens, n) for n in NS})


def base_doc(text) -> Base:
    return _base_cached(text if isinstance(text, str) else "")


# ------------------------------------------------------- learned resources
def learn_resources(source_groups: Dict[object, Iterable[str]], min_sources: int) -> Resources:
    """Boilerplate + IDF from SOURCE texts only, counted per distinct source_id.

    `source_groups` maps source_id -> that article's source texts. Candidates are
    never counted (a candidate copied from article A also shows up under others).
    """
    per_article = [np.unique(np.concatenate([base_doc(t).hashes[3] for t in texts] or [np.zeros(0, np.int64)]))
                   for texts in source_groups.values()]
    n_articles = len(per_article)
    keys, df = np.unique(np.concatenate(per_article or [np.zeros(0, np.int64)]), return_counts=True)
    idf = np.log((n_articles + 1.0) / df) if len(keys) else np.zeros(0)
    return Resources(keys[df >= min_sources], keys, idf, float(np.log(n_articles + 1.0)))


EMPTY_RESOURCES = Resources(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), 1.0)


# ---------------------------------------------------------------- prepared docs
class Doc(NamedTuple):
    base: Base
    boiler: np.ndarray                   # bool per token: inside a common 3-gram
    valid: Dict[int, np.ndarray]         # n -> bool per shingle: contains no boilerplate token
    uniq: Dict[int, np.ndarray]          # n -> unique valid shingle hashes


def prepare_doc(text, res: Resources) -> Doc:
    base = base_doc(text)
    boiler = np.zeros(len(base.tokens), dtype=bool)
    for i in np.flatnonzero(np.isin(base.hashes[3], res.common)):
        boiler[i:i + 3] = True
    cs = np.concatenate([[0], np.cumsum(boiler)])
    valid, uniq = {}, {}
    for n in NS:
        k = len(base.hashes[n])
        valid[n] = (cs[n:n + k] - cs[:k]) == 0 if k else np.zeros(0, dtype=bool)
        uniq[n] = np.unique(base.hashes[n][valid[n]])
    return Doc(base, boiler, valid, uniq)


# ------------------------------------------------------------- feature groups
def _ratio(a: float, b: float) -> float:
    return float(a) / float(b) if b else 0.0


def _overlap_feats(ds: Doc, dc: Doc) -> List[float]:
    out: List[float] = []
    for n in NS:
        inter = len(np.intersect1d(ds.uniq[n], dc.uniq[n], assume_unique=True))
        union = len(ds.uniq[n]) + len(dc.uniq[n]) - inter
        out += [_ratio(inter, len(dc.uniq[n])), _ratio(inter, len(ds.uniq[n])),
                _ratio(inter, union), float(np.log1p(inter))]
    return out


def _idf_of(hashes: np.ndarray, res: Resources) -> np.ndarray:
    """IDF per hash; unseen (and df=1) hashes get the default, so train == test."""
    out = np.full(len(hashes), res.idf_default)
    if len(res.idf_keys):
        pos = np.clip(np.searchsorted(res.idf_keys, hashes), 0, len(res.idf_keys) - 1)
        found = res.idf_keys[pos] == hashes
        out[found] = np.minimum(res.idf_vals[pos[found]], res.idf_default)
    return out


def _idf_feats(ds: Doc, dc: Doc, res: Resources) -> List[float]:
    out = []
    for a, b in ((dc, ds), (ds, dc)):
        u = a.uniq[3]
        w = _idf_of(u, res)
        out.append(_ratio(w[np.isin(u, b.uniq[3])].sum(), w.sum()))
    return out


def run_lengths(hit: np.ndarray) -> np.ndarray:
    """Lengths of maximal consecutive-True blocks."""
    if not hit.any():
        return np.zeros(0, dtype=np.int64)
    d = np.diff(np.concatenate([[0], hit.astype(np.int8), [0]]))
    return np.flatnonzero(d == -1) - np.flatnonzero(d == 1)


def window_stats(hit: np.ndarray, width: int, stride: int) -> Tuple[float, float]:
    """(max, mean of top-3) hit fraction over sliding windows of `width` items."""
    if len(hit) == 0:
        return 0.0, 0.0
    width = min(width, len(hit))
    cs = np.concatenate([[0], np.cumsum(hit, dtype=np.int64)])
    starts = np.arange(0, len(hit) - width + 1, stride)
    if starts[-1] != len(hit) - width:
        starts = np.append(starts, len(hit) - width)
    frac = np.sort((cs[starts + width] - cs[starts]) / width)[::-1]
    return float(frac[0]), float(frac[:3].mean())


def _covered_fraction(hit: np.ndarray, n: int, n_tokens: int) -> float:
    """Share of candidate words lying inside at least one hit shingle."""
    starts = np.zeros(n_tokens + n + 1)
    idx = np.flatnonzero(hit)
    np.add.at(starts, idx, 1)
    np.add.at(starts, idx + n, -1)
    return _ratio((np.cumsum(starts)[:n_tokens] > 0).sum(), n_tokens)


def _hits_feats(ds: Doc, dc: Doc) -> List[float]:
    len_s, len_c = len(ds.base.tokens), len(dc.base.tokens)
    out: List[float] = []
    for n in HIT_NS:
        hit = dc.valid[n] & np.isin(dc.base.hashes[n], ds.uniq[n])
        runs = run_lengths(hit)
        longest = float(runs.max() + n - 1) if len(runs) else 0.0
        n_passages = float((runs + n - 1 >= MIN_RUN_WORDS).sum())
        win = window_stats(hit, WINDOW_WORDS - n + 1, WINDOW_STRIDE)
        out += [longest, _ratio(longest, min(len_s, len_c)), float(np.log1p(n_passages)),
                _covered_fraction(hit, n, len_c), win[0], win[1]]
    return out


def _length_feats(ds: Doc, dc: Doc) -> List[float]:
    a, b = len(ds.base.tokens), len(dc.base.tokens)
    return [float(np.log1p(a)), float(np.log1p(b)), _ratio(min(a, b), max(a, b))]


def _masked_grams(doc: Doc) -> np.ndarray:
    """Masked 3-gram hashes with >= 2 real words and no boilerplate token."""
    m = doc.base.masked
    keep = [i for i in range(len(m) - 2)
            if sum(1 for g in m[i:i + 3] if g not in (NUM, ENT)) >= 2 and not doc.boiler[i:i + 3].any()]
    return np.unique([zlib.crc32(" ".join(m[i:i + 3]).encode("utf-8")) for i in keep]) if keep else np.zeros(0)


def _fact_tokens(doc: Doc) -> set:
    return {t for t, m in zip(doc.base.tokens, doc.base.masked) if m in (NUM, ENT)}


def _masked_feats(ds: Doc, dc: Doc) -> List[float]:
    ms, mc = _masked_grams(ds), _masked_grams(dc)
    fs, fc = _fact_tokens(ds), _fact_tokens(dc)
    return [_ratio(len(np.intersect1d(ms, mc)), len(mc)), _ratio(len(fs & fc), len(fc))]


def _char_hashes(doc: Doc) -> np.ndarray:
    """Hashes of char n-grams over the space-free text, boilerplate cut ('|' joins)."""
    parts: List[str] = []
    for tok, b in zip(doc.base.tokens, doc.boiler):
        if not b:
            parts.append(tok)
        elif parts and parts[-1] != "|":
            parts.append("|")
    raw = "".join(parts).encode("utf-8")
    return np.array([zlib.crc32(raw[i:i + CHAR_N]) for i in range(len(raw) - CHAR_N + 1)], dtype=np.int64)


def _char_feats(ds: Doc, dc: Doc) -> List[float]:
    hs, hc = _char_hashes(ds), _char_hashes(dc)
    hit = np.isin(hc, hs)
    win = window_stats(hit, CHAR_WINDOW - CHAR_N + 1, CHAR_STRIDE)
    return [_ratio(hit.sum(), len(hc)), win[0], win[1]]


def feature_names(groups: Sequence[str]) -> List[str]:
    names: Dict[str, List[str]] = {
        "overlap": [f"{k}{n}" for n in NS for k in ("cand_cont", "src_cont", "jaccard", "log_matched")],
        "idf": ["idf_cand_cont3", "idf_src_cont3"],
        "hits": [f"{k}{n}" for n in HIT_NS for k in
                 ("longest_run", "longest_over_min", "log_passages", "covered", "win_max", "win_top3")],
        "length": ["log_len_src", "log_len_cand", "len_ratio"],
        "masked": ["mask_cont_cand", "fact_overlap_cand"],
        "char": ["char_cont_cand", "char_win_max", "char_win_top3"],
    }
    return [name for g in groups for name in names[g]]


def extract_features(source_text, candidate_text, res: Resources,
                     groups: Sequence[str] = BASE_GROUPS) -> np.ndarray:
    """Feature vector for one pair (zeros if either side has no words)."""
    width = len(feature_names(groups))
    ds, dc = prepare_doc(source_text, res), prepare_doc(candidate_text, res)
    if not ds.base.tokens or not dc.base.tokens:
        return np.zeros(width)
    funcs = {"overlap": lambda: _overlap_feats(ds, dc), "idf": lambda: _idf_feats(ds, dc, res),
             "hits": lambda: _hits_feats(ds, dc), "length": lambda: _length_feats(ds, dc),
             "masked": lambda: _masked_feats(ds, dc), "char": lambda: _char_feats(ds, dc)}
    return np.array([v for g in groups for v in funcs[g]()], dtype=np.float64)
