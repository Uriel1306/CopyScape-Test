"""Feature extraction for the reuse-detection baseline."""
import re
from collections import Counter
from typing import Iterable, List, Set

import numpy as np

NGRAM = 3
COMMON_MIN_COUNT = 25  # shingles seen in at least this many texts are treated as boilerplate


def clean(text: str) -> str:
    """Lowercase the text and strip punctuation / special characters."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9 ]", "", text)
    return text


def shingles(text: str, n: int = NGRAM) -> Set[int]:
    """Return the set of hashed word n-grams for a text."""
    tokens = clean(text).split()
    return {hash(" ".join(tokens[i:i + n])) for i in range(len(tokens) - n + 1)}


def find_common_shingles(texts: Iterable[str], min_count: int = COMMON_MIN_COUNT) -> Set[int]:
    """Find boilerplate shingles: n-grams that occur in many different texts."""
    counts: Counter = Counter()
    for text in texts:
        counts.update(shingles(text))
    return {h for h, c in counts.items() if c >= min_count}


def jaccard(a: Set[int], b: Set[int]) -> float:
    """Jaccard similarity between two shingle sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(candidate: Set[int], source: Set[int]) -> float:
    """Fraction of the candidate's shingles that also appear in the source."""
    if not candidate:
        return 0.0
    return len(candidate & source) / len(candidate)


def pair_features(source_text: str, candidate_text: str, common: Set[int]) -> List[float]:
    """Hand-crafted similarity features for one (source, candidate) pair."""
    s = shingles(source_text) - common
    c = shingles(candidate_text) - common
    len_s = len(clean(source_text).split())
    len_c = len(clean(candidate_text).split())
    return [
        jaccard(s, c),
        containment(c, s),
        min(len_s, len_c) / max(len_s, len_c, 1),
        float(np.log1p(abs(len_s - len_c))),
    ]
