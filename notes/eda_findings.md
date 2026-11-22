# New EDA (on top of REPORT.md section 1 / analysis/partA_checks.py)

- Source-only boilerplate: number of articles containing each distinct source 3-gram. 96% occur in 1 article.
  Counts 10..17 are empty (10 shingles in 4..9, then a block at 18-39: the stock footers). Any min_sources in 5..17
  gives the same set on train.csv; train.py uses 10 (421 common shingles, 4.1% of distinct 3-grams).
- (a) Low-overlap positives (tr_00148, tr_01578, tr_01801, tr_00453, tr_00009): tr_00148 is a lightly PARAPHRASED copy
  (measure->gauge, every->each); the others look like candidates made mostly of unrelated text. Exact 5-gram runs miss them
  (31 of 574 positives have longest run < 8 words). Not yet verified whether the others contain a copied sentence.
- (b) High-overlap negatives (tr_00790, tr_01144, tr_00019, tr_01655, tr_01537): longest raw run 46-52 words, but these
  are same-subject paraphrases plus shared stock footers; 296 negatives have a raw run >= 8 words before masking.
- (c) Repeated candidates: 69 distinct texts in 143 rows, only 5 with mixed labels, so the same candidate is
  usually labelled consistently; pair features (not candidate identity) must decide.
- Homoglyph map measured from data: Cyrillic а е о р с у х і ѕ ј т н м к в, Greek ο ν ε α τ ι κ β μ; fullwidth Latin via NFKC;
  invisibles also include U+2060 (word joiner), U+200C/D.
