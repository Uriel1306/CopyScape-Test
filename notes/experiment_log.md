# Experiment log (expected & why | change | grouped-CV F1 before -> after | kept?)

NOT YET RUN: scikit-learn is broken on this machine (missing sklearn.utils; disk was full), so no grouped-CV number exists.

| # | Expected & why | Change | F1 before -> after | Kept |
|---|---|---|---|---|
| 0 | Fixing normalisation + source-only boilerplate masking should beat baseline single-feature 0.80 | detector.py v1 | in-sample single-feature, hand-picked threshold (NOT CV): cand_cont3 0.895, idf_cand_cont3 0.917, covered5 0.931, longest_run5 0.948, win_max5 0.967, cand_cont5 0.967 | yes |
