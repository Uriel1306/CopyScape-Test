# Report

Name:
Time you started / finished: 17:00
AI tools used: claude code pro

## 1. Baseline review

One row per problem, most important first. "How I confirmed it" should be something you ran or looked at, not only something you read in the code.

Everything below comes from one script, `analysis/partA_checks.py` (`python -X utf8 analysis/partA_checks.py`). It runs 18 checks, prints the evidence and a verdict for each, computes every key number in two independent ways, and gives identical output on two runs. The check number in brackets refers to it.

| # | Problem | Where | How I confirmed it | How much it matters |
|---|---|---|---|---|
| 1 | The boilerplate filter deletes the evidence of copying. It counts the number of rows containing a shingle, but one article appears in up to 86 rows, so its real content passes the threshold of 25 and is removed. | `features.py:25-30`, used at `train.py:35` and in `pair_features` | Ran the baseline's own function on `train.csv` [3]. It removes 74.4% of source 3-grams; counted by distinct articles only 8.8% look like boilerplate. For 307 of 574 true positives no shared shingle is left after the filter. Same result with string shingles and with hashes. | Very high. Jaccard and containment are computed on what survives the filter. |
| 2 | The 0.915 validation score is inflated by a random row split. The data has 78 articles with 2 to 86 rows each, and the split puts the same article on both sides. The hidden test uses new articles. | `train.py:50` | Rebuilt the split [1]. With a random 80/20 split, 100% of validation rows have their article in train and 98% of their source 3-grams were already seen, against 19% when whole articles are held out. A predictor that sees only the article id gets F1 0.62 on random rows and 0.00 on held-out articles. I also ran the baseline's own pipeline under both splits [2]. I reproduced its 0.915 exactly, but that is weighted F1. Its F1 for class 1 on that same split is 0.861, it is 0.891 on random rows, and 0.827 when whole articles are held out. | High. The 0.915 is not the task metric, and on new articles the real number is about 0.06 lower. |
| 3 | TF-IDF vectors of both texts are fed to the classifier (thousands of topic columns) with C=20, almost no regularisation. They describe the topic, not the wording, and let the model recognise articles (the channel in row 2). | `train.py:41-46, 53` | Cosine of TF-IDF alone reaches F1 0.68 against 0.80 for plain containment, and 28% of independent texts have cosine above 0.5 because they share the subject [12]. I also removed the TF-IDF columns from the baseline pipeline [2]. On random rows F1 drops from 0.891 to 0.808, but on held-out articles it rises from 0.827 to 0.839. So the columns only help when the same articles are on both sides. | High. They explain why the random split looks good and why it does not carry over. |
| 4 | `clean()` deletes characters instead of replacing them, and the data hides copying with look-alike letters. 427 candidates contain non-ASCII characters, mostly Cyrillic and Greek letters that look Latin, plus no-break spaces. | `features.py:12-16` | Example: "Lаne" (Cyrillic а) becomes "lne", "оn" becomes "n" [7]. After replacing with spaces, applying NFKC and mapping look-alike letters to Latin, mean containment on positives goes from 0.444 to 0.511 and the best single-feature F1 from 0.755 to 0.795. | High. The disguise lowers overlap exactly for the copied texts. |
| 5 | The reported metric is weighted F1, not F1 of the positive class. The threshold is tuned and reported on the same validation set. | `train.py:60-64` | On identical predictions weighted F1 is 0.849 and F1 of class 1 is 0.755 [6]. Two implementations agree. | Medium. The reported number is inflated and the threshold optimises the wrong objective. |
| 6 | No feature for partial copying. Only global Jaccard and containment of 3-grams, so one copied passage in a long text gives a tiny score. | `features.py:47-58` | 424 of 574 positives have a baseline Jaccard below 0.10 [9]. The median longest shared run is 18 words for positives against 4 for negatives, and "longest run" alone reaches F1 0.76. Checked against `difflib`. | Medium to high. It partly overlaps with row 1. |
| 7 | `hash()` of strings changes in every Python process, and the boilerplate set is saved as hashes in `model.pkl`. `predict.py` should never match it, and results are not reproducible. | `features.py:22` | I trained the baseline and opened `model.pkl` in a fresh Python process [4]. 0 of the 5,031 saved boilerplate hashes are found there. With the hash seed fixed to the training run's, all 5,031 are found. A plain test agrees: the same string gives a different hash under a different seed. | Medium. The saved filter does nothing when `predict.py` runs, so prediction differs from training. |
| 8 | `predict.py` re-implements the features by hand and passes the arguments of `containment` in the opposite order. | `predict.py:28` against `features.py:55` | Static analysis shows `containment(c, s)` against `containment(s, c)` [5]. The two values differ by more than 0.10 on 8.4% of pairs, and the single-feature F1 goes from 0.755 to 0.739 at the same threshold. | Low to medium. A real bug with a small effect. |
| 9 | `dropna` removes rows, so the output can have fewer rows than the input. | `train.py:30`, `predict.py:49` | 5 rows have an empty candidate [8]. The real `baseline/predict.py` turns an 11-row file into 10 output rows. | Low for the score here, because all 5 dropped rows are label 0 and a missing row counts as 0. It breaks the one-row-per-input rule. |
| 10 | Smaller issues: the boilerplate list and the TF-IDF vocabulary are fitted on all data before the split (lines 35 and 42, split at line 50) [10]. No feature scaling and no class weights [14]. `pair_features` raises `AttributeError` on None or NaN [15]. The handover note admits it was tested only on the training file [16]. Relative default paths [17]. | `train.py`, `features.py`, `HANDOVER.md` | Static analysis with line numbers, and running the function on 6 edge inputs. | Low. |

**Anything an AI tool flagged in the baseline that turned out to be wrong or unimportant?**

- Length features as a shortcut: the AI-assisted review suspected it, and I tested it. It is not confirmed: the best F1 from length alone is 0.479 against 0.477 for predicting everything as 1 [13].
- The flipped `containment` argument is real but small (row 8). The 5 dropped rows are all negatives (row 9). Loading with `pickle` is not a problem here [18].
- Two of the AI's own first statements were corrected after measuring. It said the filter removes 82% of the content, which came from a different tokenisation, and the baseline's own code gives 74.4%. It also proposed counting boilerplate by distinct `source_id` over all texts. That would be contaminated, because a candidate copied from article A also appears in rows of other articles, so the count must use source texts only.

**Limits of this review.** I did not fix the baseline. My first run of the script happened while scikit-learn was broken (the disk was full), so I ran checks 2 and 4 after I repaired it, and the numbers above come from that second run. The held-out numbers use one 80/20 split by article per seed, so they move by a few points (0.802 and 0.852 for the two seeds). Single-feature F1 values are the best threshold on all of `train.csv`, so I only use them to compare cleaning variants, not as estimates of performance.

## 2. Approach (max 200 words)

What does your detector do, and why is that the right approach for this problem and this data? What did you consider and reject?

The detector measures how much exact wording the candidate shares with the source. The task is about wording, not meaning, so I look for runs of identical words and ignore topic.

I clean both texts (HTML, invisible characters, and look-alike Cyrillic or Greek letters mapped to Latin, because the data hides copies this way) and build word n-grams of 2, 3, 5 and 8 words with crc32, so runs are repeatable. N-grams found in the sources of 10 or more articles (stock footers) are masked.

Each pair becomes 33 numbers: overlap in both directions, IDF-weighted overlap, the longest shared run, the share of the candidate that is covered, and the best 30-word window. The window numbers catch one copied sentence in a long text. A HistGradientBoosting model gives the score, with a cut-off of 0.42 chosen on out-of-fold scores.

This fits the data: texts on the same subject share facts but almost never five words in a row. It is linear in length, so 1,000 pairs with 25,000-word documents take about 8 seconds.

Rejected: TF-IDF (it measures topic and lets the model memorise articles), Python's `hash()` (changes every run), full alignment (too slow), neural networks (too little data).

## 3. Experiment log

Fill this in as you go. Include experiments that did not work.

All results are F1 for class 1. Rows 5 to 12 are grouped cross-validation (5 folds, 3 seeds), shown for the two models I compared, `lr` (logistic regression) and `hgb` (HistGradientBoosting). Rows 2 to 4 use one feature with the best threshold picked on all of `train.csv`, so they are only good for comparing steps with each other.

| # | What I expected and why | What I changed | Result (F1, before → after) | Kept? |
|---|---|---|---|---|
| 1 | The baseline's 0.915 looked too good, because its split keeps the same article on both sides. | Ran the baseline's own pipeline with whole articles held out, and scored class 1 instead of weighted F1. | 0.915 (weighted, as reported) → 0.827 (class 1, new articles; 0.802 and 0.852 for two splits) | No. This is the number I had to beat. |
| 2 | Shared wording alone should be a strong signal, since that is what the task asks about. | One feature: the share of the candidate's 3-grams that are in the source. | 0.477 (everything is 1) → 0.755 | Yes |
| 3 | The data hides copies with look-alike letters, so cleaning should matter. | Replace characters with a space instead of deleting them, NFKC, and map look-alike Cyrillic and Greek letters to Latin. | 0.755 → 0.795 | Yes |
| 4 | Texts on the same subject share many 3-grams by chance, and stock footers add fake matches. Longer n-grams and masked footers should fix both. | 5-word n-grams, and footers masked using a list of n-grams that appear in the sources of many different articles. | 0.795 → 0.968 (`cand_cont5`) | Yes |
| 5 | I needed a real number, so I trained models on overlap features only (n = 2, 3, 5, 8). | Grouped cross-validation. | lr 0.9667, hgb 0.9673 | Yes, as the base |
| 6 | IDF weighting should make generic phrases count less. | Added 2 IDF-weighted overlap features. | lr 0.9712, hgb 0.9679 | Yes. Small, and inside the noise. |
| 7 | Overlap numbers dilute one copied sentence in a long text. The longest run, the covered share and 30-word windows should catch it. | Added 12 run and window features and 3 length features. This is the submitted feature set. | lr 0.9682, hgb 0.9714 | Yes. Compared with overlap only the gain is +0.007 (95% interval −0.001 to +0.015), so it is not proven. I kept it because the task says one copied sentence counts. |
| 8 | If the local features are enough, the global ones could go. | Only runs, windows and lengths. | lr 0.9652, hgb 0.9630 | No, slightly worse |
| 9 | Masking footers should matter. | Turned footer learning off. | lr 0.9594, hgb 0.9523 (hgb −0.019 against row 7) | Kept the masking. This is the only change with a clear effect: +0.013 against the submitted model, interval +0.004 to +0.022. |
| 10 | Masking numbers and names, and character n-grams, should separate shared facts from shared phrasing. | Switched on the two optional feature groups. | lr 0.9723, hgb 0.9710 | No. Nothing gained for hgb, and they are slower. |
| 11 | A footer threshold of 10 articles might be too strict. My EDA said anything from 5 to 17 gives the same list. | Used 5 instead of 10. | lr 0.9739, hgb 0.9712 | No. The difference is −0.003 (interval −0.016 to +0.008), inside the noise. If I tried many variants and kept the best one, my estimate would be too optimistic. |
| 12 | Boosting should combine short runs and overlap better than a straight line. | Compared lr, hgb and their average on the final features, and took the simplest model within 0.002 of the best. | lr 0.9682, hgb 0.9714, average 0.9691 → hgb | Yes |

How did you estimate performance, and why that way?

I used grouped cross-validation: 5 folds split by `source_id`, so a whole article is either in training or in validation, repeated with 3 seeds. The hidden test comes from new articles, so this is the same situation. In every fold the footer list and the IDF are learned from the training articles only. The cut-off 0.42 comes from the out-of-fold scores of all seeds. The F1 curve is flat (0.964 to 0.972 for cut-offs between 0.2 and 0.7), so choosing it adds very little optimism.

The ± printed by `train.py` is only the spread between seeds, so it is too small. To see the real uncertainty I resampled articles: 0.972 with a 95% interval of 0.956 to 0.984. For comparing variants I used the same resampling on the difference between two models.

**My prediction for F1 on the hidden test set:** about 0.96 (somewhere between 0.94 and 0.98). I put it a little under the cross-validation score because new articles may carry footers I have not seen.

Where does your detector still fail? Give two or three concrete examples (`pair_id`s from the training data).

Out of 1,835 pairs it makes 19 misses and 13 false alarms. All 19 missed copies have no shared run of 8 words, so they are edited copies and not verbatim ones:

- `tr_01578` (score 0.001): one sentence of the source is copied into a text about a lighthouse, with three words swapped (carried → hauled, afterwards → later, nearly → almost). The longest shared run is 3 words, so no 5-word n-gram matches.
- `tr_00148` (score 0.003): the whole text is a light paraphrase (measure → gauge, every → each, one sentence reworded). Again no 5-word match.
- `tr_00514` (score 0.993, but independent): a text about the same composer with the same facts. It shares phrases like "the daughter of a church organist", which is a 6-word run. 7 of the 13 false alarms come from this one article (`src_4789`), and its F1 is 0.63.

The two kinds of error look alike on word overlap. The median bigram overlap is 0.29 for the missed copies and 0.37 for the false alarms, against 0.66 for found copies and 0.18 for true negatives. Exact n-grams cannot tell a copy with synonym swaps from independent writing of the same facts.

## 4. Working with AI tools (max 200 words)

Your session export is the evidence for this section. Point to it where you can.

- Name up to three decisions you made yourself that changed the result. For each, what had the tool proposed?
1. not trusting the ai at all.
2. Before the test I tried to guess the problem and designed a similar system with an AI, then turned it into a blueprint for a separate build session. The tool had proposed a smaller plan (overlap features, longest run, small model). My blueprint was too big for three hours and had wrong details (`test/` folder, footers counted over all texts, slow `difflib`) that the tool corrected. Its multi-size n-grams and windows stayed; entity masking and character n-grams did not help.

- Describe one or two moments when the tool was wrong or misleading. Where in your session did you notice, and what did you do?
I asked why `source_id` matters and whether pairs are one to one. The tool's footer counting was contaminated by copied candidates, so the code now counts source texts only. Footer masking is the only change with a clear effect (+0.013).

- Is there any part of your submission you would not be comfortable explaining line by line? Which?
Because of the time limit, and my limited knowladge before hand, i took to much to understand instead of execute. i will still be comfotable to explain everything, because my need to understnd every by myself, and acheritecth every thing to be the best i can produce.
## 5. With another day (max 150 words)

What would you do next, in order? What would have to change for this to work on Copyscape's real traffic?
With additional time, I would focus on four high-leverage improvements while preserving fast, offline CPU inference:

1. **Larger Boilerplate & IDF Corpus:** Computing `common` shingles and IDF weights from only 1,835 pairs is noisy. Estimating them from a large, diverse web corpus would filter out standard disclaimers and templates much more reliably.
2. **Offline Hard-Case Augmentation:** Use LLMs offline to generate tricky training pairs from real articles: independent rewrites sharing the exact same facts and numbers (hard negatives), and heavily disguised copies with synonym swaps or reordering (hard positives).
3. **Entity-Masked & Syntactic Features:** Since the task is about shared wording rather than shared meaning, logical entailment models would cause false positives on independent news reports. Instead, I would compare sentence structure after masking out numbers and proper nouns to separate copied phrasing from shared facts.
4. **MinHash / LSH Pre-Filtering:** For 25,000-word documents, pre-filtering candidate windows with MinHash before running fine-grained sequence alignments would keep latency flat at much larger scales.
## 6. Anything else (optional)

Anything you built, measured or noticed that the sections above do not cover.

## Assumptions

Anything you had to assume because the instructions were unclear.
