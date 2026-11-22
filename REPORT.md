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
| 2 | The 0.915 validation score is inflated by a random row split. The data has 78 articles with 2 to 86 rows each, and the split puts the same article on both sides. The hidden test uses new articles. | `train.py:50` | Rebuilt the split [1]. With a random 80/20 split, 100% of validation rows have their article in train and 98% of their source 3-grams were already seen, against 19% when whole articles are held out. A predictor that sees only the article id gets F1 0.62 on random rows and 0.00 on held-out articles. | High. The score does not estimate performance on new articles. I did not measure the baseline's own F1 under a correct split (see the limits below). |
| 3 | TF-IDF vectors of both texts are fed to the classifier (thousands of topic columns) with C=20, almost no regularisation. They describe the topic, not the wording, and let the model recognise articles (the channel in row 2). | `train.py:41-46, 53` | Cosine of TF-IDF alone reaches F1 0.68 against 0.80 for plain containment, and 28% of independent texts have cosine above 0.5 because they share the subject [12]. | High in principle. The mechanism is confirmed, but the effect on the baseline's own F1 is not measured. |
| 4 | `clean()` deletes characters instead of replacing them, and the data hides copying with look-alike letters. 427 candidates contain non-ASCII characters, mostly Cyrillic and Greek letters that look Latin, plus no-break spaces. | `features.py:12-16` | Example: "Lаne" (Cyrillic а) becomes "lne", "оn" becomes "n" [7]. After replacing with spaces, applying NFKC and mapping look-alike letters to Latin, mean containment on positives goes from 0.444 to 0.511 and the best single-feature F1 from 0.755 to 0.795. | High. The disguise lowers overlap exactly for the copied texts. |
| 5 | The reported metric is weighted F1, not F1 of the positive class. The threshold is tuned and reported on the same validation set. | `train.py:60-64` | On identical predictions weighted F1 is 0.849 and F1 of class 1 is 0.755 [6]. Two implementations agree. | Medium. The reported number is inflated and the threshold optimises the wrong objective. |
| 6 | No feature for partial copying. Only global Jaccard and containment of 3-grams, so one copied passage in a long text gives a tiny score. | `features.py:47-58` | 424 of 574 positives have a baseline Jaccard below 0.10 [9]. The median longest shared run is 18 words for positives against 4 for negatives, and "longest run" alone reaches F1 0.76. Checked against `difflib`. | Medium to high. It partly overlaps with row 1. |
| 7 | `hash()` of strings changes in every Python process, and the boilerplate set is saved as hashes in `model.pkl`. `predict.py` should never match it, and results are not reproducible. | `features.py:22` | Only the plain Python behaviour was run: the same string gives different values under different `PYTHONHASHSEED` and the same value under the same seed [4]. The end-to-end test on `model.pkl` did not run. | Medium, expected rather than measured. It also breaks "same input, same output". |
| 8 | `predict.py` re-implements the features by hand and passes the arguments of `containment` in the opposite order. | `predict.py:28` against `features.py:55` | Static analysis shows `containment(c, s)` against `containment(s, c)` [5]. The two values differ by more than 0.10 on 8.4% of pairs, and the single-feature F1 goes from 0.755 to 0.739 at the same threshold. | Low to medium. A real bug with a small effect. |
| 9 | `dropna` removes rows, so the output can have fewer rows than the input. | `train.py:30`, `predict.py:49` | 5 rows have an empty candidate [8]. A simulated `predict.py` turns 11 input rows into 10 output rows. | Low for the score here, because all 5 dropped rows are label 0 and a missing row counts as 0. It breaks the one-row-per-input rule. |
| 10 | Smaller issues: the boilerplate list and the TF-IDF vocabulary are fitted on all data before the split (lines 35 and 42, split at line 50) [10]. No feature scaling and no class weights [14]. `pair_features` raises `AttributeError` on None or NaN [15]. The handover note admits it was tested only on the training file [16]. Relative default paths [17]. | `train.py`, `features.py`, `HANDOVER.md` | Static analysis with line numbers, and running the function on 6 edge inputs. | Low. |

**Anything an AI tool flagged in the baseline that turned out to be wrong or unimportant?**

- Length features as a shortcut: the AI-assisted review suspected it, and I tested it. It is not confirmed: the best F1 from length alone is 0.479 against 0.477 for predicting everything as 1 [13].
- The flipped `containment` argument is real but small (row 8). The 5 dropped rows are all negatives (row 9). Loading with `pickle` is not a problem here [18].
- Two of the AI's own first statements were corrected after measuring. It said the filter removes 82% of the content, which came from a different tokenisation, and the baseline's own code gives 74.4%. It also proposed counting boilerplate by distinct `source_id` over all texts. That would be contaminated, because a candidate copied from article A also appears in rows of other articles, so the count must use source texts only.

**What I did not do.** The scikit-learn install on my machine broke because the disk was full (the `sklearn/utils` folder is missing), so `baseline/train.py` did not run. I therefore did not reproduce the 0.915, and I did not measure the baseline's F1 under a split by article, or with the TF-IDF columns removed. Check 2 (the real pipeline, random split against held-out articles, with and without TF-IDF) and the `model.pkl` part of check 4 are written in the script and will run once scikit-learn is repaired. Until then rows 2, 3 and 7 are supported by the mechanism, not by a measured drop in the baseline's score. Single-feature F1 values are the best threshold on all of `train.csv`, so they are only used to compare cleaning variants, not as estimates of performance.

## 2. Approach (max 200 words)

What does your detector do, and why is that the right approach for this problem and this data? What did you consider and reject?

## 3. Experiment log

Fill this in as you go. Include experiments that did not work.

| # | What I expected and why | What I changed | Result (your validation F1, before → after) | Kept? |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |

How did you estimate performance, and why that way?

**My prediction for F1 on the hidden test set:** ___

Where does your detector still fail? Give two or three concrete examples (`pair_id`s from the training data).

## 4. Working with AI tools (max 200 words)

Your session export is the evidence for this section. Point to it where you can.

- Name up to three decisions you made yourself that changed the result. For each, what had the tool proposed?
- Describe one or two moments when the tool was wrong or misleading. Where in your session did you notice, and what did you do?
- Is there any part of your submission you would not be comfortable explaining line by line? Which?

## 5. With another day (max 150 words)

What would you do next, in order? What would have to change for this to work on Copyscape's real traffic?

## 6. Anything else (optional)

Anything you built, measured or noticed that the sections above do not cover.

## Assumptions

Anything you had to assume because the instructions were unclear.
