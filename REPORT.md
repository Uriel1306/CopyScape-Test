# Report

Name:

Time you started / finished: 17:00 20:00

AI tools used: Claude Code Pro


## 1. Baseline review

*One row per problem, most important first. "How I confirmed it" should be something you ran or looked at, not only something you read in the code.*

All the evidence comes from one script, `analysis/partA_checks.py` (run it with `python -X utf8 analysis/partA_checks.py`). It runs 18 checks, prints a verdict for each, and computes every key number in two independent ways. The number in brackets, like [3], is the check number.

| # | Problem | Where | How I confirmed it | How much it matters |
|---|---|---|---|---|
| 1 | The boilerplate filter deletes real content. It counts rows, and one article has up to 86 rows. | `features.py:25-30` | Ran it on `train.csv` [3]. It removes 74.4% of source 3-grams, but only about 9% look like footers. 307 of 574 copied pairs lose every shared n-gram. | Very high. All overlap features use what is left. |
| 2 | The 0.915 comes from a random row split. The same article is on both sides, and the test has new articles. | `train.py:50` | Rebuilt the split [1]. With random rows, every validation article is also in training. A predictor that sees only the article id gets F1 0.62 (and 0.00 on new articles). I also ran the baseline under both splits [2]. I reproduced 0.915, but that is weighted F1. For class 1 it is 0.861 on that split, 0.891 on random rows and 0.827 on new articles. | High. On new articles the real score is about 0.06 lower. |
| 3 | TF-IDF of both texts (thousands of columns, C=20) lets the model recognise articles. It measures topic, not wording. | `train.py:41-46, 53` | TF-IDF cosine alone gets F1 0.68, against 0.80 for containment [12]. I removed the TF-IDF columns from the baseline [2]. On random rows F1 goes 0.891 → 0.808, but on new articles 0.827 → 0.839. | High. It explains why the random split looks good. |
| 4 | `clean()` deletes characters, and the data hides copies with look-alike letters (427 candidates). | `features.py:12-16` | "Lаne" (Cyrillic а) becomes "lne" [7]. With replace, NFKC and a look-alike map, mean containment on copies goes 0.444 → 0.511, and single-feature F1 goes 0.755 → 0.795. | High. The disguise lowers overlap exactly on the copied texts. |
| 5 | The metric is weighted F1, and the threshold is tuned and reported on the same data. | `train.py:60-64` | Same predictions: weighted F1 is 0.849, class-1 F1 is 0.755 [6]. | Medium. The reported score is inflated. |
| 6 | No feature for partial copying. One copied passage in a long text gives a tiny score. | `features.py:47-58` | 424 of 574 copies have Jaccard below 0.10 [9]. The median longest shared run is 18 words for copies and 4 for the rest. That run alone gives F1 0.76. | Medium to high. It partly overlaps with row 1. |
| 7 | `hash()` changes in every Python process, but the footer list is saved as hashes. | `features.py:22` | Trained the baseline and opened `model.pkl` in a new process [4]. 0 of 5,031 saved hashes match. With the same hash seed all of them match. | Medium. The saved filter does nothing in `predict.py`. |
| 8 | `predict.py` re-implements the features and flips the arguments of `containment`. | `predict.py:28` and `features.py:55` | Code check [5]: `containment(c, s)` against `containment(s, c)`. The values differ by more than 0.10 on 8.4% of pairs, and F1 goes 0.755 → 0.739 at the same threshold. | Low to medium. A real bug, with a small effect. |
| 9 | `dropna` removes rows, so the output can be shorter than the input. | `train.py:30`, `predict.py:49` | 5 rows have an empty candidate [8]. The real `baseline/predict.py` turns an 11-row file into 10 rows. | Low for the score, because all 5 are label 0. It breaks the one-row-per-input rule. |
| 10 | Smaller issues: the footer list and TF-IDF are fitted before the split [10]. No scaling and no class weights [14]. None or NaN input raises `AttributeError` [15]. Only tested on the training file [16]. Relative paths [17]. | `train.py`, `features.py`, `HANDOVER.md` | Code analysis with line numbers, and 6 edge inputs. | Low. |

*Anything an AI tool flagged in the baseline that turned out to be wrong or unimportant?*

- Length as a shortcut: not confirmed. The best F1 from length alone is 0.479, and "everything is 1" gives 0.477 [13].
- The flipped `containment` is real but small (row 8). The 5 dropped rows are all negatives (row 9). `pickle` is not a problem here [18].
- Two first claims of the AI were wrong. It said the filter removes 82% of the content, but the real number is 74.4%. It also proposed counting footers by `source_id` over all texts. That is contaminated, because a copied candidate also appears in rows of other articles, so the count must use source texts only.

**Limits of this review**

- I did not fix the baseline.
- I ran checks 2 and 4 after I repaired scikit-learn (the disk was full the first time).
- The held-out numbers come from one 80/20 split by article per seed, so they move a few points (0.802 and 0.852).
- Single-feature F1 values use the best threshold on all of `train.csv`. I only use them to compare steps.


## 2. Approach (max 200 words)

*What does your detector do, and why is that the right approach for this problem and this data? What did you consider and reject?*

The detector measures how much exact wording the candidate shares with the source. The task is about wording, not meaning, so I look for runs of identical words and ignore topic.

I clean both texts (HTML, invisible characters, and look-alike Cyrillic or Greek letters mapped to Latin, because the data hides copies this way) and build word n-grams of 2, 3, 5 and 8 words with crc32, so runs are repeatable. N-grams found in the sources of 10 or more articles (stock footers) are masked.

Each pair becomes 33 numbers: overlap in both directions, IDF-weighted overlap, the longest shared run, the share of the candidate that is covered, and the best 30-word window. The window numbers catch one copied sentence in a long text. A HistGradientBoosting model gives the score, with a cut-off of 0.42 chosen on out-of-fold scores.

This fits the data: texts on the same subject share facts but almost never five words in a row. It is linear in length, so 1,000 pairs with 25,000-word documents take about 8 seconds.

Rejected: TF-IDF (it measures topic and lets the model memorise articles), Python's `hash()` (changes every run), full alignment (too slow), neural networks (too little data).


## 3. Experiment log

*Fill this in as you go. Include experiments that did not work.*

All numbers are F1 for class 1. Rows 5 to 12 use grouped cross-validation (5 folds, 3 seeds) for two models: `lr` (logistic regression) and `hgb` (HistGradientBoosting). Rows 2 to 4 use one feature with the best threshold on all of `train.csv`, so I only use them to compare steps.

| # | What I expected and why | What I changed | Result (F1, before → after) | Kept? |
|---|---|---|---|---|
| 1 | The 0.915 looked too good, because the split keeps the same article on both sides. | Ran the baseline with whole articles held out, and scored class 1. | 0.915 (weighted) → 0.827 (class 1, new articles) | No. This is the score to beat. |
| 2 | Shared wording should be a strong signal. | One feature: the share of the candidate's 3-grams found in the source. | 0.477 (everything is 1) → 0.755 | Yes |
| 3 | Copies hide behind look-alike letters, so cleaning should matter. | Replace instead of delete, NFKC, and a look-alike letter map. | 0.755 → 0.795 | Yes |
| 4 | Texts on the same subject share 3-grams by chance, and footers add fake matches. | 5-word n-grams, and footers masked with a list built from many articles. | 0.795 → 0.968 (`cand_cont5`) | Yes |
| 5 | I needed a real number. | Grouped CV on overlap features only (n = 2, 3, 5, 8). | lr 0.9667, hgb 0.9673 | Yes, as the base |
| 6 | IDF should make generic phrases count less. | Added 2 IDF-weighted features. | lr 0.9712, hgb 0.9679 | Yes. Small, inside the noise. |
| 7 | Overlap numbers dilute one copied sentence. The longest run and 30-word windows should catch it. | Added 12 run and window features and 3 length features. This is the submitted set. | lr 0.9682, hgb 0.9714 | Yes. The gain over overlap only is +0.007 (interval −0.001 to +0.015), so it is not proven. I kept it because one copied sentence counts. |
| 8 | Local features alone might be enough. | Only runs, windows and lengths. | lr 0.9652, hgb 0.9630 | No, slightly worse |
| 9 | Masking footers should matter. | Turned footer learning off. | lr 0.9594, hgb 0.9523 | Kept the masking. It is the only clear effect: +0.013 (interval +0.004 to +0.022). |
| 10 | Masking numbers and names, and character n-grams, should separate facts from wording. | Switched on both optional groups. | lr 0.9723, hgb 0.9710 | No. No gain for hgb, and they are slower. |
| 11 | A footer threshold of 10 articles might be too strict. | Used 5 instead of 10. | lr 0.9739, hgb 0.9712 | No. The difference (−0.003, interval −0.016 to +0.008) is noise. Keeping the best of many variants would make my estimate too optimistic. |
| 12 | Boosting should combine short runs and overlap better than a straight line. | Compared lr, hgb and their average. Took the simplest within 0.002 of the best. | lr 0.9682, hgb 0.9714, average 0.9691 → hgb | Yes |

*How did you estimate performance, and why that way?*

I used grouped cross-validation: 5 folds split by `source_id`, repeated with 3 seeds. A whole article is either in training or in validation. The hidden test has new articles, so this is the same situation.

In every fold the footer list and the IDF are learned from the training articles only. The cut-off 0.42 comes from the out-of-fold scores. The F1 curve is flat (0.964 to 0.972 for cut-offs from 0.2 to 0.7), so this choice adds very little optimism.

The ± printed by `train.py` is only the spread between seeds, so it is too small. Resampling articles gives 0.972 with a 95% interval of 0.956 to 0.984. I used the same resampling to compare variants.

**My prediction for F1 on the hidden test set:** about 0.96 (between 0.94 and 0.98). It is a little under the cross-validation score, because new articles may have footers I have not seen.

*Where does your detector still fail? Give two or three concrete examples (`pair_id`s from the training data).*

It makes 19 misses and 13 false alarms out of 1,835 pairs. All 19 missed copies are edited ones: none of them has a shared run of 8 words.

- `tr_01578` (score 0.001): one source sentence is copied into a text about a lighthouse, with three words swapped (carried → hauled, afterwards → later, nearly → almost). The longest shared run is 3 words.
- `tr_00148` (score 0.003): a light paraphrase of the whole text (measure → gauge, every → each). No 5-word match.
- `tr_00514` (score 0.993, but independent): a text about the same composer with the same facts. It shares 6-word phrases like "the daughter of a church organist". 7 of the 13 false alarms come from this article (`src_4789`), and its F1 is 0.63.

Both kinds of error look alike in word overlap. The median bigram overlap is 0.29 for missed copies and 0.37 for false alarms, against 0.66 for found copies and 0.18 for true negatives. Exact n-grams cannot tell a copy with swapped words from independent writing of the same facts.


## 4. Working with AI tools (max 200 words)

*Your session export is the evidence for this section. Point to it where you can.*

*Name up to three decisions you made yourself that changed the result. For each, what had the tool proposed?*

1. I did not trust the tool's claims. I asked for every one to be tested in code and checked twice. The tool had written checks for only some of them.
2. Before the test I tried to guess the problem and designed a similar system with an AI, then turned it into a blueprint for a separate build session. The tool had proposed a smaller plan (overlap features, longest run, small model). My blueprint was too big for three hours and had wrong details (`test/` folder, footers counted over all texts, slow `difflib`) that the tool corrected. Its multi-size n-grams and windows stayed; entity masking and character n-grams did not help.

*Describe one or two moments when the tool was wrong or misleading. Where in your session did you notice, and what did you do?*

The tool proposed counting footers by `source_id` over all texts, which is contaminated by copied candidates. I noticed when I asked why `source_id` matters and whether pairs are one to one. I measured it, and the code now counts source texts only (+0.013).

*Is there any part of your submission you would not be comfortable explaining line by line? Which?*

Because of the time limit and my limited background, I spent too much time understanding instead of building. I can still explain everything, because I needed to understand every part myself and design the system as well as I could.


## 5. With another day (max 150 words)

*What would you do next, in order? What would have to change for this to work on Copyscape's real traffic?*

Next steps, in order:

1. **Bigger dataset.** 1,835 pairs from 78 articles is small. I would collect more articles and label them with several strong LLMs, only to make training data and never inside `predict.py`.
2. **Catch edited copies.** All 19 missed copies are edited, with no shared 8-word run. I would add fuzzy matching, for example n-grams with gaps.
3. **Handle shared facts.** 7 of the 13 false alarms come from one article that restates the same facts. I would mask numbers and names and compare the wording around them.
4. **Test on held-out articles** that I never tune on.

For Copyscape's real traffic, comparing one pair at a time is not enough. I would index n-gram fingerprints of all pages, find candidate sources by lookup, and run this detector only on the top matches. The footer list would come from a large crawl, not from 78 articles.


## 6. Anything else (optional)

*Anything you built, measured or noticed that the sections above do not cover.*

- **Look-alike letters.** The data hides copies by swapping Latin letters for Cyrillic or Greek ones. Mapping them back raises a one-feature F1 from 0.755 to 0.795.
- **Robustness.** `predict.py` keeps every row on 12 odd inputs (empty, HTML, a 200,000-character word, quotes). 1,000 pairs, including 20 documents of 25,000 words, take about 8 seconds. The output is identical across runs and hash seeds, and two trainings give the same model.
- **Reproducing section 1.** `analysis/partA_checks.py` regenerates every number in it.


## Assumptions

*Anything you had to assume because the instructions were unclear.*

- The hidden test uses new articles from the same process, so I validated by whole articles.
- `predict.py` only uses `source_text` and `candidate_text`. Empty or missing text gets score 0 and label 0.
- `model.joblib` was saved with Python 3.14, scikit-learn 1.9.1 and numpy 2.4.2. If it does not load with other versions, run `python train.py` to rebuild it. `requirements.txt` does not pin versions.
- A copy with edited wording counts as reused, as the task says.
