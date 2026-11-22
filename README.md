# Reuse or coincidence?

**Copyscape coding exercise: Algorithms & AI Developer (student position)**

**Time limit: 3 hours**, from the moment you receive this pack to the moment you send your submission. Late submissions are still read, but we note the overrun.

Use any AI coding tools you like (Claude Code, Cursor, Copilot, ChatGPT and so on). That is how we work, and it is part of what we want to see. A capable coding agent can produce a working detector quickly, so the detector on its own will not tell us much about you. We are more interested in what you notice, what you check, what you decide, and what you make of the problem.

## The problem

Two web pages say similar things. Did one copy from the other, or did two people simply write about the same subject?

You get pairs of texts: a **source** and a **candidate**. For each pair, decide:

- **1 (reused):** the candidate was produced by copying from the source text, in whole or in part, possibly followed by editing to disguise it. Copying a single sentence into an otherwise original text counts.
- **0 (independent):** the candidate was written without copying from this source. This includes texts about the same subject that state the same facts, names and numbers in their own words, and texts copied from somewhere else.

The question is about shared wording, not shared meaning.

This is a small, self-contained version of a question Copyscape answers every day at a much larger scale, where a page is checked against billions of others, usually without knowing in advance which one it might have been copied from.

## The data

`data/train.csv` has 1,835 labelled pairs:

| column | meaning |
|---|---|
| `pair_id` | unique id of the row |
| `source_id` | id of the article the source text comes from |
| `source_text` | the original text |
| `candidate_text` | the text being checked |
| `label` | 1 = reused, 0 = independent |

The texts were collected from web pages and are as messy as web text usually is. The labels are correct.

We will score your work on a **hidden test set** of about 700 pairs. It was produced by the same process as the training data, from articles that do not appear in the training data. You will not see it.

`data/sample_input.csv` shows the format your code will be given (the same columns without `label`).

## What to do

### Part A: Review the baseline (suggested: 40 minutes)

`baseline/` contains a detector that a previous developer built in an afternoon with an AI assistant. Their note is in `baseline/HANDOVER.md`. It reports a validation F1 of 0.915. When we ran it on fresh data it did far worse, and nobody has had time to find out why.

Work out what is wrong with it. For each problem you find, we want to know:

1. what the problem is and where it is,
2. how you confirmed it is real (not just that it looks suspicious),
3. roughly how much it matters.

There are several problems. Some are in the code, some only show up when you look at the data, and they are not equally important. You do not have to fix the baseline, although you may reuse any part of it in Part B.

### Part B: Build a better detector (suggested: 1 hour 45 minutes)

Build the best detector you can in the time. The approach is entirely up to you: rules, classical algorithms, machine learning, or a mix.

Your submission must include a `predict.py` at the top level that we can run like this:

```
python predict.py --input some_pairs.csv --output predictions.csv
```

`predictions.csv` must have one row for **every** input row, with these columns:

```
pair_id,score,label
te_00001,0.9312,1
te_00002,0.0410,0
```

`score` is a number between 0 and 1 (higher means more likely reused) and `label` is your final 0/1 decision.

Rules for `predict.py`:

- **Offline and CPU only.** No network calls, no LLM or other API calls, and no downloaded pretrained models. At Copyscape's scale a per-pair model call is not affordable, and we score submissions in a sandbox without network access.
- **Python 3.10+** for `predict.py` itself, with `numpy`, `pandas`, `scipy` and `scikit-learn` available. If you need another pip package, list it in `requirements.txt` and keep it small. You may write the detector in any language and have `predict.py` call it. If you do, document the build steps and anything we need to install (standard, freely available tools only).
- **Fast enough:** the prediction run must finish within 3 minutes for 1,000 pairs on an ordinary laptop. Any build step does not count towards this.
- **It must not fall over.** The input is untrusted web text. Expect empty fields, markup, unusual characters, and the occasional very long document (up to about 25,000 words). A missing row counts as a wrong answer, and a crash scores zero.
- **Same input, same output.** Running `predict.py` twice on the same file must give identical predictions.
- If your approach needs training, include the training script and commit the trained artefact, so that `predict.py` works straight after we unpack your submission. We should be able to re-run your training and get the same result.

These rules are what we need in order to run and score your detector. They are not a limit on what you build around it.

The score is **F1 for label 1 on the hidden test set**. `tools/evaluate.py` computes the same metric locally:

```
python tools/evaluate.py --pred predictions.csv --truth data/train.csv
```

You have only the training file, so how you estimate your real performance is up to you, and is part of the exercise.

### Part C: Write it up (suggested: 25 minutes)

Fill in `REPORT.md`. It has five short sections with word limits: your baseline review, your approach, your experiment log, how you worked with AI tools, and what you would do next. A sixth section is optional. Keep the experiment log as you go rather than reconstructing it at the end.

Please do not skip this part to squeeze out a slightly better score. A clear report with a decent score beats a better score we cannot understand.

## What to send

Reply to the email that brought you this pack with a zip file (or, if it is too large to attach, a Google Drive link to it) containing:

- `predict.py`, plus any training code and trained artefacts
- `REPORT.md`
- `requirements.txt` if you added dependencies
- an export of your AI tool session(s). If your tool cannot export, include your prompts. A messy session is fine; a missing one is not.
- anything else you think is worth showing us

If you use git, commit as you go and keep the history, and include the `.git` folder in your zip. We like to see how the work developed.

## How we evaluate

We look at five things, in roughly equal measure:

1. **Speed and reliability.** How far you got in three hours, and whether your code runs correctly on data it has not seen.
2. **Your own strategy.** Whether the approach fits this particular problem and this particular data, and whether you can say why you chose it.
3. **Understanding.** Whether you understand the code you are submitting, whoever or whatever wrote it.
4. **Catching mistakes.** Whether you notice when a tool, a baseline or your own first idea is wrong, and what you do about it.
5. **Iteration.** Whether each experiment taught you something that shaped the next one.

The hidden-test score tells us whether your detector works. Near the top, small differences in it are mostly luck, so we don't put much weight on them. What separates strong submissions is what you understood and what you decided, and we may explore that in a follow-up call.

Candidates who do well may be invited to a follow-up call. We might ask you to walk us through your submission, predict what your detector does on a few new pairs, explain parts of the code that we pick, or find and fix a small problem in it, without AI assistance. Nothing in it is a trick, but you do need to know your own submission.

## Practical notes

- You may not share this exercise or ask another person for help. AI tools are fine; other humans are not.
- Do not try to obtain the hidden test set or its labels.
- If something in these instructions is ambiguous, make a reasonable assumption, write it down in `REPORT.md`, and carry on.
- Running the baseline: `python baseline/train.py`, then `python baseline/predict.py --input data/sample_input.csv --output preds.csv` (both from the top-level folder).

Good luck. We hope you enjoy it.
