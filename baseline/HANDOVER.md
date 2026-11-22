# Handover note: reuse detection baseline

Put this together on Thursday afternoon with an AI assistant. Summary:

- `features.py`: text cleaning, hashed word 3-gram shingles, Jaccard and containment similarity, plus a filter that learns boilerplate shingles (cookie banners and so on) from the data and ignores them.
- `train.py`: builds the similarity features, adds TF-IDF vectors of both texts, trains a logistic regression, tunes the decision threshold, and saves everything to `model.pkl`.
- `predict.py`: loads `model.pkl` and writes `pair_id,score,label`.

Results: **validation F1 0.915**, on a held-out 20% of the data. Training takes a couple of seconds.

The assistant reviewed the code afterwards and said it follows best practice (stratified split, fixed random seed, no magic numbers). I think it is ready to ship. I did not get round to trying it on anything other than the training file.

Run it from the top-level folder:

```
python baseline/train.py
python baseline/predict.py --input data/sample_input.csv --output preds.csv
```
