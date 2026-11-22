"""Train the reuse-detection baseline.

Usage:
    python baseline/train.py --data data/train.csv --model baseline/model.pkl
"""
import argparse
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from features import clean, find_common_shingles, pair_features

SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--model", default="baseline/model.pkl")
    args = parser.parse_args()

    # 1. Load and clean the data
    df = pd.read_csv(args.data)
    df = df.dropna().reset_index(drop=True)
    print(f"Loaded {len(df)} pairs ({df.label.mean():.1%} positive)")

    # 2. Learn which shingles are boilerplate
    all_texts = pd.concat([df.source_text, df.candidate_text])
    common = find_common_shingles(all_texts)
    print(f"Found {len(common)} boilerplate shingles")

    # 3. Build features: similarity features + TF-IDF of both texts,
    #    so the model can also pick up on vocabulary and writing style
    sim = np.array([pair_features(s, c, common) for s, c in zip(df.source_text, df.candidate_text)])
    vectorizer = TfidfVectorizer(preprocessor=clean, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    vectorizer.fit(all_texts)
    tfidf_s = vectorizer.transform(df.source_text)
    tfidf_c = vectorizer.transform(df.candidate_text)
    cosine = np.asarray(tfidf_s.multiply(tfidf_c).sum(axis=1))
    X = hstack([csr_matrix(sim), csr_matrix(cosine), tfidf_s, tfidf_c]).tocsr()
    y = df.label.values

    # 4. Train / validation split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    # 5. Fit the classifier
    clf = LogisticRegression(C=20.0, max_iter=5000)
    clf.fit(X_train, y_train)

    # 6. Pick the decision threshold that maximises F1
    val_scores = clf.predict_proba(X_val)[:, 1]
    best_threshold, best_f1 = 0.5, 0.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        f1 = f1_score(y_val, val_scores >= threshold, average="weighted")
        if f1 > best_f1:
            best_threshold, best_f1 = float(threshold), f1
    print(f"Best threshold: {best_threshold:.2f}")
    print(f"Validation F1:  {best_f1:.3f}")

    # 7. Save everything predict.py needs
    with open(args.model, "wb") as f:
        pickle.dump({"clf": clf, "vectorizer": vectorizer, "common": common, "threshold": best_threshold}, f)
    print(f"Saved model to {args.model}")


if __name__ == "__main__":
    main()
