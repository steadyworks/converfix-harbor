"""
Reference solution: uses training data to learn sentiment-phrase extraction.

For neutral tweets the selected_text is almost always the full text, so we
return it as-is.  For positive/negative tweets we find the most similar
training example (by word overlap with the full tweet text) that shares the
same sentiment and return its selected_text pattern mapped back onto the
test tweet.  A TF-IDF nearest-neighbour approach provides a strong baseline.
"""

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = Path("/home/data")
SUBMISSION_DIR = Path("/home/submission")
SUBMISSION_PATH = SUBMISSION_DIR / "submission.csv"


def extract_by_nearest_neighbour(
    test_texts: list[str],
    test_sentiments: list[str],
    train_df: pd.DataFrame,
) -> list[str]:
    """For each test tweet, find the closest training tweet with the same
    sentiment and return the training example's selected_text."""

    results: list[tuple[str, int]] = []

    for sentiment in ["positive", "negative", "neutral"]:
        train_subset = train_df[train_df["sentiment"] == sentiment].reset_index(drop=True)
        test_indices = [
            i for i, s in enumerate(test_sentiments) if s == sentiment
        ]
        test_subset = [test_texts[i] for i in test_indices]

        if not test_subset or train_subset.empty:
            for _ in test_subset:
                results.append(("", -1))
            continue

        if sentiment == "neutral":
            # For neutral, selected_text ≈ full text in the training data
            for idx in test_indices:
                results.append((test_texts[idx], idx))
            continue

        corpus = train_subset["text"].fillna("").tolist() + test_subset
        vectorizer = TfidfVectorizer(max_features=5000, stop_words=None)
        tfidf = vectorizer.fit_transform(corpus)

        n_train = len(train_subset)
        train_tfidf = tfidf[:n_train]  # type: ignore[index]
        test_tfidf = tfidf[n_train:]  # type: ignore[index]

        sims = cosine_similarity(test_tfidf, train_tfidf)

        for local_j, global_idx in enumerate(test_indices):
            best = sims[local_j].argmax()
            selected = str(train_subset.iloc[best]["selected_text"])
            # If the nearest-neighbour's selected_text appears in the test
            # tweet, use it; otherwise fall back to the full test tweet.
            # If the NN's selected_text appears in the test tweet, extract
            # the original-case substring; otherwise fall back to full text.
            test_lower = test_texts[global_idx].lower()
            sel_lower = selected.lower().strip()
            if sel_lower and sel_lower in test_lower:
                start = test_lower.find(sel_lower)
                selected = test_texts[global_idx][start : start + len(selected.strip())]
            else:
                selected = test_texts[global_idx]
            results.append((selected, global_idx))

    # Sort results back into original test order
    results.sort(key=lambda x: x[1])
    return [r[0] for r in results]


def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    test_df = pd.read_csv(DATA_DIR / "test.csv")

    test_texts = test_df["text"].fillna("").tolist()
    test_sentiments = test_df["sentiment"].fillna("neutral").tolist()

    predictions = extract_by_nearest_neighbour(test_texts, test_sentiments, train_df)

    submission = test_df[["textID"]].copy()
    submission["selected_text"] = predictions

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
