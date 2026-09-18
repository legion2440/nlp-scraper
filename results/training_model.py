#!/usr/bin/env python3
"""Train and evaluate the BBC News topic classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

TRAIN_URL = "https://raw.githubusercontent.com/01-edu/public/master/subjects/ai/nlp-scraper/bbc_news_train.csv"
TEST_URL = "https://raw.githubusercontent.com/01-edu/public/master/subjects/ai/nlp-scraper/bbc_news_tests.csv"


def download_if_missing(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"Downloaded {path}")


def load_dataset(
    path: Path,
    *,
    deduplicate: bool = False,
) -> tuple[pd.Series, pd.Series]:
    frame = pd.read_csv(path)
    required = {"Text", "Category"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["Text", "Category"])

    if deduplicate:
        before = len(frame)
        frame = frame.drop_duplicates(subset=["Text"], keep="first")
        removed = before - len(frame)
        print(f"Removed {removed} duplicate training texts")

    frame = frame.reset_index(drop=True)
    return frame["Text"].astype(str), frame["Category"].astype(str)


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_df=0.95,
                    sublinear_tf=True,
                    max_features=60_000,
                ),
            ),
            ("classifier", LinearSVC(C=1.5)),
        ]
    )


def save_learning_curves(
    model: Pipeline,
    texts: pd.Series,
    labels: pd.Series,
    output: Path,
) -> None:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_sizes, train_scores, validation_scores = learning_curve(
        model,
        texts,
        labels,
        cv=cv,
        scoring="accuracy",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1,
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    validation_mean = validation_scores.mean(axis=1)
    validation_std = validation_scores.std(axis=1)

    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, marker="o", label="Training accuracy")
    plt.plot(
        train_sizes,
        validation_mean,
        marker="o",
        label="Cross-validation accuracy",
    )
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.15,
    )
    plt.fill_between(
        train_sizes,
        validation_mean - validation_std,
        validation_mean + validation_std,
        alpha=0.15,
    )
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("BBC News topic classifier learning curves")
    plt.ylim(0.80, 1.01)
    plt.grid(alpha=0.2)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()

    gap = train_mean[-1] - validation_mean[-1]
    print(
        "Learning curve at full training size: "
        f"train={train_mean[-1]:.2%}, "
        f"cv={validation_mean[-1]:.2%}, "
        f"gap={gap:.2%}"
    )


def train(args: argparse.Namespace) -> int:
    download_if_missing(args.train, TRAIN_URL)
    download_if_missing(args.test, TEST_URL)

    x_train, y_train = load_dataset(args.train, deduplicate=True)
    x_test, y_test = load_dataset(args.test)

    model = build_classifier()
    save_learning_curves(model, x_train, y_train, args.learning_curves)

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Test accuracy: {accuracy:.4%}")
    print(classification_report(y_test, predictions, digits=4))

    train_texts = set(x_train.tolist())
    non_overlap_mask = ~x_test.isin(train_texts).to_numpy()
    overlap_count = int((~non_overlap_mask).sum())
    if non_overlap_mask.any():
        non_overlap_accuracy = accuracy_score(
            y_test.to_numpy()[non_overlap_mask],
            predictions[non_overlap_mask],
        )
        print(
            "Test accuracy excluding train/test text overlap: "
            f"{non_overlap_accuracy:.4%} "
            f"({overlap_count} overlapping test rows excluded)"
        )

    if accuracy < args.min_accuracy:
        raise RuntimeError(
            f"Accuracy {accuracy:.4%} is below required {args.min_accuracy:.2%}"
        )

    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"Saved classifier to {args.model}")
    print(f"Saved learning curves to {args.learning_curves}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train",
        type=Path,
        default=Path("data/bbc_news_train.csv"),
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=Path("data/bbc_news_tests.csv"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("topic_classifier.pkl"),
    )
    parser.add_argument(
        "--learning-curves",
        type=Path,
        default=Path("results/learning_curves.png"),
    )
    parser.add_argument("--min-accuracy", type=float, default=0.95)
    return parser


if __name__ == "__main__":
    raise SystemExit(train(build_parser().parse_args()))
