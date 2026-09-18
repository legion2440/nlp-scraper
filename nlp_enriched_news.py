#!/usr/bin/env python3
"""Enrich scraped news with NER, topic, sentiment, and scandal scores."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import joblib
import nltk
import numpy as np
import pandas as pd
import spacy
from nltk.sentiment import SentimentIntensityAnalyzer

KEYWORDS = [
    "oil spill",
    "water pollution",
    "air pollution",
    "chemical contamination",
    "industrial pollution",
    "deforestation",
    "environmental disaster",
    "ecological damage",
]


def load_nlp(name):
    try:
        return spacy.load(name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{name}' is missing. "
            f"Run: python -m spacy download {name}"
        ) from exc


def load_sentiment():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)


def build_keyword_vectors(nlp):
    vectors = [nlp(keyword).vector for keyword in KEYWORDS]
    vectors = [vector for vector in vectors if np.linalg.norm(vector) > 0]
    if not vectors:
        raise RuntimeError(
            "The spaCy model has no usable vectors; use en_core_web_md."
        )
    return vectors


def analyze_document(nlp, text, keyword_vectors):
    doc = nlp(text)
    organizations = sorted(
        {
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "ORG" and ent.text.strip()
        }
    )

    best_similarity = 0.0
    for sentence in doc.sents:
        if not any(ent.label_ == "ORG" for ent in sentence.ents):
            continue
        if np.linalg.norm(sentence.vector) == 0:
            continue
        best_similarity = max(
            best_similarity,
            max(cosine(sentence.vector, vector) for vector in keyword_vectors),
        )

    return organizations, float(np.clip(1.0 - best_similarity, 0.0, 2.0))


def load_articles(database, limit):
    if not database.exists():
        raise FileNotFoundError(
            f"{database} does not exist. Run python scraper_news.py first."
        )
    with sqlite3.connect(database) as connection:
        frame = pd.read_sql_query(
            """
            SELECT article_id, url, date_scraped, headline, body
            FROM articles
            ORDER BY date_scraped ASC
            LIMIT ?
            """,
            connection,
            params=(limit,),
        )
    if len(frame) < limit:
        raise RuntimeError(
            f"Expected {limit} articles, found only {len(frame)}."
        )
    return frame


def sentiment_label(score):
    if score >= 0.05:
        return "positive"
    if score <= -0.05:
        return "negative"
    return "neutral"


def enrich(args):
    articles = load_articles(args.db, args.limit)
    classifier = joblib.load(args.model)
    nlp = load_nlp(args.spacy_model)
    sentiment = load_sentiment()
    keyword_vectors = build_keyword_vectors(nlp)

    rows = []
    for _, article in articles.iterrows():
        document = f"{article['headline']}\n{article['body']}"
        print(f"\nEnriching {article['url']}:")

        print("---------- Detect entities ----------")
        organizations, scandal_distance = analyze_document(
            nlp, document, keyword_vectors
        )
        print(
            f"Detected {len(organizations)} companies/organizations: "
            + ", ".join(organizations[:10])
        )

        print("---------- Topic detection ----------")
        topic = str(classifier.predict([document])[0])
        print(f"The topic of the article is: {topic}")

        print("---------- Sentiment analysis ----------")
        sentiment_score = float(
            sentiment.polarity_scores(document)["compound"]
        )
        print(
            f"The article has a {sentiment_label(sentiment_score)} sentiment "
            f"({sentiment_score:.3f})"
        )

        print("---------- Scandal detection ----------")
        print(f"Environmental scandal distance: {scandal_distance:.4f}")

        rows.append(
            {
                "Unique ID": article["article_id"],
                "URL": article["url"],
                "Date scraped": article["date_scraped"],
                "Headline": article["headline"],
                "Body": article["body"],
                "Org": json.dumps(organizations, ensure_ascii=False),
                "Topics": json.dumps([topic]),
                "Sentiment": sentiment_score,
                "Scandal_distance": scandal_distance,
                "Top_10": False,
            }
        )

    result = pd.DataFrame(rows)
    result.loc[
        result.nsmallest(10, "Scandal_distance").index, "Top_10"
    ] = True

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"\nSaved {len(result)} enriched articles to {args.output}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/news.db"))
    parser.add_argument(
        "--model", type=Path, default=Path("topic_classifier.pkl")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/enhanced_news.csv"),
    )
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--spacy-model", default="en_core_web_md")
    return parser


if __name__ == "__main__":
    raise SystemExit(enrich(build_parser().parse_args()))
