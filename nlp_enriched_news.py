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

SCANDAL_KEYWORDS = [
    "oil spill",
    "water pollution",
    "air pollution",
    "chemical contamination",
    "toxic chemical leak",
    "hazardous waste dumping",
    "industrial pollution",
    "illegal deforestation",
    "environmental damage",
    "ecological disaster",
]

# Strong evidence of environmental harm. Generic topic words such as "oil",
# "water", and "air" are intentionally absent to avoid market/news false positives.
HARM_LEMMAS = {
    "contaminate",
    "contamination",
    "deforest",
    "deforestation",
    "discharge",
    "dump",
    "hazardous",
    "leak",
    "poison",
    "pollutant",
    "pollute",
    "pollution",
    "sewage",
    "spill",
    "toxic",
}
HARM_PHRASES = {
    "ecological damage",
    "environmental damage",
    "hazardous waste",
    "oil slick",
}
IGNORED_ORGS = {"euronews"}


def load_nlp(name):
    try:
        return spacy.load(name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{name}' is missing. Install requirements.txt first."
        ) from exc


def load_sentiment():
    try:
        return SentimentIntensityAnalyzer()
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
        return SentimentIntensityAnalyzer()


def normalize_org(value):
    return " ".join(value.casefold().split()).strip(" .")


def is_ignored_org(value):
    normalized = normalize_org(value)
    return normalized in IGNORED_ORGS or normalized.startswith("euronews ")


def cosine(left, right):
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return 0.0 if denominator == 0 else float(np.dot(left, right) / denominator)


def build_keyword_vectors(nlp):
    vectors = [nlp(keyword).vector for keyword in SCANDAL_KEYWORDS]
    vectors = [vector for vector in vectors if np.linalg.norm(vector) > 0]
    if not vectors:
        raise RuntimeError(
            "The spaCy model has no usable vectors; use en_core_web_md."
        )
    return vectors


def sentence_has_harm_marker(sentence):
    lemmas = {
        token.lemma_.casefold()
        for token in sentence
        if token.is_alpha
    }
    if lemmas & HARM_LEMMAS:
        return True

    lowered = sentence.text.casefold()
    return any(phrase in lowered for phrase in HARM_PHRASES)


def sentence_organizations(sentence):
    return sorted(
        {
            ent.text.strip()
            for ent in sentence.ents
            if ent.label_ == "ORG"
            and ent.text.strip()
            and not is_ignored_org(ent.text)
        }
    )


def analyze_document(nlp, text, keyword_vectors):
    doc = nlp(text)
    organizations = sorted(
        {
            ent.text.strip()
            for ent in doc.ents
            if ent.label_ == "ORG"
            and ent.text.strip()
            and not is_ignored_org(ent.text)
        }
    )

    best_distance = 2.0
    best_org = None
    best_sentence = None

    for sentence in doc.sents:
        orgs = sentence_organizations(sentence)
        if not orgs or np.linalg.norm(sentence.vector) == 0:
            continue

        semantic_similarity = max(
            cosine(sentence.vector, vector) for vector in keyword_vectors
        )
        semantic_similarity = float(np.clip(semantic_similarity, 0.0, 1.0))
        semantic_distance = 1.0 - semantic_similarity

        # All entity-containing sentences are compared semantically as required.
        # A sentence without an explicit environmental-harm marker is placed in
        # the [1, 2] distance band, while harm-bearing sentences stay in [0, 1].
        # This prevents generic oil/water/air business stories from outranking
        # actual pollution, spills, contamination, or deforestation.
        if sentence_has_harm_marker(sentence):
            adjusted_distance = semantic_distance
        else:
            adjusted_distance = 1.0 + semantic_distance

        if adjusted_distance < best_distance:
            best_distance = adjusted_distance
            best_org = orgs[0]
            best_sentence = sentence.text.strip()

    return doc, organizations, float(best_distance), best_org, best_sentence


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


def article_sentiment(analyzer, doc):
    scores = [
        analyzer.polarity_scores(sentence.text)["compound"]
        for sentence in doc.sents
        if sentence.text.strip()
    ]
    if not scores:
        return 0.0
    return float(np.mean(scores))


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
        (
            doc,
            organizations,
            scandal_distance,
            scandal_org,
            scandal_sentence,
        ) = analyze_document(nlp, document, keyword_vectors)
        print(
            f"Detected {len(organizations)} companies/organizations: "
            + ", ".join(organizations[:10])
        )

        print("---------- Topic detection ----------")
        print("Text preprocessing ...")
        topic = str(classifier.predict([document])[0])
        print(f"The topic of the article is: {topic}")

        print("---------- Sentiment analysis ----------")
        sentiment_score = article_sentiment(sentiment, doc)
        print(
            f"The article {article['headline']} has a "
            f"{sentiment_label(sentiment_score)} sentiment "
            f"({sentiment_score:.3f})"
        )

        print("---------- Scandal detection ----------")
        print("Computing embeddings and distance ...")
        print(f"Environmental scandal distance: {scandal_distance:.4f}")
        if scandal_org:
            print(f"Closest candidate organization: {scandal_org}")

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
                "_Scandal_org": scandal_org,
                "_Scandal_sentence": scandal_sentence,
            }
        )

    result = pd.DataFrame(rows)
    top_indices = result.nsmallest(10, "Scandal_distance").index
    result.loc[top_indices, "Top_10"] = True

    print("\n---------- Top environmental scandal candidates ----------")
    for _, row in result.loc[top_indices].sort_values(
        "Scandal_distance"
    ).iterrows():
        organization = row["_Scandal_org"]
        if organization:
            print(f"Environmental scandal detected for {organization}")
            print(f"  {row['Headline']}")
            if row["_Scandal_sentence"]:
                print(f"  Evidence: {row['_Scandal_sentence']}")

    output = result.drop(columns=["_Scandal_org", "_Scandal_sentence"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"\nSaved {len(output)} enriched articles to {args.output}")
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
