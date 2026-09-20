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

# These markers are specific enough to count as environmental-harm evidence
# without extra context.
STRONG_HARM_LEMMAS = {
    "contaminate",
    "contamination",
    "deforest",
    "deforestation",
    "pollutant",
    "pollute",
    "pollution",
    "sewage",
}

# These words are ambiguous in general news. They count as harm evidence only
# when an environmental-context term occurs in the same sentence.
AMBIGUOUS_HARM_LEMMAS = {
    "damage",
    "discharge",
    "dump",
    "hazardous",
    "leak",
    "poison",
    "slick",
    "spill",
    "toxic",
}

ENVIRONMENT_CONTEXT_LEMMAS = {
    "air",
    "beach",
    "chemical",
    "coast",
    "ecological",
    "ecosystem",
    "environment",
    "environmental",
    "farmland",
    "forest",
    "gas",
    "groundwater",
    "habitat",
    "lake",
    "land",
    "ocean",
    "oil",
    "pipeline",
    "river",
    "sea",
    "soil",
    "waste",
    "water",
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


def clean_org(value):
    text = " ".join(value.split()).strip(" .")
    if text.casefold().startswith("the "):
        text = text[4:].strip()
    if text.endswith("'s") or text.endswith("’s"):
        text = text[:-2].rstrip()
    return text.strip(" .")


def org_key(value):
    return clean_org(value).casefold()


def is_ignored_org(value):
    normalized = org_key(value)
    return normalized in IGNORED_ORGS or normalized.startswith("euronews ")


def normalized_orgs(entities):
    by_key = {}
    for ent in entities:
        if ent.label_ != "ORG" or not ent.text.strip() or is_ignored_org(ent.text):
            continue
        cleaned = clean_org(ent.text)
        if cleaned:
            by_key.setdefault(cleaned.casefold(), cleaned)
    return sorted(by_key.values(), key=str.casefold)


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


def harm_marker_positions(sentence):
    # Ignore words inside named entities when deciding environmental context.
    # This prevents names such as "Organisation for the Prohibition of
    # Chemical Weapons" from turning "poison" into an environmental match.
    tokens = [
        token
        for token in sentence
        if token.is_alpha and token.ent_iob_ == "O"
    ]
    lemmas = {token.lemma_.casefold() for token in tokens}

    positions = [
        token.i
        for token in tokens
        if token.lemma_.casefold() in STRONG_HARM_LEMMAS
    ]

    if lemmas & ENVIRONMENT_CONTEXT_LEMMAS:
        positions.extend(
            token.i
            for token in tokens
            if token.lemma_.casefold() in AMBIGUOUS_HARM_LEMMAS
        )

    return sorted(set(positions))


def sentence_org_entities(sentence):
    entities = []
    seen = set()
    for ent in sentence.ents:
        if ent.label_ != "ORG" or not ent.text.strip() or is_ignored_org(ent.text):
            continue
        cleaned = clean_org(ent.text)
        key = cleaned.casefold()
        if not cleaned or key in seen:
            continue
        seen.add(key)
        entities.append((cleaned, ent.start, ent.end))
    return entities


def marker_distance_to_entity(marker_position, entity_start, entity_end):
    if marker_position < entity_start:
        return entity_start - marker_position
    if marker_position >= entity_end:
        return marker_position - entity_end + 1
    return 0


def closest_org_to_markers(org_entities, marker_positions):
    if not org_entities or not marker_positions:
        return None

    best = min(
        org_entities,
        key=lambda item: min(
            marker_distance_to_entity(marker, item[1], item[2])
            for marker in marker_positions
        ),
    )
    return best[0]


def build_document(headline, body):
    headline = str(headline).strip()
    if headline and headline[-1] not in ".!?…":
        headline += "."
    return f"{headline}\n{body}"


def analyze_document(nlp, text, keyword_vectors):
    doc = nlp(text)
    organizations = normalized_orgs(doc.ents)

    best_distance = 2.0
    best_org = None
    best_sentence = None
    best_has_harm_evidence = False

    for sentence in doc.sents:
        org_entities = sentence_org_entities(sentence)
        if not org_entities or np.linalg.norm(sentence.vector) == 0:
            continue

        semantic_similarity = max(
            cosine(sentence.vector, vector) for vector in keyword_vectors
        )
        semantic_similarity = float(np.clip(semantic_similarity, 0.0, 1.0))
        semantic_distance = 1.0 - semantic_similarity

        marker_positions = harm_marker_positions(sentence)
        has_harm_evidence = bool(marker_positions)

        # Every ORG-containing sentence is compared semantically as required.
        # Sentences with explicit harm evidence remain in [0, 1]. Others receive
        # a +1 penalty and therefore stay in [1, 2].
        if has_harm_evidence:
            adjusted_distance = semantic_distance
            candidate_org = closest_org_to_markers(
                org_entities, marker_positions
            )
        else:
            adjusted_distance = 1.0 + semantic_distance
            candidate_org = None

        if adjusted_distance < best_distance:
            best_distance = adjusted_distance
            best_org = candidate_org
            best_sentence = sentence.text.strip()
            best_has_harm_evidence = has_harm_evidence

    return (
        doc,
        organizations,
        float(best_distance),
        best_org,
        best_sentence,
        best_has_harm_evidence,
    )


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
        document = build_document(article["headline"], article["body"])
        print(f"\nEnriching {article['url']}:")

        print("---------- Detect entities ----------")
        (
            doc,
            organizations,
            scandal_distance,
            closest_org,
            evidence_sentence,
            has_harm_evidence,
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
        if has_harm_evidence:
            print("Environmental-harm candidate")
            if closest_org:
                print(f"Closest organization in evidence: {closest_org}")
        else:
            print("No explicit environmental-harm evidence")

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
                "_Closest_org": closest_org,
                "_Evidence_sentence": evidence_sentence,
                "_Has_harm_evidence": has_harm_evidence,
            }
        )

    result = pd.DataFrame(rows)
    top_indices = result.nsmallest(10, "Scandal_distance").index
    result.loc[top_indices, "Top_10"] = True

    print("\n---------- Top environmental-harm candidates ----------")
    for _, row in result.loc[top_indices].sort_values(
        "Scandal_distance"
    ).iterrows():
        print(f"  {row['Headline']}")
        print(f"  Distance: {row['Scandal_distance']:.4f}")
        if row["_Has_harm_evidence"]:
            print("  Environmental-harm candidate")
            if row["_Closest_org"]:
                print(
                    "  Closest organization in evidence: "
                    f"{row['_Closest_org']}"
                )
            if row["_Evidence_sentence"]:
                print(f"  Evidence: {row['_Evidence_sentence']}")
        else:
            print("  No explicit environmental-harm evidence")

    output = result.drop(
        columns=[
            "_Closest_org",
            "_Evidence_sentence",
            "_Has_harm_evidence",
        ]
    )
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
