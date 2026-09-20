# NLP Scraper

An NLP-enriched news pipeline for the 01-edu AI track. It scrapes recent Euronews articles, trains a BBC-topic classifier, detects organizations, scores sentiment, ranks environmental-harm candidates, and exports a unified audit-ready CSV.

## 📋 TOC

- [🚀 Quick start](#-quick-start)
- [📝 About](#-about)
- [🔄 Pipeline](#-pipeline)
- [📰 News scraping](#-news-scraping)
- [🧠 Topic classifier](#-topic-classifier)
- [🔎 NLP enrichment](#-nlp-enrichment)
- [🌍 Environmental-harm ranking](#-environmental-harm-ranking)
- [📊 Generated artifacts](#-generated-artifacts)
- [📁 Project structure](#-project-structure)
- [⚠️ Known limitations](#️-known-limitations)
- [🧑‍💻 Author](#-author)

## 🚀 Quick start

### Requirements

- Python 3.11–3.13
- internet access for the news scraper and first BBC-dataset download

Create an environment and install the pinned dependencies:

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS / WSL
source .venv/bin/activate

python -m pip install -r requirements.txt
python -m pip check
```

The medium spaCy model is pinned directly in `requirements.txt`, so no separate model-download command is required.

Run the complete pipeline:

```bash
python scraper_news.py
python results/training_model.py
python nlp_enriched_news.py
```

## 📝 About

The project satisfies the main assignment flow:

- scrape at least 300 recent English-language news articles;
- store a stable UUID, URL, scrape date, headline, and body;
- detect `ORG` entities with spaCy;
- classify topics as `business`, `entertainment`, `politics`, `sport`, or `tech`;
- score sentiment with NLTK VADER;
- compute a unified environmental-harm distance;
- flag exactly ten articles with the smallest distance;
- save the enriched 300-row result as CSV.

Sentiment is averaged over article sentences rather than calculated once over the full document, which avoids whole-document VADER saturation on long news articles.

## 🔄 Pipeline

```text
Euronews sitemap
    |
    v
scraper_news.py
    |
    v
SQLite: data/news.db
    |
    +------------------------------+
    |                              |
    v                              v
results/training_model.py      nlp_enriched_news.py
    |                              ^
    v                              |
topic_classifier.pkl --------------+
    |
    v
results/learning_curves.png

nlp_enriched_news.py
    |
    v
results/enhanced_news.csv
```

## 📰 News scraping

Run:

```bash
python scraper_news.py
```

Defaults:

- source: Euronews English article sitemap;
- time window: last 7 days;
- target: 300 stored articles;
- storage: `data/news.db`.

The scraper:

- follows the Euronews sitemap index;
- skips malformed child sitemaps instead of aborting the complete run;
- checks publication dates against the requested window;
- excludes `/video/` pages before downloading because short bulletins and mixed-story video pages add noise to topic and sentiment analysis;
- normalizes headline whitespace;
- stores URLs uniquely in SQLite, making reruns restart-safe;
- stops once the database contains the requested total number of articles.

Example with a wider audit window if a quiet news week does not provide enough non-video articles:

```bash
python scraper_news.py --days 10 --limit 300
```

Alternative sitemap:

```bash
python scraper_news.py --sitemap <URL> --days 7 --limit 300
```

## 🧠 Topic classifier

Run:

```bash
python results/training_model.py
```

The script downloads the BBC train/test datasets supplied by the subject when they are missing from `data/`.

Model:

- representation: TF-IDF with unigrams and bigrams;
- classifier: `LinearSVC`;
- validation: 5-fold stratified learning curve;
- required held-out test accuracy: at least 95%;
- persisted model: `topic_classifier.pkl`;
- learning curve: `results/learning_curves.png`.

### Overfitting check

Exact duplicate training texts are removed before fitting and before cross-validation. The script reports:

- training accuracy;
- cross-validation accuracy;
- train/CV gap;
- held-out test accuracy;
- a second test accuracy with exact train/test text overlaps excluded.

The validated local run reached approximately 97.5% cross-validation accuracy and 98.37% held-out test accuracy, with a 2.5 percentage-point train/CV gap.

### Topic-domain limitation

The BBC classifier is intentionally kept single-label because the supplied training data is single-label.

Its taxonomy is also old and source-specific. In the live Euronews run, EU-policy articles can systematically be predicted as `business` because the BBC `politics` class is dominated by British-parliament vocabulary. This is a known domain-shift limitation rather than something hidden by adding an arbitrary second-topic threshold.

## 🔎 NLP enrichment

Run:

```bash
python nlp_enriched_news.py
```

The script loads 300 scraped articles and adds:

- normalized organization entities;
- one BBC topic label serialized as a JSON list;
- sentence-averaged VADER sentiment;
- environmental-harm distance;
- `Top_10` flag.

The output columns are:

```text
Unique ID
URL
Date scraped
Headline
Body
Org
Topics
Sentiment
Scandal_distance
Top_10
```

`Org` and `Topics` are serialized JSON lists so their structure remains unambiguous in CSV.

### Organization normalization

spaCy `ORG` entities are cleaned before export:

- a leading `the ` is removed;
- trailing possessive `'s` / `’s` is removed;
- duplicate names are collapsed case-insensitively;
- entities beginning with the publisher name `Euronews` are excluded, including possessive forms such as `Euronews’ Forum`.

The remaining entities are still spaCy predictions, not guaranteed company identities. NER can classify NGOs, universities, agencies, people, or other phrases as `ORG`.

## 🌍 Environmental-harm ranking

The subject calls the metric `Scandal_distance`, but the implementation is deliberately conservative: it ranks environmental-harm evidence and does **not** claim that an organization caused a scandal.

The semantic reference phrases include examples such as:

- `oil spill`;
- `water pollution`;
- `air pollution`;
- `chemical contamination`;
- `hazardous waste dumping`;
- `illegal deforestation`;
- `environmental damage`;
- `ecological disaster`.

For every article:

1. the headline is given an explicit sentence boundary before it is joined with the body;
2. spaCy detects and normalizes `ORG` entities;
3. every sentence containing at least one remaining `ORG` is compared with the environmental phrases using `en_core_web_md` vectors and cosine similarity;
4. strong harm markers such as pollution, contamination, deforestation, and sewage qualify on their own;
5. ambiguous markers such as spill, leak, dump, discharge, poison, toxic, hazardous, damage, and slick qualify only when environmental context appears in the same sentence;
6. tokens inside named entities are ignored when establishing that environmental context;
7. harm-bearing sentences remain in the `[0, 1]` distance band;
8. sentences without explicit harm evidence receive a +1 penalty and therefore remain in `[1, 2]`;
9. for harm-bearing evidence, the nearest `ORG` to the harm marker is retained only as contextual information;
10. the ten articles with the smallest final distance receive `Top_10 = True`.

### Why cosine distance?

Cosine similarity compares vector direction rather than magnitude, which is useful for sentences of different lengths. Static word-vector averages can still confuse a general topic such as oil markets with an environmental incident, so semantic distance is combined with the explicit harm-evidence gate.

### Why the console says "Environmental-harm candidate"

The assignment example contains:

```text
Environmental scandal detected for <entity>
```

This implementation intentionally avoids that assertion. spaCy's `ORG` label does not prove that an entity is a company, and semantic proximity does not establish responsibility.

The console therefore reports:

```text
Environmental-harm candidate
Closest organization in evidence: <ORG>
```

or:

```text
No explicit environmental-harm evidence
```

A weekly sample may legitimately contain no explicit company-related environmental incident. The pipeline still produces the required top-10 ranking without fabricating a scandal.

## 📊 Generated artifacts

The final audit artifacts are:

```text
results/learning_curves.png
results/enhanced_news.csv
topic_classifier.pkl
```

The committed `results/enhanced_news.csv` is a snapshot generated on **2026-09-20** from that day's rolling Euronews news window. A later audit run is expected to collect different articles and can therefore produce a different top-10 ranking.

`results/learning_curves.png` comes from the validated BBC-classifier run and does not depend on the weekly Euronews sample.

The raw SQLite database and downloaded BBC datasets are local working data and are not committed by default.

## 📁 Project structure

```text
nlp-scraper/
├── data/
│   └── .gitkeep
├── results/
│   ├── .gitkeep
│   ├── training_model.py
│   ├── enhanced_news.csv
│   └── learning_curves.png
├── .gitignore
├── nlp_enriched_news.py
├── README.md
├── requirements.txt
└── scraper_news.py
```

## ⚠️ Known limitations

- spaCy `ORG` is a named-entity label, not a verified-company registry.
- `en_core_web_md` uses static pretrained vectors; the explicit evidence gate compensates for common semantic false positives but does not turn the metric into causal attribution.
- The BBC topic model experiences domain shift on modern Euronews content, especially EU-policy stories.
- A quiet seven-day news window may contain fewer than 300 usable non-video articles; in that case `--days` can be widened while keeping the target at 300.
- The top 10 are the nearest ranked articles by definition; some may explicitly report that no environmental-harm evidence was found.

## 🧑‍💻 Author

- Nazar Yestayev (@nyestaye)
