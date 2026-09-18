# NLP Scraper

NLP-enriched news intelligence pipeline for the 01 Edu AI track.

## Pipeline

```text
Euronews sitemap
    -> scraper_news.py
    -> SQLite (data/news.db)
    -> results/training_model.py -> topic_classifier.pkl
    -> nlp_enriched_news.py
    -> results/enhanced_news.csv
```

The project:

- scrapes at least 300 recent English-language news articles;
- stores a stable UUID, URL, scrape date, headline, and body in SQLite;
- detects organizations with spaCy NER;
- classifies topics as business, entertainment, politics, sport, or tech;
- scores sentiment with NLTK VADER;
- ranks articles by environmental-scandal semantic distance;
- flags exactly ten articles with the smallest scandal distance.

## Setup

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

The medium spaCy model is intentional: it provides both `ORG` NER and real word vectors used by the scandal detector.

## 1. Scrape news

```bash
python scraper_news.py
```

Defaults:

- source: Euronews English article sitemap;
- time window: last 7 days;
- target: 300 stored articles;
- storage: `data/news.db`.

The scraper is restart-safe. URLs are unique in SQLite, so rerunning it does not duplicate already stored articles.

Alternative sitemap:

```bash
python scraper_news.py --sitemap <URL> --days 7 --limit 300
```

## 2. Train the topic classifier

```bash
python results/training_model.py
```

The script downloads the two BBC datasets supplied by the subject when they are not already present in `data/`.

Model:

- representation: TF-IDF with unigrams and bigrams;
- classifier: `LinearSVC`;
- required held-out test accuracy: >= 95%;
- persisted model: `topic_classifier.pkl`;
- learning curves: `results/learning_curves.png`.

### Overfitting check

Five-fold stratified cross-validation is used to generate learning curves. Training and validation accuracy are plotted against training-set size. A persistent large gap would indicate overfitting; convergence at high accuracy indicates that the classifier generalizes to unseen samples.

## 3. Enrich 300 articles

```bash
python nlp_enriched_news.py
```

The output is written to `results/enhanced_news.csv` with the required columns:

- `Unique ID`
- `URL`
- `Date scraped`
- `Headline`
- `Body`
- `Org`
- `Topics`
- `Sentiment`
- `Scandal_distance`
- `Top_10`

`Org` and `Topics` are serialized JSON lists so they remain unambiguous when stored in CSV.

## Scandal detection

The detector uses environmental-disaster phrases such as `oil spill`, `water pollution`, `chemical contamination`, `deforestation`, and `environmental disaster`.

For every article:

1. spaCy detects `ORG` entities;
2. only sentences containing at least one `ORG` entity are considered;
3. each sentence and each disaster phrase is represented with spaCy's `en_core_web_md` vectors;
4. cosine similarity is computed between every relevant sentence and disaster phrase;
5. the strongest similarity is retained for the article;
6. `Scandal_distance = 1 - max_cosine_similarity`;
7. the ten articles with the smallest distance are flagged with `Top_10 = True`.

### Why cosine distance?

Cosine similarity measures semantic direction rather than vector magnitude, which is more useful for comparing textual embeddings of different sentence lengths. Converting it to `1 - similarity` gives an intuitive distance where smaller values mean stronger semantic proximity to the environmental-disaster concepts.

## Generated artifacts

After a complete run the relevant deliverables are:

```text
results/learning_curves.png
results/enhanced_news.csv
topic_classifier.pkl
```

The raw SQLite database and downloaded BBC datasets are local working data and are not committed by default.
