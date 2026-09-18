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
- scores sentiment with NLTK VADER, averaged over article sentences to avoid whole-document saturation;
- ranks articles by environmental-scandal semantic distance;
- flags exactly ten articles with the smallest scandal distance.

## Setup

Use Python 3.11–3.13. The pinned stack is tested against this range.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

The medium spaCy model is pinned in `requirements.txt`. It provides both `ORG` NER and static pretrained vectors used by the scandal detector. These vectors are not treated as a complete semantic model on their own: the scandal score also requires explicit environmental-harm evidence so generic oil, water, or air stories do not dominate the ranking.

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

Duplicate training texts are removed before fitting and before the five-fold stratified cross-validation used to generate learning curves. Training and validation accuracy are plotted against training-set size, and the final gap is printed. The script also reports a second test accuracy with exact train/test text overlaps excluded. A persistent large train/CV gap would indicate overfitting; a small gap together with similar held-out accuracy supports generalization.

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

1. spaCy detects `ORG` entities; the publisher name `Euronews` is excluded;
2. every sentence containing at least one remaining `ORG` entity is considered;
3. each such sentence and each disaster phrase is represented with spaCy's `en_core_web_md` vectors;
4. cosine similarity is computed between every relevant sentence and disaster phrase;
5. the semantic distance is `1 - cosine_similarity`, clipped to the `[0, 1]` band;
6. sentences with an explicit harm marker such as pollution, contamination, spill, leak, toxic waste, dumping, or deforestation remain in the `[0, 1]` distance band;
7. sentences without a harm marker receive a +1 penalty and therefore occupy the `[1, 2]` band;
8. the smallest adjusted distance is retained for the article together with the corresponding organization and evidence sentence;
9. the ten articles with the smallest final distance are flagged with `Top_10 = True`.

### Why cosine distance?

Cosine similarity measures semantic direction rather than vector magnitude, which is useful for comparing textual embeddings of different sentence lengths. Static word-vector averages can still confuse a topic such as oil markets with an environmental incident, so cosine distance is combined with an explicit harm-evidence penalty. Smaller final values therefore mean both semantic proximity to the disaster concepts and stronger evidence of actual environmental harm.

## Generated artifacts

After a complete run the relevant deliverables are:

```text
results/learning_curves.png
results/enhanced_news.csv
topic_classifier.pkl
```

The raw SQLite database and downloaded BBC datasets are local working data and are not committed by default. `learning_curves.png` and `enhanced_news.csv` should be committed only after a real successful run, not generated from synthetic audit data.
