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
- ranks articles by environmental-harm semantic distance;
- flags exactly ten articles with the smallest scandal distance required by the subject.

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

The scraper is restart-safe. URLs are unique in SQLite, so rerunning it does not duplicate already stored articles. Euronews `/video/` pages are excluded before downloading because short bulletins and mixed-story video pages add noise to topic and sentiment analysis. Headline whitespace is normalized during extraction.

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

### Topic-domain limitation

The BBC classifier is intentionally kept as a single-label classifier because that is how the supplied training data is defined. Its taxonomy is also old and source-specific. In the live Euronews run, EU-policy articles can systematically be predicted as `business` because the BBC `politics` class is dominated by British-parliament vocabulary. This is treated as a known domain-shift limitation rather than hidden by adding an arbitrary second-topic threshold.

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

The subject calls this field `Scandal_distance`, but the implementation is deliberately conservative: it ranks environmental-harm evidence and does **not** claim that an organization caused a scandal.

The detector uses environmental-disaster phrases such as `oil spill`, `water pollution`, `chemical contamination`, `deforestation`, and `environmental disaster`.

For every article:

1. spaCy detects `ORG` entities. Display names are normalized by removing a leading `the` and trailing possessive `'s` / `’s`, then deduplicated case-insensitively. The publisher name `Euronews` is excluded.
2. The headline is given an explicit sentence boundary before it is joined with the body, so it is not merged with the first body sentence.
3. Every sentence containing at least one remaining `ORG` entity is represented with spaCy's `en_core_web_md` vectors and compared with the disaster phrases using cosine similarity.
4. Strong environmental-harm markers such as pollution, contamination, deforestation, and sewage qualify on their own.
5. Ambiguous words such as spill, leak, dump, discharge, poison, toxic, hazardous, damage, and slick qualify only when an environmental-context term such as water, river, sea, soil, waste, oil, pipeline, forest, ecosystem, or chemical occurs in the same sentence.
6. Tokens inside named entities are ignored when deciding whether an ambiguous harm word has environmental context. This avoids entity names accidentally creating context.
7. A harm-bearing sentence remains in the `[0, 1]` distance band. A sentence without explicit harm evidence receives a +1 penalty and therefore stays in `[1, 2]`.
8. When harm evidence exists, the reported organization is the `ORG` entity nearest to the harm marker in that sentence. This is context only, not attribution of responsibility.
9. The ten articles with the smallest final distance are flagged with `Top_10 = True`, even if fewer than ten contain explicit environmental-harm evidence.

### Why cosine distance?

Cosine similarity measures semantic direction rather than vector magnitude, which is useful for comparing textual embeddings of different sentence lengths. Static word-vector averages can still confuse a topic such as oil markets with an environmental incident, so cosine distance is combined with the explicit harm-evidence gate above.

### Why the console says "Environmental-harm candidate"

The subject's example output contains `Environmental scandal detected for <entity>`. This implementation intentionally does not reproduce that assertion. spaCy's `ORG` label can represent companies, NGOs, universities, agencies, or even occasional NER mistakes, and semantic proximity does not establish responsibility.

Therefore the console uses:

- `Environmental-harm candidate` when explicit harm evidence is present;
- `Closest organization in evidence: <ORG>` only as contextual information;
- `No explicit environmental-harm evidence` when the article is in the top 10 only because it is the next-closest article.

A weekly news sample may legitimately contain no explicit company-related environmental incident at all. In that case the detector still produces the required top-10 ranking and clearly reports that no explicit environmental-harm evidence was found rather than fabricating a scandal.

## Generated artifacts

After a complete run the relevant deliverables are:

```text
results/learning_curves.png
results/enhanced_news.csv
topic_classifier.pkl
```

The raw SQLite database and downloaded BBC datasets are local working data and are not committed by default. `learning_curves.png` and `enhanced_news.csv` should be committed only after a real successful run, not generated from synthetic audit data.
