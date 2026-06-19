# NLP Scraper

Small project to scrape news articles and run simple NLP processing and training.

## Contents

- `data/` — CSV datasets: `articles.csv`, `bbc_news_train.csv`, `bbc_news_tests.csv`.
- `scripts/` — main processing scripts: `scraper_news.py`, `nlp_enriched_news.py`.
- `results/` — outputs and `training_model.py` for training/persistence.
- `test.py` — small test runner.

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. Run the scraper to collect articles:

```bash
python scripts/scraper_news.py
```

2. Enrich/clean the scraped data:

```bash
python scripts/nlp_enriched_news.py
```

3. Train a model or run training utilities:

```bash
python results/training_model.py
```


## Project notes

- Dependencies are listed in `requirements.txt`.
- Ignore heavy outputs and virtual envs via `.gitignore`.

## License

MIT
# NLP Enriched News Intelligence Platform

## Overview
An NLP pipeline that scrapes BBC News articles and enriches them
with entity detection, topic classification, sentiment analysis,
and environmental scandal detection.

## Project Structure
```
project/
├── data/               # Raw and training data
├── results/            # Model outputs and visualizations
├── scraper_news.py     # BBC News scraper
├── nlp_enriched_news.py # NLP analysis pipeline
└── requirements.txt    # Dependencies
```
## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Step 2 — Scrape articles (min 300)
```bash
python scraper_news.py
```

### Step 3 — Train topic classifier
```bash
python results/training_model.py
```

### Step 4 — Run NLP engine
```bash
python nlp_enriched_news.py
```

## NLP Components

### 1. Named Entity Recognition
Tool: SpaCy `en_core_web_md`
Target: ORG entities (companies and organizations)

### 2. Topic Detection
Tool: TF-IDF + LinearSVC pipeline
Accuracy: 98.64% on test set
Topics: Tech, Sport, Business, Politics, Entertainment

### 3. Sentiment Analysis
Tool: NLTK VADER (pre-trained)
Output: compound score (-1.0 to +1.0)
- Positive: score >= 0.05
- Negative: score <= -0.05
- Neutral: between -0.05 and 0.05

### 4. Scandal Detection

#### Embeddings chosen: SpaCy en_core_web_md vectors
Reason: 300-dimensional word vectors trained on large corpus,
available without additional installation, and supports
sentence-level embedding through averaging.

#### Distance metric chosen: Cosine Similarity
Reason: Cosine similarity measures the angle between vectors,
making it robust to sentence length differences. Unlike
Euclidean distance, it focuses on direction (meaning) rather
than magnitude, which is ideal for semantic comparison.

#### Methodology:
1. Define 15 environmental disaster keywords
2. Compute average keyword embedding vector
3. For each sentence containing an ORG entity:
   - Compute sentence embedding
   - Calculate cosine similarity with keyword vector
4. Article scandal score = max similarity across all ORG sentences
5. Flag top 10 articles by scandal score

#### Threshold: 0.55
Articles scoring above 0.55 are flagged as potential scandals.

## Output Format (enhanced_news.csv)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Unique article identifier |
| url | str | Source URL |
| date | date | Scraping date |
| headline | str | Article headline |
| body | str | Article body text |
| org | list | Detected organizations |
| topic | str | Predicted topic |
| sentiment | float | VADER compound score |
| scandal_distance | float | Cosine similarity score |
| top_10 | bool | True if in top 10 scandal articles |## How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_md
```

### Step 2 — Scrape articles (min 300)
```bash
python scraper_news.py
```

### Step 3 — Train topic classifier
```bash
python results/training_model.py
```

### Step 4 — Run NLP engine
```bash
python nlp_enriched_news.py
```

## NLP Components

### 1. Named Entity Recognition
Tool: SpaCy `en_core_web_md`
Target: ORG entities (companies and organizations)

### 2. Topic Detection
Tool: TF-IDF + LinearSVC pipeline
Accuracy: 98.64% on test set
Topics: Tech, Sport, Business, Politics, Entertainment

### 3. Sentiment Analysis
Tool: NLTK VADER (pre-trained)
Output: compound score (-1.0 to +1.0)
- Positive: score >= 0.05
- Negative: score <= -0.05
- Neutral: between -0.05 and 0.05

### 4. Scandal Detection

#### Embeddings chosen: SpaCy en_core_web_md vectors
Reason: 300-dimensional word vectors trained on large corpus,
available without additional installation, and supports
sentence-level embedding through averaging.

#### Distance metric chosen: Cosine Similarity
Reason: Cosine similarity measures the angle between vectors,
making it robust to sentence length differences. Unlike
Euclidean distance, it focuses on direction (meaning) rather
than magnitude, which is ideal for semantic comparison.

#### Methodology:
1. Define 15 environmental disaster keywords
2. Compute average keyword embedding vector
3. For each sentence containing an ORG entity:
   - Compute sentence embedding
   - Calculate cosine similarity with keyword vector
4. Article scandal score = max similarity across all ORG sentences
5. Flag top 10 articles by scandal score

#### Threshold: 0.55
Articles scoring above 0.55 are flagged as potential scandals.

## Output Format (enhanced_news.csv)
| Column | Type | Description |
|--------|------|-------------|
| id | uuid | Unique article identifier |
| url | str | Source URL |
| date | date | Scraping date |
| headline | str | Article headline |
| body | str | Article body text |
| org | list | Detected organizations |
| topic | str | Predicted topic |
| sentiment | float | VADER compound score |
| scandal_distance | float | Cosine similarity score |
| top_10 | bool | True if in top 10 scandal articles |