# nlp_enriched_news.py
import pandas as pd
import numpy as np
import spacy
import nltk
import joblib
import os

nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment import SentimentIntensityAnalyzer

# ============================================================
# Setup
# ============================================================
print("Loading models...")
sia = SentimentIntensityAnalyzer()
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Downloading 'en_core_web_md' model for spaCy...")
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

try:
    topic_model = joblib.load("results/topic_classifier.pkl")
except FileNotFoundError:
    print("ERROR: Topic model not found. Please run 'python3 results/training_model.py' first.")
    exit(1)
print("All models loaded!\n")

# ============================================================
# Scandal Detection Keywords
# ============================================================
SCANDAL_KEYWORDS = [
    "pollution", "contamination", "contaminate", "pollute",
    "deforestation", "oil spill", "oil leak", "toxic waste",
    "hazardous waste", "environmental disaster", "ecological damage",
    "habitat destruction", "chemical leak", "chemical spill",
    "nuclear waste", "water contamination", "air pollution",
    "soil contamination", "illegal dumping", "toxic spill",
    "sewage discharge", "wastewater discharge", "effluent", "smog",
]

SINGLE_KEYWORDS = {k for k in SCANDAL_KEYWORDS if " " not in k}
PHRASE_KEYWORDS = {k for k in SCANDAL_KEYWORDS if " " in k}


KEYWORD_VECTORS = [nlp(kw).vector for kw in SCANDAL_KEYWORDS]


ORG_BLOCK_PREFIXES = (
    "bbc", "reuters", "afp", "cnn", "sky news", "the guardian",
    "itv", "pa media", "getty", "associated press",
)


def _is_blocked_org(name):
    low = name.lower().strip()
    return any(low == p or low.startswith(p + " ") for p in ORG_BLOCK_PREFIXES)


SCANDAL_THRESHOLD = 0.50


# ============================================================
# Helpers
# ============================================================
def cosine_similarity(vec1, vec2):
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / norm)


def content_vector(span):
    """Average vector over content words only (drop stopwords/punct).
    This removes the function-word noise that flattened similarities."""
    vecs = [t.vector for t in span
            if t.has_vector and not t.is_stop and not t.is_punct and t.is_alpha]
    if not vecs:
        return None
    return np.mean(vecs, axis=0)


# ============================================================
# Step 1: Detect ORG entities (de-duplicated, noise-filtered)
# ============================================================
def detect_entities(text):
    doc = nlp(text)
    orgs = {}
    for ent in doc.ents:
        if ent.label_ != "ORG":
            continue
        name = ent.text.strip()
        if len(name) < 2:
            continue
        if _is_blocked_org(name):
            continue
        key = name.lower()
        orgs.setdefault(key, name)  
    return list(orgs.values())


# ============================================================
# Step 2: Detect topic
# ============================================================
def detect_topic(headline, body):
    combined = headline + " " + body[:500]
    return topic_model.predict([combined])[0]


# ============================================================
# Step 3: Sentiment analysis
# ============================================================
def detect_sentiment(text):
    scores = sia.polarity_scores(text[:1000])
    return round(scores["compound"], 4)


# ============================================================
# Step 4: Scandal detection (lexical gate + semantic ranking)
# ============================================================
def detect_scandal(body_doc, orgs):
    orgs = [o for o in orgs if not _is_blocked_org(o)]
    if not orgs:
        return 0.0
    orgs_lower = [o.lower() for o in orgs]

    best = 0.0
    for sent in body_doc.sents:
        low = sent.text.lower()
        if not any(o in low for o in orgs_lower):
            continue

        lemmas = {t.lemma_.lower() for t in sent}
        lexical_hit = (any(k in lemmas for k in SINGLE_KEYWORDS)
                       or any(p in low for p in PHRASE_KEYWORDS))
        if not lexical_hit:
            continue

        sv = content_vector(sent)
        if sv is None:
            continue

        sim = max(cosine_similarity(sv, kv) for kv in KEYWORD_VECTORS)
        best = max(best, sim)

    return round(float(best), 4)


# ============================================================
# Main pipeline
# ============================================================
def main():
    df = pd.read_csv("data/articles.csv")
    total = len(df)
    results = []

    for i, row in df.iterrows():
        url      = row["url"]
        date     = row["date"]
        headline = str(row["headline"])
        body     = str(row["body"])

        print(f"\nEnriching [{i+1}/{total}]: {url}")

        # ---------- Detect entities ----------
        print("\n---------- Detect entities ----------")
        orgs = detect_entities(headline + " " + body[:3000])
        if orgs:
            print(f"Detected {len(orgs)} companies which are {', '.join(orgs[:5])}")
        else:
            print("No organizations detected")

        # ---------- Topic detection ----------
        print("\n---------- Topic detection ----------")
        print("Text preprocessing ...")
        topic = detect_topic(headline, body)
        print(f"The topic of the article is: {topic}")

        # ---------- Sentiment analysis ----------
        print("\n---------- Sentiment analysis ----------")
        sentiment = detect_sentiment(body)
        if sentiment >= 0.05:
            sent_label = "POSITIVE"
        elif sentiment <= -0.05:
            sent_label = "NEGATIVE"
        else:
            sent_label = "NEUTRAL"
        print(f"The article '{headline[:40]}' has a {sent_label} sentiment ({sentiment})")

        # ---------- Scandal detection ----------
        print("\n---------- Scandal detection ----------")
        print("Computing embeddings and distance ...")
        body_doc      = nlp(body[:5000])
        scandal_score = detect_scandal(body_doc, orgs)

        if scandal_score >= SCANDAL_THRESHOLD and orgs:
            print(f"Environmental scandal detected for: {', '.join(orgs[:3])}")
        else:
            print(f"No scandal detected (score={scandal_score})")

        results.append({
            "id":               row["id"],
            "url":              url,
            "date":             date,
            "headline":         headline,
            "body":             body,
            "org":              orgs,
            "topic":            topic,
            "sentiment":        sentiment,
            "scandal_distance": scandal_score,
            "top_10":           False,
        })

    # --------------------------------------------------------
    # Flag the top 10 scandal articles.
    # --------------------------------------------------------
    results_df = pd.DataFrame(results)
    candidates = results_df[results_df["scandal_distance"] >= SCANDAL_THRESHOLD]
    top_idx = candidates["scandal_distance"].nlargest(10).index
    results_df.loc[top_idx, "top_10"] = True

    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/enhanced_news.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Done! Processed {len(results_df)} articles")
    print(f"Saved to: results/enhanced_news.csv")
    flagged = results_df[results_df["top_10"]].sort_values(
        "scandal_distance", ascending=False)
    print(f"\nFlagged {len(flagged)} scandal article(s):")
    if flagged.empty:
        print("  (none cleared the threshold in this dataset)")
    for _, r in flagged.iterrows():
        print(f"  {r['scandal_distance']:.3f} | {r['headline'][:60]}")
    print("=" * 60)


if __name__ == "__main__":
    main()