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
nlp      = spacy.load("en_core_web_md")
sia      = SentimentIntensityAnalyzer()
topic_model = joblib.load("results/topic_classifier.pkl")
print("All models loaded!\n")

# ============================================================
# Scandal Detection Keywords
# ============================================================
SCANDAL_KEYWORDS = [
    "pollution", "contamination", "deforestation",
    "oil spill", "toxic waste", "environmental disaster",
    "emissions", "ecological damage", "habitat destruction",
    "chemical leak", "nuclear waste", "water contamination",
    "air pollution", "soil contamination", "illegal dumping",
]

SCANDAL_THRESHOLD = 0.55   # raised to reduce false positives

# Precompute keyword vector (average of all keywords)
keyword_vectors  = [nlp(kw).vector for kw in SCANDAL_KEYWORDS]
KEYWORD_VECTOR   = np.mean(keyword_vectors, axis=0)


# ============================================================
# Helper: cosine similarity
# ============================================================
def cosine_similarity(vec1, vec2):
    norm = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if norm == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / norm)


# ============================================================
# Step 1: Detect ORG entities
# ============================================================
def detect_entities(text):
    doc  = nlp(text)
    orgs = list({ent.text for ent in doc.ents if ent.label_ == "ORG"})
    return orgs


# ============================================================
# Step 2: Detect topic
# ============================================================
def detect_topic(headline, body):
    combined = headline + " " + body[:500]
    topic    = topic_model.predict([combined])[0]
    return topic


# ============================================================
# Step 3: Sentiment analysis
# ============================================================
def detect_sentiment(text):
    scores   = sia.polarity_scores(text[:1000])
    return round(scores["compound"], 4)


# ============================================================
# Step 4: Scandal detection
# ============================================================
def detect_scandal(body, orgs):
    if not orgs:
        return 0.0

    doc       = nlp(body)
    sentences = [sent.text for sent in doc.sents]

    # Keep only sentences that contain at least one ORG
    org_sentences = []
    for sent in sentences:
        sent_lower = sent.lower()
        if any(org.lower() in sent_lower for org in orgs):
            org_sentences.append(sent)

    if not org_sentences:
        return 0.0

    # Compute similarity for each ORG sentence
    scores = []
    for sent in org_sentences:
        sent_vec = nlp(sent).vector
        score    = cosine_similarity(sent_vec, KEYWORD_VECTOR)
        scores.append(score)

    # Return max score as the scandal distance for this article
    return round(float(np.max(scores)), 4)


# ============================================================
# Main pipeline
# ============================================================
def main():
    # Load articles
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
        orgs = detect_entities(headline + " " + body)
        if orgs:
            print(f"Detected {len(orgs)} companies: {', '.join(orgs[:5])}")
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
        scandal_score = detect_scandal(body, orgs)

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
            "top_10":           False,   # will update below
        })

    # --------------------------------------------------------
    # Flag top 10 scandal articles
    # --------------------------------------------------------
    results_df = pd.DataFrame(results)
    top10_idx  = results_df["scandal_distance"].nlargest(10).index
    results_df.loc[top10_idx, "top_10"] = True

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    results_df.to_csv("results/enhanced_news.csv", index=False)

    print("\n" + "=" * 60)
    print(f"Done! Processed {len(results_df)} articles")
    print(f"Saved to: results/enhanced_news.csv")
    print("\nTop 10 scandal articles:")
    top10 = results_df[results_df["top_10"]].sort_values(
        "scandal_distance", ascending=False
    )
    for _, r in top10.iterrows():
        print(f"  {r['scandal_distance']:.3f} | {r['headline'][:60]}")
    print("=" * 60)


if __name__ == "__main__":
    main()