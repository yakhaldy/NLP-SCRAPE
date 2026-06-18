# test_scandal.py
import spacy
import numpy as np

nlp = spacy.load("en_core_web_md")

# Environmental disaster keywords
KEYWORDS = [
    "pollution", "contamination", "deforestation",
    "oil spill", "toxic waste", "environmental disaster",
    "emissions", "ecological damage", "habitat destruction",
]

# Test sentences
sentences = [
    "BP caused massive oil spill polluting the entire coastline",
    "Amazon linked to deforestation and habitat destruction in Brazil",
    "Apple released new iPhone with improved camera and battery",
    "The factory dumped toxic waste into the river causing contamination",
    "Manchester United won the Premier League championship title",
    "Shell faced accusations of illegal emissions and ecological damage",
]

print("=" * 60)
print("Scandal Detection Test — Cosine Similarity")
print("=" * 60)

# Compute keyword embeddings (average of all keywords)
keyword_vectors = [nlp(kw).vector for kw in KEYWORDS]
keyword_vector  = np.mean(keyword_vectors, axis=0)

for sentence in sentences:
    doc = nlp(sentence)

    # Cosine similarity between sentence and keyword vector
    sent_vec = doc.vector
    
    # Avoid division by zero
    norm = np.linalg.norm(sent_vec) * np.linalg.norm(keyword_vector)
    if norm == 0:
        score = 0.0
    else:
        score = float(np.dot(sent_vec, keyword_vector) / norm)

    flag = "SCANDAL ⚠️ " if score >= 0.4 else "Safe ✅   "
    print(f"\n{flag} score={score:.3f}")
    print(f"  {sentence}")

print("\n" + "=" * 60)
print("Scandal detection working!")
print("=" * 60)