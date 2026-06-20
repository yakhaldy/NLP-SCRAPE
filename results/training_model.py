# results/training_model.py
# Train a topic classifier on BBC News data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.pipeline                import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm                     import LinearSVC
from sklearn.calibration             import CalibratedClassifierCV
from sklearn.model_selection         import learning_curve, StratifiedKFold
from sklearn.metrics                 import classification_report, accuracy_score

# ============================================================
# Step 1: Load data
# ============================================================
print("=" * 55)
print("Step 1: Loading data")
print("=" * 55)

train_df = pd.read_csv("data/bbc_news_train.csv")
test_df  = pd.read_csv("data/bbc_news_tests.csv")

X_train = train_df["Text"]
y_train = train_df["Category"]
X_test  = test_df["Text"]
y_test  = test_df["Category"]

print(f"Training examples : {len(X_train)}")
print(f"Test examples     : {len(X_test)}")
print(f"Categories        : {list(y_train.unique())}")

# ============================================================
# Step 2: Build pipeline
# ============================================================
print("\n" + "=" * 55)
print("Step 2: Building pipeline")
print("=" * 55)

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        max_features=10000,    
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,     # apply log normalization to TF
        min_df=1,
    )),
    ("clf", CalibratedClassifierCV(
        LinearSVC(
            max_iter=2000,
            random_state=42,
            C=1.0,
        )
    )),
])

print("Pipeline steps:")
print("  1. TfidfVectorizer (10000 features, bigrams, log-norm)")
print("  2. LinearSVC with CalibratedClassifierCV")

# ============================================================
# Step 3: Train
# ============================================================
print("\n" + "=" * 55)
print("Step 3: Training model...")
print("=" * 55)

pipeline.fit(X_train, y_train)
print("Training complete!")

# ============================================================
# Step 4: Evaluate
# ============================================================
print("\n" + "=" * 55)
print("Step 4: Evaluating on test set")
print("=" * 55)

y_pred = pipeline.predict(X_test)
score  = accuracy_score(y_test, y_pred)

print(f"\nTest accuracy: {score:.2%}")


if score >= 0.95:
    print("Target achieved: score > 95%")
else:
    print(f"Warning: score {score:.2%} is below 95% target")

# ============================================================
# Step 5: Learning curves
# ============================================================
print("\n" + "=" * 55)
print("Step 5: Generating learning curves...")
print("=" * 55)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

train_sizes, train_scores, val_scores = learning_curve(
    pipeline,
    X_train, y_train,
    cv=cv,
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="accuracy", 
    n_jobs=-1,
)

train_mean = train_scores.mean(axis=1)
train_std  = train_scores.std(axis=1)
val_mean   = val_scores.mean(axis=1)
val_std    = val_scores.std(axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, "o-", color="blue",   label="Training score")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.plot(train_sizes, val_mean,   "o-", color="orange", label="Validation score")
plt.fill_between(train_sizes, val_mean - val_std,   val_mean + val_std,   alpha=0.1, color="orange")
plt.title("Learning Curves — Topic Classifier (LinearSVC)", fontsize=14)
plt.xlabel("Training examples")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.ylim(0, 1.1)

os.makedirs("results", exist_ok=True)
plt.savefig("results/learning_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: results/learning_curves.png")

# ============================================================
# Step 6: Save model
# ============================================================
print("\n" + "=" * 55)
print("Step 6: Saving model...")
print("=" * 55)

joblib.dump(pipeline, "results/topic_classifier.pkl")
print("Saved: results/topic_classifier.pkl")


print("=" * 55)
print("Done!")
print("=" * 55)