"""
Fit a TF-IDF vectorizer + LogisticRegression classifier on your résumé corpus
and save them to disk as clf.pkl and tfidf.pkl
"""

import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ──────────────────────────────────────────────────────────────────────────────
#  1.  Load & prepare your training data
#      The CSV must have at least two columns:  "Resume"  and  "Category"
# ──────────────────────────────────────────────────────────────────────────────
DATA_PATH = "final_dataset.csv"           # <- change if your file is elsewhere
df = pd.read_csv(DATA_PATH).dropna(subset=["Resume", "Category"])

X = df["Resume"].astype(str)
y = df["Category"].astype(str)

# ──────────────────────────────────────────────────────────────────────────────
#  2.  Train
# ──────────────────────────────────────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.9,
    min_df=2,
    ngram_range=(1, 2)
)
X_vec = vectorizer.fit_transform(X)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_vec, y)

# Optional – see quick metrics
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
print(classification_report(y_te, clf.predict(vectorizer.transform(X_te))))

# ──────────────────────────────────────────────────────────────────────────────
#  3.  Save artefacts
# ──────────────────────────────────────────────────────────────────────────────
joblib.dump(clf,  "clf.pkl")
joblib.dump(vectorizer, "tfidf.pkl")
print("✅  Saved clf.pkl and tfidf.pkl")
