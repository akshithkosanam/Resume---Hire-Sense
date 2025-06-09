"""
build_models.py  – fit TF-IDF + k-NN classifier, then save them as pkl

Assumes:
• final_dataset.csv  with columns  "Resume" and "Category"
"""

import pickle, joblib, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ──────────────────────────────────────────────
# 1.  Load data
# ──────────────────────────────────────────────
df = pd.read_csv("final_dataset.csv")

X_text = df["Resume"].astype(str)
y      = df["Category"]

# ──────────────────────────────────────────────
# 2.  Fit TF-IDF on ALL resumes
# ──────────────────────────────────────────────
tfidf = TfidfVectorizer(stop_words="english")
X_all = tfidf.fit_transform(X_text)

#  ▸ Optionally split for accuracy check
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y, test_size=0.2, random_state=42, stratify=y
)

# ──────────────────────────────────────────────
# 3.  Train classifier
# ──────────────────────────────────────────────
clf = OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train, y_train)

print("Validation accuracy:",
      accuracy_score(y_test, clf.predict(X_test)))

# ──────────────────────────────────────────────
# 4.  Save artefacts
# ──────────────────────────────────────────────
with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f)          # or joblib.dump(tfidf, f)

with open("clf.pkl", "wb") as f:
    pickle.dump(clf, f)            # or joblib.dump(clf, f)

print("✅  Saved  tfidf.pkl  and  clf.pkl")
