# main.py
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

from constants import DATASET, MODELS_DIR

# -----------------------------
# 1. Load the dataset
# -----------------------------
df = pd.read_csv(DATASET)

# Expect columns: target, tweet
df = df.rename(columns={"tweet": "text"})
df = df[["target", "text"]]

# Keep only positive (4) and negative (0) tweets
df = df[df["target"].isin([0, 4])]
df["target"] = df["target"].map({0: 0, 4: 1})  # 0=Negative, 1=Positive
df = df.reset_index(drop=True)

# -----------------------------
# 2. Preprocess text
# -----------------------------
def _lowercase(text: str) -> str:
    return str(text).lower()

df["text"] = df["text"].apply(_lowercase)

# -----------------------------
# 3. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["target"], test_size=0.2, random_state=42, stratify=df["target"]
)

# -----------------------------
# 4. TF-IDF Vectorization
# -----------------------------
vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vec.fit_transform(X_train)
X_test_vec = vec.transform(X_test)

# -----------------------------
# 5. Train Models
# -----------------------------
# Bernoulli Naive Bayes
bnb = BernoulliNB()
bnb.fit(X_train_vec, y_train)
y_bnb_pred = bnb.predict(X_test_vec)

# Linear SVC
lsvc = LinearSVC(max_iter=1000, random_state=42)
lsvc.fit(X_train_vec, y_train)
y_lsvc_pred = lsvc.predict(X_test_vec)

# Logistic Regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_vec, y_train)
y_lr_pred = lr.predict(X_test_vec)

# -----------------------------
# 6. Print Accuracies
# -----------------------------
print(f"BNB Accuracy (test): {accuracy_score(y_test, y_bnb_pred) * 100:.2f}%")
print(f"SVC Accuracy (test): {accuracy_score(y_test, y_lsvc_pred) * 100:.2f}%")
print(f"LR Accuracy (test): {accuracy_score(y_test, y_lr_pred) * 100:.2f}%")

# -----------------------------
# 7. Save Models (ONLY models, not vectorizer)
# -----------------------------
joblib.dump(bnb, MODELS_DIR / "bnb.pkl")
joblib.dump(lsvc, MODELS_DIR / "lsvc.pkl")
joblib.dump(lr, MODELS_DIR / "lr.pkl")

# -----------------------------
# 8. Inference Demo (using in-memory vectorizer)
# -----------------------------
print("\nSample Inference:")

sample_texts = [
    "I love you!",
    "I hate you but I love you also.",
    "I love your code, it's so clean. :)",
]

sample_vec = vec.transform(sample_texts)
models = {"BNB": bnb, "LSVC": lsvc, "LR": lr}

for text in sample_texts:
    print(f"\nText: {text}")
    for name, model in models.items():
        pred = model.predict(vec.transform([text]))[0]
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"{name} → {sentiment}")
