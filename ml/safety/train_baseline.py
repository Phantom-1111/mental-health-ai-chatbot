
import joblib
import os
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


TRAIN_PATH = "../datasets/safety/safety_train_clean.csv"
VAL_PATH = "../datasets/safety/safety_validation_clean.csv"
TEST_PATH = "../datasets/safety/safety_test_clean.csv"


train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)


X_train = train_df["text"]
y_train = train_df["label"]

X_val = val_df["text"]
y_val = val_df["label"]

X_test = test_df["text"]
y_test = test_df["label"]


print("Training TF-IDF...")


vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)


X_train_tfidf = vectorizer.fit_transform(X_train)
X_val_tfidf = vectorizer.transform(X_val)
X_test_tfidf = vectorizer.transform(X_test)


print("Train shape:", X_train_tfidf.shape)
print("Validation shape:", X_val_tfidf.shape)
print("Test shape:", X_test_tfidf.shape)


print("\nTraining Logistic Regression...")


model = LogisticRegression(
    max_iter=1000,
    C=2.0,
    solver="liblinear"
)


model.fit(
    X_train_tfidf,
    y_train
)

os.makedirs("safety_model", exist_ok=True)

joblib.dump(
    vectorizer,
    "safety_model/tfidf_vectorizer.pkl"
)

joblib.dump(
    model,
    "safety_model/logistic_regression.pkl"
)

print("\nModel and vectorizer saved.")

print("\nValidation results:")

val_predictions = model.predict(X_val_tfidf)

print(
    classification_report(
        y_val,
        val_predictions,
        target_names=[
            "non-suicide",
            "suicide"
        ],
        zero_division=0
    )
)


print("\nTest results:")

test_predictions = model.predict(X_test_tfidf)

print(
    "Accuracy:",
    accuracy_score(
        y_test,
        test_predictions
    )
)


print(
    classification_report(
        y_test,
        test_predictions,
        target_names=[
            "non-suicide",
            "suicide"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        test_predictions
    )
)