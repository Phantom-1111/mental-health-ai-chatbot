import pandas as pd
import numpy as np

from transformers import pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


TEST_PATH = "../datasets/emotion/emotion_test_clean.csv"


label_names = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "shame",
    "surprise"
]


model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    device=-1
)


test_df = pd.read_csv(TEST_PATH)

test_df = test_df[["Text", "label"]]


texts = test_df["Text"].tolist()
true_labels = test_df["label"].tolist()


predictions = []

print("\nRunning pretrained emotion model...\n")


for i, text in enumerate(texts):

    result = model(
        text,
        truncation=True
    )[0]

    predicted_emotion = result["label"]

    predicted_label = label_names.index(
        predicted_emotion
    )

    predictions.append(predicted_label)

    if (i + 1) % 100 == 0:
        print(
            f"Processed {i + 1}/{len(texts)}"
        )


predictions = np.array(predictions)
true_labels = np.array(true_labels)


accuracy = accuracy_score(
    true_labels,
    predictions
)


weighted_precision, weighted_recall, weighted_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0
    )
)


macro_precision, macro_recall, macro_f1, _ = (
    precision_recall_fscore_support(
        true_labels,
        predictions,
        average="macro",
        zero_division=0
    )
)


print("\n==============================")
print("PRETRAINED MODEL RESULTS")
print("==============================")

print(
    f"\nAccuracy: {accuracy:.4f}"
)

print(
    f"Weighted F1: {weighted_f1:.4f}"
)

print(
    f"Macro F1: {macro_f1:.4f}"
)


print("\nClassification Report:\n")

print(
    classification_report(
        true_labels,
        predictions,
        labels=[0, 1, 2, 3, 4, 5, 7],
        target_names=[
            "anger",
            "disgust",
            "fear",
            "joy",
            "neutral",
            "sadness",
            "surprise"
        ],
        zero_division=0
    )
)


print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        true_labels,
        predictions,
        labels=[0, 1, 2, 3, 4, 5, 7]
    )
)