import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "../datasets/emotion/emotion_dataset.csv"
OUTPUT = "../datasets/emotion/"

df = pd.read_csv(INPUT)

df = df[["Text", "Emotion"]].copy()

bad = df.groupby("Text")["Emotion"].nunique()
bad_texts = bad[bad > 1].index

df = df[~df["Text"].isin(bad_texts)]
df = df.drop_duplicates(subset="Text")

labels = {
    "anger": 0,
    "disgust": 1,
    "fear": 2,
    "joy": 3,
    "neutral": 4,
    "sadness": 5,
    "shame": 6,
    "surprise": 7
}

df["label"] = df["Emotion"].map(labels)

train, temp = train_test_split(
    df,
    test_size=0.20,
    stratify=df["label"],
    random_state=42
)

validation, test = train_test_split(
    temp,
    test_size=0.50,
    stratify=temp["label"],
    random_state=42
)

train.to_csv(OUTPUT + "emotion_train_clean.csv", index=False)
validation.to_csv(OUTPUT + "emotion_validation_clean.csv", index=False)
test.to_csv(OUTPUT + "emotion_test_clean.csv", index=False)

print("Train:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

print("\nTrain distribution:")
print(train["Emotion"].value_counts())

print("\nValidation distribution:")
print(validation["Emotion"].value_counts())

print("\nTest distribution:")
print(test["Emotion"].value_counts())