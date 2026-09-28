import pandas as pd
from sklearn.model_selection import train_test_split


INPUT = "../datasets/safety/Suicide_Detection.csv"
OUTPUT = "../datasets/safety/"


df = pd.read_csv(INPUT)

df = df[["text", "class"]].copy()

df = df.dropna(subset=["text", "class"])

df["text"] = df["text"].astype(str).str.strip()

df = df[df["text"] != ""]

df = df.drop_duplicates(subset="text")


labels = {
    "non-suicide": 0,
    "suicide": 1
}

df["label"] = df["class"].map(labels)

df = df.dropna(subset=["label"])

df["label"] = df["label"].astype(int)


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


train.to_csv(
    OUTPUT + "safety_train_clean.csv",
    index=False
)

validation.to_csv(
    OUTPUT + "safety_validation_clean.csv",
    index=False
)

test.to_csv(
    OUTPUT + "safety_test_clean.csv",
    index=False
)


print("Total:", df.shape)

print("\nTrain:", train.shape)
print("Validation:", validation.shape)
print("Test:", test.shape)

print("\nTrain distribution:")
print(train["class"].value_counts())

print("\nValidation distribution:")
print(validation["class"].value_counts())

print("\nTest distribution:")
print(test["class"].value_counts())