import pandas as pd
import numpy as np

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)


MODEL_NAME = "vinai/bertweet-base"


train_df = pd.read_csv("../datasets/sentiment/sentiment_train.csv")
valid_df = pd.read_csv("../datasets/sentiment/sentiment_validation.csv")
test_df = pd.read_csv("../datasets/sentiment/sentiment_test.csv")


train_df = train_df[["text", "label"]]
valid_df = valid_df[["text", "label"]]
test_df = test_df[["text", "label"]]


train_df = (
    train_df
    .groupby("label", group_keys=False)
    .sample(n=5000, random_state=42)
)


print("Training samples:", len(train_df))
print("Validation samples:", len(valid_df))
print("Test samples:", len(test_df))

print("\nTraining class distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation class distribution:")
print(valid_df["label"].value_counts().sort_index())

print("\nTest class distribution:")
print(test_df["label"].value_counts().sort_index())


train_dataset = Dataset.from_pandas(
    train_df,
    preserve_index=False
)

valid_dataset = Dataset.from_pandas(
    valid_df,
    preserve_index=False
)

test_dataset = Dataset.from_pandas(
    test_df,
    preserve_index=False
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    normalization=True
)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=64
    )


train_dataset = train_dataset.map(
    tokenize,
    batched=True
)

valid_dataset = valid_dataset.map(
    tokenize,
    batched=True
)

test_dataset = test_dataset.map(
    tokenize,
    batched=True
)


train_dataset = train_dataset.remove_columns(["text"])
valid_dataset = valid_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label={
        0: "Negative",
        1: "Neutral",
        2: "Positive"
    },
    label2id={
        "Negative": 0,
        "Neutral": 1,
        "Positive": 2
    }
)


def compute_metrics(eval_pred):

    predictions, labels = eval_pred

    predictions = np.argmax(
        predictions,
        axis=1
    )

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="weighted",
            zero_division=0
        )
    )

    precision_macro, recall_macro, f1_macro, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro
    }


training_args = TrainingArguments(
    output_dir="./results_15000",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,

    num_train_epochs=2,

    weight_decay=0.01,

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    report_to="none",

    dataloader_num_workers=0,

    save_total_limit=1
)


trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


print("\n========================================")
print("Starting BERTweet training")
print("========================================\n")


trainer.train()


print("\n========================================")
print("FINAL VALIDATION RESULTS")
print("========================================")

validation_results = trainer.evaluate(
    valid_dataset
)

print(validation_results)


print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

test_results = trainer.evaluate(
    test_dataset
)

print(test_results)


print("\n========================================")
print("DETAILED TEST CLASSIFICATION REPORT")
print("========================================")


test_output = trainer.predict(
    test_dataset
)

test_predictions = np.argmax(
    test_output.predictions,
    axis=1
)

test_labels = test_output.label_ids


print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=[
            "Negative",
            "Neutral",
            "Positive"
        ],
        digits=4,
        zero_division=0
    )
)


print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")


cm = confusion_matrix(
    test_labels,
    test_predictions
)


print("\nRows = Actual")
print("Columns = Predicted\n")

print(
    pd.DataFrame(
        cm,
        index=[
            "Negative",
            "Neutral",
            "Positive"
        ],
        columns=[
            "Negative",
            "Neutral",
            "Positive"
        ]
    )
)


trainer.save_model(
    "./sentiment_model_15000"
)

tokenizer.save_pretrained(
    "./sentiment_model_15000"
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print("./sentiment_model_15000")