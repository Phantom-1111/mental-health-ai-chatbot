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
    classification_report,
    confusion_matrix
)

MODEL_NAME = "vinai/bertweet-base"

TRAIN_PATH = "../../datasets/emotion/emotion_train.csv"
VAL_PATH = "../../datasets/emotion/emotion_validation.csv"
TEST_PATH = "../../datasets/emotion/emotion_test.csv"

MODEL_PATH = "./tweeteval_model"

labels = [
    "anger",
    "joy",
    "optimism",
    "sadness"
]

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = train_df[["text", "label"]]
val_df = val_df[["text", "label"]]
test_df = test_df[["text", "label"]]

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=64
    )

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)

train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])
test_dataset = test_dataset.remove_columns(["text"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=4,
    id2label={i: name for i, name in enumerate(labels)},
    label2id={name: i for i, name in enumerate(labels)}
)

def compute_metrics(pred):

    predictions = np.argmax(pred.predictions, axis=1)
    actual = pred.label_ids

    accuracy = accuracy_score(actual, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        actual,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

training_args = TrainingArguments(
    output_dir="./training",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    num_train_epochs=1,

    weight_decay=0.01,

    logging_steps=50,

    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    greater_is_better=True,

    report_to="none"
)

data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)

trainer = Trainer(
    model=model,
    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,
    data_collator=data_collator,

    compute_metrics=compute_metrics
)

print("\nStarting TweetEval emotion training...\n")

trainer.train()

print("\nValidation results:")
print(trainer.evaluate())

print("\nTest results:")

test_results = trainer.predict(test_dataset)

predictions = np.argmax(
    test_results.predictions,
    axis=1
)

print("\nClassification Report:\n")

print(
    classification_report(
        test_results.label_ids,
        predictions,
        target_names=labels,
        zero_division=0
    )
)

print("\nConfusion Matrix:\n")

print(
    confusion_matrix(
        test_results.label_ids,
        predictions
    )
)

trainer.save_model(MODEL_PATH)
tokenizer.save_pretrained(MODEL_PATH)

print("\nModel saved to:", MODEL_PATH)