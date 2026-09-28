import pandas as pd
import numpy as np
import torch

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

from sklearn.utils.class_weight import compute_class_weight


MODEL_NAME = "vinai/bertweet-base"

TRAIN_PATH = "../datasets/emotion/emotion_train_clean.csv"
VAL_PATH = "../datasets/emotion/emotion_validation_clean.csv"
TEST_PATH = "../datasets/emotion/emotion_test_clean.csv"

MODEL_PATH = "./emotion_model_v2"


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


train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df = train_df[["Text", "label"]]
val_df = val_df[["Text", "label"]]
test_df = test_df[["Text", "label"]]


train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True
)


def tokenize(batch):
    return tokenizer(
        batch["Text"],
        truncation=True,
        max_length=64
    )


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)
test_dataset = test_dataset.map(tokenize, batched=True)


train_dataset = train_dataset.remove_columns(["Text"])
val_dataset = val_dataset.remove_columns(["Text"])
test_dataset = test_dataset.remove_columns(["Text"])


train_dataset.set_format("torch")
val_dataset.set_format("torch")
test_dataset.set_format("torch")


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=8,
    id2label={i: name for i, name in enumerate(label_names)},
    label2id={name: i for i, name in enumerate(label_names)}
)


classes = np.arange(8)

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=train_df["label"]
)

weights = np.clip(weights, 0, 5)

class_weights = torch.tensor(
    weights,
    dtype=torch.float
)


print("\nClass weights:")

for i, weight in enumerate(class_weights):
    print(
        f"{label_names[i]}: "
        f"{weight.item():.2f}"
    )


class WeightedTrainer(Trainer):

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None
    ):

        labels = inputs.pop("labels")

        outputs = model(**inputs)

        loss_function = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(outputs.logits.device)
        )

        loss = loss_function(
            outputs.logits,
            labels
        )

        return (
            (loss, outputs)
            if return_outputs
            else loss
        )


def compute_metrics(pred):

    predictions = np.argmax(
        pred.predictions,
        axis=1
    )

    labels = pred.label_ids

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            labels,
            predictions,
            average="macro",
            zero_division=0
        )
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
    output_dir="./emotion_training_v2",

    eval_strategy="epoch",
    save_strategy="epoch",

    learning_rate=2e-5,

    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,

    num_train_epochs=2,

    weight_decay=0.01,

    logging_steps=100,

    load_best_model_at_end=True,

    metric_for_best_model="macro_f1",
    greater_is_better=True,

    report_to="none"
)


data_collator = DataCollatorWithPadding(
    tokenizer=tokenizer
)


trainer = WeightedTrainer(
    model=model,

    args=training_args,

    train_dataset=train_dataset,
    eval_dataset=val_dataset,

    processing_class=tokenizer,

    data_collator=data_collator,

    compute_metrics=compute_metrics
)


print("\nStarting emotion model training...\n")

trainer.train()


print("\nValidation results:")

validation_results = trainer.evaluate()

print(validation_results)


print("\nTest results:")

test_results = trainer.predict(
    test_dataset
)


predictions = np.argmax(
    test_results.predictions,
    axis=1
)


print("\nClassification Report:\n")

print(
    classification_report(
        test_results.label_ids,
        predictions,
        target_names=label_names,
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


print(
    "\nModel saved to:",
    MODEL_PATH
)