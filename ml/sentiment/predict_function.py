import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification



MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "sentiment_model_15000"
)

labels = [
    "negative",
    "neutral",
    "positive"
]


tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.eval()


def analyze_sentiment(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1
    )[0]

    index = torch.argmax(probabilities).item()

    return {
        "label": labels[index],
        "confidence": float(probabilities[index])
    }


if __name__ == "__main__":

    print("Sentiment model loaded.")
    print("Type 'exit' to stop.\n")

    while True:

        text = input("You: ")

        if text.lower() == "exit":
            break

        result = analyze_sentiment(text)

        print("\nSentiment:")
        print(
            f"{result['label']} "
            f"({result['confidence'] * 100:.2f}%)"
        )
        print()