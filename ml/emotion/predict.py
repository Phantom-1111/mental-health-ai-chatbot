import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_PATH = "./emotion_model"

labels = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "shame",
    "surprise"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

model.eval()

print("Emotion model loaded.")
print("Type a sentence or type 'exit' to stop.\n")

while True:

    text = input("You: ")

    if text.lower() == "exit":
        break

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=64
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=1)

    top_values, top_indices = torch.topk(probabilities[0], 3)

    print("\nTop predictions:")

    for value, index in zip(top_values, top_indices):
        print(
            f"{labels[index.item()]}: "
            f"{value.item() * 100:.2f}%"
        )

    print()