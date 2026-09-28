from sentiment.predict_function import analyze_sentiment
from emotion.predict_function import analyze_emotion
from safety.predict_function import analyze_safety


def analyze_text(text):

    sentiment = analyze_sentiment(text)
    emotion = analyze_emotion(text)
    safety = analyze_safety(text)

    safety_confidence = safety["suicide_probability"]

    if safety_confidence >= 0.80:
        safety_level = "high"

    elif safety_confidence >= 0.50:
        safety_level = "moderate"

    else:
        safety_level = "low"

    return {
        "text": text,

        "sentiment": {
            "label": sentiment["label"],
            "confidence": sentiment["confidence"]
        },

        "emotion": {
            "label": emotion["label"],
            "confidence": emotion["confidence"]
        },

        "safety": {
            "label": safety["label"],
            "suicide_probability": safety["suicide_probability"],
            "non_suicide_probability": safety["non_suicide_probability"],
            "level": safety_level
        }
    }


if __name__ == "__main__":

    print("Combined analyzer loaded.")
    print("Type 'exit' to stop.\n")

    while True:

        text = input("You: ")

        if text.lower() == "exit":
            break

        result = analyze_text(text)

        print("\nAnalysis:")

        print(
            f"Sentiment: "
            f"{result['sentiment']['label']} "
            f"({result['sentiment']['confidence'] * 100:.2f}%)"
        )

        print(
            f"Emotion: "
            f"{result['emotion']['label']} "
            f"({result['emotion']['confidence'] * 100:.2f}%)"
        )

        print(
            f"Safety: "
            f"{result['safety']['level']} "
            f"({result['safety']['suicide_probability'] * 100:.2f}%)"
        )

        print()