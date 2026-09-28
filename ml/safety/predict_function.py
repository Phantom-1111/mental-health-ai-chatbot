import os
import joblib

BASE_DIR = os.path.dirname(__file__)

VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "safety_model",
    "tfidf_vectorizer.pkl"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "safety_model",
    "logistic_regression.pkl"
)


vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


def analyze_safety(text):

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    if prediction == 1:
        label = "suicide"
    else:
        label = "non-suicide"

    return {
        "label": label,
        "confidence": float(probabilities[prediction]),
        "suicide_probability": float(probabilities[1]),
        "non_suicide_probability": float(probabilities[0])
    }


if __name__ == "__main__":

    print("Safety model loaded.")
    print("Type 'exit' to stop.\n")

    while True:

        text = input("You: ")

        if text.lower() == "exit":
            break

        result = analyze_safety(text)

        print("\nSafety:")
        print(
            f"{result['label']} "
            f"({result['confidence'] * 100:.2f}%)"
        )

        print(
            f"Suicide probability: "
            f"{result['suicide_probability'] * 100:.2f}%"
        )

        print(
            f"Non-suicide probability: "
            f"{result['non_suicide_probability'] * 100:.2f}%"
        )

        print()