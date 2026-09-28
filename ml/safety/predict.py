import joblib


VECTORIZER_PATH = "./safety_model/tfidf_vectorizer.pkl"
MODEL_PATH = "./safety_model/logistic_regression.pkl"


vectorizer = joblib.load(VECTORIZER_PATH)
model = joblib.load(MODEL_PATH)


print("Safety model loaded.")
print("Type a message or type 'exit' to stop.\n")


while True:

    text = input("You: ")

    if text.lower() == "exit":
        break

    text_vector = vectorizer.transform([text])

    prediction = model.predict(text_vector)[0]

    probabilities = model.predict_proba(text_vector)[0]

    non_suicide_probability = probabilities[0]
    suicide_probability = probabilities[1]

    if prediction == 1:
        label = "suicide"
    else:
        label = "non-suicide"

    print("\nSafety prediction:")
    print(f"Class: {label}")
    print(f"Non-suicide: {non_suicide_probability * 100:.2f}%")
    print(f"Suicide: {suicide_probability * 100:.2f}%")
    print()