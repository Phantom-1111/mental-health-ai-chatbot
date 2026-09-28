import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "ml"))
sys.path.append(os.path.join(os.path.dirname(__file__), "llm"))

from combined.analyze import analyze_text
from response import generate_response


history = []

print("Mental Health AI Chatbot")
print("Type 'exit' to stop.\n")

while True:

    text = input("You: ")

    if text.lower() == "exit":
        break

    analysis = analyze_text(text)

    response = generate_response(
        analysis,
        history
    )

    history.append({
        "role": "user",
        "content": text
    })

    history.append({
        "role": "assistant",
        "content": response
    })

    print("\nAssistant:")
    print(response)
    print()