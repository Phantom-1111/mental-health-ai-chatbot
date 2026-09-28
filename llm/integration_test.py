import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))

from combined.analyze import analyze_text
from response import generate_response


text = input("You: ")

analysis = analyze_text(text)

print("\nML Analysis:")
print(analysis)

response = generate_response(analysis)

print("\nAssistant:")
print(response)