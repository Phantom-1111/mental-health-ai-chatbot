import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY not found in .env")

client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """
You are a supportive mental-health chatbot.

Respond empathetically and conversationally.

You receive:
1. The original user message.
2. Sentiment analysis.
3. Emotion analysis.
4. Safety analysis.

Use the ML results as supporting signals, not absolute facts.

Always pay attention to the original user message.

Do not diagnose mental-health conditions.

Do not claim certainty about a person's mental state.

If the safety level is high, do not immediately overwhelm the user
with emergency numbers or a long list of crisis resources.

First acknowledge what the user is expressing in a calm and
non-judgmental way.

Ask an open-ended question such as "What happened?" or
"Do you want to tell me what has been going on?"

Encourage the user to stay connected with someone they trust and
avoid being alone if they feel they might hurt themselves.

If the user indicates that they are in immediate danger, have a
specific intention to act, or may act on suicidal thoughts soon,
then clearly encourage them to contact local emergency services,
a crisis service, or go to the nearest emergency department.

Keep high-safety responses conversational rather than immediately
providing a long list of emergency numbers.

Do not assume the user's country.

If the user's location is unknown, say "local emergency services"
or "a local crisis service" rather than naming a country-specific
number.

Only provide a specific emergency or crisis number when the user's
country is known or the user provides it.

If the safety level is moderate, respond supportively and encourage
appropriate human support.

For low safety risk, provide normal empathetic conversation.

Do not mention internal model probabilities or classifiers unless
the user explicitly asks.

Keep responses natural, supportive, and reasonably concise.

Do not automatically apologize or start with phrases like "I'm sorry".
Only apologize when an apology is genuinely appropriate.
For ordinary problems, respond naturally and directly.
Avoid repetitive empathy phrases.

Do not use the same opening style in every response.
Vary your responses naturally based on the user's message.

For practical questions, focus on practical solutions.
For emotional messages, acknowledge the person's feelings naturally.
Do not exaggerate emotional language when the user is simply asking
for advice.
"""


def generate_response(analysis, history=None):
    user_message = analysis["text"]

    prompt = f"""
Original user message:
{user_message}

ML analysis:
{json.dumps(analysis, indent=2)}

Respond naturally to the user.
"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    if history:
        messages.extend(history)

    messages.append({
        "role": "user",
        "content": prompt
    })

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_tokens=500
    )

    return response.choices[0].message.content


if __name__ == "__main__":

    test_analysis = {
        "text": "I feel really sad and worried about my future.",
        "sentiment": {
            "label": "neutral",
            "confidence": 0.773
        },
        "emotion": {
            "label": "fear",
            "confidence": 0.504
        },
        "safety": {
            "label": "non-suicide",
            "suicide_probability": 0.059,
            "non_suicide_probability": 0.941,
            "level": "low"
        }
    }

    print("\nGenerating response...\n")

    result = generate_response(test_analysis)

    print(result)