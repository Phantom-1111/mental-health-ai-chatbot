import sys
import os

from fastapi import FastAPI
from pydantic import BaseModel

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "llm"))

from combined.analyze import analyze_text
from response import generate_response


app = FastAPI(title="Mental Health AI Chatbot")
conversation_history = []


class ChatRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"message": "Mental Health AI Chatbot API is running"}



@app.post("/chat")
def chat(request: ChatRequest):

    analysis = analyze_text(request.message)

    response = generate_response(
        analysis,
        conversation_history
    )

    conversation_history.append({
        "role": "user",
        "content": request.message
    })

    conversation_history.append({
        "role": "assistant",
        "content": response
    })

    return {
        "response": response,
        "analysis": analysis
    }