import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="Local AI assistant - Backend")

# Configuring OpenAI' client
client = OpenAI(
    base_url="http://llama-server:8080/v1",
    api_key="sk-no-key-required" # llama.cpp doesnt require api key in a local environment
)

class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7

@app.get("/")
def read_root():
    return {"status": "online", "service": "Assistant - Backend"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        response = client.chat.completions.create(
            model="deepseek-r1", 
            messages=[
                {"role": "system", "content": "You are a private AI assistant"},
                {"role": "user", "content": request.message}
            ],
            temperature=request.temperature,
            max_tokens=2048,
        )
        ai_message = response.choices[0].message.content
        return {"response": ai_message}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
