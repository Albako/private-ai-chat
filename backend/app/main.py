import glob
import os
import datetime
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from openai import OpenAI
from typing import List, Optional
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, select
from sqlalchemy.orm import sessionmaker, Session, declarative_base
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# database for the llm. it is created with sqlite
# database will create itself in backend/app/
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = "/app/models"
DB_PATH = os.path.join(BASE_DIR, "chat_history.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

# db engine
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class MessageDB(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True) # conversation index
    role = Column(String) # "user" or "assistant" or "system"
    content = Column(Text) # message content
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)

Base.metadata.create_all(bind=engine) # creating table in the database

# RAG configuration
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

try:
    qdrant_client = QdrantClient(url=QDRANT_URL)
except Exception as e:
    print(f"Warning: Could not connect to Qdrant: {e}")
    qdrant_client = None

# Embedded model
print("Loading embedding model...")
embed_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
COLLECTION_NAME = "knowledge"

# dependency for downloading DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# API
app = FastAPI(title="Local AI assistant - Backend")

# Configuring OpenAI' client
client = OpenAI(
    base_url="http://llama-server:8080/v1",
    api_key="sk-no-key-required" # llama.cpp doesnt require api key in a local environment
)

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"
    temperature: float = 0.7
    model_name: str

# logic of finding knowledge
def search_knowledge(query_text: str, limit: int = 3) -> str:
    if not qdrant_client:
        return ""
    try:
        cols = qdrant_client.get_collections()
        if not any(c.name == COLLECTION_NAME for c in cols.collections):
            return ""
        query_vector = embed_model.encode(query_text).tolist()
        search_result = qdrant_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )
        found_texts = [hit.payload['text'] for hit in search_result if 'text' in hit.payload]
        if not found_texts:
            return ""

        return "\n---\n".join(found_texts)
    except Exception as e:
        print(f"Error searching knowledge: {e}")
        return ""

@app.get("/")
def read_root():
    return {"status": "online", "service": "Assistant - Backend"}

@app.get("/models")
def list_models():
    models_list = []

    if not os.path.exists(MODELS_DIR):
        return {"error": f"Directory {MODELS_DIR} not found. Check docker volumes."}

    # we are looking for .gguf files
    gguf_files = glob.glob(os.path.join(MODELS_DIR, "*.gguf"))

    for file_path in gguf_files:
        try:
            filename = os.path.basename(file_path)
            size_bytes = os.path.getsize(file_path)
            size_gb = round(size_bytes / (1024 ** 3), 2)
            models_list.append({
                "name": filename,
                "size_gb": f"{size_gb} GB",
                "path": file_path
            })
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")

    return {"models": models_list, "count": len(models_list)}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        # 1. Download history of the conversations for this session
        # last 10 messages
        history_query = select(MessageDB).where(MessageDB.session_id == request.session_id).order_by(MessageDB.id)
        result = db.execute(history_query)
        stored_messages = result.scalars().all()

        # 2. RAG
        context_knowledge = search_knowledge(request.message)
        system_instruction = "You're a private and uncensored AI assistant."
        if context_knowledge:
            system_instruction += f"\n\nUSE THIS KNOWLEDGE TO GIVE AN ANSWEAR:\n{context_knowledge}"

        # list of messages for the llm
        messages_for_llm = [{"role": "system", "content": system_instruction}]

        for msg in stored_messages:
            messages_for_llm.append({"role": msg.role, "content": msg.content})

        # adding current user's message
        messages_for_llm.append({"role": "user", "content": request.message})

        # 2. Save the message in to the DB
        user_msg_db = MessageDB(session_id=request.session_id, role="user", content=request.message)
        db.add(user_msg_db)
        db.commit()

        # 3. Send the prompt to the llm
        print(f"Sending to LLM (Model: {request.model_name})...")
        response = client.chat.completions.create(
            model=request.model_name,
            messages=messages_for_llm,
            temperature=request.temperature,
            max_tokens=2048,
        )

        ai_message_content = response.choices[0].message.content

        # 4. Save LLM's answear to the DB
        ai_msg_db = MessageDB(session_id=request.session_id, role="assistant", content=ai_message_content)
        db.add(ai_msg_db)
        db.commit()

        return {
            "response": ai_message_content,
            "history_used": len(stored_messages)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
