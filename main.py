from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
import os
import uvicorn

# Load environment variables
load_dotenv()

# FastAPI App
app = FastAPI(
    title="Gemini Chatbot API with ChromaDB",
    description="PDF + RAG chatbot using Gemini, MongoDB, and Chroma",
    version="4.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB Setup
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
client = MongoClient(MONGO_URI)
db = client["gemini_chatbot"]
chats_collection = db["chat_history"]
documents_collection = db["documents"]

# Gemini Setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY in environment variables")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

# Embedding Model
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# ChromaDB Setup
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="pdf_chunks", embedding_function=embedding_function)

# Pydantic Models
class UserQuery(BaseModel):
    query: str
    user_id: str = "default"
    chat_history: list[dict] = []

class BotResponse(BaseModel):
    response: str
    chat_history: list[dict]

# Root
@app.get("/")
def read_root():
    return {"message": "Gemini Chatbot API with ChromaDB and PDF RAG"}

# List Models
@app.get("/models")
async def list_models():
    return genai.list_models()

# Get Chat History
@app.get("/history/{user_id}")
async def get_history(user_id: str, limit: int = 10):
    history = list(chats_collection.find(
        {"user_id": user_id},
        {"_id": 0, "history": 1}
    ).sort("timestamp", -1).limit(limit))
    return {"history": [doc["history"] for doc in history]}

# Upload PDF and Embed to ChromaDB
@app.post("/upload_pdf/")
async def upload_pdf(file: UploadFile = File(...), user_id: str = "default"):
    try:
        pdf_reader = PyPDF2.PdfReader(file.file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        chunks = [text[i:i + 500] for i in range(0, len(text), 500)]

        # Save embeddings to Chroma with metadata
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"user_id": user_id, "chunk_index": i, "filename": file.filename}],
                ids=[f"{user_id}_{file.filename}_{i}"]
            )

        # Store full doc in MongoDB
        documents_collection.insert_one({
            "user_id": user_id,
            "filename": file.filename,
            "content": text,
            "timestamp": datetime.utcnow()
        })

        return {"message": "PDF uploaded and embedded", "chunks_stored": len(chunks)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chat without context
@app.post("/chat", response_model=BotResponse)
async def chat(user_query: UserQuery):
    try:
        if not user_query.chat_history:
            last_chat = chats_collection.find_one(
                {"user_id": user_query.user_id},
                sort=[("timestamp", -1)]
            )
            if last_chat:
                user_query.chat_history = last_chat["history"]

        if user_query.chat_history:
            chat_session = model.start_chat(history=user_query.chat_history)
            response = chat_session.send_message(user_query.query)
        else:
            response = model.generate_content(user_query.query)

        bot_response = response.text
        updated_history = user_query.chat_history.copy()
        updated_history.extend([
            {"role": "user", "parts": [user_query.query]},
            {"role": "model", "parts": [bot_response]}
        ])

        chats_collection.insert_one({
            "user_id": user_query.user_id,
            "query": user_query.query,
            "response": bot_response,
            "history": updated_history,
            "timestamp": datetime.utcnow()
        })

        return {
            "response": bot_response,
            "chat_history": updated_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# RAG Chat using ChromaDB
@app.post("/rag_chat", response_model=BotResponse)
async def rag_chat(user_query: UserQuery):
    try:
        # 1. Similarity Search in Chroma
        results = collection.query(
            query_texts=[user_query.query],
            n_results=5,
            where={"user_id": user_query.user_id}
        )
        retrieved_chunks = results["documents"][0]
        context = "\n".join(retrieved_chunks)

        # 2. Build prompt
        prompt = f"Context:\n{context}\n\nQuestion: {user_query.query}"

        # 3. Generate response
        response = model.generate_content(prompt)
        bot_response = response.text

        # 4. Update and save chat history
        updated_history = user_query.chat_history.copy()
        updated_history.extend([
            {"role": "user", "parts": [user_query.query]},
            {"role": "model", "parts": [bot_response]}
        ])

        # Save to MongoDB
        chats_collection.insert_one({
            "user_id": user_query.user_id,
            "query": user_query.query,
            "response": bot_response,
            "history": updated_history,
            "context_used": retrieved_chunks,
            "timestamp": datetime.utcnow()
        })

        # 5. Return with history
        return {
            "response": bot_response,
            "chat_history": updated_history
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
