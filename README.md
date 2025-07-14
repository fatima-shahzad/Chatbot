# 🤖 Gemini Chatbot with PDF RAG

A FastAPI-powered chatbot using Google's Gemini LLM, ChromaDB for semantic search, and MongoDB for chat/document storage. Upload PDF files, embed text using free transformer models, and ask questions with document-aware responses powered by Retrieval-Augmented Generation (RAG).

---

## 🚀 Features

- 🔐 Google Gemini integration via Generative AI API
- 📄 Upload PDFs and split text into 500-character chunks
- 🧠 Embed text using `all-MiniLM-L6-v2` (Sentence Transformers)
- 🔎 Semantic similarity search using ChromaDB (Cosine Similarity)
- 🗂️ MongoDB for storing chat history and documents
- 💬 Intelligent Q&A using Gemini + retrieved context (RAG)
- 🌐 FastAPI backend with full CORS support

---

## 🛠️ Tech Stack

- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Generative AI (Gemini)](https://ai.google.dev/)
- [ChromaDB](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [MongoDB](https://www.mongodb.com/)
- [PyPDF2](https://pypi.org/project/PyPDF2/)

---

## 📦 Installation

1. Clone the Repository
   ```bash
   git clone https://github.com/yourusername/Chatbot.git
   cd Chatbot


2. Create and Activate Virtual Environment
   ```bash
   python -m venv venv
   source venv/bin/activate

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   
4. Set Environment Variables
  Create a .env file in the root directory and add your credentials:
  GEMINI_API_KEY=your_google_gemini_api_key.
  MONGO_URI=mongodb://localhost:27017
  
6. Run the Application
   ```bash
   uvicorn main:app --reload
   
The FastAPI server will start at http://127.0.0.1:8000
