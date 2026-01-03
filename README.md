# 📄 Chat with Multiple PDFs (RAG Application)

A Retrieval-Augmented Generation (RAG) based web application that allows users to upload multiple PDF documents and ask context-aware questions using **Google Gemini** and **Hugging Face embeddings**.

---

## 🚀 Features

- Upload and chat with multiple PDF files
- Semantic search using FAISS vector database
- Hugging Face embeddings (`all-MiniLM-L6-v2`)
- Google Gemini (`gemini-2.0-flash`) for answer generation
- Context-aware responses (no hallucinations)
- Conversation history with timestamps
- Download chat history as CSV
- Interactive Streamlit UI

---

## 🧠 Tech Stack

- **Language:** Python  
- **Frontend:** Streamlit  
- **LLM:** Google Gemini  
- **Embeddings:** Hugging Face Sentence Transformers  
- **Vector Database:** FAISS  
- **Framework:** LangChain  

---

## 📁 Project Structure

chat-with-multiple-pdfs/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore


---

## ⚙️ Installation & Setup

1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/chat-with-multiple-pdfs.git
cd chat-with-multiple-pdfs

2️⃣ Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Run the application
streamlit run app.py

🔑 API Key Setup

Get your Google API key from: https://ai.google.dev/

Enter the API key in the Streamlit sidebar when prompted

🧩 How It Works

User uploads one or more PDF files

PDF text is extracted and split into overlapping chunks

Embeddings are generated using Hugging Face models

Chunks are stored in a FAISS vector database

Relevant chunks are retrieved using semantic search

Google Gemini generates answers strictly from retrieved context

📌 Important Notes

FAISS index is generated at runtime and is not committed to GitHub

API keys are never stored in the codebase

PDFs are processed locally
