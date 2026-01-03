import streamlit as st
from pypdf import PdfReader
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA


# -------------------------------------------------
# Environment
# -------------------------------------------------
load_dotenv()
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# -------------------------------------------------
# Global Embeddings (Load Once)
# -------------------------------------------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


EMBEDDINGS = load_embeddings()


# -------------------------------------------------
# PDF Processing
# -------------------------------------------------
def read_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    db = FAISS.from_texts(chunks, EMBEDDINGS)
    db.save_local("faiss_db")


# -------------------------------------------------
# RAG QA
# -------------------------------------------------
def answer_question(question):
    if not os.path.exists("faiss_db"):
        return "Please upload and process PDFs first."

    db = FAISS.load_local(
        "faiss_db",
        EMBEDDINGS,
        allow_dangerous_deserialization=True
    )

    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = Ollama(
    model="mistral",
    temperature=0.3
)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=False
    )

    return qa_chain.run(question)


# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
def main():
    st.set_page_config(
        page_title="Chat with PDF (Ollama)",
        layout="wide"
    )

    st.header("📄 RAG-Based Chat with PDF ")

    user_question = st.text_input("Ask a question from the uploaded PDFs")

    if user_question:
        with st.spinner("Generating answer..."):
            response = answer_question(user_question)
            st.subheader("Answer")
            st.write(response)

    with st.sidebar:
        st.title("Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process PDFs"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = read_pdfs(pdf_docs)
                    chunks = split_text(raw_text)
                    create_vector_store(chunks)
                    st.success("PDFs processed successfully!")


if __name__ == "__main__":
    main()
