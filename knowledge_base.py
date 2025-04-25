from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def load_and_process_pdfs(pdf_folder="insurance_pdfs"):
    """Step 1a: Document Ingestion"""
    text = ""
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    return text

def chunk_documents(text):
    """Step 1b: Text Splitting"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_text(text)

def create_vector_store(chunks):
    """Steps 1c-1d: Embedding & Vector Storage"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)

def get_relevant_context(query, vector_store, k=3):
    """Step 2: Retrieval"""
    docs = vector_store.similarity_search(query, k=k)
    return "\n".join([d.page_content for d in docs])

#print()