from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

def ingest_docs():
    # Load text documents
    loader = TextLoader("data/data.txt", encoding="utf8")
    documents = loader.load()

    # Split text
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # Ensure CPU compatibility
    )

    # Create vector store
    db = FAISS.from_documents(texts, embeddings)

    # Save FAISS index
    db.save_local("faiss_index")

if __name__ == "__main__":
    ingest_docs()
