from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()  # Load environment variables from .env file

def create_agent():
    # Load vector database
    db = FAISS.load_local(
        "faiss_index",
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}  # avoid meta tensor error
        ),
        allow_dangerous_deserialization=True
    )
    
    # Fetch Hugging Face API token from environment
    hf_token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
    if not hf_token:
        raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in environment variables")

    # Load LLM from HuggingFace Hub
    llm = HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0.7}
    )

    # Create the Retrieval QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(),
        return_source_documents=True
    )

    return qa_chain
