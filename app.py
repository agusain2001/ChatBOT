import os
import pandas as pd
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import CSVLoader, JSONLoader, TextLoader  # Import necessary loaders
from langchain.llms import HuggingFacePipeline
import streamlit as st
from transformers import pipeline

# Step 1: Data Loading
def load_data(file_path):
    """Load and preprocess data from a CSV, JSON, or TXT file."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    if file_path.endswith(".csv"):
        loader = CSVLoader(file_path)
    elif file_path.endswith(".json"):
        loader = JSONLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV, JSON, or TXT file.")

    documents = loader.load()
    return documents

# Step 2: Setup Retrieval-Augmented Generation (RAG) Pipeline
def setup_rag_pipeline(documents):
    """Set up a Retrieval-Augmented Generation pipeline using LangChain with open-source tools."""
    # Use Hugging Face embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    # Use a Hugging Face pipeline for the language model
    hf_pipeline = pipeline("text-generation", model="gpt2")
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain

# Step 3: Streamlit Chatbot Interface
def chatbot_interface(qa_chain):
    """Streamlit interface for the chatbot."""
    st.title("RAG-Based Chatbot")
    st.write("Ask me anything based on the knowledge base!")

    user_question = st.text_input("Your Question:")

    if user_question:
        response = qa_chain.run(user_question)
        st.write("**Answer:**", response)

# Main Script
if __name__ == "__main__":
    st.sidebar.title("Chatbot Configuration")
    dataset_path = st.sidebar.text_input("Dataset Path (CSV):", "data.csv") 

    if st.sidebar.button("Load Dataset"):
        try:
            st.sidebar.write("Loading dataset...")
            documents = load_data(dataset_path)
            st.sidebar.success("Dataset loaded successfully!")

            st.sidebar.write("Setting up the RAG pipeline...")
            qa_chain = setup_rag_pipeline(documents)
            st.sidebar.success("RAG pipeline set up successfully!")

            chatbot_interface(qa_chain)
        except Exception as e:
            st.sidebar.error(f"Error: {e}")