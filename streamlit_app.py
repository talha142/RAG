import streamlit as st
from agromind_rag_chatbot import load_pdfs, split_docs, create_vectorstore, ask_question
import os

st.set_page_config(page_title="🌾 AgroMind RAG Chatbot", page_icon="🤖")
st.title("🌾 AgroMind RAG Chatbot")
st.markdown("Ask any agriculture-related question from the PDF knowledge base.")

# Build vectorstore if not already done
DATA_PATH = "data"
VECTOR_DB_PATH = "db/agromind_vectorstore"

if not os.path.exists(VECTOR_DB_PATH):
    with st.spinner("🔄 Creating vector store from PDF documents..."):
        docs = load_pdfs(DATA_PATH)
        chunks = split_docs(docs)
        create_vectorstore(chunks)
    st.success("✅ Vector store created!")
else:
    st.info("✅ Vector store already exists.")

# Text input for questions
user_input = st.text_input("🔍 Ask your agriculture question:")

# Show response
if user_input:
    with st.spinner("🤖 Generating answer..."):
        response = ask_question(user_input)
    st.markdown("**🧠 Answer:**")
    st.write(response)
