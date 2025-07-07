import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

# ✅ Step 1: Load all PDFs from data folder
def load_pdfs(folder_path):
    all_docs = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            path = os.path.join(folder_path, file)
            loader = PyPDFLoader(path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = file
            all_docs.extend(documents)
    return all_docs

# ✅ Step 2: Split into chunks
def split_docs(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(documents)

# ✅ Step 3: Embed and save vector DB using HuggingFace
def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("db/agromind_vectorstore")
    print("✅ Vector store created and saved.")

# ✅ Step 4: Load vectorstore and answer questions
def ask_question(query):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local("db/agromind_vectorstore", embeddings)
    docs = vectorstore.similarity_search(query, k=3)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo")  # Optional: works with OpenAI key
    chain = load_qa_chain(llm, chain_type="stuff")

    return chain.run(input_documents=docs, question=query)
