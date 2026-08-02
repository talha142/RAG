"""
Builds the retrieval-augmented generation chain:
  user question -> retrieve top-k chunks from Chroma -> prompt Ollama -> grounded answer

Import get_rag_chain() from app.py.
"""

import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"  # change to whatever model you've pulled, e.g. "mistral"
TOP_K = 4

PROMPT_TEMPLATE = """You are a careful medical information assistant. Answer the
question using ONLY the context below, which comes from MedlinePlus. If the
context doesn't contain enough information to answer, say so clearly instead
of guessing.

Always:
- Give a clear, plain-language answer.
- Stay strictly grounded in the provided context.
- Remind the user this is general health information, not a diagnosis, and
  that they should consult a healthcare professional for medical advice.

Context:
{context}

Question: {question}

Answer:"""


def load_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        raise FileNotFoundError(
            f"No vector store found at {CHROMA_DIR}. Run `python ingest.py` first."
        )
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_rag_chain():
    vectordb = load_vectorstore()
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    llm = Ollama(model=OLLAMA_MODEL, temperature=0.2)

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


def ask(chain, question: str):
    """Run a question through the chain. Returns (answer, sources)."""
    result = chain.invoke(question)
    answer = result["answer"]
    sources = []
    seen = set()
    for doc in result.get("context", []):
        title = doc.metadata.get("title", "Unknown source")
        url = doc.metadata.get("source", "")
        key = (title, url)
        if key not in seen:
            seen.add(key)
            sources.append({"title": title, "url": url})
    return answer, sources
