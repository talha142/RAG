"""
Ingestion pipeline for the Medical RAG Chatbot.

Steps:
1. Query the MedlinePlus Web Service for each topic in topics.py
2. Parse the XML response, extract title / url / full-summary
3. Clean the text (strip any residual HTML tags from the summary)
4. Save each topic as its own .txt file in data/raw/ (useful for debugging/audit)
5. Chunk the text with LangChain's RecursiveCharacterTextSplitter
6. Embed chunks with a local HuggingFace sentence-transformer
7. Persist everything into a local Chroma vector store (data/chroma_db/)

Run:
    python ingest.py
"""

import os
import re
import time
import requests
import xml.etree.ElementTree as ET

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from topics import HEALTH_TOPICS

MEDLINEPLUS_URL = "https://wsearch.nlm.nih.gov/ws/query"
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "data", "chroma_db")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 700
CHUNK_OVERLAP = 100


def strip_html(text: str) -> str:
    """Remove HTML tags that sometimes appear inside the full-summary field."""
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_topic(term: str, retmax: int = 3):
    """
    Query MedlinePlus for a given term. Returns a list of dicts:
    [{title, url, summary}, ...]
    """
    params = {"db": "healthTopics", "term": term, "retmax": retmax}
    try:
        resp = requests.get(MEDLINEPLUS_URL, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  ! request failed for '{term}': {e}")
        return []

    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        print(f"  ! XML parse failed for '{term}': {e}")
        return []

    results = []
    for doc in root.findall(".//document"):
        url = doc.attrib.get("url", "")
        title, summary = "", ""
        for content in doc.findall("content"):
            name = content.attrib.get("name", "")
            text = "".join(content.itertext())
            if name == "title":
                title = strip_html(text)
            elif name == "FullSummary":
                summary = strip_html(text)
        if title and summary:
            results.append({"title": title, "url": url, "summary": summary})
    return results


def save_raw(title: str, url: str, summary: str):
    os.makedirs(RAW_DIR, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]+", "_", title.lower()).strip("_")
    path = os.path.join(RAW_DIR, f"{safe_name}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"TITLE: {title}\nSOURCE: {url}\n\n{summary}")
    return path


def build_documents() -> list[Document]:
    """Fetch all topics and turn them into LangChain Documents."""
    seen_titles = set()
    documents = []

    for i, term in enumerate(HEALTH_TOPICS, 1):
        print(f"[{i}/{len(HEALTH_TOPICS)}] fetching '{term}'...")
        results = fetch_topic(term)

        for item in results:
            if item["title"] in seen_titles:
                continue
            seen_titles.add(item["title"])

            save_raw(item["title"], item["url"], item["summary"])

            documents.append(
                Document(
                    page_content=item["summary"],
                    metadata={"title": item["title"], "source": item["url"]},
                )
            )

        time.sleep(0.3)  # be polite to the API

    print(f"\nFetched {len(documents)} unique topic documents.")
    return documents


def chunk_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks: list[Document]):
    print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Embedding chunks and writing to Chroma (this can take a few minutes)...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
    )
    vectordb.persist()
    print(f"Vector store persisted to {CHROMA_DIR}")


def main():
    documents = build_documents()
    if not documents:
        print("No documents fetched — check your network connection and try again.")
        return
    chunks = chunk_documents(documents)
    build_vectorstore(chunks)
    print("\nIngestion complete. You can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
