# Medical RAG Chatbot

A Retrieval-Augmented Generation chatbot that answers general health questions
grounded in [MedlinePlus](https://medlineplus.gov) content. Fully local and free:
no paid APIs required.

**Stack:** LangChain · ChromaDB · HuggingFace sentence-transformers · Ollama · Streamlit

## How it works

1. `ingest.py` queries the official MedlinePlus Web Service for ~100+ curated
   health topics (see `topics.py`), extracts each topic's full summary, chunks
   the text, embeds it with a local sentence-transformer, and stores it in a
   local Chroma vector database.
2. `rag_chain.py` builds a LangChain `RetrievalQA` chain: given a question, it
   retrieves the most relevant chunks from Chroma and passes them to a local
   Ollama model as context, so answers stay grounded in the source material.
3. `app.py` is a Streamlit chat UI on top of that chain, with source citations
   shown under each answer.

## Setup

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Make sure Ollama is running with a model pulled

```bash
ollama pull llama3.1:8b
```

(You can swap in a smaller model like `mistral` or `llama3.2` — just update
`OLLAMA_MODEL` in `rag_chain.py`.)

### 3. Build the vector store

```bash
python ingest.py
```

This fetches all topics from MedlinePlus, saves raw text under `data/raw/`
(useful for auditing what the bot actually knows), and persists embeddings to
`data/chroma_db/`. Takes a few minutes the first time (downloads the embedding
model, then embeds ~100+ documents).

### 4. Run the app

```bash
streamlit run app.py
```

## Customizing

- **Add/remove topics** — edit `HEALTH_TOPICS` in `topics.py`, then re-run
  `python ingest.py`.
- **Change the LLM** — edit `OLLAMA_MODEL` in `rag_chain.py`.
- **Tune retrieval** — edit `TOP_K` (chunks retrieved per question) and
  `CHUNK_SIZE` / `CHUNK_OVERLAP` in `ingest.py`.
- **Change the prompt / tone** — edit `PROMPT_TEMPLATE` in `rag_chain.py`.

## Notes

- All data used is from MedlinePlus (National Library of Medicine), a public,
  authoritative consumer health resource — attribution is included via the
  "Sources" section under each answer.
- This is an educational project, not a medical device. The UI includes a
  standing disclaimer that answers are not medical advice.
- If `ingest.py` returns fewer documents than expected, some topic queries may
  not have matched — check the console output for topics that returned zero
  results and adjust the term in `topics.py`.
