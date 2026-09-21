<p align="center">
  <img src="docs/banner.svg" alt="Medical RAG Chatbot: grounded health Q&A over MedlinePlus" width="100%">
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white">
  <img alt="LangChain" src="https://img.shields.io/badge/LangChain-RAG-2ea44f">
  <img alt="ChromaDB" src="https://img.shields.io/badge/ChromaDB-vector%20store-8957e5">
  <img alt="Ollama" src="https://img.shields.io/badge/Ollama-local%20LLM-000000">
  <img alt="Hugging Face" src="https://img.shields.io/badge/sentence--transformers-ffd21e?logo=huggingface&logoColor=black">
  <img alt="Streamlit" src="https://img.shields.io/badge/Streamlit-ff4b4b?logo=streamlit&logoColor=white">
</p>

# Medical RAG Chatbot

**A Retrieval-Augmented Generation chatbot that answers general health questions using only content from [MedlinePlus](https://medlineplus.gov), and shows the source pages under every answer.**

The health content is fetched once from the official MedlinePlus Web Service. After that, embeddings and the language model both run on your own machine (sentence-transformers and Ollama), so there are no paid APIs and no API keys.

> **Educational project, not a medical device.** The app shows a standing disclaimer, and the prompt tells the model to answer only from the retrieved context and to remind users to consult a healthcare professional.

---

## Key features

- **Grounded answers**: the prompt restricts the model to the retrieved MedlinePlus context and tells it to say so when the context is not enough, instead of guessing.
- **Source citations**: every answer lists the titles and links of the pages it was built from, with duplicates removed.
- **Curated knowledge base**: 129 search terms in `topics.py` (chronic conditions, cardiovascular, and more), each queried against MedlinePlus and de-duplicated by title.
- **Auditable data**: every fetched topic is also saved as a plain `.txt` file in `data/raw/`, so you can see exactly what the bot knows.
- **Runs locally**: `all-MiniLM-L6-v2` embeddings, a local Chroma vector store, and an Ollama model.
- **Simple chat UI**: Streamlit chat with history, a Sources expander, a Clear chat button, and error handling if the vector store or Ollama is missing.

---

## Architecture

```mermaid
flowchart LR
    subgraph ING["Offline: python ingest.py"]
        topics["topics.py<br/>129 search terms"] --> fetch["MedlinePlus Web Service<br/>db=healthTopics, up to 3 results per term"]
        fetch --> parse["Parse XML: title, URL, FullSummary<br/>strip HTML, de-duplicate by title"]
        parse --> raw["data/raw/*.txt<br/>audit copy"]
        parse --> chunk["Chunk text<br/>700 characters, 100 overlap"]
        chunk --> embed["Embed chunks<br/>all-MiniLM-L6-v2"]
    end

    embed --> chroma[("Chroma vector store<br/>data/chroma_db")]

    subgraph QRY["Online: streamlit run app.py"]
        q["User question<br/>Streamlit chat"] --> ret["Retriever<br/>top 4 chunks"]
        ret --> prompt["Prompt: answer ONLY from context,<br/>say so if unsure, add advice reminder"]
        prompt --> llm["Ollama<br/>llama3.2:3b, temperature 0.2"]
        llm --> ans["Answer + Sources (title, URL)<br/>+ medical disclaimer"]
    end

    chroma --> ret

    classDef step fill:#1f6feb,stroke:#0b3d91,color:#ffffff;
    classDef data fill:#8957e5,stroke:#512a97,color:#ffffff;
    classDef out fill:#2da44e,stroke:#116329,color:#ffffff;
    class topics,fetch,parse,chunk,embed,q,ret,prompt,llm step;
    class raw,chroma data;
    class ans out;
```

### How a question is answered

1. The question is embedded with the same model used at ingestion and matched against the Chroma store.
2. The four most similar chunks are joined into the prompt as context.
3. The Ollama model answers with a low temperature (0.2).
4. The app displays the answer and the unique source pages of the retrieved chunks.

The chain is built with LangChain's expression language (`RunnableParallel` and `RunnablePassthrough`) so the retrieved documents flow through to the UI as sources.

---

## Tech stack

| Layer | Choice |
|---|---|
| Orchestration | LangChain |
| Data source | MedlinePlus Web Service (U.S. National Library of Medicine) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persisted locally) |
| LLM | Ollama, default `llama3.2:3b` |
| UI | Streamlit |

## Project structure

```
RAG/
├── app.py            # Streamlit chat UI
├── rag_chain.py      # retriever + prompt + Ollama chain, source extraction
├── ingest.py         # MedlinePlus fetch, clean, chunk, embed, persist
├── topics.py         # the 129 health search terms
├── requirements.txt
├── docs/banner.svg
└── data/             # created by ingest.py (git-ignored)
    ├── raw/          # one .txt per fetched topic
    └── chroma_db/    # the vector store
```

---

## Setup

Requires Python 3.9 or newer and [Ollama](https://ollama.com).

```bash
git clone https://github.com/talha142/RAG.git
cd RAG

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**1. Pull a model with Ollama** (make sure Ollama is running):

```bash
ollama pull llama3.2:3b
```

**2. Build the vector store** (needs internet access to MedlinePlus; the first run also downloads the embedding model):

```bash
python ingest.py
```

**3. Start the app:**

```bash
streamlit run app.py
```

## Configuration

| What | Where | Default |
|---|---|---|
| Health topics to ingest | `HEALTH_TOPICS` in `topics.py` (re-run `ingest.py` after editing) | 129 terms |
| LLM model | `OLLAMA_MODEL` in `rag_chain.py` | `llama3.2:3b` |
| Chunks retrieved per question | `TOP_K` in `rag_chain.py` | 4 |
| Chunk size and overlap | `CHUNK_SIZE`, `CHUNK_OVERLAP` in `ingest.py` | 700 / 100 |
| Embedding model | `EMBEDDING_MODEL` in `ingest.py` and `rag_chain.py` (keep both the same) | `all-MiniLM-L6-v2` |
| Prompt and tone | `PROMPT_TEMPLATE` in `rag_chain.py` | context-only, with advice reminder |

Any model you have pulled in Ollama works, for example `mistral` or `llama3.1:8b`. A larger model usually answers better but needs more memory.

## Example questions

- What is type 2 diabetes?
- What are the symptoms of hypothyroidism?
- How is high cholesterol managed?

## Limitations

- **Coverage**: the bot only knows the topics listed in `topics.py`, using MedlinePlus topic summaries. Anything outside them should get an "I don't have enough information" reply, but that depends on the model following the prompt.
- **Answer quality depends on the local model**, and small models can still paraphrase inaccurately. Check the cited sources.
- **General information only**: no diagnosis and no personalization. There is no conversation memory either: each question is answered independently, and the chat history is only displayed.
- If some topics return no results during ingestion, adjust their search term in `topics.py`.
- Some LangChain imports used here (`langchain_community`) are being replaced by dedicated packages in recent LangChain releases, so you may see deprecation warnings.
- There is no automated evaluation or test suite yet.

## Data source and attribution

All content comes from MedlinePlus, a public consumer health resource of the U.S. National Library of Medicine. Each answer links back to the MedlinePlus pages it used.

## Future improvements

- Add an evaluation set of health questions with expected sources, and measure retrieval quality.
- Move to the dedicated `langchain-chroma`, `langchain-huggingface` and `langchain-ollama` packages.
- Use conversation history when retrieving follow-up questions.
- Add a screenshot or short demo GIF of the chat UI.
