import streamlit as st
from rag_chain import get_rag_chain, ask

st.set_page_config(page_title="Medical RAG Chatbot", page_icon="🩺", layout="centered")

st.title("🩺 Medical RAG Chatbot")
st.caption(
    "Answers are grounded in MedlinePlus content using local retrieval + a local LLM (Ollama)."
)

st.info(
    "⚠️ This tool provides general health information for educational purposes only. "
    "It is **not** a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider.",
    icon="⚠️",
)


@st.cache_resource(show_spinner="Loading vector store and connecting to Ollama...")
def load_chain():
    return get_rag_chain()


try:
    chain = load_chain()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- [{s['title']}]({s['url']})")

# Chat input
question = st.chat_input("Ask a question about a disease, condition, or symptom...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, sources = ask(chain, question)
            except Exception as e:
                answer = f"Something went wrong while generating an answer: {e}"
                sources = []
        st.markdown(answer)
        if sources:
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- [{s['title']}]({s['url']})")

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )

with st.sidebar:
    st.header("About")
    st.markdown(
        "This chatbot retrieves relevant passages from a local vector store "
        "(built from MedlinePlus health topics) and passes them to a local "
        "LLM running via **Ollama** to generate grounded answers."
    )
    st.markdown("**Stack:** LangChain · Chroma · HuggingFace embeddings · Ollama · Streamlit")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
