import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_core.messages import SystemMessage, HumanMessage
from db_utils import get_retriever
import json
import os

# -------- CONFIG --------
CONFIG_FILE = "user_config.json"
DEFAULT_CONFIG = {
    "model_name": "llama-3.1-8b-instant",
    "temperature": 0.7,
    "max_tokens": 512,
    "personal": "You are a helpful assistant."
}

# -------- HELPERS --------
@st.cache_resource
def get_llm(model_name, temperature, max_tokens):
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

@st.cache_resource
def get_cached_retriever():
    return get_retriever()

def create_rag_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

def enhance_answer(answer, sources, llm, persona):
    if not sources:
        return answer

    context = "\n".join([doc.page_content for doc in sources])

    response = llm.invoke([
        SystemMessage(content=persona),
        HumanMessage(content=f"Enhance this answer:\n{answer}\n\nContext:\n{context}")
    ])

    return response.content

def load_user_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_user_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f)

# -------- UI --------
st.set_page_config(page_title="🚀 RAG App (Groq)", layout="wide")
st.title("🚀 RAG Application (Groq - FREE)")

config = load_user_config()

# Sidebar
st.sidebar.title("Model Settings")

model_name = st.sidebar.selectbox(
    "Model",
    ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "mixtral-8x7b-32768"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, config.get("temperature", 0.7))
max_tokens = st.sidebar.slider("Max Tokens", 64, 2048, config.get("max_tokens", 512))

persona_options = {
    "Helpful": "You are a helpful assistant.",
    "Technical": "You are a technical expert.",
    "Simple": "Explain in simple terms."
}

persona_label = st.sidebar.selectbox("Persona", list(persona_options.keys()))
prompt_prefix = persona_options[persona_label]

save_user_config({
    "model_name": model_name,
    "temperature": temperature,
    "max_tokens": max_tokens,
    "persona": prompt_prefix
})

# Load LLM + Retriever
llm = get_llm(model_name, temperature, max_tokens)
retriever = get_cached_retriever()
rag_chain = create_rag_chain(llm, retriever)

# Mode toggle
use_kb = st.radio("Mode", ["With KB", "LLM Only"]) == "With KB"

query = st.text_input("🔍 Ask something")

if st.button("Submit") and query:
    try:
        if use_kb:
            result = rag_chain.invoke({"query": query})

            answer = enhance_answer(
                result["result"],
                result["source_documents"],
                llm,
                prompt_prefix
            )

            sources = result["source_documents"]
            provenance = "📚 Knowledge Base + Groq"

        else:
            response = llm.invoke([
                SystemMessage(content=prompt_prefix),
                HumanMessage(content=query)
            ])
            answer = response.content
            sources = []
            provenance = "🧠 Groq Only"

        st.markdown("### 💡 Answer")
        st.success(answer)
        st.caption(provenance)

        # Save history
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append((query, answer, provenance))

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Chat history
if "history" in st.session_state:
    st.markdown("---")
    st.markdown("### 💬 History")

    for q, a, p in reversed(st.session_state.history):
        st.markdown(f"**Q:** {q}")
        st.markdown(f"**A:** {a}")
        st.caption(p)
        st.markdown("---")

st.markdown("---")
st.caption("Groq RAG App - Free Version 🚀")
