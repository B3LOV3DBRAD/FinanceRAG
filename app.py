# app_chat.py — FinanceRAG Conversational UI

import streamlit as st
from src.retriever import retrieve_context, generate_answer
from src.config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DIR

st.set_page_config(page_title="FinanceRAG Chat", page_icon="💬", layout="centered")

st.title("FinanceRAG Chat")
st.caption("Ask questions about 10-K filings and company performance.")

# --- Initialize chat memory ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi 👋 I'm FinanceRAG. Ask me about any company’s 10-K or financial performance!"}
    ]

# --- Display chat history ---
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.chat_message("user").write(msg["content"])
    else:
        st.chat_message("assistant").write(msg["content"])

# --- User input ---
if question := st.chat_input("Type your financial question..."):
    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Retrieve + answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing SEC filings..."):
            trimmed_docs = retrieve_context(question, k=6)
            answer = generate_answer(question, trimmed_docs)

            # Display answer
            st.markdown(answer, unsafe_allow_html=False)

            # Optionally show context
        with st.expander("View retrieved context"):
            for i, snippet in enumerate(trimmed_docs, 1):
                text = getattr(snippet, "page_content", snippet)
                st.markdown(f"**Source {i}:** {text[:400]}...")

            # Save answer to history
            st.session_state.messages.append({"role": "assistant", "content": answer})
