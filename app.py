import streamlit as st
from src.retriever import generate_answer, retrieve_relevant_docs, clear_vectorstore_cache

# --- Page setup ---
st.set_page_config(page_title="FinanceRAG", page_icon="💼")
st.title("💼 FinanceRAG")
st.caption("Ask questions about 10-K filings and company financial performance.")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Clear Chat button ---
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("🗑 Clear Chat"):
        st.session_state.chat_history = []
        clear_vectorstore_cache()
        st.rerun()

# --- Chat input ---
question = st.chat_input("Ask a question...")

if question:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Combine all prior user messages for context
    conversation_context = " ".join(
        [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]
    )

    # --- Dynamic progress messages ---
    status = st.empty()

    # Step 1: Retrieval
    status.info("📂 Retrieving relevant filings...")
    retrieved_docs = retrieve_relevant_docs(conversation_context)
    trimmed_docs = retrieved_docs[:6]

    # Confidence bar
    confidence = min(1.0, len(trimmed_docs) / 6)
    st.caption(f"Retrieval confidence: {confidence:.0%}")
    st.progress(confidence)

    # Step 2: Generate answer
    status.info("🧠 Analyzing filings and generating response...")
    answer = generate_answer(conversation_context, trimmed_docs)

    # Step 3: Done
    status.success("✅ Analysis complete!")
    st.session_state.chat_history.append({"role": "assistant", "content": answer})

# --- Display chat history ---
for message in st.session_state.chat_history:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["content"])
    else:
        with st.chat_message("assistant"):
            content = message["content"]

            # Split "Key Facts" and "Summary"
            if "## Summary" in content:
                parts = content.split("## Summary", 1)
                key_facts = parts[0].replace("## Key Facts", "").strip()
                summary = parts[1].strip()
            else:
                key_facts, summary = content, ""

            # Render clean layout
            st.markdown("### Key Facts")
            st.markdown(key_facts)
            st.markdown("### Summary")
            st.markdown(summary)
            st.markdown("---")
