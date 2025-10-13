import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import streamlit as st
from retriever import generate_answer, retrieve_relevant_docs, clear_vectorstore_cache


# --- Page setup ---
st.set_page_config(page_title="FinanceRAG", page_icon="📊")
st.title("FinanceRAG — S&P 500")
st.caption("Ask questions about S&P 500 companies’ 10-K filings and financial performance.")

# --- Initialize chat history ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Clear Chat button ---
col1, col2 = st.columns([6, 1])
with col2:
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        clear_vectorstore_cache()
        st.rerun()

question = st.chat_input("Ask a question...")

if question:
    # Save user message
    st.session_state.chat_history.append({"role": "user", "content": question})

    # --- Build contextual conversation for the model ---
    past_questions = [
        msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"
    ]
    conversation_context = (
        "Previous conversation:\n"
        + "\n".join(f"- {q}" for q in past_questions[:-1])
        + f"\n\nCurrent question:\n{question}"
    )

    # --- Dynamic progress messages ---
    status = st.empty()

    # Step 1: Retrieval (use only the current question to avoid diluting relevance)
    status.info("Retrieving relevant filings...")
    retrieved_docs = retrieve_relevant_docs(question)
    trimmed_docs = retrieved_docs[:6]

    # Confidence bar + retrieved doc count caption
    retrieved_count = len(retrieved_docs)
    if retrieved_count > 0:
        confidence = min(1.0, len(trimmed_docs) / 6)
        st.caption(f"Retrieval confidence: {confidence:.0%}  •  Retrieved {retrieved_count} docs")
        st.progress(confidence)
    else:
        st.caption("Retrieval confidence: 0%  •  Retrieved 0 docs")
        st.progress(0)

    # Step 2: Generate answer
    status.info("Analyzing filings and generating response...")
    answer = generate_answer(conversation_context, trimmed_docs)

    # Step 3: Done
    status.success("Analysis complete!")
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

            # Render clean layout with better formatting
            st.markdown("### Key Facts")
            # Clean up the key facts section
            if key_facts:
                # Remove any remaining markdown artifacts and clean up formatting
                import re
                key_facts_clean = re.sub(r'[*_]{1,}', '', key_facts)  # Remove any remaining markdown
                key_facts_clean = re.sub(r'—\s*\*\s*\*', '', key_facts_clean)  # Remove "— * *"
                key_facts_clean = re.sub(r'\*\*:\s*', '', key_facts_clean)  # Remove "**:"
                key_facts_clean = re.sub(r'\s{2,}', ' ', key_facts_clean)  # Normalize spaces
                # Fix specific formatting issues
                key_facts_clean = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', key_facts_clean)  # Fix "394. 3" -> "394.3"
                key_facts_clean = re.sub(r'Co\.\s*mpany', 'Company', key_facts_clean, flags=re.IGNORECASE)  # Fix "Co. mpany" -> "Company"
                key_facts_clean = key_facts_clean.strip()
                st.markdown(key_facts_clean)
            else:
                st.markdown("No key facts available.")
            
            st.markdown("### Summary")
            # Clean up the summary section
            if summary:
                summary_clean = re.sub(r'[*_]{1,}', '', summary)  # Remove any remaining markdown
                summary_clean = re.sub(r'—\s*\*\s*\*', '', summary_clean)  # Remove "— * *"
                summary_clean = re.sub(r'\*\*:\s*', '', summary_clean)  # Remove "**:"
                summary_clean = re.sub(r'\s{2,}', ' ', summary_clean)  # Normalize spaces
                # Fix specific formatting issues
                summary_clean = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', summary_clean)  # Fix "394. 3" -> "394.3"
                summary_clean = re.sub(r'Co\.\s*mpany', 'Company', summary_clean, flags=re.IGNORECASE)  # Fix "Co. mpany" -> "Company"
                summary_clean = summary_clean.strip()
                st.markdown(summary_clean)
            else:
                st.markdown("No summary available.")
            
            st.markdown("---")
