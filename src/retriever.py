# src/retriever.py

import os
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DIR


# --- Cached global vectorstore (faster retrieval) ---
_vectorstore = None

def get_vectorstore():
    """Load and cache the Chroma vectorstore once for reuse across multiple queries."""
    global _vectorstore
    if _vectorstore is None:
        print("Initializing vectorstore...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        _vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
        print("✅ Vectorstore loaded successfully.\n")
    return _vectorstore

def clear_vectorstore_cache():
    """Clear the cached Chroma vectorstore (used when resetting chat or reloading data)."""
    global _vectorstore
    _vectorstore = None
    print("🗑 Vectorstore cache cleared.")

# --- Retrieve top chunks (debug-friendly) ---
def retrieve_context(question, k=6):
    """Retrieve relevant chunks and print summary info (useful for terminal debugging)."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    print(f"Retrieved {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown").split("\\")[-1]
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"[{i}] ({source}) — {preview}...\n")

    return docs


# --- Retrieve relevant docs (used by Streamlit app) ---
def retrieve_relevant_docs(question, k: int = 6):
    """Retrieve the most relevant document chunks for a query (used in Streamlit)."""
    try:
        vectorstore = get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        docs = retriever.get_relevant_documents(question)
        return docs
    except Exception as e:
        print(f"⚠️ Retrieval error: {e}")
        return []


# --- Generate structured answer ---
def generate_answer(question, docs):
    """Generate a concise, well-formatted answer using the retrieved context."""
    # --- Combine and clean context ---
    raw_texts = [getattr(d, "page_content", str(d)) for d in docs]
    combined = "\n\n".join(raw_texts)
    cleaned = re.sub(r"([A-Za-z])\s+([A-Za-z])", r"\1 \2", combined)
    cleaned = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", cleaned)
    cleaned = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    context = cleaned[:9000]

    # --- Prompt Template ---
    template = """
You are FinanceRAG, an AI financial analyst that interprets SEC 10-K filings.

Use the CONTEXT below to answer the QUESTION clearly and concisely.
Be specific and back up your statements with exact figures from the filings.
Cite the source company for each fact you present.
Ask if there is any other information you need.

Format the response exactly like this:

## Key Facts
- Bullet 1
- Bullet 2
(Each bullet covers one company or key point. Use bold for company names.)

## Summary
One short paragraph (2–4 sentences) summarizing the findings.

Rules:
• Fix any joined words (e.g., “100onDecember31,2019” → “100 on December 31, 2019”)
• Keep spacing and capitalization normal.
• No asterisks, underscores, or markdown artifacts.
• Always begin each key fact with the company name or its ticker symbol.
• If you cannot find data for the requested year, identify the **most recent fiscal year** mentioned and clearly state that it is the latest available information.
• Only use "No relevant details were found in the provided filings." when nothing in the filings relates to the question at all.
• Always report numeric data in the **most appropriate scale or denomination** (e.g., use billions instead of millions when values exceed 1,000 million).
• If nothing relevant exists, write:
  **No relevant details were found in the provided filings.**

CONTEXT:
{context}

QUESTION:
{question}
"""

    # --- Run LLM Chain ---
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    chain = LLMChain(prompt=prompt, llm=llm)
    result = chain.run({"context": context, "question": question})

    # --- Post-format cleanup ---
    result = re.sub(r"[*_]+", "", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"##\s*Summary", "\n\n## Summary", result)
    result = result.strip()

    # Ensure both sections exist
    if not result.startswith("## Key Facts"):
        result = "## Key Facts\n" + result
    if "## Summary" not in result:
        result += "\n\n## Summary\n(No summary section detected.)"

    return result


# --- Terminal debug entry point ---
def ask_question(question):
    print("Retrieving relevant chunks...")
    trimmed_docs = retrieve_context(question)
    print("Generating analytical answer...\n")
    answer = generate_answer(question, trimmed_docs)
    print("Final Answer:\n")
    print(answer)


if __name__ == "__main__":
    ask_question("Which tech company had the highest revenue growth last year?")
