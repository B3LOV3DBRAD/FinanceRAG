# src/retriever.py – Lazy Batch BM25 (safe for large corpora)

import os, random
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, DATA_DIR, VECTOR_DIR

# --- Load vectorstore (semantic) ---
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    return Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)


# --- Lightweight keyword retriever (lazy batch) ---
def make_keyword_retriever(data_dir, sample_size=400):
    """Loads a random sample of 10-Ks for keyword search (to save RAM)."""
    txts = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    sample = random.sample(txts, min(sample_size, len(txts)))

    docs = []
    for name in sample:
        path = os.path.join(data_dir, name)
        try:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception:
            pass

    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 5
    return retriever


# --- Combined retrieval (semantic + keyword) ---
def retrieve_context(question, k=5):
    print("🔍 Retrieving relevant chunks...")

    # Semantic retrieval
    vectorstore = load_vectorstore()
    semantic_docs = vectorstore.similarity_search(question, k=k)

    # Keyword retrieval on a small random batch
    keyword_retriever = make_keyword_retriever(DATA_DIR, sample_size=400)
    keyword_docs = keyword_retriever.get_relevant_documents(question)

    # Merge results
    combined = semantic_docs + keyword_docs
    print(f"✅ Retrieved {len(combined)} total chunks (semantic + keyword).")

    for i, doc in enumerate(combined[:5], 1):
        snippet = doc.page_content[:200].replace("\n", " ")
        print(f"[{i}] {snippet}...\n")

    return combined


# --- Generate answer using GPT ---
def generate_answer(question, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""
    You are a financial analysis assistant.
    Use the CONTEXT below to answer the QUESTION accurately and concisely.
    If possible, cite which company or filing the info came from.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    try:
        llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        print(f"⚠️ OpenAI issue: {e}")
        return "Based on the context, here’s a summarized response:\n\n" + context[:400] + "..."


# --- Entry point ---
def ask_question(question):
    docs = retrieve_context(question)
    answer = generate_answer(question, docs)
    print("\nFinal Answer:\n")
    print(answer)


if __name__ == "__main__":
    ask_question("Which tech company had the highest revenue growth last year?")
