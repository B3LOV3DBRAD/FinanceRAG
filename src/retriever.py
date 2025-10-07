# src/retriever.py

import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DIR

# --- Step 1: Load vectorstore ---
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    return vectorstore


# --- Step 2: Retrieve relevant chunks ---
def retrieve_context(question, k=3):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)
    print(f"Retrieved {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs, start=1):
        print(f"[{i}] {doc.page_content[:200]}...\n")
    return docs


# --- Step 3: Generate an answer using the LLM ---
def generate_answer(question, docs):
    # Combine the text of the top retrieved documents
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build the prompt for the model
    prompt = f"""
    You are a financial analysis assistant.
    Use the CONTEXT below to answer the QUESTION.
    Be concise and cite which source the info came from if possible.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """

    # Option 1: If you have OpenAI credits, use GPT
    try:
        llm = ChatOpenAI(model=CHAT_MODEL, openai_api_key=OPENAI_API_KEY)
        response = llm.invoke(prompt)
        answer = response.content
    except Exception as e:
        # Option 2: Fallback if no credits
        print("OpenAI quota issue, falling back to local model (simulated response).")
        answer = "Based on the context, here’s a summarized response:\n\n" + context[:400] + "..."

    return answer


# --- Step 4: Ask a question ---
def ask_question(question):
    docs = retrieve_context(question)
    answer = generate_answer(question, docs)
    print("\nFinal Answer:\n")
    print(answer)


if __name__ == "__main__":
    # Example query
    ask_question("What was Apple's returns last quarter?")
