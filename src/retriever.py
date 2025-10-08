# src/retriever.py

import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from src.config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DIR

# --- Load vectorstore ---
def load_vectorstore():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    return Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)


# --- Retrieve top chunks ---
def retrieve_context(question, k=6):
    vectorstore = load_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    print(f"Retrieved {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown").split("\\")[-1]
        print(f"[{i}] ({source}) — {doc.page_content[:200]}...\n")

    return docs


def generate_answer(question, docs):
    import re
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain

    # --- combine and lightly clean messy context ---
    raw = "\n".join(getattr(d, "page_content", str(d)) for d in docs)
    # strip obvious tags / identifiers but keep words
    cleaned = re.sub(r"<[^>]+>", " ", raw)
    cleaned = re.sub(r"[A-Za-z0-9_-]+:[A-Za-z0-9_-]+", " ", cleaned)   # remove xbrl/us-gaap: tags
    cleaned = re.sub(r"[^A-Za-z0-9.,%$() \n]", " ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    context = cleaned[:8000]   # keep it short enough

    # --- simple reasoning prompt ---
    template = """
You are FinanceRAG, an AI financial analyst reading SEC 10-K filings.

Use the CONTEXT to answer the QUESTION in 3-6 plain English sentences.
• If numbers or performance clues appear, summarize them clearly.
• If qualitative risks or themes appear, summarize them clearly.
• Do not say "no relevant details" unless absolutely nothing relates to the question.
• Ignore gibberish or codes like "us-gaap:" or "xbrli:".

QUESTION: {question}

CONTEXT:
{context}

Answer:
"""
    prompt = PromptTemplate(input_variables=["context", "question"], template=template)
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = LLMChain(prompt=prompt, llm=llm)
    result = chain.run({"context": context, "question": question})
    return result.strip()







# --- Main ---
def ask_question(question):
    print("Retrieving relevant chunks...")
    trimmed_docs = retrieve_context(question)
    print("Generating analytical answer...\n")
    answer = generate_answer(question, trimmed_docs)
    print("Final Answer:\n")
    print(answer)


if __name__ == "__main__":
    ask_question("Which tech company had the highest revenue growth last year?")
