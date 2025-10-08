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

    # ---------- Combine & sanitize context ----------
    unique_texts = []
    for d in docs:
        txt = getattr(d, "page_content", str(d))
        if txt not in unique_texts:
            unique_texts.append(txt)
    context = "\n\n".join(unique_texts)

    context = re.sub(r"\s{2,}", " ", context)
    context = context.replace("*", "").replace("_", "")

    # ---------- Universal Prompt ----------
    template = """
You are **FinanceRAG**, an expert AI analyst that interprets SEC 10-K filings.

Your task: Answer the QUESTION using only the information in CONTEXT.
If the question is about:
- **Performance / Returns** → extract quantitative data like revenue, growth, or total return.
- **Risks / Challenges / Outlook** → summarize qualitative risk factors and management commentary.
- **Comparisons** → describe relative performance or risk positioning among companies.

Follow these formatting rules:

## Key Facts
- One bullet per company or relevant topic.
- Clean and readable; fix spacing (e.g., “100onDecember31,2019” → “100 on December 31, 2019”).
- When discussing qualitative topics (like risk), summarize the main ideas in 1–2 short sentences.
- When discussing quantitative metrics (like returns or revenue), include clean numbers.

## Summary
Write 1–3 sentences summarizing the key takeaway or comparison.

If there’s truly no relevant data, return:
**No relevant details were found in the provided filings.**

CONTEXT:
{context}

QUESTION:
{question}
"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=template)

    llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = LLMChain(prompt=prompt, llm=llm)
    result = chain.run({"context": context, "question": question})

    # ---------- Post-cleanup ----------
    result = re.sub(r"[*_]+", "", result)
    result = re.sub(r"([A-Za-z])\s*([A-Za-z])", r"\1\2", result)
    result = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", result)
    result = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", result)
    result = re.sub(r"##\s*Summary", "\n\n## Summary", result)
    result = re.sub(r"\s{2,}", " ", result)
    result = re.sub(r"\n{3,}", "\n\n", result)
    result = result.strip()
    if not result.startswith("##"):
        result = "## Key Facts\n" + result
    if "## Summary" not in result:
        result += "\n\n## Summary\n(No summary section detected.)"
    return result






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
