# src/retriever.py
import os
import re
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.config import OPENAI_API_KEY, VECTOR_DIR


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
        _vectorstore = Chroma(
            persist_directory=VECTOR_DIR,
            embedding_function=embeddings
        )
        print("✅ Vectorstore loaded successfully.\n")
    return _vectorstore


def clear_vectorstore_cache():
    """Clear the cached Chroma vectorstore (used when resetting chat or reloading data)."""
    global _vectorstore
    _vectorstore = None
    print("🗑 Vectorstore cache cleared.")


# --- Retrieve top chunks (debug-friendly) ---
def retrieve_context(question, k=6):
    """Retrieve relevant chunks and print summary info (for debugging)."""
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(question, k=k)

    print(f"Retrieved {len(docs)} relevant chunks:\n")
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown").split("\\")[-1]
        preview = doc.page_content[:200].replace("\n", " ")
        print(f"[{i}] ({source}) — {preview}...\n")
    return docs


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
    """Generate a concise, well-formatted answer using retrieved 10-K context."""
    import re
    from langchain_openai import ChatOpenAI
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from src.config import OPENAI_API_KEY

    # --- Combine and pre-clean context (defensive against SEC/OCR spacing) ---
    raw_texts = [getattr(d, "page_content", str(d)) for d in docs]
    combined = "\n\n".join(raw_texts)
    combined = re.sub(r"([0-9])([A-Za-z])", r"\1 \2", combined)
    combined = re.sub(r"([A-Za-z])([0-9])", r"\1 \2", combined)
    combined = re.sub(r"\s{2,}", " ", combined)
    context = combined[:9000]

    # --- Prompt Template with full 10-K interpretation rules ---
    template = """
You are FinanceRAG, an AI financial analyst specializing in SEC 10-K, 10-Q, and 8-K filings.

Use the CONTEXT below to answer the QUESTION clearly and precisely.

### Output Format
## Key Facts
- **<Company/Ticker>**: <Fact 1 (specific number, year, and qualifier)>
- **<Company/Ticker>**: <Fact 2>

## Summary
2–4 sentences summarizing the key findings, highlighting year, direction of change, and drivers.

---

### Interpretation Rules & Edge Cases (follow all)
1) **Period discipline**
   • Always specify the fiscal year (FY) or period end (e.g., FY2024, year ended Sep 28, 2024).  
   • If “last year” appears, interpret it as the latest fiscal year in filings.  
   • If the requested period isn’t in the context, use the latest available year.

2) **Units, scale, and currency**
   • Use the currency shown in filings; include the code (USD, EUR, etc.) if not USD.  
   • Normalize large values to billions if ≥ 1,000 million.  
   • Maintain correct magnitude (thousands, millions, billions).  
   • Distinguish basic vs diluted per-share figures.  
   • Preserve negatives (parentheses or minus sign).

3) **Definition discipline**
   • Do not mix “net sales” and “total revenue.”  
   • Distinguish between operating vs net income, cash flow from ops vs free cash flow.  
   • Label non-GAAP figures and mention nearest GAAP anchor.

4) **Consolidated vs segment**
   • Use consolidated totals unless segment data is explicitly asked for.  
   • State segment name when cited and avoid implying equality with consolidated totals.

5) **Continuing ops / discontinued / reclasses**
   • If results exclude discontinued operations or include reclassifications, note that explicitly.

6) **52/53-week years**
   • Mention extra weeks or fiscal-year-end changes if disclosed.

7) **Comparisons**
   • If direction/magnitude are mentioned, include (% or absolute).  
   • Use reported basis by default; label constant-currency explicitly if used.

8) **FX, M&A, one-offs**
   • Mention disclosed FX effects, acquisitions, divestitures, impairments, or one-time events.

9) **Subsequent events & guidance**
   • Avoid mixing guidance with historicals unless the question asks.  
   • Mention material subsequent events when disclosed.

10) **Ambiguity & absence**
   • If a value isn’t disclosed, state that explicitly and give the closest related metric.  
   • Do not infer or invent missing data.

11) **Numbers & typography hygiene**
   • Always use a space between numbers and words (e.g., “383.3 billion”).  
   • Correct joined tokens like “in2024” → “in 2024” or “comparedto” → “compared to.”  
   • Avoid markdown artifacts—no italics, no underscores.

12) **Multi-company context**
   • If multiple companies are referenced, use one bullet per company.  
   • If only one company is relevant, focus on that.

13) **Compliance**
   • Stay factual, concise, and neutral.  
   • Prefer the most recent filing when inconsistencies appear.

---

### Formatting rules
• Bold company/ticker names only.  
• No italics, underscores, or single asterisks.  
• Add a space between numbers and units.  
• End every sentence with a period.  
• Ensure bullets are readable, not run-together.

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

    # --- Post-format cleanup for Markdown/Streamlit rendering ---
    # Remove italics and stray markdown chars but preserve **bold**
    sentinel = "§§"
    result = result.replace("**", sentinel)
    result = re.sub(r'(\w)[*_](?=\w)', r'\1 ', result)
    result = re.sub(r'(?<=\w)[*_](\w)', r' \1', result)
    result = result.replace("*", "").replace("_", "")
    result = result.replace(sentinel, "**")

    # Fix run-together words and numbers
    replacements = [
        (r'(?<=\d)(?=[A-Za-z])', ' '),
        (r'(?<=[A-Za-z])(?=\d)', ' '),
        (r'(?i)yearended', 'year ended'),
        (r'(?i)fiscalyear', 'fiscal year'),
        (r'(?i)reflectinga', 'reflecting a'),
        (r'(?i)representsa', 'represents a'),
        (r'(?i)declineof', 'decline of '),
        (r'(?i)increaseof', 'increase of '),
        (r'(?i)comparedto', 'compared to'),
        (r'(?i)in(?=\d{4})', 'in '),
        (r'(?i)FY\s*(?=\d{4})', 'FY '),
        (r'(?i)million(?=[A-Za-z])', 'million '),
        (r'(?i)billion(?=[A-Za-z])', 'billion '),
        # Additional targeted fixes seen in UI
        (r'(?i)grossmargin', 'gross margin'),
        (r'(?i)totalgrossmargin', 'total gross margin'),
    ]
    for pat, repl in replacements:
        result = re.sub(pat, repl, result)

    # Clean punctuation/spacing
    result = re.sub(r'\s*,\s*', ', ', result)
    result = re.sub(r'\s*\.\s*', '. ', result)
    result = re.sub(r'\.\s*\.', '.', result)
    # Fix decimals with stray space: 394. 3 -> 394.3
    result = re.sub(r'(\d+)\.\s+(\d+)', r'\1.\2', result)
    # Fix stuck month+day: September30 -> September 30
    result = re.sub(r'(?i)(January|February|March|April|May|June|July|August|September|October|November|December)(\d{1,2})', r'\1 \2', result)
    # Fix possessive spacing: Company 's -> Company's
    result = re.sub(r"\s+'s\b", "'s", result)
    result = re.sub(r'\s{2,}', ' ', result)
    result = re.sub(r'\n{3,}', '\n\n', result)

    # Normalize headers
    result = re.sub(r'#+\s*Key Facts', '## Key Facts', result)
    result = re.sub(r'#+\s*Summary', '\n\n## Summary', result)
    result = result.strip()

    # Ensure both sections exist
    if not result.startswith("## Key Facts"):
        result = "## Key Facts\n" + result
    if "## Summary" not in result:
        result += "\n\n## Summary\n(Data summary unavailable in this excerpt.)"

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
