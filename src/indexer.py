# src/indexer.py — Balanced, budget-capped indexer (≈$10 total cost)

import os, math, json, time, random, openai
from collections import defaultdict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DIR

MODEL = "text-embedding-3-small"
COST_PER_1M = 0.02          # $ per 1M tokens
MAX_BUDGET = 10.00          # 💵 total cap
AVG_TOKENS_PER_CHUNK = 525  # empirical estimate
BATCH_SIZE = 1000
CHECKPOINT_FILE = "progress_checkpoint.json"


# --- Checkpoint helpers ---
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"last_batch": -1, "spent_usd": 0.0}


def save_checkpoint(batch_id, spent_usd):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_batch": batch_id, "spent_usd": spent_usd}, f)


# --- Load all 10-K text files ---
def load_documents(data_dir):
    docs = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    print(f"Found {len(files)} 10-K files in {data_dir}")
    for i, name in enumerate(files, 1):
        path = os.path.join(data_dir, name)
        try:
            company_docs = TextLoader(path, encoding="utf-8").load()
            # Tag each doc with its source company (for balanced sampling)
            for d in company_docs:
                d.metadata["source"] = name.replace("_10K.txt", "")
            docs.extend(company_docs)
        except Exception as e:
            print(f"Skipping {name}: {e}")
        if i % 500 == 0:
            print(f"Processed {i}/{len(files)} filings...")
    print(f"Finished loading {len(files)} documents.")
    return docs


# --- Split into chunks ---
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks):,} chunks.")
    return chunks


# --- Group chunks by company for balanced sampling ---
def group_by_company(chunks):
    grouped = defaultdict(list)
    for c in chunks:
        company = c.metadata.get("source", "Unknown")
        grouped[company].append(c)
    print(f"Grouped chunks by {len(grouped):,} companies.")
    return grouped


# --- Estimate total cost ---
def estimate_cost(chunks):
    total_tokens = len(chunks) * AVG_TOKENS_PER_CHUNK
    est_cost = total_tokens / 1_000_000 * COST_PER_1M
    return total_tokens, est_cost


# --- Balanced sampling per company ---
def balanced_sample(grouped):
    max_tokens_allowed = MAX_BUDGET / COST_PER_1M * 1_000_000
    max_chunks = int(max_tokens_allowed / AVG_TOKENS_PER_CHUNK)
    print(f"Budget allows about {max_chunks:,} chunks (≈ ${MAX_BUDGET:.2f}).")

    # Split budget evenly across companies
    companies = list(grouped.keys())
    per_company = max_chunks // len(companies)
    sampled_chunks = []

    for company, docs in grouped.items():
        random.seed(42)
        if len(docs) <= per_company:
            sampled = docs
        else:
            sampled = random.sample(docs, per_company)
        sampled_chunks.extend(sampled)

    print(f"Sampled ~{per_company:,} chunks per company ({len(sampled_chunks):,} total).")
    return sampled_chunks


# --- Create embeddings ---
def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model=MODEL, openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)

    total_batches = math.ceil(len(chunks) / BATCH_SIZE)
    checkpoint = load_checkpoint()
    last_done, spent_usd = checkpoint["last_batch"], checkpoint["spent_usd"]
    print(f"Resuming from batch {last_done+1}, already spent ≈${spent_usd:.2f}")

    for batch_id in range(total_batches):
        if batch_id <= last_done:
            continue

        start = batch_id * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = chunks[start:end]
        est_batch_tokens = len(batch) * AVG_TOKENS_PER_CHUNK
        est_batch_cost = est_batch_tokens / 1_000_000 * COST_PER_1M

        if spent_usd + est_batch_cost > MAX_BUDGET:
            print(f"Stopping at ${spent_usd:.2f} — reached ${MAX_BUDGET:.2f} cap.")
            break

        try:
            vectorstore.add_documents(batch)
            vectorstore.persist()
            spent_usd += est_batch_cost
            save_checkpoint(batch_id, spent_usd)
            print(f"Batch {batch_id+1}/{total_batches} saved | +${est_batch_cost:.2f} | Total ${spent_usd:.2f}")
        except openai.RateLimitError:
            print("Rate limit hit — sleeping 60s...")
            time.sleep(60)
        except Exception as e:
            print(f"Batch {batch_id} failed: {e}")
            time.sleep(10)

    print(f"Done. Total spent ≈${spent_usd:.2f} within budget (${MAX_BUDGET:.2f}).")


# --- Main ---
def build_index():
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    total_tokens, full_cost = estimate_cost(chunks)
    print(f"Estimated total tokens: {total_tokens:,}")
    print(f"Estimated full-corpus cost: ${full_cost:.2f}")

    confirm = input(f"Proceed with balanced $10 sample (full ≈${full_cost:.2f})? [y/n]: ")
    if confirm.lower() != "y":
        print("Cancelled.")
        return

    grouped = group_by_company(chunks)
    sampled_chunks = balanced_sample(grouped)
    create_vectorstore(sampled_chunks)
    print("🚀 Indexing complete!")


if __name__ == "__main__":
    build_index()
