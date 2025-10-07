# src/indexer.py – cost-safe version (<= $10)

import os, time, math, json, openai
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import OPENAI_API_KEY, DATA_DIR, VECTOR_DIR

MODEL = "text-embedding-3-small"
COST_PER_1M = 0.02           # $ per 1 M tokens
MAX_BUDGET = 10.00           # 💰 hard limit
AVG_TOKENS_PER_CHUNK = 525   # empirical from last run
BATCH_SIZE = 1000
CHECKPOINT_FILE = "progress_checkpoint.json"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"last_batch": -1, "spent_usd": 0.0}


def save_checkpoint(batch_id, spent_usd):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"last_batch": batch_id, "spent_usd": spent_usd}, f)


def load_documents(data_dir):
    docs, files = [], [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    print(f"📂 Found {len(files)} 10-K files in {data_dir}")
    for i, name in enumerate(files, 1):
        path = os.path.join(data_dir, name)
        try:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        except Exception as e:
            print(f"⚠️ Skipping {name}: {e}")
        if i % 500 == 0:
            print(f"✅ Processed {i}/{len(files)} filings…")
    print(f"✅ Finished loading {len(files)} documents.")
    return docs


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    print(f"📖 Split into {len(chunks)} chunks (same logic as before).")
    return chunks


def estimate_cost(chunks):
    total_tokens = len(chunks) * AVG_TOKENS_PER_CHUNK
    est_cost = total_tokens / 1_000_000 * COST_PER_1M
    print(f"💰 Estimated total tokens: {total_tokens:,}")
    print(f"💵 Estimated full-corpus cost: ${est_cost:.2f}")
    return total_tokens, est_cost


def create_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model=MODEL, openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)
    try:
        existing = vectorstore._collection.count()
        print(f"📊 Found {existing:,} chunks already in Chroma.")
    except Exception:
        existing = 0

    total_tokens, _ = estimate_cost(chunks)
    total_batches = math.ceil(len(chunks) / BATCH_SIZE)
    checkpoint = load_checkpoint()
    last_done, spent_usd = checkpoint["last_batch"], checkpoint["spent_usd"]
    print(f"🔁 Resuming from batch {last_done+1}, already spent ≈${spent_usd:.2f}")

    start_time = time.time()
    for batch_id in range(total_batches):
        if batch_id <= last_done:
            continue

        start = batch_id * BATCH_SIZE
        end = start + BATCH_SIZE
        batch = chunks[start:end]
        est_batch_tokens = len(batch) * AVG_TOKENS_PER_CHUNK
        est_batch_cost = est_batch_tokens / 1_000_000 * COST_PER_1M

        # 💸 Stop if adding this batch would exceed budget
        if spent_usd + est_batch_cost > MAX_BUDGET:
            print(f"⛔ Reached ${spent_usd:.2f} spent — stopping before exceeding ${MAX_BUDGET:.2f}.")
            break

        try:
            vectorstore.add_documents(batch)
            vectorstore.persist()
            spent_usd += est_batch_cost
            elapsed = time.time() - start_time
            print(f"✅ Saved batch {batch_id+1}/{total_batches} | +${est_batch_cost:.2f} | Total ${spent_usd:.2f}")
            save_checkpoint(batch_id, spent_usd)
        except openai.RateLimitError:
            print("⚠️ Rate limit — sleeping 60 s.")
            time.sleep(60)
        except Exception as e:
            print(f"❌ Batch {batch_id} failed: {e}")
            time.sleep(10)

    print(f"🎯 Done. Embedded within budget (${spent_usd:.2f}). Data saved to {VECTOR_DIR}")


def build_index():
    docs = load_documents(DATA_DIR)
    chunks = split_documents(docs)
    _, full_cost = estimate_cost(chunks)
    confirm = input(f"Proceed? (full dataset ≈${full_cost:.2f}, budget ${MAX_BUDGET:.2f}) [y/n]: ")
    if confirm.lower() != "y":
        print("❌ Cancelled.")
        return
    create_vectorstore(chunks)
    print("🚀 Indexing complete!")


if __name__ == "__main__":
    build_index()
