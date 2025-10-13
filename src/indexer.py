# src/indexer.py — Balanced, budget-capped indexer (≈$10 total cost)

import os, math, json, time, random, openai
from collections import defaultdict
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.config import OPENAI_API_KEY, CHAT_MODEL, VECTOR_DIR, DATA_DIR
import argparse
import shutil
import pandas as pd
import requests

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
def load_documents(data_dir, tickers_filter=None):
    docs = []
    files = [f for f in os.listdir(data_dir) if f.endswith(".txt")]
    if tickers_filter:
        tickers_filter = set(t.upper() for t in tickers_filter)
        files = [f for f in files if f.split("_", 1)[0].upper() in tickers_filter]
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
def balanced_sample(grouped, sample_per_company=None, max_companies=None):
    max_tokens_allowed = MAX_BUDGET / COST_PER_1M * 1_000_000
    max_chunks = int(max_tokens_allowed / AVG_TOKENS_PER_CHUNK)
    # Avoid unicode approx symbol for Windows console compatibility
    print(f"Budget allows about {max_chunks:,} chunks (~ ${MAX_BUDGET:.2f}).")

    # Split budget evenly across companies
    companies = list(grouped.keys())
    if max_companies is not None:
        companies = companies[:max_companies]
    per_company = sample_per_company if sample_per_company is not None else max_chunks // len(companies)
    sampled_chunks = []

    random.seed(42)
    for company in companies:
        docs = grouped.get(company, [])
        if len(docs) <= per_company:
            sampled = docs
        else:
            sampled = random.sample(docs, per_company)
        sampled_chunks.extend(sampled)

    print(f"Sampled ~{per_company:,} chunks per company ({len(sampled_chunks):,} total).")
    return sampled_chunks


# --- Create embeddings ---
def create_vectorstore(chunks, batch_size=BATCH_SIZE):
    embeddings = OpenAIEmbeddings(model=MODEL, openai_api_key=OPENAI_API_KEY)
    vectorstore = Chroma(persist_directory=VECTOR_DIR, embedding_function=embeddings)

    total_batches = math.ceil(len(chunks) / batch_size)
    checkpoint = load_checkpoint()
    last_done, spent_usd = checkpoint["last_batch"], checkpoint["spent_usd"]
    # Avoid unicode approx for Windows console
    print(f"Resuming from batch {last_done+1}, already spent ~${spent_usd:.2f}")

    for batch_id in range(total_batches):
        if batch_id <= last_done:
            continue

        start = batch_id * batch_size
        end = start + batch_size
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

    print(f"Done. Total spent ~${spent_usd:.2f} within budget (${MAX_BUDGET:.2f}).")


# --- Main ---
def load_sp500_tickers(cache_path=os.path.join("data", "sp500_tickers.txt")):
    """Load S&P 500 tickers; fetch from Wikipedia if cache missing."""
    try:
        if os.path.exists(cache_path):
            with open(cache_path, "r") as f:
                return [t.strip() for t in f if t.strip()]
        # Try fetching from Wikipedia with a browser-like User-Agent to avoid 403
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(url, headers=headers, timeout=20)
        if resp.status_code == 200:
            tables = pd.read_html(resp.text)
            tickers = tables[0]["Symbol"].astype(str).str.strip().tolist()
        else:
            # Fallback to a public dataset if Wikipedia blocks us
            csv_url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv"
            resp2 = requests.get(csv_url, timeout=20)
            resp2.raise_for_status()
            df = pd.read_csv(pd.compat.StringIO(resp2.text)) if hasattr(pd, 'compat') else pd.read_csv(pd.io.common.StringIO(resp2.text))
            tickers = df["Symbol"].astype(str).str.strip().tolist()
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            f.write("\n".join(tickers))
        return tickers
    except Exception as e:
        print(f"Failed to load S&P 500 tickers: {e}")
        return []


def build_index(args=None):
    parser = argparse.ArgumentParser(description="Build Chroma vectorstore from 10-K text files")
    parser.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    parser.add_argument("--reset", action="store_true", help="Delete existing vectorstore before indexing")
    parser.add_argument("--tickers", type=str, default="", help="Comma-separated tickers to include (e.g., AAPL,MSFT)")
    parser.add_argument("--tickers-file", type=str, default="", help="Path to a file with one ticker per line")
    parser.add_argument("--sp500", action="store_true", help="Limit to S&P 500 constituents")
    parser.add_argument("--sample-per-company", type=int, default=None, help="Limit number of chunks per company")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Embedding batch size for indexing (lower reduces memory spikes)")
    parser.add_argument("--max-companies", type=int, default=None, help="Hard cap on number of companies processed this run")
    cli_args = parser.parse_args(args=args)

    if cli_args.reset and os.path.exists(VECTOR_DIR):
        print(f"Removing existing vectorstore at {VECTOR_DIR}...")
        shutil.rmtree(VECTOR_DIR, ignore_errors=True)
        # Also reset checkpoint so we don't skip new batches
        if os.path.exists(CHECKPOINT_FILE):
            try:
                os.remove(CHECKPOINT_FILE)
                print("Reset progress checkpoint.")
            except Exception as e:
                print(f"Could not remove checkpoint: {e}")

    tickers = [t.strip() for t in cli_args.tickers.split(",") if t.strip()] if cli_args.tickers else []
    if cli_args.tickers_file:
        try:
            with open(cli_args.tickers_file, "r") as f:
                tickers += [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"Failed to read tickers file {cli_args.tickers_file}: {e}")
    if cli_args.sp500:
        tickers += load_sp500_tickers()
    tickers = [t.upper() for t in tickers]
    tickers = list(dict.fromkeys(tickers))  # de-dupe preserving order
    tickers_filter = tickers if tickers else None

    # Guard: if user requested S&P 500 but we couldn't load any tickers, abort safely
    if cli_args.sp500 and not tickers_filter:
        print("S&P 500 tickers unavailable. Install 'lxml' (pip install lxml) or pass --tickers-file.")
        return

    docs = load_documents(DATA_DIR, tickers_filter=tickers_filter)
    chunks = split_documents(docs)
    total_tokens, full_cost = estimate_cost(chunks)
    print(f"Estimated total tokens: {total_tokens:,}")
    print(f"Estimated full-corpus cost: ${full_cost:.2f}")

    if not cli_args.yes:
        confirm = input(f"Proceed with balanced $10 sample (full ≈${full_cost:.2f})? [y/n]: ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    grouped = group_by_company(chunks)
    sampled_chunks = balanced_sample(
        grouped,
        sample_per_company=cli_args.sample_per_company,
        max_companies=cli_args.max_companies,
    )
    create_vectorstore(sampled_chunks, batch_size=cli_args.batch_size)
    print("Indexing complete!")


if __name__ == "__main__":
    build_index()
