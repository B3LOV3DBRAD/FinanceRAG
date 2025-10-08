
import os
import json
import asyncio
import aiohttp
from aiohttp import ClientSession
from bs4 import BeautifulSoup
from time import time

# === CONFIGURATION ===
HEADERS = {"User-Agent": "FinanceRAG (contact: your_email@example.com)"}
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
CIK_FILE = os.path.join(os.path.dirname(__file__), "company_ciks.json")
PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "progress.json")
SKIP_FILE = os.path.join(os.path.dirname(__file__), "skip_list.json")
BASE_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
CONCURRENT_LIMIT = 25   # number of concurrent requests
RETRY_LIMIT = 3         # retry per ticker
BATCH_SIZE = 250        # how many tickers to process at once

# ensure data folder exists
os.makedirs(DATA_DIR, exist_ok=True)

# === UTILITIES ===
def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

async def fetch_json(session: ClientSession, url: str):
    for attempt in range(RETRY_LIMIT):
        try:
            async with session.get(url, headers=HEADERS, timeout=30) as res:
                if res.status != 200:
                    await asyncio.sleep(1.5)
                    continue
                return await res.json()
        except Exception:
            await asyncio.sleep(2)
    return None

async def fetch_10k_url(session, cik):
    """Find latest 10-K filing URL"""
    data = await fetch_json(session, BASE_URL.format(cik=cik))
    if not data:
        return None
    filings = data.get("filings", {}).get("recent", {})
    for i, form in enumerate(filings.get("form", [])):
        if form == "10-K":
            accession = filings["accessionNumber"][i].replace("-", "")
            primary_doc = filings["primaryDocument"][i]
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
    return None

async def download_and_save(session, ticker, cik):
    """Download filing text and save to /data"""
    url = await fetch_10k_url(session, cik)
    if not url:
        return (ticker, False)

    try:
        async with session.get(url, headers=HEADERS, timeout=45) as res:
            if res.status != 200:
                return (ticker, False)
            html = await res.text()
            soup = BeautifulSoup(html, "html.parser")
            text = soup.get_text(separator="\n", strip=True)

            out_path = os.path.join(DATA_DIR, f"{ticker}_10K.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            return (ticker, True)
    except Exception:
        return (ticker, False)

async def worker(ticker_batch, company_data, progress, skip_list, sem):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for ticker in ticker_batch:
            if ticker in progress or ticker in skip_list:
                continue

            cik = company_data[ticker]["cik"]
            tasks.append(run_with_semaphore(session, ticker, cik, sem))

        results = await asyncio.gather(*tasks)
        success_count = 0
        for ticker, ok in results:
            if ok:
                progress[ticker] = True
                success_count += 1
            else:
                skip_list[ticker] = skip_list.get(ticker, 0) + 1

        print(f"Batch complete — {success_count} new filings saved.")
        save_json(PROGRESS_FILE, progress)
        save_json(SKIP_FILE, skip_list)

async def run_with_semaphore(session, ticker, cik, sem):
    async with sem:
        return await download_and_save(session, ticker, cik)

# === MAIN ===
async def main():
    company_data = load_json(CIK_FILE, {})
    progress = load_json(PROGRESS_FILE, {})
    skip_list = load_json(SKIP_FILE, {})

    print(f"Loaded {len(company_data)} companies from {CIK_FILE}")
    print(f"{len(progress)} already downloaded, {len(company_data) - len(progress)} remaining.")

    # create semaphore to limit concurrency
    sem = asyncio.Semaphore(CONCURRENT_LIMIT)
    tickers = list(company_data.keys())

    # break into batches
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        print(f"Starting batch {i // BATCH_SIZE + 1} ({len(batch)} companies)...")
        start = time()
        await worker(batch, company_data, progress, skip_list, sem)
        duration = time() - start
        print(f"⏱Batch done in {duration:.1f}s\n")
        await asyncio.sleep(5)

    print("All done!")

if __name__ == "__main__":
    asyncio.run(main())
