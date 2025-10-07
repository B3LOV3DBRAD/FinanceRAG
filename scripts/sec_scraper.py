import os
import json
import requests
import random
from bs4 import BeautifulSoup
from time import sleep
from concurrent.futures import ThreadPoolExecutor, as_completed

# === Configuration ===
HEADERS = {"User-Agent": "FinanceRAG (contact: your_email@example.com)"}
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
CIK_FILE = os.path.join(os.path.dirname(__file__), "company_ciks.json")
SKIP_FILE = os.path.join(os.path.dirname(__file__), "no_10k.json")
BASE_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

# === Setup ===
os.makedirs(DATA_DIR, exist_ok=True)


# === Utility Functions ===
def load_ciks(limit=None):
    """Load all CIKs (or a limited subset if limit is set)."""
    with open(CIK_FILE, "r", encoding="utf-8") as f:
        company_data = json.load(f)
    if limit:
        company_data = dict(list(company_data.items())[:limit])
    print(f"📁 Loaded {len(company_data)} companies from {CIK_FILE}")
    return company_data


def load_skipped():
    """Load tickers that previously had no 10-K found."""
    if os.path.exists(SKIP_FILE):
        with open(SKIP_FILE, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_skipped(skipped):
    """Save updated skip-list."""
    with open(SKIP_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(list(skipped)), f, indent=2)
    print(f"💾 Saved skip-list with {len(skipped)} tickers.")


def get_latest_10k_or_20f_url(cik):
    """Fetch the latest 10-K or 20-F filing URL for a given CIK."""
    url = BASE_URL.format(cik=cik)
    res = requests.get(url, headers=HEADERS)
    if res.status_code != 200:
        return None

    data = res.json()
    filings = data.get("filings", {}).get("recent", {})

    for i, form in enumerate(filings.get("form", [])):
        if form in ("10-K", "20-F"):  # include foreign issuer reports
            accession = filings["accessionNumber"][i].replace("-", "")
            primary_doc = filings["primaryDocument"][i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{primary_doc}"
            return filing_url
    return None


def download_filing(url, ticker):
    """Download and save the text of a filing."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=20)
        if res.status_code != 200:
            print(f"❌ Failed to download {ticker}: {url}")
            return False

        soup = BeautifulSoup(res.text, "html.parser")
        text = soup.get_text(separator="\n", strip=True)

        filename = os.path.join(DATA_DIR, f"{ticker}_10K.txt")
        with open(filename, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved {ticker} 10-K to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error downloading {ticker}: {e}")
        return False


def process_company(ticker, info, skipped, existing):
    """Process a single company (thread-safe)."""
    if ticker in existing or ticker in skipped:
        return None

    cik = info["cik"]
    url = get_latest_10k_or_20f_url(cik)
    if not url:
        print(f"⚠️ No 10-K/20-F found for {ticker}")
        return ticker  # mark as failed/skip next time

    print(f"⬇️  Downloading {ticker} from {url}")
    success = download_filing(url, ticker)
    sleep(random.uniform(1.5, 3.0))
    if not success:
        return ticker
    return None


def main(batch_size=100, max_threads=10):
    companies = load_ciks()
    skipped = load_skipped()

    # Skip already downloaded
    existing = {f.split('_')[0] for f in os.listdir(DATA_DIR) if f.endswith(".txt")}
    remaining = [item for item in companies.items() if item[0] not in existing and item[0] not in skipped]

    print(f"🧾 {len(existing)} already done, {len(remaining)} remaining.")
    if not remaining:
        print("✅ Nothing left to download.")
        return

    # Process in batches
    for i in range(0, len(remaining), batch_size):
        batch = remaining[i:i + batch_size]
        print(f"🚀 Starting batch of {len(batch)} companies using {max_threads} threads...")

        failed_tickers = []
        with ThreadPoolExecutor(max_workers=max_threads) as executor:
            futures = {executor.submit(process_company, ticker, info, skipped, existing): ticker for ticker, info in batch}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    failed_tickers.append(result)

        skipped.update(failed_tickers)
        save_skipped(skipped)
        print(f"✅ Batch complete! Sleeping before next run...")
        sleep(random.uniform(8, 12))  # short cooldown between batches

    print("🎯 All batches complete! You can rerun anytime to resume.")


if __name__ == "__main__":
    main(batch_size=100, max_threads=10)
