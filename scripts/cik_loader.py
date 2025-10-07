import requests
import json
import os

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "company_ciks.json")
SEC_URL = "https://www.sec.gov/files/company_tickers.json"

def fetch_cik_data():
    print("🔍 Fetching company list from SEC...")
    response = requests.get(SEC_URL, headers={"User-Agent": "FinanceRAG (contact: your_email@example.com)"})
    response.raise_for_status()
    data = response.json()

    company_ciks = {}
    for entry in data.values():
        name = entry["title"]
        cik = str(entry["cik_str"]).zfill(10)
        ticker = entry["ticker"]
        company_ciks[ticker] = {"name": name, "cik": cik}

    with open(OUTPUT_FILE, "w") as f:
        json.dump(company_ciks, f, indent=2)

    print(f"Saved {len(company_ciks)} company CIKs to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_cik_data()
