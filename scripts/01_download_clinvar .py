import requests
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_DIR / "RAW DATA"

RAW_DATA_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RAW_DATA_DIR / "variant_summary.txt.gz"

# --------------------------------------------------
# Official ClinVar variant summary URL
# --------------------------------------------------

URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/"
    "variant_summary.txt.gz"
)

print("Downloading official ClinVar variant summary...")
print("This file is large, so it may take some time.")

response = requests.get(URL, stream=True, timeout=300)
response.raise_for_status()

with open(OUTPUT_FILE, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            file.write(chunk)

print("\nDownload completed successfully.")
print("Saved to:")
print(OUTPUT_FILE)

print("\nFile size:")
print(f"{OUTPUT_FILE.stat().st_size / (1024 * 1024):.2f} MB")