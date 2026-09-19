import pandas as pd
from pathlib import Path

# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = PROJECT_DIR / "RAW DATA" / "variant_summary.txt.gz"

PROCESSED_DIR = PROJECT_DIR / "PROCESSED DATA"
PROCESSED_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = PROCESSED_DIR / "clinvar_pnpla3_raw_filtered.csv"


# --------------------------------------------------
# Load ClinVar dataset
# --------------------------------------------------

print("Loading ClinVar variant summary dataset...")
print("This may take a little time because the file is large.")

df = pd.read_csv(
    RAW_FILE,
    sep="\t",
    compression="gzip",
    low_memory=False
)

print(f"\nTotal ClinVar records loaded: {len(df):,}")
print(f"Total columns: {len(df.columns)}")


# --------------------------------------------------
# Display columns for verification
# --------------------------------------------------

print("\nAvailable columns:")
print(df.columns.tolist())


# --------------------------------------------------
# Filter PNPLA3 records
# --------------------------------------------------

print("\nFiltering PNPLA3 variants...")

pnpla3_df = df[
    df["GeneSymbol"].astype(str).str.upper() == "PNPLA3"
].copy()


print(f"PNPLA3 records found: {len(pnpla3_df)}")


# --------------------------------------------------
# Save filtered dataset
# --------------------------------------------------

pnpla3_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFiltered PNPLA3 dataset saved successfully:")
print(OUTPUT_FILE)


# --------------------------------------------------
# Quick preview
# --------------------------------------------------

print("\nFirst 10 PNPLA3 records:")

preview_columns = [
    "VariationID",
    "Name",
    "GeneSymbol",
    "ClinicalSignificance",
    "ReviewStatus",
    "Type"
]

# Only display columns that actually exist
preview_columns = [
    col for col in preview_columns
    if col in pnpla3_df.columns
]

print(
    pnpla3_df[preview_columns]
    .head(10)
    .to_string(index=False)
)

print("\nStep 5 initial filtering completed successfully.")