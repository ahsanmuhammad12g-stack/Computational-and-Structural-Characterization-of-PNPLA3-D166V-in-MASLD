import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "PROCESSED DATA"
    / "clinvar_pnpla3_raw_filtered.csv"
)

PROCESSED_DIR = PROJECT_DIR / "PROCESSED DATA"

OUTPUT_FILE = (
    PROCESSED_DIR
    / "clinvar_pnpla3_master.csv"
)


# --------------------------------------------------
# Load raw filtered PNPLA3 dataset
# --------------------------------------------------

print("Loading PNPLA3 ClinVar dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Initial rows: {len(df)}")
print(f"Initial unique VariationIDs: {df['VariationID'].nunique()}")


# --------------------------------------------------
# Keep GRCh38 only
# --------------------------------------------------

print("\nKeeping GRCh38 records only...")

df_grch38 = df[
    df["Assembly"] == "GRCh38"
].copy()

print(f"Rows after GRCh38 selection: {len(df_grch38)}")
print(f"Unique VariationIDs: {df_grch38['VariationID'].nunique()}")


# --------------------------------------------------
# Remove duplicate Variation IDs
# --------------------------------------------------

df_master = df_grch38.drop_duplicates(
    subset=["VariationID"],
    keep="first"
).copy()

print(f"\nRows after deduplication: {len(df_master)}")


# --------------------------------------------------
# Sort by VariationID
# --------------------------------------------------

df_master = df_master.sort_values(
    by="VariationID"
).reset_index(drop=True)


# --------------------------------------------------
# Save master dataset
# --------------------------------------------------

df_master.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nMaster PNPLA3 dataset saved successfully:")
print(OUTPUT_FILE)


# --------------------------------------------------
# Final summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("MASTER DATASET SUMMARY")
print("=" * 70)

print(f"Total unique PNPLA3 variants: {len(df_master)}")

print("\nClinical significance:")

print(
    df_master["ClinicalSignificance"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nVariant types:")

print(
    df_master["Type"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nAssembly:")

print(
    df_master["Assembly"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nMaster dataset creation completed successfully.")