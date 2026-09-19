import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "PROCESSED DATA"
    / "clinvar_pnpla3_raw_filtered.csv"
)


# --------------------------------------------------
# Load PNPLA3 dataset
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

print("=" * 70)
print("PNPLA3 CLINVAR DATASET INSPECTION")
print("=" * 70)

print(f"\nTotal rows: {len(df)}")

print(f"Unique VariationIDs: {df['VariationID'].nunique()}")

print(f"Duplicate rows based on VariationID: "
      f"{len(df) - df['VariationID'].nunique()}")


# --------------------------------------------------
# Variant type distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("VARIANT TYPE DISTRIBUTION")
print("=" * 70)

print(
    df["Type"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# Clinical significance distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("CLINICAL SIGNIFICANCE DISTRIBUTION")
print("=" * 70)

print(
    df["ClinicalSignificance"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# Review status distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("REVIEW STATUS DISTRIBUTION")
print("=" * 70)

print(
    df["ReviewStatus"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# Molecular origin distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("ORIGIN DISTRIBUTION")
print("=" * 70)

print(
    df["Origin"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# Assembly distribution
# --------------------------------------------------

print("\n" + "=" * 70)
print("ASSEMBLY DISTRIBUTION")
print("=" * 70)

print(
    df["Assembly"]
    .value_counts(dropna=False)
    .to_string()
)


# --------------------------------------------------
# Example duplicate VariationIDs
# --------------------------------------------------

duplicate_ids = (
    df["VariationID"]
    .value_counts()
)

duplicate_ids = duplicate_ids[
    duplicate_ids > 1
].head(10)

print("\n" + "=" * 70)
print("EXAMPLE DUPLICATE VARIATION IDs")
print("=" * 70)

print(duplicate_ids.to_string())


# --------------------------------------------------
# Missense-like records based on protein notation
# --------------------------------------------------

protein_variants = df[
    df["Name"]
    .astype(str)
    .str.contains(r"\(p\.", regex=True, na=False)
]

print("\n" + "=" * 70)
print("RECORDS CONTAINING PROTEIN CHANGE NOTATION")
print("=" * 70)

print(f"Rows containing p. notation: {len(protein_variants)}")

print(
    protein_variants[
        ["VariationID", "Name", "ClinicalSignificance", "Type"]
    ]
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("INSPECTION COMPLETED")
print("=" * 70)