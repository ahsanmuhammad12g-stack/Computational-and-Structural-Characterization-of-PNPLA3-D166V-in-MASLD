import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "PROCESSED DATA"
    / "clinvar_pnpla3_master.csv"
)

OUTPUT_DIR = PROJECT_DIR / "04_VARIANT ANALYSIS"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "pnpla3_missense_variants.csv"
)


# --------------------------------------------------
# Load master dataset
# --------------------------------------------------

print("Loading PNPLA3 master ClinVar dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total PNPLA3 variants: {len(df)}")


# --------------------------------------------------
# Define true missense pattern
# --------------------------------------------------
#
# Examples matched:
# p.Glu331Gly
# p.Arg447Gln
# p.Thr216Pro
#
# Excluded:
# p.Leu324=
# p.Lys434Ter
# intronic variants
#

missense_pattern = r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})\)"


# --------------------------------------------------
# Extract protein information
# --------------------------------------------------

protein_extract = df["Name"].astype(str).str.extract(
    missense_pattern
)

protein_extract.columns = [
    "Reference_AA",
    "Protein_Position",
    "Alternate_AA"
]


# --------------------------------------------------
# Identify true missense variants
# --------------------------------------------------

missense_mask = protein_extract.notna().all(axis=1)

missense_df = df.loc[missense_mask].copy()

missense_df["Reference_AA"] = (
    protein_extract.loc[missense_mask, "Reference_AA"].values
)

missense_df["Protein_Position"] = (
    protein_extract.loc[missense_mask, "Protein_Position"]
    .astype(int)
    .values
)

missense_df["Alternate_AA"] = (
    protein_extract.loc[missense_mask, "Alternate_AA"].values
)


# --------------------------------------------------
# Create simplified protein change column
# --------------------------------------------------

missense_df["Protein_Change"] = (
    "p."
    + missense_df["Reference_AA"]
    + missense_df["Protein_Position"].astype(str)
    + missense_df["Alternate_AA"]
)


# --------------------------------------------------
# Sort variants by protein position
# --------------------------------------------------

missense_df = missense_df.sort_values(
    by="Protein_Position"
).reset_index(drop=True)


# --------------------------------------------------
# Save results
# --------------------------------------------------

missense_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("PNPLA3 MISSENSE VARIANT ANALYSIS")
print("=" * 70)

print(f"\nTotal PNPLA3 variants: {len(df)}")
print(f"True missense variants identified: {len(missense_df)}")


print("\nClinical significance distribution:")

print(
    missense_df["ClinicalSignificance"]
    .value_counts(dropna=False)
    .to_string()
)


print("\nFirst 20 missense variants:")

display_columns = [
    "VariationID",
    "Name",
    "Protein_Change",
    "ClinicalSignificance",
    "ReviewStatus"
]

print(
    missense_df[display_columns]
    .head(20)
    .to_string(index=False)
)


print("\nMissense dataset saved successfully:")
print(OUTPUT_FILE)

print("\nStep 6 completed successfully.")