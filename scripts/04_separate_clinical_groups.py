import pandas as pd
from pathlib import Path


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "04_VARIANT ANALYSIS"
    / "pnpla3_missense_variants.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "04_VARIANT ANALYSIS"
    / "clinical_groups"
)

OUTPUT_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# Load missense dataset
# --------------------------------------------------

print("Loading PNPLA3 missense variant dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Total missense variants: {len(df)}")


# --------------------------------------------------
# Normalize clinical significance
# --------------------------------------------------

df["ClinicalSignificance"] = (
    df["ClinicalSignificance"]
    .fillna("-")
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# Create clinical groups
# --------------------------------------------------

vus_df = df[
    df["ClinicalSignificance"]
    .str.lower()
    .eq("uncertain significance")
].copy()


conflicting_df = df[
    df["ClinicalSignificance"]
    .str.contains(
        "conflicting",
        case=False,
        na=False
    )
].copy()


benign_df = df[
    df["ClinicalSignificance"]
    .str.lower()
    .isin([
        "benign",
        "likely benign",
        "benign/likely benign"
    ])
].copy()


unclassified_df = df[
    df["ClinicalSignificance"]
    .eq("-")
].copy()


# --------------------------------------------------
# Save separate datasets
# --------------------------------------------------

vus_df.to_csv(
    OUTPUT_DIR / "pnpla3_vus_missense.csv",
    index=False
)

conflicting_df.to_csv(
    OUTPUT_DIR / "pnpla3_conflicting_missense.csv",
    index=False
)

benign_df.to_csv(
    OUTPUT_DIR / "pnpla3_benign_missense.csv",
    index=False
)

unclassified_df.to_csv(
    OUTPUT_DIR / "pnpla3_unclassified_missense.csv",
    index=False
)


# --------------------------------------------------
# Create summary table
# --------------------------------------------------

summary = pd.DataFrame({
    "Clinical_Group": [
        "VUS",
        "Conflicting",
        "Benign/Likely Benign",
        "Unclassified"
    ],
    "Variant_Count": [
        len(vus_df),
        len(conflicting_df),
        len(benign_df),
        len(unclassified_df)
    ]
})

summary.to_csv(
    OUTPUT_DIR / "clinical_group_summary.csv",
    index=False
)


# --------------------------------------------------
# Print summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("PNPLA3 MISSENSE CLINICAL GROUPS")
print("=" * 70)

print(f"\nTotal missense variants: {len(df)}")

print(f"VUS variants: {len(vus_df)}")
print(f"Conflicting variants: {len(conflicting_df)}")
print(f"Benign/Likely benign variants: {len(benign_df)}")
print(f"Unclassified variants: {len(unclassified_df)}")


# --------------------------------------------------
# Display conflicting variants
# --------------------------------------------------

print("\n" + "=" * 70)
print("CONFLICTING VARIANTS")
print("=" * 70)

if len(conflicting_df) > 0:

    columns = [
        "VariationID",
        "Name",
        "Protein_Change",
        "ClinicalSignificance",
        "ReviewStatus"
    ]

    print(
        conflicting_df[columns]
        .to_string(index=False)
    )

else:
    print("No conflicting variants found.")


print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(f"Output directory: {OUTPUT_DIR}")

print("\nStep 7 completed successfully.")