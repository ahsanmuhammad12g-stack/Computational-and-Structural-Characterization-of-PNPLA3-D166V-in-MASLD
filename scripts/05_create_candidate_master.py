import pandas as pd
from pathlib import Path

# ============================================================
# STEP 8A — CREATE PNPLA3 CANDIDATE MASTER TABLE
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = BASE_DIR / "04_VARIANT ANALYSIS" / "clinical_groups"
OUTPUT_DIR = BASE_DIR / "04_VARIANT ANALYSIS"

VUS_FILE = INPUT_DIR / "pnpla3_vus_missense.csv"
CONFLICTING_FILE = INPUT_DIR / "pnpla3_conflicting_missense.csv"

OUTPUT_FILE = OUTPUT_DIR / "pnpla3_candidate_master.csv"

print("=" * 70)
print("PNPLA3 CANDIDATE MASTER TABLE")
print("=" * 70)

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading VUS variants...")
vus = pd.read_csv(VUS_FILE)

print("Loading conflicting variants...")
conflicting = pd.read_csv(CONFLICTING_FILE)

print(f"VUS variants: {len(vus)}")
print(f"Conflicting variants: {len(conflicting)}")

# ------------------------------------------------------------
# Add candidate group
# ------------------------------------------------------------

vus["Candidate_Group"] = "VUS"
conflicting["Candidate_Group"] = "Conflicting"

# ------------------------------------------------------------
# Combine
# ------------------------------------------------------------

candidates = pd.concat(
    [vus, conflicting],
    ignore_index=True
)

# ------------------------------------------------------------
# Remove duplicate VariationIDs
# ------------------------------------------------------------

candidates = candidates.drop_duplicates(
    subset="VariationID"
).reset_index(drop=True)

# ------------------------------------------------------------
# Select useful columns
# ------------------------------------------------------------

preferred_columns = [
    "VariationID",
    "Name",
    "GeneSymbol",
    "HGNC_ID",
    "ClinicalSignificance",
    "ReviewStatus",
    "Origin",
    "OriginSimple",
    "Assembly",
    "ChromosomeAccession",
    "Chromosome",
    "Start",
    "Stop",
    "ReferenceAllele",
    "AlternateAllele",
    "RS# (dbSNP)",
    "Type",
    "Protein_Change",
    "Reference_AA",
    "Protein_Position",
    "Alternate_AA",
    "Candidate_Group",
    "RCVaccession",
    "PhenotypeList",
    "OtherIDs"
]

# Keep only columns that actually exist
available_columns = [
    col for col in preferred_columns
    if col in candidates.columns
]

candidates = candidates[available_columns]

# ------------------------------------------------------------
# Sort: VUS first, then conflicting
# ------------------------------------------------------------

group_order = {
    "VUS": 0,
    "Conflicting": 1
}

candidates["_sort"] = candidates["Candidate_Group"].map(group_order)

candidates = candidates.sort_values(
    by=["_sort", "Protein_Position", "VariationID"]
)

candidates = candidates.drop(columns=["_sort"])

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

candidates.to_csv(
    OUTPUT_FILE,
    index=False
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CANDIDATE SUMMARY")
print("=" * 70)

print(f"Total candidate variants: {len(candidates)}")
print(
    candidates["Candidate_Group"]
    .value_counts()
    .to_string()
)

print("\nCandidate protein changes:")
print(
    candidates[
        ["VariationID", "Protein_Change", "Candidate_Group"]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("FILE SAVED")
print("=" * 70)

print(f"Output: {OUTPUT_FILE}")
print("\nStep 8A completed successfully.")