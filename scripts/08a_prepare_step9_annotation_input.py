import pandas as pd
from pathlib import Path


# ============================================================
# STEP 9A
# PREPARE PNPLA3 MASTER ANNOTATION INPUT
# ============================================================

print("=" * 80)
print("STEP 9A — PREPARE PNPLA3 MASTER ANNOTATION INPUT")
print("=" * 80)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

ANNOTATION_DIR = PROJECT_ROOT / "04_VARIANT ANALYSIS" / "annotations"

GNOMAD_FINAL = (
    ANNOTATION_DIR /
    "pnpla3_gnomad_population_frequency_FINAL.csv"
)

OUTPUT_FILE = (
    ANNOTATION_DIR /
    "pnpla3_step9_annotation_input.csv"
)


# ============================================================
# 2. CHECK INPUT
# ============================================================

print("\nChecking project paths...")

if not GNOMAD_FINAL.exists():
    raise FileNotFoundError(
        f"\nFinalized gnomAD file not found:\n{GNOMAD_FINAL}"
    )

ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

print(f"Input file : {GNOMAD_FINAL}")
print(f"Output file: {OUTPUT_FILE}")


# ============================================================
# 3. LOAD FINAL GNOMAD DATA
# ============================================================

print("\nLoading finalized Step 8 gnomAD dataset...")

df = pd.read_csv(GNOMAD_FINAL, dtype=str)

print(f"Records loaded: {len(df)}")


# ============================================================
# 4. REQUIRED INPUT COLUMNS
# ============================================================

required_columns = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "gnomAD_Variant_ID",
    "Exome_AF",
    "Genome_AF",
    "Final_gnomAD_Status",
    "Population_Frequency_Interpretation",
    "Name",
    "Candidate_Group",
    "cDNA_Position",
    "RS# (dbSNP)",
    "gnomAD_rsIDs",
]


missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(f"  - {c}" for c in missing_columns)
    )

print("All required columns are present.")


# ============================================================
# 5. KEEP ONLY STEP 9 MASTER COLUMNS
# ============================================================

master_columns = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "gnomAD_Variant_ID",
    "Exome_AF",
    "Genome_AF",
    "Final_gnomAD_Status",
    "Population_Frequency_Interpretation",
    "Name",
    "Candidate_Group",
    "cDNA_Position",
    "RS# (dbSNP)",
    "gnomAD_rsIDs",
]

df = df[master_columns].copy()


# ============================================================
# 6. CLEAN BASIC VALUES
# ============================================================

for col in df.columns:
    df[col] = df[col].astype("string").str.strip()

# Convert common missing-value representations to NA
missing_tokens = {
    "",
    "NA",
    "N/A",
    "nan",
    "NaN",
    "None",
    "none",
    "null",
    "NULL",
}

for col in df.columns:
    df[col] = df[col].replace(list(missing_tokens), pd.NA)


# ============================================================
# 7. REMOVE ACCIDENTAL DUPLICATE VARIANTS
# ============================================================

print("\nChecking duplicate VariationIDs...")

duplicate_ids = df[
    df["VariationID"].duplicated(keep=False)
]

if len(duplicate_ids) > 0:
    print("\nWARNING: Duplicate VariationIDs detected:")
    print(
        duplicate_ids[
            ["VariationID", "Protein_Change",
             "Chromosome", "Position_GRCh38",
             "Reference", "Alternate"]
        ].to_string(index=False)
    )

    raise ValueError(
        "\nDuplicate VariationIDs detected. "
        "Step 9 input was not generated."
    )

print("VariationID uniqueness: PASS")


# ============================================================
# 8. CHECK EXACT GENOMIC VARIANT IDENTITY
# ============================================================

print("\nChecking exact genomic variant identities...")

identity_columns = [
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
]

for col in identity_columns:
    if df[col].isna().any():
        bad = df[df[col].isna()][
            ["VariationID", "Protein_Change"] + identity_columns
        ]

        print("\nVariants with missing genomic identity:")
        print(bad.to_string(index=False))

        raise ValueError(
            f"\nMissing values detected in authoritative "
            f"variant identity column: {col}"
        )


# Normalize chromosome representation
df["Chromosome"] = (
    df["Chromosome"]
    .str.replace("chr", "", case=False, regex=False)
)

# Normalize alleles
df["Reference"] = df["Reference"].str.upper()
df["Alternate"] = df["Alternate"].str.upper()


# ============================================================
# 9. CREATE STANDARDIZED EXACT VARIANT KEY
# ============================================================

df["Exact_Variant_Key"] = (
    df["Chromosome"].astype(str)
    + "-"
    + df["Position_GRCh38"].astype(str)
    + "-"
    + df["Reference"].astype(str)
    + "-"
    + df["Alternate"].astype(str)
)


# ============================================================
# 10. CHECK EXACT VARIANT KEY DUPLICATES
# ============================================================

duplicate_keys = df[
    df["Exact_Variant_Key"].duplicated(keep=False)
]

if len(duplicate_keys) > 0:
    print("\nERROR: Duplicate exact genomic variant identities:")
    print(
        duplicate_keys[
            [
                "VariationID",
                "Protein_Change",
                "Exact_Variant_Key"
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        "\nDuplicate exact genomic variant identities detected."
    )

print("Exact genomic variant identity uniqueness: PASS")


# ============================================================
# 11. CHECK gnomAD STATUS COUNTS
# ============================================================

print("\nChecking finalized gnomAD status...")

status_counts = (
    df["Final_gnomAD_Status"]
    .value_counts(dropna=False)
)

print("\nFinal gnomAD status:")
print(status_counts.to_string())


# ============================================================
# 12. VERIFY THE 9 NOT-DETECTED VARIANTS
# ============================================================

not_detected = df[
    df["Final_gnomAD_Status"]
    == "NOT_DETECTED_IN_QUERIED_RELEASES"
].copy()

print("\nVariants not detected in queried gnomAD releases:")
print(f"Count: {len(not_detected)}")

if len(not_detected) > 0:
    print(
        not_detected[
            [
                "VariationID",
                "Protein_Change",
                "Exact_Variant_Key",
                "Population_Frequency_Interpretation",
            ]
        ].to_string(index=False)
    )


# ============================================================
# 13. IMPORTANT AF QUALITY CONTROL
# ============================================================

print("\nChecking population-frequency values...")

# Not-detected variants must NOT have AF = 0
not_detected_af_zero = not_detected[
    (
        pd.to_numeric(
            not_detected["Exome_AF"],
            errors="coerce"
        ).fillna(-1)
        == 0
    )
    |
    (
        pd.to_numeric(
            not_detected["Genome_AF"],
            errors="coerce"
        ).fillna(-1)
        == 0
    )
]

if len(not_detected_af_zero) > 0:
    print(not_detected_af_zero.to_string(index=False))

    raise ValueError(
        "\nQC FAILURE: A gnomAD-not-detected variant has AF=0."
    )

print(
    "Not-detected variants incorrectly assigned AF=0: 0"
)


# ============================================================
# 14. CHECK POPULATION FREQUENCY NUMERIC VALUES
# ============================================================

for col in ["Exome_AF", "Genome_AF"]:

    numeric_values = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    invalid = numeric_values[
        (numeric_values < 0) |
        (numeric_values > 1)
    ]

    if len(invalid) > 0:
        print(
            f"\nInvalid values detected in {col}:"
        )
        print(invalid)

        raise ValueError(
            f"\nQC FAILURE: Invalid AF values in {col}."
        )

print("AF range check: PASS")


# ============================================================
# 15. VERIFY CANDIDATE GROUPS
# ============================================================

print("\nCandidate group distribution:")

print(
    df["Candidate_Group"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# 16. ADD EMPTY STEP 9 ANNOTATION COLUMNS
# ============================================================

annotation_columns = [

    # AlphaMissense
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Status",

    # CADD
    "CADD_Score",
    "CADD_Status",

    # REVEL
    "REVEL_Score",
    "REVEL_Status",

    # Conservation
    "Conservation_Score",
    "Conservation_Source",
    "Conservation_Annotation",

    # Protein/domain/function
    "Protein_Domain",
    "Functional_Annotation",
    "Protein_Annotation_Source",

    # Overall annotation QC
    "Annotation_Notes",
]


for col in annotation_columns:
    df[col] = pd.NA


# ============================================================
# 17. FINAL COLUMN ORDER
# ============================================================

final_columns = [
    "VariationID",
    "Protein_Change",

    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",

    "cDNA_Position",

    "RS# (dbSNP)",
    "gnomAD_rsIDs",
    "gnomAD_Variant_ID",

    "Exome_AF",
    "Genome_AF",
    "Final_gnomAD_Status",
    "Population_Frequency_Interpretation",

    "Name",
    "Candidate_Group",

    # Step 9 annotations
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Status",

    "CADD_Score",
    "CADD_Status",

    "REVEL_Score",
    "REVEL_Status",

    "Conservation_Score",
    "Conservation_Source",
    "Conservation_Annotation",

    "Protein_Domain",
    "Functional_Annotation",
    "Protein_Annotation_Source",

    "Annotation_Notes",
]

df = df[final_columns]


# ============================================================
# 18. FINAL 82-VARIANT CHECK
# ============================================================

print("\n" + "=" * 80)
print("FINAL STEP 9A QUALITY CONTROL")
print("=" * 80)

print(
    f"Total variants:                     {len(df)}"
)

print(
    f"Unique VariationIDs:                "
    f"{df['VariationID'].nunique()}"
)

print(
    f"Unique exact variant identities:     "
    f"{df['Exact_Variant_Key'].nunique()}"
)

print(
    f"Not detected in queried releases:   "
    f"{len(not_detected)}"
)

print(
    f"Annotation columns created:         "
    f"{len(annotation_columns)}"
)


# ============================================================
# 19. ASSERT 82 VARIANTS
# ============================================================

if len(df) != 82:
    raise ValueError(
        f"\nQC FAILURE: Expected 82 variants, found {len(df)}."
    )

if df["VariationID"].nunique() != 82:
    raise ValueError(
        "\nQC FAILURE: VariationID count is not 82."
    )

if df["Exact_Variant_Key"].nunique() != 82:
    raise ValueError(
        "\nQC FAILURE: Exact variant identity count is not 82."
    )

print("\n82/82 variants retained: PASS")


# ============================================================
# 20. SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 21. CONFIRM OUTPUT
# ============================================================

print("\n" + "=" * 80)
print("STEP 9A SUCCESSFULLY COMPLETED")
print("=" * 80)

print(f"\nOutput file:")
print(OUTPUT_FILE)

print(
    f"\nFinal records written: {len(df)}"
)

print("\nStep 9 master annotation input is ready.")

print(
    "\nNext stage:"
    "\n  Step 9B — AlphaMissense annotation"
)

print("=" * 80)