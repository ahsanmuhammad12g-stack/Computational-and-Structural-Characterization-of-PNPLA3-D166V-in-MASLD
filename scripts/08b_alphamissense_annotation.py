import pandas as pd
from pathlib import Path


print("=" * 78)
print("STEP 9B — FINAL ALPHAMISSENSE ANNOTATION REPAIR")
print("=" * 78)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

ANNOTATION_DIR = PROJECT_ROOT / "04_VARIANT ANALYSIS" / "annotations"

INPUT_FILE = (
    ANNOTATION_DIR /
    "pnpla3_step9_alphamissense_annotated.csv"
)

OUTPUT_FILE = (
    ANNOTATION_DIR /
    "pnpla3_step9_alphamissense_annotated_FINAL.csv"
)

ALPHAMISSENSE_ANNOTATION_FILE = (
    ANNOTATION_DIR /
    "pnpla3_alphamissense_annotation.csv"
)


# ============================================================
# CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )


print(f"\nInput file:")
print(INPUT_FILE)

print("\nReading existing Step 9B master...")
df = pd.read_csv(INPUT_FILE, low_memory=False)

print(f"Records loaded: {len(df)}")


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Status",
    "AlphaMissense_AM",
    "AlphaMissense_Classification_AM",
    "Genome_Build",
    "UniProt_ID",
    "Transcript_ID",
    "Protein_Variant",
]


missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(f"  - {x}" for x in missing)
    )

print("\nRequired columns: PASS")


# ============================================================
# BASIC QC
# ============================================================

print("\n" + "-" * 78)
print("BASIC INPUT QC")
print("-" * 78)

if df["VariationID"].duplicated().any():
    duplicates = df.loc[
        df["VariationID"].duplicated(keep=False),
        ["VariationID", "Protein_Change"]
    ]

    print("\nDuplicate VariationIDs detected:")
    print(duplicates.to_string(index=False))

    raise ValueError(
        "\nVariationID uniqueness QC FAILED."
    )

print("VariationID uniqueness: PASS")


if df["Exact_Variant_Key"].duplicated().any():
    duplicates = df.loc[
        df["Exact_Variant_Key"].duplicated(keep=False),
        ["VariationID", "Exact_Variant_Key"]
    ]

    print("\nDuplicate genomic identities detected:")
    print(duplicates.to_string(index=False))

    raise ValueError(
        "\nExact genomic variant identity uniqueness QC FAILED."
    )

print("Exact genomic variant identity uniqueness: PASS")


# ============================================================
# VERIFY THE PREVIOUS EXTRACTION
# ============================================================

print("\n" + "-" * 78)
print("CHECKING PREVIOUS ALPHAMISSENSE EXTRACTION")
print("-" * 78)


am_scores = pd.to_numeric(
    df["AlphaMissense_AM"],
    errors="coerce"
)

am_classes = (
    df["AlphaMissense_Classification_AM"]
    .astype("string")
    .str.strip()
)


score_found = am_scores.notna()
class_found = am_classes.notna() & (am_classes != "")


print(
    f"\nAlphaMissense scores already present: "
    f"{score_found.sum()}"
)

print(
    f"AlphaMissense classifications already present: "
    f"{class_found.sum()}"
)


# ============================================================
# IMPORTANT SAFETY CHECK
# ============================================================

if score_found.sum() != len(df):
    missing_scores = df.loc[
        ~score_found,
        [
            "VariationID",
            "Protein_Change",
            "Exact_Variant_Key"
        ]
    ]

    print("\nVariants without AlphaMissense scores:")
    print(missing_scores.to_string(index=False))

    raise ValueError(
        "\nNot all 82 variants contain the previously extracted "
        "AlphaMissense score. Stop rather than silently filling values."
    )


if class_found.sum() != len(df):
    missing_classes = df.loc[
        ~class_found,
        [
            "VariationID",
            "Protein_Change",
            "Exact_Variant_Key"
        ]
    ]

    print("\nVariants without AlphaMissense classifications:")
    print(missing_classes.to_string(index=False))

    raise ValueError(
        "\nNot all variants contain AlphaMissense classifications."
    )


print("\nPrevious AlphaMissense extraction: VALID")
print("All candidate variants have AlphaMissense values.")


# ============================================================
# COPY THE VALIDATED VALUES INTO THE CORRECT COLUMNS
# ============================================================

print("\n" + "-" * 78)
print("REPAIRING ALPHAMISSENSE COLUMNS")
print("-" * 78)


df["AlphaMissense"] = am_scores

df["AlphaMissense_Classification"] = am_classes


# ============================================================
# ALPHAMISSENSE INTERPRETATION
# ============================================================

def interpret_alphamissense(score):
    """
    AlphaMissense computational interpretation.

    < 0.34       = likely benign
    0.34–0.564   = ambiguous
    > 0.564      = likely pathogenic

    These are computational classifications and are NOT
    equivalent to clinical diagnostic classifications.
    """

    if pd.isna(score):
        return pd.NA

    if score < 0.34:
        return "Likely benign"

    elif score <= 0.564:
        return "Ambiguous"

    else:
        return "Likely pathogenic"


df["AlphaMissense_Interpretation"] = (
    df["AlphaMissense"]
    .apply(interpret_alphamissense)
)


# ============================================================
# STATUS
# ============================================================

df["AlphaMissense_Status"] = (
    "EXACT_MATCH"
)


# ============================================================
# ALPHAMISSENSE SCORE QC
# ============================================================

print("\n" + "-" * 78)
print("ALPHAMISSENSE SCORE QC")
print("-" * 78)


invalid_scores = df[
    (df["AlphaMissense"] < 0) |
    (df["AlphaMissense"] > 1)
]

if len(invalid_scores) > 0:

    print("\nInvalid AlphaMissense scores detected:")
    print(
        invalid_scores[
            [
                "VariationID",
                "Protein_Change",
                "AlphaMissense"
            ]
        ].to_string(index=False)
    )

    raise ValueError(
        "\nAlphaMissense score range QC FAILED."
    )

print("Score range 0–1: PASS")


print(
    f"Minimum AlphaMissense score: "
    f"{df['AlphaMissense'].min():.4f}"
)

print(
    f"Maximum AlphaMissense score: "
    f"{df['AlphaMissense'].max():.4f}"
)


# ============================================================
# CLASSIFICATION QC
# ============================================================

print("\n" + "-" * 78)
print("ALPHAMISSENSE CLASSIFICATION QC")
print("-" * 78)

print(
    "\nAlphaMissense supplied classifications:"
)

print(
    df["AlphaMissense_Classification"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# INTERPRETATION DISTRIBUTION
# ============================================================

print("\n" + "-" * 78)
print("COMPUTATIONAL INTERPRETATION")
print("-" * 78)

print(
    df["AlphaMissense_Interpretation"]
    .value_counts(dropna=False)
    .to_string()
)


# ============================================================
# VERIFY ALL 82 WERE RETAINED
# ============================================================

print("\n" + "-" * 78)
print("FINAL RETENTION QC")
print("-" * 78)


expected_n = 82
actual_n = len(df)

print(f"\nExpected variants: {expected_n}")
print(f"Actual variants:   {actual_n}")


if actual_n != expected_n:
    raise ValueError(
        f"\nVariant retention QC FAILED: "
        f"expected {expected_n}, found {actual_n}."
    )

print("82/82 variants retained: PASS")


# ============================================================
# VERIFY REQUIRED ALPHAMISSENSE FIELDS
# ============================================================

required_am_fields = [
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Interpretation",
    "AlphaMissense_Status",
    "Genome_Build",
    "UniProt_ID",
    "Transcript_ID",
    "Protein_Variant",
]


for col in required_am_fields:

    missing_count = df[col].isna().sum()

    if missing_count > 0:

        raise ValueError(
            f"\nField QC FAILED for {col}: "
            f"{missing_count} missing values."
        )


print(
    "All required AlphaMissense fields populated: PASS"
)


# ============================================================
# VERIFY NO NA SCORES
# ============================================================

if df["AlphaMissense"].isna().any():

    raise ValueError(
        "\nAlphaMissense missing-score QC FAILED."
    )

print("AlphaMissense missing-score QC: PASS")


# ============================================================
# CREATE CLEAN ALPHAMISSENSE ANNOTATION TABLE
# ============================================================

annotation_columns = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Interpretation",
    "AlphaMissense_Status",
    "Genome_Build",
    "UniProt_ID",
    "Transcript_ID",
    "Protein_Variant",
]


am_annotation = df[annotation_columns].copy()


# ============================================================
# SAVE ALPHAMISSENSE ANNOTATION TABLE
# ============================================================

am_annotation.to_csv(
    ALPHAMISSENSE_ANNOTATION_FILE,
    index=False
)

print(
    f"\nAlphaMissense annotation table written:\n"
    f"{ALPHAMISSENSE_ANNOTATION_FILE}"
)


# ============================================================
# REMOVE TEMPORARY / DUPLICATE AM COLUMNS
# ============================================================

temporary_columns = [
    "AlphaMissense_AM",
    "AlphaMissense_Classification_AM",
]

for col in temporary_columns:

    if col in df.columns:
        df.drop(columns=col, inplace=True)


# ============================================================
# FINAL COLUMN ORDER
# ============================================================

preferred_order = [
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

    # AlphaMissense
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Interpretation",
    "AlphaMissense_Status",

    # Other Step 9 fields
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

    # Reference metadata
    "Genome_Build",
    "UniProt_ID",
    "Transcript_ID",
    "Protein_Variant",
]


final_order = [
    col for col in preferred_order
    if col in df.columns
]

remaining_columns = [
    col for col in df.columns
    if col not in final_order
]

df = df[
    final_order + remaining_columns
]


# ============================================================
# SAVE FINAL STEP 9B MASTER
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL READ-BACK QC
# ============================================================

print("\n" + "=" * 78)
print("FINAL STEP 9B QUALITY CONTROL")
print("=" * 78)


check = pd.read_csv(
    OUTPUT_FILE,
    low_memory=False
)


print(
    f"\nPNPLA3 variants analyzed: "
    f"{len(check)}"
)

print(
    f"AlphaMissense exact matches: "
    f"{(check['AlphaMissense_Status'] == 'EXACT_MATCH').sum()}"
)

print(
    f"AlphaMissense unmatched: "
    f"{(check['AlphaMissense_Status'] != 'EXACT_MATCH').sum()}"
)

print(
    f"Unique VariationIDs: "
    f"{check['VariationID'].nunique()}"
)

print(
    f"Unique exact variant identities: "
    f"{check['Exact_Variant_Key'].nunique()}"
)


# ============================================================
# FINAL ASSERTIONS
# ============================================================

assert len(check) == 82
assert check["VariationID"].nunique() == 82
assert check["Exact_Variant_Key"].nunique() == 82
assert check["AlphaMissense"].notna().all()
assert check["AlphaMissense_Classification"].notna().all()
assert check["AlphaMissense_Interpretation"].notna().all()
assert (
    check["AlphaMissense_Status"]
    .eq("EXACT_MATCH")
    .all()
)


print("\n" + "-" * 78)
print("FINAL STEP 9B QC: ALL TESTS PASSED")
print("-" * 78)


print("\nAlphaMissense score distribution:")
print(
    check["AlphaMissense"]
    .describe()
    .to_string()
)


print("\nComputational interpretation distribution:")
print(
    check["AlphaMissense_Interpretation"]
    .value_counts()
    .to_string()
)


print("\n" + "=" * 78)
print("STEP 9B SUCCESSFULLY COMPLETED")
print("=" * 78)

print(
    "\nFinal master file:"
    f"\n{OUTPUT_FILE}"
)

print(
    "\nClean AlphaMissense annotation file:"
    f"\n{ALPHAMISSENSE_ANNOTATION_FILE}"
)

print("\nNo AlphaMissense redownload was required.")
print("No AlphaMissense dataset rescan was required.")