# ============================================================
# STEP 9C — FINAL CADD v1.7 ANNOTATION AND QUALITY CONTROL
# ============================================================
# Handles headerless CADD result files automatically.
#
# Input:
#   pnpla3_step9_alphamissense_annotated_FINAL.csv
#   CADD_GRCh38_v1.7_result.tsv
#
# Output:
#   pnpla3_cadd_annotation.csv
#   pnpla3_step9_cadd_annotated_FINAL.csv
# ============================================================

import pandas as pd
from pathlib import Path
import sys


print("=" * 78)
print("STEP 9C — FINAL CADD v1.7 ANNOTATION AND QUALITY CONTROL")
print("=" * 78)


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

ANNOTATION_DIR = PROJECT_DIR / "04_VARIANT ANALYSIS" / "annotations"

INPUT_MASTER = (
    ANNOTATION_DIR /
    "pnpla3_step9_alphamissense_annotated_FINAL.csv"
)

CADD_RESULT = (
    ANNOTATION_DIR /
    "CADD_GRCh38_v1.7_result.tsv"
)

OUTPUT_ANNOTATION = (
    ANNOTATION_DIR /
    "pnpla3_cadd_annotation.csv"
)

OUTPUT_MASTER = (
    ANNOTATION_DIR /
    "pnpla3_step9_cadd_annotated_FINAL.csv"
)


# ============================================================
# REQUIRED MASTER COLUMNS
# ============================================================

REQUIRED_MASTER_COLUMNS = [
    "VariationID",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate"
]


# ============================================================
# CHECK INPUT FILES
# ============================================================

print("\nChecking input files...")

if not INPUT_MASTER.exists():
    print(f"\nERROR: Step 9B master file not found:\n{INPUT_MASTER}")
    sys.exit(1)

print("Step 9B master: PASS")

if not CADD_RESULT.exists():
    print(f"\nERROR: CADD result file not found:\n{CADD_RESULT}")
    print("\nPlace the downloaded CADD result here with filename:")
    print("CADD_GRCh38_v1.7_result.tsv")
    sys.exit(1)

print("CADD result file: PASS")


# ============================================================
# READ STEP 9B MASTER
# ============================================================

print("\nReading validated Step 9B master...")

master = pd.read_csv(INPUT_MASTER)

print(f"Records loaded: {len(master)}")


# ============================================================
# MASTER QC
# ============================================================

print("\n" + "-" * 78)
print("BASIC INPUT QC")
print("-" * 78)


missing_master_cols = [
    col for col in REQUIRED_MASTER_COLUMNS
    if col not in master.columns
]

if missing_master_cols:
    print("\nERROR: Required Step 9B columns missing:")
    for col in missing_master_cols:
        print(f"  - {col}")
    sys.exit(1)

print("Required Step 9B columns: PASS")


EXPECTED_VARIANTS = 82

if len(master) != EXPECTED_VARIANTS:
    print(
        f"WARNING: Expected {EXPECTED_VARIANTS} variants, "
        f"found {len(master)}"
    )
else:
    print(f"{EXPECTED_VARIANTS} variants: PASS")


if master["VariationID"].duplicated().any():
    print("ERROR: Duplicate VariationIDs detected.")
    sys.exit(1)

print("VariationID uniqueness: PASS")


# Normalize master genomic fields
master["Chromosome"] = (
    master["Chromosome"]
    .astype(str)
    .str.replace("chr", "", case=False, regex=False)
    .str.strip()
)

master["Position_GRCh38"] = pd.to_numeric(
    master["Position_GRCh38"],
    errors="coerce"
).astype("Int64")

master["Reference"] = (
    master["Reference"]
    .astype(str)
    .str.upper()
    .str.strip()
)

master["Alternate"] = (
    master["Alternate"]
    .astype(str)
    .str.upper()
    .str.strip()
)


master["Variant_Key"] = (
    master["Chromosome"].astype(str)
    + "-"
    + master["Position_GRCh38"].astype(str)
    + "-"
    + master["Reference"]
    + "-"
    + master["Alternate"]
)


if master["Variant_Key"].duplicated().any():
    print("ERROR: Duplicate genomic variant identities detected.")
    sys.exit(1)

print("Exact genomic identity uniqueness: PASS")


# ============================================================
# READ CADD RESULT
# ============================================================

print("\nReading official CADD v1.7 result...")


# ------------------------------------------------------------
# CADD downloads may have:
#   1. A proper header:
#      Chrom Pos Ref Alt RawScore PHRED
#
#   2. No header:
#      22 43923921 G A 0.578266 6.246
#
# Read without assuming headers first.
# ------------------------------------------------------------

try:
    cadd_raw = pd.read_csv(
        CADD_RESULT,
        sep="\t",
        header=None,
        dtype=str,
        comment="#"
    )
except Exception as e:
    print("\nERROR: Unable to read CADD result file.")
    print(e)
    sys.exit(1)


# Remove completely empty rows
cadd_raw = cadd_raw.dropna(how="all")


if len(cadd_raw) == 0:
    print("\nERROR: CADD result file is empty.")
    sys.exit(1)


print(f"CADD records loaded (raw): {len(cadd_raw)}")


# ============================================================
# DETECT HEADER VS HEADERLESS FILE
# ============================================================

first_row = [
    str(x).strip().lower()
    for x in cadd_raw.iloc[0].tolist()
]

header_keywords = {
    "chrom",
    "chr",
    "#chrom",
    "pos",
    "position",
    "ref",
    "alt",
    "rawscore",
    "raw_score",
    "phred"
}


has_header = any(
    value in header_keywords
    for value in first_row
)


if has_header:

    print("CADD file format: HEADER DETECTED")

    header = cadd_raw.iloc[0].tolist()

    cadd = cadd_raw.iloc[1:].copy()

    cadd.columns = [
        str(x).strip().replace("#", "")
        for x in header
    ]

else:

    print("CADD file format: HEADERLESS DATA DETECTED")

    cadd = cadd_raw.copy()

    if cadd.shape[1] < 6:
        print(
            "\nERROR: Headerless CADD result has fewer than "
            "6 required columns."
        )
        print(f"Columns detected: {cadd.shape[1]}")
        sys.exit(1)

    # Official expected structure:
    # Chrom | Pos | Ref | Alt | RawScore | PHRED
    cadd = cadd.iloc[:, :6].copy()

    cadd.columns = [
        "Chrom",
        "Pos",
        "Ref",
        "Alt",
        "RawScore",
        "PHRED"
    ]


print(f"CADD records available: {len(cadd)}")

print("\nCADD columns detected:")
for col in cadd.columns:
    print(f"  {col}")


# ============================================================
# STANDARDIZE CADD COLUMN NAMES
# ============================================================

column_mapping = {}

for col in cadd.columns:

    normalized = (
        str(col)
        .strip()
        .lower()
        .replace("#", "")
        .replace("_", "")
        .replace(" ", "")
    )

    if normalized in ["chrom", "chr", "chromosome"]:
        column_mapping[col] = "Chrom"

    elif normalized in ["pos", "position"]:
        column_mapping[col] = "Pos"

    elif normalized == "ref":
        column_mapping[col] = "Ref"

    elif normalized == "alt":
        column_mapping[col] = "Alt"

    elif normalized in ["rawscore", "raw"]:
        column_mapping[col] = "RawScore"

    elif normalized in ["phred", "phredscore"]:
        column_mapping[col] = "PHRED"


cadd = cadd.rename(columns=column_mapping)


REQUIRED_CADD_COLUMNS = [
    "Chrom",
    "Pos",
    "Ref",
    "Alt",
    "RawScore",
    "PHRED"
]


missing_cadd_cols = [
    col for col in REQUIRED_CADD_COLUMNS
    if col not in cadd.columns
]


if missing_cadd_cols:

    print("\nERROR: Required CADD columns are missing:")

    for col in missing_cadd_cols:
        print(f"  - {col}")

    print("\nAvailable columns:")
    print(list(cadd.columns))

    sys.exit(1)


print("\nRequired CADD columns: PASS")


# ============================================================
# CLEAN CADD DATA
# ============================================================

print("\n" + "-" * 78)
print("PROCESSING CADD RESULTS")
print("-" * 78)


cadd["Chrom"] = (
    cadd["Chrom"]
    .astype(str)
    .str.replace("chr", "", case=False, regex=False)
    .str.strip()
)

cadd["Pos"] = pd.to_numeric(
    cadd["Pos"],
    errors="coerce"
).astype("Int64")

cadd["Ref"] = (
    cadd["Ref"]
    .astype(str)
    .str.upper()
    .str.strip()
)

cadd["Alt"] = (
    cadd["Alt"]
    .astype(str)
    .str.upper()
    .str.strip()
)

cadd["RawScore"] = pd.to_numeric(
    cadd["RawScore"],
    errors="coerce"
)

cadd["PHRED"] = pd.to_numeric(
    cadd["PHRED"],
    errors="coerce"
)


# Remove invalid genomic rows
invalid_position = cadd["Pos"].isna().sum()

if invalid_position > 0:
    print(
        f"WARNING: Removing {invalid_position} rows "
        f"with invalid positions."
    )

    cadd = cadd.dropna(subset=["Pos"])


# Build exact variant key
cadd["Variant_Key"] = (
    cadd["Chrom"].astype(str)
    + "-"
    + cadd["Pos"].astype(str)
    + "-"
    + cadd["Ref"]
    + "-"
    + cadd["Alt"]
)


# ============================================================
# CADD DUPLICATE QC
# ============================================================

duplicate_cadd = cadd["Variant_Key"].duplicated().sum()

if duplicate_cadd > 0:

    print(
        f"WARNING: {duplicate_cadd} duplicate CADD genomic "
        f"identities detected."
    )

    cadd = (
        cadd
        .sort_values("PHRED", ascending=False)
        .drop_duplicates(
            subset="Variant_Key",
            keep="first"
        )
    )

    print("Duplicate CADD records resolved: PASS")

else:

    print("CADD genomic identity uniqueness: PASS")


# ============================================================
# MATCH CADD RESULTS TO MASTER
# ============================================================

print("\n" + "-" * 78)
print("MATCHING CADD RESULTS TO PNPLA3 VARIANTS")
print("-" * 78)


cadd_subset = cadd[
    [
        "Variant_Key",
        "RawScore",
        "PHRED"
    ]
].copy()


cadd_subset = cadd_subset.rename(
    columns={
        "RawScore": "CADD_RawScore",
        "PHRED": "CADD_PHRED"
    }
)


merged = master.merge(
    cadd_subset,
    on="Variant_Key",
    how="left"
)


# ============================================================
# CADD MATCH STATUS
# ============================================================

merged["CADD_Status"] = pd.NA

merged.loc[
    merged["CADD_PHRED"].notna(),
    "CADD_Status"
] = "EXACT_MATCH"

merged.loc[
    merged["CADD_PHRED"].isna(),
    "CADD_Status"
] = "NOT_RETURNED"


# ============================================================
# CADD INTERPRETATION
# ============================================================

def interpret_cadd(phred):

    if pd.isna(phred):
        return pd.NA

    # Common CADD interpretation framework
    if phred >= 30:
        return "Highly deleterious"

    elif phred >= 20:
        return "Potentially deleterious"

    elif phred >= 10:
        return "Moderate deleteriousness"

    else:
        return "Low predicted deleteriousness"


merged["CADD_Interpretation"] = (
    merged["CADD_PHRED"]
    .apply(interpret_cadd)
)


# ============================================================
# CADD SCORE QC
# ============================================================

print("\n" + "-" * 78)
print("CADD SCORE QC")
print("-" * 78)


available_scores = merged["CADD_PHRED"].notna().sum()

unavailable_scores = merged["CADD_PHRED"].isna().sum()


print(f"CADD scores available: {available_scores}")
print(f"CADD scores unavailable: {unavailable_scores}")


if available_scores > 0:

    print(
        f"Minimum CADD PHRED score: "
        f"{merged['CADD_PHRED'].min():.3f}"
    )

    print(
        f"Maximum CADD PHRED score: "
        f"{merged['CADD_PHRED'].max():.3f}"
    )

    if (merged["CADD_PHRED"] < 0).any():
        print("ERROR: Negative CADD PHRED scores detected.")
        sys.exit(1)

    print("CADD PHRED score range: PASS")


# ============================================================
# MATCHING QC
# ============================================================

print("\n" + "-" * 78)
print("CADD MATCHING QC")
print("-" * 78)


exact_matches = (
    merged["CADD_Status"] == "EXACT_MATCH"
).sum()


not_returned = (
    merged["CADD_Status"] == "NOT_RETURNED"
).sum()


print(f"Expected PNPLA3 variants: {len(master)}")
print(f"Exact CADD matches:       {exact_matches}")
print(f"Not returned by CADD:      {not_returned}")


# ============================================================
# CHECK WHICH VARIANT IS MISSING
# ============================================================

if not_returned > 0:

    print("\nVariants not returned by CADD:")

    missing_variants = merged.loc[
        merged["CADD_Status"] == "NOT_RETURNED",
        [
            "VariationID",
            "Chromosome",
            "Position_GRCh38",
            "Reference",
            "Alternate"
        ]
    ]

    for _, row in missing_variants.iterrows():

        print(
            f"VariationID={row['VariationID']} | "
            f"{row['Chromosome']}-"
            f"{row['Position_GRCh38']}-"
            f"{row['Reference']}-"
            f"{row['Alternate']}"
        )


# ============================================================
# INTERPRETATION DISTRIBUTION
# ============================================================

print("\n" + "-" * 78)
print("CADD COMPUTATIONAL INTERPRETATION")
print("-" * 78)


print(
    merged["CADD_Interpretation"]
    .value_counts(dropna=False)
)


# ============================================================
# CREATE CLEAN ANNOTATION TABLE
# ============================================================

annotation_columns = [
    "VariationID",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate"
]


# Include useful optional columns if present
optional_columns = [
    "Protein_Change",
    "HGVS_p",
    "rsID",
    "ClinVar_Significance"
]


for col in optional_columns:

    if col in merged.columns:
        annotation_columns.append(col)


annotation_columns.extend(
    [
        "CADD_RawScore",
        "CADD_PHRED",
        "CADD_Interpretation",
        "CADD_Status"
    ]
)


annotation_columns = [
    col for col in annotation_columns
    if col in merged.columns
]


annotation = merged[
    annotation_columns
].copy()


# ============================================================
# SAVE OUTPUTS
# ============================================================

# Remove temporary merge key before saving master
final_master = merged.drop(
    columns=["Variant_Key"],
    errors="ignore"
)


annotation.to_csv(
    OUTPUT_ANNOTATION,
    index=False
)


final_master.to_csv(
    OUTPUT_MASTER,
    index=False
)


print("\nCADD annotation table written:")
print(OUTPUT_ANNOTATION)

print("\nFinal Step 9C master written:")
print(OUTPUT_MASTER)


# ============================================================
# FINAL QUALITY CONTROL
# ============================================================

print("\n" + "=" * 78)
print("FINAL STEP 9C QUALITY CONTROL")
print("=" * 78)


print(f"\nPNPLA3 variants analyzed:       {len(final_master)}")
print(f"CADD exact matches:             {exact_matches}")
print(f"CADD unavailable/not returned:  {not_returned}")
print(
    f"Unique VariationIDs:             "
    f"{final_master['VariationID'].nunique()}"
)
print(
    f"Unique exact variant identities: "
    f"{final_master['Variant_Key'].nunique() if 'Variant_Key' in final_master.columns else len(final_master)}"
)


# Retention QC
if len(final_master) == len(master):
    print(
        f"\n{len(final_master)}/{len(master)} variants retained: PASS"
    )
else:
    print("\nERROR: Variant retention failure.")
    sys.exit(1)


if final_master["VariationID"].duplicated().any():
    print("ERROR: Duplicate VariationIDs in final master.")
    sys.exit(1)

print("VariationID uniqueness retained: PASS")


# ============================================================
# FINAL SUCCESS STATUS
# ============================================================

print("\n" + "-" * 78)

if exact_matches == len(master):

    print("FINAL STEP 9C QC: ALL TESTS PASSED")

elif exact_matches > 0:

    print("STEP 9C PARTIALLY COMPLETED")
    print(
        f"{exact_matches}/{len(master)} variants received "
        f"CADD scores."
    )
    print(
        f"{not_returned} variant(s) were not present in "
        f"the returned CADD result."
    )

else:

    print("STEP 9C NOT COMPLETED")
    print("No PNPLA3 variants matched usable CADD results.")


print("-" * 78)


# ============================================================
# SCORE DISTRIBUTION
# ============================================================

if available_scores > 0:

    print("\nCADD PHRED score distribution:")

    print(
        merged["CADD_PHRED"]
        .describe()
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("STEP 9C SUMMARY")
print("=" * 78)

print(f"\nTotal variants:          {len(master)}")
print(f"CADD exact matches:      {exact_matches}")
print(f"CADD not returned:       {not_returned}")

print("\nOutput:")
print(OUTPUT_MASTER)

print("\nClean CADD annotation:")
print(OUTPUT_ANNOTATION)

print("\n" + "=" * 78)

if exact_matches == len(master):
    print("STEP 9C SUCCESSFULLY COMPLETED")
else:
    print("STEP 9C COMPLETED WITH QC FLAGS")
    
print("=" * 78)