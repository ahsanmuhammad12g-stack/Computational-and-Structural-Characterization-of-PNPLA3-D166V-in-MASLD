import pandas as pd
from pathlib import Path


print("=" * 80)
print("STEP 8 — FINAL PNPLA3 GNOMAD POPULATION FREQUENCY ANNOTATION")
print("=" * 80)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

INPUT_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_gnomad_final.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_gnomad_population_frequency_FINAL.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading gnomAD annotation table...")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE, dtype=str)

print(f"Records loaded: {len(df)}")


# ============================================================
# CLEAN COLUMN VALUES
# ============================================================

df = df.fillna("")

# Convert the important numeric fields later
numeric_columns = [
    "Exome_AC",
    "Exome_AN",
    "Exome_AF",
    "Genome_AC",
    "Genome_AN",
    "Genome_AF",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")


# ============================================================
# EXACT VARIANT IDENTITY CHECK
# ============================================================

print("\nChecking exact variant identities...")

def build_variant_id(row):
    return (
        f"{row['Chromosome']}-"
        f"{row['Position_GRCh38']}-"
        f"{row['Reference']}-"
        f"{row['Alternate']}"
    )


df["Expected_Variant_ID"] = df.apply(build_variant_id, axis=1)


# Use the authoritative ClinVar-derived identity
if "gnomAD_variant_id_returned" in df.columns:
    returned = df["gnomAD_variant_id_returned"].fillna("").astype(str)

    exact_match = (
        (returned != "") &
        (returned == df["Expected_Variant_ID"])
    )
else:
    exact_match = pd.Series(False, index=df.index)


# ============================================================
# FINAL STATUS CLASSIFICATION
# ============================================================

print("\nClassifying population-frequency status...")


def classify_status(row):
    original_status = str(row.get("gnomAD_Status", "")).strip()
    error = str(row.get("gnomAD_Error", "")).strip()

    # --------------------------------------------------------
    # Exact successful gnomAD match
    # --------------------------------------------------------
    if original_status.lower() == "found":
        returned_id = str(
            row.get("gnomAD_variant_id_returned", "")
        ).strip()

        expected_id = str(
            row.get("Expected_Variant_ID", "")
        ).strip()

        if returned_id == expected_id and returned_id != "":
            return "EXACT_MATCH"

        return "IDENTITY_MISMATCH"

    # --------------------------------------------------------
    # Explicit gnomAD "Variant not found"
    # --------------------------------------------------------
    if error.lower() == "variant not found":
        return "NOT_DETECTED_IN_QUERIED_RELEASES"

    # --------------------------------------------------------
    # Anything else is genuinely unresolved
    # --------------------------------------------------------
    return "UNRESOLVED_API_OR_DATASET_STATUS"


df["Final_gnomAD_Status"] = df.apply(classify_status, axis=1)


# ============================================================
# PREVENT FALSE AF = 0
# ============================================================

# For variants that were not detected, frequencies remain NA.
not_detected = (
    df["Final_gnomAD_Status"]
    == "NOT_DETECTED_IN_QUERIED_RELEASES"
)

for col in numeric_columns:
    df.loc[not_detected, col] = pd.NA


# ============================================================
# POPULATION FREQUENCY INTERPRETATION
# ============================================================

def frequency_interpretation(row):

    status = row["Final_gnomAD_Status"]

    if status == "EXACT_MATCH":
        exome_af = row["Exome_AF"]
        genome_af = row["Genome_AF"]

        if pd.notna(exome_af) and pd.notna(genome_af):
            return "Detected in gnomAD exome and genome datasets"

        elif pd.notna(exome_af):
            return "Detected in gnomAD exome dataset; genome AF not reported"

        elif pd.notna(genome_af):
            return "Detected in gnomAD genome dataset; exome AF not reported"

        else:
            return "Exact gnomAD variant identified; AF not reported"

    elif status == "NOT_DETECTED_IN_QUERIED_RELEASES":
        return "Not detected in queried gnomAD releases"

    elif status == "IDENTITY_MISMATCH":
        return "Excluded because returned allele identity did not match"

    else:
        return "Population frequency unresolved"


df["Population_Frequency_Interpretation"] = df.apply(
    frequency_interpretation,
    axis=1
)


# ============================================================
# REMOVE INTERNAL COLUMN
# ============================================================

df.drop(
    columns=["Expected_Variant_ID"],
    inplace=True,
    errors="ignore"
)


# ============================================================
# REORDER IMPORTANT COLUMNS
# ============================================================

preferred_columns = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "gnomAD_variant_id_returned",
    "RSID",
    "Exome_AC",
    "Exome_AN",
    "Exome_AF",
    "Genome_AC",
    "Genome_AN",
    "Genome_AF",
    "Final_gnomAD_Status",
    "Population_Frequency_Interpretation",
    "gnomAD_Error",
]

existing_preferred = [
    col for col in preferred_columns
    if col in df.columns
]

remaining_columns = [
    col for col in df.columns
    if col not in existing_preferred
]

df = df[existing_preferred + remaining_columns]


# ============================================================
# SORT BY GENOMIC POSITION
# ============================================================

df["Position_GRCh38_numeric"] = pd.to_numeric(
    df["Position_GRCh38"],
    errors="coerce"
)

df = df.sort_values(
    by=["Position_GRCh38_numeric", "VariationID"]
)

df.drop(
    columns=["Position_GRCh38_numeric"],
    inplace=True
)


# ============================================================
# SAVE FINAL TABLE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL QC
# ============================================================

exact_count = (
    df["Final_gnomAD_Status"] == "EXACT_MATCH"
).sum()

not_detected_count = (
    df["Final_gnomAD_Status"]
    == "NOT_DETECTED_IN_QUERIED_RELEASES"
).sum()

identity_mismatch_count = (
    df["Final_gnomAD_Status"] == "IDENTITY_MISMATCH"
).sum()

unresolved_count = (
    df["Final_gnomAD_Status"]
    == "UNRESOLVED_API_OR_DATASET_STATUS"
).sum()


print("\n")
print("=" * 80)
print("FINAL STEP 8 QUALITY CONTROL")
print("=" * 80)

print(f"Total variants:                         {len(df)}")
print(f"Exact gnomAD matches:                   {exact_count}")
print(f"Not detected in queried releases:      {not_detected_count}")
print(f"Identity mismatches:                    {identity_mismatch_count}")
print(f"Other unresolved records:              {unresolved_count}")


# ============================================================
# CHECK AF VALUES
# ============================================================

print("\nAF QUALITY CONTROL")

false_zero_count = 0

for col in ["Exome_AF", "Genome_AF"]:

    if col not in df.columns:
        continue

    for idx, row in df.iterrows():

        if (
            row["Final_gnomAD_Status"]
            == "NOT_DETECTED_IN_QUERIED_RELEASES"
        ):
            if pd.notna(row[col]) and row[col] == 0:
                false_zero_count += 1


print(
    f"Not-detected variants incorrectly assigned AF=0: "
    f"{false_zero_count}"
)


# ============================================================
# SHOW 9 NOT-DETECTED VARIANTS
# ============================================================

print("\n")
print("=" * 80)
print("VARIANTS NOT DETECTED IN QUERIED GNOMAD RELEASES")
print("=" * 80)

display_columns = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Final_gnomAD_Status",
]

print(
    df.loc[
        df["Final_gnomAD_Status"]
        == "NOT_DETECTED_IN_QUERIED_RELEASES",
        display_columns
    ].to_string(index=False)
)


# ============================================================
# FINAL VALIDATION
# ============================================================

if len(df) != 82:
    print(
        f"\nWARNING: Expected 82 variants but found {len(df)}."
    )
else:
    print("\n82/82 variants successfully retained.")


if exact_count != 73:
    print(
        f"WARNING: Expected 73 exact gnomAD matches, "
        f"but found {exact_count}."
    )


if not_detected_count != 9:
    print(
        f"WARNING: Expected 9 not-detected variants, "
        f"but found {not_detected_count}."
    )


if (
    len(df) == 82
    and exact_count == 73
    and not_detected_count == 9
    and identity_mismatch_count == 0
    and unresolved_count == 0
    and false_zero_count == 0
):
    print("\n" + "=" * 80)
    print("STEP 8 SUCCESSFULLY FINALIZED")
    print("=" * 80)
else:
    print("\nSTEP 8 QC REQUIRES REVIEW.")


print("\nFinal output:")
print(OUTPUT_FILE)

print("\n")