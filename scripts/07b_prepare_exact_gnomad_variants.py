import pandas as pd
import re
from pathlib import Path

print("=" * 70)
print("PREPARING EXACT PNPLA3 VARIANT REPRESENTATIONS FOR GNOMAD")
print("=" * 70)

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

INPUT_FILE = Path(
    r"04_VARIANT ANALYSIS\pnpla3_candidate_master.csv"
)

OUTPUT_FILE = Path(
    r"04_VARIANT ANALYSIS\annotations\pnpla3_exact_variant_mapping.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------

print("\nLoading PNPLA3 candidate variants...")

df = pd.read_csv(INPUT_FILE)

print(f"Total variants: {len(df)}")


# ------------------------------------------------------------
# Extract nucleotide change from ClinVar Name
#
# Example:
# NM_025225.3(PNPLA3):c.16C>G (p.Arg6Gly)
#
# Extract:
# cDNA_Position = 16
# Reference = C
# Alternate = G
# ------------------------------------------------------------

pattern = r":c\.(\d+)([ACGT])>([ACGT])"

extracted = df["Name"].astype(str).str.extract(pattern)

df["cDNA_Position"] = extracted[0]
df["Reference"] = extracted[1]
df["Alternate"] = extracted[2]


# ------------------------------------------------------------
# Convert position to numeric
# ------------------------------------------------------------

df["cDNA_Position"] = pd.to_numeric(
    df["cDNA_Position"],
    errors="coerce"
)


# ------------------------------------------------------------
# Standardize chromosome
# ------------------------------------------------------------

df["Chromosome"] = df["Chromosome"].astype(str).str.replace(
    ".0",
    "",
    regex=False
)


# ------------------------------------------------------------
# Rename genomic position
# ------------------------------------------------------------

df["Position_GRCh38"] = df["Start"]


# ------------------------------------------------------------
# Create exact gnomAD variant ID
#
# Format:
# chromosome-position-reference-alternate
# ------------------------------------------------------------

df["gnomAD_Variant_ID"] = (
    df["Chromosome"].astype(str)
    + "-"
    + df["Position_GRCh38"].astype(str)
    + "-"
    + df["Reference"].astype(str)
    + "-"
    + df["Alternate"].astype(str)
)


# ------------------------------------------------------------
# Mapping status
# ------------------------------------------------------------

df["Allele_Mapping_Status"] = "Mapped from ClinVar cDNA notation"

missing = (
    df["Reference"].isna()
    | df["Alternate"].isna()
    | df["Position_GRCh38"].isna()
)

df.loc[
    missing,
    "Allele_Mapping_Status"
] = "Unable to map"


# ------------------------------------------------------------
# Select useful columns
# ------------------------------------------------------------

output_columns = [

    "VariationID",
    "Name",
    "Protein_Change",
    "Candidate_Group",

    "Chromosome",
    "Position_GRCh38",

    "cDNA_Position",

    "Reference",
    "Alternate",

    "RS# (dbSNP)",

    "gnomAD_Variant_ID",

    "Allele_Mapping_Status"
]


output_df = df[output_columns].copy()


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXACT VARIANT MAPPING SUMMARY")
print("=" * 70)

print(f"\nTotal variants: {len(output_df)}")

print("\nAllele mapping status:")

print(
    output_df["Allele_Mapping_Status"]
    .value_counts(dropna=False)
)


print("\nVariants successfully assigned exact alleles:")

success = output_df[
    output_df["Allele_Mapping_Status"]
    == "Mapped from ClinVar cDNA notation"
]

print(len(success))


print("\nFirst 15 variants:")

print(
    output_df.head(15).to_string(
        index=False
    )
)


print("\nOutput saved to:")

print(OUTPUT_FILE.resolve())


print("\nDone.")