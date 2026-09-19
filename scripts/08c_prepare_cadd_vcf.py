import pandas as pd
from pathlib import Path


print("=" * 78)
print("STEP 9C — PREPARE PNPLA3 GRCh38 VCF FOR CADD v1.7")
print("=" * 78)


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
    / "pnpla3_step9_alphamissense_annotated_FINAL.csv"
)

OUTPUT_VCF = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_CADD_input_GRCh38.vcf"
)


EXPECTED_VARIANTS = 82


# ============================================================
# LOAD MASTER
# ============================================================

print("\nReading validated Step 9B master...")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")


# ============================================================
# BASIC QC
# ============================================================

required = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
]

missing = [x for x in required if x not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

if len(df) != EXPECTED_VARIANTS:
    raise ValueError(
        f"Expected {EXPECTED_VARIANTS} variants, "
        f"found {len(df)}"
    )

if df["VariationID"].nunique() != EXPECTED_VARIANTS:
    raise ValueError(
        "VariationID uniqueness failed."
    )

if df["Exact_Variant_Key"].nunique() != EXPECTED_VARIANTS:
    raise ValueError(
        "Exact genomic identity uniqueness failed."
    )

print("Required columns: PASS")
print("82 variants: PASS")
print("VariationID uniqueness: PASS")
print("Exact genomic identity uniqueness: PASS")


# ============================================================
# PREPARE VCF FIELDS
# ============================================================

df["CHROM"] = (
    df["Chromosome"]
    .astype(str)
    .str.replace("chr", "", case=False, regex=False)
    .str.strip()
)

df["POS"] = pd.to_numeric(
    df["Position_GRCh38"],
    errors="coerce"
)

df["REF"] = (
    df["Reference"]
    .astype(str)
    .str.strip()
    .str.upper()
)

df["ALT"] = (
    df["Alternate"]
    .astype(str)
    .str.strip()
    .str.upper()
)


# ============================================================
# VALIDATE VARIANTS
# ============================================================

if df["POS"].isna().any():
    raise ValueError(
        "One or more positions are invalid."
    )

valid_bases = {"A", "C", "G", "T"}

for column in ["REF", "ALT"]:

    invalid = ~df[column].isin(valid_bases)

    if invalid.any():

        bad = df.loc[
            invalid,
            ["VariationID", column]
        ]

        raise ValueError(
            f"Invalid SNV alleles detected:\n{bad}"
        )


# Confirm all variants are SNVs

if not (
    (df["REF"].str.len() == 1)
    & (df["ALT"].str.len() == 1)
).all():

    raise ValueError(
        "At least one variant is not a single-nucleotide substitution."
    )

print("GRCh38 positions: PASS")
print("REF alleles: PASS")
print("ALT alleles: PASS")
print("All 82 variants are SNVs: PASS")


# ============================================================
# CHECK DUPLICATE GENOMIC VARIANTS
# ============================================================

vcf_identity = (
    df["CHROM"].astype(str)
    + ":"
    + df["POS"].astype(str)
    + ":"
    + df["REF"]
    + ":"
    + df["ALT"]
)

if vcf_identity.nunique() != EXPECTED_VARIANTS:
    raise ValueError(
        "Duplicate genomic variants detected."
    )

print("VCF genomic identity uniqueness: PASS")


# ============================================================
# WRITE VCF
# ============================================================

print("\nWriting CADD input VCF...")

with open(OUTPUT_VCF, "w", encoding="utf-8") as f:

    f.write("##fileformat=VCFv4.2\n")
    f.write("##reference=GRCh38\n")
    f.write(
        "##source=PNPLA3_MASLD_Variant_Project\n"
    )
    f.write(
        '##INFO=<ID=VID,Number=1,Type=String,'
        'Description="ClinVar VariationID">\n'
    )
    f.write(
        '##INFO=<ID=PROT,Number=1,Type=String,'
        'Description="PNPLA3 protein change">\n'
    )

    f.write(
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
    )

    for _, row in df.iterrows():

        info = (
            f"VID={row['VariationID']};"
            f"PROT={row['Protein_Change']}"
        )

        f.write(
            f"{row['CHROM']}\t"
            f"{int(row['POS'])}\t"
            f"PNPLA3_{row['VariationID']}\t"
            f"{row['REF']}\t"
            f"{row['ALT']}\t"
            f".\t"
            f"PASS\t"
            f"{info}\n"
        )


# ============================================================
# FINAL CHECK
# ============================================================

with open(OUTPUT_VCF, "r", encoding="utf-8") as f:

    variant_lines = [
        line
        for line in f
        if line.strip()
        and not line.startswith("#")
    ]

if len(variant_lines) != EXPECTED_VARIANTS:
    raise ValueError(
        "VCF does not contain exactly 82 variants."
    )


print("\n" + "=" * 78)
print("VCF PREPARATION COMPLETE")
print("=" * 78)

print(f"\nVariants written: {len(variant_lines)}")

print("\nOutput:")
print(OUTPUT_VCF)

print("\nQC:")
print("82/82 variants written: PASS")
print("GRCh38 reference: PASS")
print("All variants are SNVs: PASS")
print("Unique genomic identities: PASS")

print("\nNext step:")
print("Upload this VCF to the CADD v1.7 scoring service.")

print("\n" + "=" * 78)