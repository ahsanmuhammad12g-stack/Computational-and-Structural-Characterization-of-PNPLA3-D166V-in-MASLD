import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# STEP 9D — INTEGRATED COMPUTATIONAL VARIANT PRIORITIZATION
# PNPLA3 MASLD PROJECT
# ============================================================

print("=" * 78)
print("STEP 9D — INTEGRATED COMPUTATIONAL VARIANT PRIORITIZATION")
print("=" * 78)


# ============================================================
# FILE PATHS
# ============================================================

PROJECT_ROOT = Path(r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3")

INPUT_FILE = (
    PROJECT_ROOT
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_step9_cadd_annotated_FINAL.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "04_VARIANT ANALYSIS"
    / "annotations"
)

OUTPUT_MASTER = (
    OUTPUT_DIR
    / "pnpla3_step9_integrated_prioritized_FINAL.csv"
)

OUTPUT_PRIORITY_TABLE = (
    OUTPUT_DIR
    / "pnpla3_integrated_variant_prioritization.csv"
)


# ============================================================
# EXPECTED INPUT
# ============================================================

REQUIRED_COLUMNS = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
    "Candidate_Group",
    "Exome_AF",
    "Genome_AF",
    "Population_Frequency_Interpretation",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Interpretation",
    "CADD_PHRED",
    "CADD_Interpretation",
]


# ============================================================
# READ INPUT FILE
# ============================================================

print("\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: Input file not found:\n{INPUT_FILE}"
    )

print("\nReading validated Step 9C master...")
df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")


# ============================================================
# BASIC INPUT QC
# ============================================================

print("\n" + "-" * 78)
print("BASIC INPUT QC")
print("-" * 78)

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nERROR: Required columns missing:\n"
        + "\n".join(f"  - {col}" for col in missing_columns)
    )

print("Required columns: PASS")


EXPECTED_VARIANTS = 82

if len(df) != EXPECTED_VARIANTS:
    raise ValueError(
        f"\nERROR: Expected {EXPECTED_VARIANTS} variants, "
        f"but found {len(df)}"
    )

print(f"Expected variant count ({EXPECTED_VARIANTS}): PASS")


if df["VariationID"].duplicated().any():
    duplicates = df.loc[
        df["VariationID"].duplicated(),
        "VariationID"
    ].tolist()

    raise ValueError(
        f"\nERROR: Duplicate VariationIDs detected:\n{duplicates}"
    )

print("VariationID uniqueness: PASS")


if df["Exact_Variant_Key"].duplicated().any():
    duplicates = df.loc[
        df["Exact_Variant_Key"].duplicated(),
        "Exact_Variant_Key"
    ].tolist()

    raise ValueError(
        f"\nERROR: Duplicate genomic identities detected:\n{duplicates}"
    )

print("Exact genomic variant identity uniqueness: PASS")


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("\n" + "-" * 78)
print("PREPARING NUMERIC ANNOTATIONS")
print("-" * 78)

numeric_columns = [
    "Exome_AF",
    "Genome_AF",
    "AlphaMissense",
    "CADD_PHRED",
]

# Optional computational columns
optional_numeric_columns = [
    "REVEL_Score",
    "Conservation_Score",
]

for col in numeric_columns + optional_numeric_columns:

    if col in df.columns:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

print("Numeric annotation conversion: PASS")


# ============================================================
# POPULATION FREQUENCY PROCESSING
# ============================================================

print("\n" + "-" * 78)
print("POPULATION FREQUENCY PRIORITIZATION")
print("-" * 78)


def get_max_population_af(row):

    values = []

    for column in ["Exome_AF", "Genome_AF"]:

        value = row.get(column)

        if pd.notna(value):
            values.append(float(value))

    if len(values) == 0:
        return np.nan

    return max(values)


df["Maximum_gnomAD_AF"] = df.apply(
    get_max_population_af,
    axis=1
)


def classify_population_rarity(af):

    if pd.isna(af):
        return "Population frequency unavailable"

    elif af == 0:
        return "Not observed"

    elif af < 0.00001:
        return "Ultra-rare"

    elif af < 0.0001:
        return "Very rare"

    elif af < 0.001:
        return "Rare"

    elif af < 0.01:
        return "Low frequency"

    else:
        return "Common"


df["Population_Rarity_Category"] = (
    df["Maximum_gnomAD_AF"]
    .apply(classify_population_rarity)
)


# Population prioritization score
def population_score(af):

    if pd.isna(af):
        return 0

    elif af == 0:
        return 3

    elif af < 0.00001:
        return 3

    elif af < 0.0001:
        return 2

    elif af < 0.001:
        return 1

    else:
        return 0


df["Population_Priority_Score"] = (
    df["Maximum_gnomAD_AF"]
    .apply(population_score)
)


print("Population frequency processing: PASS")

print("\nPopulation rarity distribution:")
print(
    df["Population_Rarity_Category"]
    .value_counts(dropna=False)
)


# ============================================================
# ALPHAMISSENSE PRIORITIZATION
# ============================================================

print("\n" + "-" * 78)
print("ALPHAMISSENSE PRIORITIZATION")
print("-" * 78)


def alphamissense_score_priority(score):

    if pd.isna(score):
        return 0

    elif score >= 0.95:
        return 3

    elif score >= 0.564:
        return 2

    elif score >= 0.34:
        return 1

    else:
        return 0


df["AlphaMissense_Priority_Score"] = (
    df["AlphaMissense"]
    .apply(alphamissense_score_priority)
)


print("AlphaMissense score integration: PASS")

print("\nAlphaMissense classification distribution:")
print(
    df["AlphaMissense_Interpretation"]
    .value_counts(dropna=False)
)


# ============================================================
# CADD PRIORITIZATION
# ============================================================

print("\n" + "-" * 78)
print("CADD PRIORITIZATION")
print("-" * 78)


def cadd_priority_score(phred):

    if pd.isna(phred):
        return 0

    elif phred >= 25:
        return 3

    elif phred >= 20:
        return 2

    elif phred >= 10:
        return 1

    else:
        return 0


df["CADD_Priority_Score"] = (
    df["CADD_PHRED"]
    .apply(cadd_priority_score)
)


print("CADD score integration: PASS")

print("\nCADD interpretation distribution:")
print(
    df["CADD_Interpretation"]
    .value_counts(dropna=False)
)


# ============================================================
# REVEL PRIORITIZATION
# ============================================================

print("\n" + "-" * 78)
print("REVEL PRIORITIZATION")
print("-" * 78)


def revel_priority_score(score):

    if pd.isna(score):
        return 0

    elif score >= 0.75:
        return 3

    elif score >= 0.50:
        return 2

    elif score >= 0.25:
        return 1

    else:
        return 0


if "REVEL_Score" in df.columns:

    df["REVEL_Priority_Score"] = (
        df["REVEL_Score"]
        .apply(revel_priority_score)
    )

else:

    df["REVEL_Priority_Score"] = 0


revel_available = df["REVEL_Score"].notna().sum()

print(f"REVEL scores available: {revel_available}")
print("REVEL integration: PASS")


# ============================================================
# CONSERVATION PRIORITIZATION
# ============================================================

print("\n" + "-" * 78)
print("CONSERVATION PRIORITIZATION")
print("-" * 78)


def conservation_priority_score(annotation):

    if pd.isna(annotation):
        return 0

    annotation = str(annotation).lower()

    if "high" in annotation or "highly conserved" in annotation:
        return 2

    elif "moderate" in annotation:
        return 1

    else:
        return 0


df["Conservation_Priority_Score"] = (
    df["Conservation_Annotation"]
    .apply(conservation_priority_score)
)


conservation_available = (
    df["Conservation_Annotation"]
    .notna()
    .sum()
)

print(f"Conservation annotations available: {conservation_available}")
print("Conservation integration: PASS")


# ============================================================
# CLINICAL / CANDIDATE GROUP PRIORITIZATION
# ============================================================

print("\n" + "-" * 78)
print("CLINICAL CANDIDATE GROUP PRIORITIZATION")
print("-" * 78)


def clinical_priority_score(candidate_group):

    if pd.isna(candidate_group):
        return 0

    text = str(candidate_group).lower()

    # Primary study focus:
    # ClinVar VUS and conflicting interpretation variants

    if (
        "conflict" in text
        or "conflicting" in text
    ):
        return 3

    elif (
        "vus" in text
        or "uncertain" in text
        or "unknown significance" in text
    ):
        return 3

    else:
        return 0


df["Clinical_Priority_Score"] = (
    df["Candidate_Group"]
    .apply(clinical_priority_score)
)


print("Candidate group integration: PASS")

print("\nCandidate group distribution:")
print(
    df["Candidate_Group"]
    .value_counts(dropna=False)
)


# ============================================================
# COMPUTATIONAL EVIDENCE AGREEMENT
# ============================================================

print("\n" + "-" * 78)
print("MULTI-PREDICTOR EVIDENCE AGREEMENT")
print("-" * 78)


def computational_agreement(row):

    positive_predictors = 0

    # AlphaMissense
    if row["AlphaMissense_Priority_Score"] >= 2:
        positive_predictors += 1

    # CADD
    if row["CADD_Priority_Score"] >= 2:
        positive_predictors += 1

    # REVEL
    if row["REVEL_Priority_Score"] >= 2:
        positive_predictors += 1

    return positive_predictors


df["Computational_Evidence_Count"] = (
    df.apply(
        computational_agreement,
        axis=1
    )
)


def agreement_category(count):

    if count >= 3:
        return "Strong multi-predictor agreement"

    elif count == 2:
        return "Moderate predictor agreement"

    elif count == 1:
        return "Limited predictor agreement"

    else:
        return "No strong computational agreement"


df["Computational_Evidence_Agreement"] = (
    df["Computational_Evidence_Count"]
    .apply(agreement_category)
)


# ============================================================
# AGREEMENT BONUS
# ============================================================

def agreement_bonus(count):

    if count >= 3:
        return 3

    elif count == 2:
        return 2

    elif count == 1:
        return 1

    else:
        return 0


df["Computational_Agreement_Bonus"] = (
    df["Computational_Evidence_Count"]
    .apply(agreement_bonus)
)


print("Multi-predictor evidence integration: PASS")

print("\nComputational evidence agreement:")
print(
    df["Computational_Evidence_Agreement"]
    .value_counts(dropna=False)
)


# ============================================================
# INTEGRATED PRIORITIZATION SCORE
# ============================================================

print("\n" + "=" * 78)
print("CALCULATING INTEGRATED VARIANT PRIORITY SCORE")
print("=" * 78)


PRIORITY_SCORE_COLUMNS = [
    "Clinical_Priority_Score",
    "Population_Priority_Score",
    "AlphaMissense_Priority_Score",
    "CADD_Priority_Score",
    "REVEL_Priority_Score",
    "Conservation_Priority_Score",
    "Computational_Agreement_Bonus",
]


df["Integrated_Priority_Score"] = (
    df[PRIORITY_SCORE_COLUMNS]
    .sum(axis=1)
)


# ============================================================
# PRIORITY CLASSIFICATION
# ============================================================

def assign_priority_category(row):

    score = row["Integrated_Priority_Score"]

    clinical_focus = row["Clinical_Priority_Score"] >= 3

    computational_support = (
        row["Computational_Evidence_Count"] >= 2
    )

    # Highest priority:
    # Study-relevant clinical category + strong computational support

    if (
        score >= 10
        and clinical_focus
        and computational_support
    ):
        return "High priority"

    elif score >= 7:
        return "Moderate priority"

    else:
        return "Low priority"


df["Integrated_Priority_Category"] = (
    df.apply(
        assign_priority_category,
        axis=1
    )
)


# ============================================================
# STRUCTURAL ANALYSIS CANDIDATE FLAG
# ============================================================

def structural_candidate_flag(row):

    if (
        row["Integrated_Priority_Category"] == "High priority"
        and row["Computational_Evidence_Count"] >= 2
    ):
        return "YES"

    return "NO"


df["Structural_Analysis_Candidate"] = (
    df.apply(
        structural_candidate_flag,
        axis=1
    )
)


# ============================================================
# PRIORITY RANK
# ============================================================

df = df.sort_values(
    by=[
        "Integrated_Priority_Score",
        "Computational_Evidence_Count",
        "AlphaMissense",
        "CADD_PHRED",
    ],
    ascending=[
        False,
        False,
        False,
        False,
    ]
).reset_index(drop=True)


df["Priority_Rank"] = (
    np.arange(1, len(df) + 1)
)


# ============================================================
# PRIORITIZATION QC
# ============================================================

print("\n" + "-" * 78)
print("INTEGRATED PRIORITIZATION QC")
print("-" * 78)


print(
    f"\nVariants analyzed: {len(df)}"
)

print(
    f"Unique VariationIDs: "
    f"{df['VariationID'].nunique()}"
)

print(
    f"Unique genomic identities: "
    f"{df['Exact_Variant_Key'].nunique()}"
)


if len(df) != EXPECTED_VARIANTS:
    raise ValueError(
        "\nERROR: Variant retention failure."
    )

print("82/82 variants retained: PASS")


if df["VariationID"].duplicated().any():
    raise ValueError(
        "\nERROR: Duplicate VariationIDs introduced."
    )

print("VariationID uniqueness retained: PASS")


if df["Exact_Variant_Key"].duplicated().any():
    raise ValueError(
        "\nERROR: Duplicate genomic identities introduced."
    )

print("Genomic identity uniqueness retained: PASS")


if df["Integrated_Priority_Score"].isna().any():
    raise ValueError(
        "\nERROR: Missing integrated priority scores detected."
    )

print("Integrated priority scores populated: PASS")


# ============================================================
# PRIORITY DISTRIBUTION
# ============================================================

print("\n" + "-" * 78)
print("FINAL PRIORITY DISTRIBUTION")
print("-" * 78)

print(
    df["Integrated_Priority_Category"]
    .value_counts()
)


# ============================================================
# STRUCTURAL CANDIDATES
# ============================================================

print("\n" + "-" * 78)
print("STRUCTURAL ANALYSIS CANDIDATES")
print("-" * 78)

structural_candidates = df[
    df["Structural_Analysis_Candidate"] == "YES"
].copy()


print(
    f"Candidates identified: "
    f"{len(structural_candidates)}"
)


if len(structural_candidates) > 0:

    display_columns = [
        "Priority_Rank",
        "VariationID",
        "Protein_Change",
        "Candidate_Group",
        "AlphaMissense",
        "AlphaMissense_Interpretation",
        "CADD_PHRED",
        "CADD_Interpretation",
        "REVEL_Score",
        "Population_Rarity_Category",
        "Integrated_Priority_Score",
        "Integrated_Priority_Category",
    ]

    available_display_columns = [
        col for col in display_columns
        if col in structural_candidates.columns
    ]

    print(
        structural_candidates[
            available_display_columns
        ].to_string(index=False)
    )

else:
    print(
        "No variants met the structural candidate threshold."
    )


# ============================================================
# CREATE CLEAN PRIORITIZATION TABLE
# ============================================================

priority_columns = [
    "Priority_Rank",
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
    "Candidate_Group",
    "Maximum_gnomAD_AF",
    "Population_Rarity_Category",
    "Population_Priority_Score",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "AlphaMissense_Interpretation",
    "AlphaMissense_Priority_Score",
    "CADD_RawScore",
    "CADD_PHRED",
    "CADD_Interpretation",
    "CADD_Priority_Score",
    "REVEL_Score",
    "REVEL_Priority_Score",
    "Conservation_Score",
    "Conservation_Annotation",
    "Conservation_Priority_Score",
    "Computational_Evidence_Count",
    "Computational_Evidence_Agreement",
    "Computational_Agreement_Bonus",
    "Clinical_Priority_Score",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
    "Structural_Analysis_Candidate",
]


available_priority_columns = [
    col for col in priority_columns
    if col in df.columns
]


priority_table = (
    df[
        available_priority_columns
    ]
    .copy()
)


# ============================================================
# WRITE OUTPUT FILES
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUTPUT_MASTER,
    index=False
)


priority_table.to_csv(
    OUTPUT_PRIORITY_TABLE,
    index=False
)


# ============================================================
# FINAL QUALITY CONTROL
# ============================================================

print("\n" + "=" * 78)
print("FINAL STEP 9D QUALITY CONTROL")
print("=" * 78)

print(
    f"\nPNPLA3 variants analyzed: {len(df)}"
)

print(
    f"High-priority variants: "
    f"{(df['Integrated_Priority_Category'] == 'High priority').sum()}"
)

print(
    f"Moderate-priority variants: "
    f"{(df['Integrated_Priority_Category'] == 'Moderate priority').sum()}"
)

print(
    f"Low-priority variants: "
    f"{(df['Integrated_Priority_Category'] == 'Low priority').sum()}"
)

print(
    f"Structural analysis candidates: "
    f"{(df['Structural_Analysis_Candidate'] == 'YES').sum()}"
)


print("\nIntegrated priority score distribution:")

print(
    df["Integrated_Priority_Score"]
    .describe()
)


# ============================================================
# FINAL SUCCESS
# ============================================================

print("\n" + "=" * 78)
print("STEP 9D SUCCESSFULLY COMPLETED")
print("=" * 78)

print("\nFinal integrated prioritized master:")
print(OUTPUT_MASTER)

print("\nClean prioritization table:")
print(OUTPUT_PRIORITY_TABLE)

print("\nNext scientific stage:")
print(
    "Review high-priority variants and select final PNPLA3 mutant "
    "candidate(s) for structural analysis."
)

print("\n" + "=" * 78)