import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# STEP 9E — BIOLOGICAL AND STRUCTURAL CONTEXTUALIZATION OF HIGH-PRIORITY
#           PNPLA3 VARIANTS
# =============================================================================

print("=" * 78)
print("STEP 9E — BIOLOGICAL AND STRUCTURAL CONTEXTUALIZATION")
print("OF HIGH-PRIORITY PNPLA3 VARIANTS")
print("=" * 78)


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

ANNOTATION_DIR = PROJECT_ROOT / "04_VARIANT ANALYSIS" / "annotations"

INPUT_FILE = (
    ANNOTATION_DIR /
    "pnpla3_step9_integrated_prioritized_FINAL.csv"
)

OUTPUT_FILE = (
    ANNOTATION_DIR /
    "pnpla3_high_priority_variant_contextualization.csv"
)

FINAL_MASTER = (
    ANNOTATION_DIR /
    "pnpla3_step9e_contextualized_FINAL.csv"
)


# =============================================================================
# CONFIGURATION
# =============================================================================

EXPECTED_TOTAL_VARIANTS = 82
EXPECTED_HIGH_PRIORITY = 9

PNPLA3_PROTEIN_LENGTH = 481

# PNPLA3 catalytic patatin domain.
# The patatin-like phospholipase domain is approximately located in this region.
PATATIN_DOMAIN_START = 10
PATATIN_DOMAIN_END = 300

# Catalytic residues commonly associated with PNPLA3 patatin-like phospholipase
# activity.
CATALYTIC_SERINE_POSITION = 47
CATALYTIC_ASPARTATE_POSITION = 166


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fail(message):
    """Print error and terminate execution."""
    print("\nERROR:")
    print(message)
    raise SystemExit(1)


def extract_protein_position(protein_change):
    """
    Extract amino-acid position from HGVS protein notation.

    Examples:
        p.Asp166Val -> 166
        p.Gly49Trp  -> 49
    """

    if pd.isna(protein_change):
        return np.nan

    protein_change = str(protein_change)

    digits = "".join(
        character
        for character in protein_change
        if character.isdigit()
    )

    if digits == "":
        return np.nan

    return int(digits)


def extract_reference_amino_acid(protein_change):
    """
    Extract three-letter reference amino acid.

    Example:
        p.Asp166Val -> Asp
    """

    if pd.isna(protein_change):
        return pd.NA

    protein_change = str(protein_change)

    if protein_change.startswith("p."):
        protein_change = protein_change[2:]

    return protein_change[:3]


def extract_alternate_amino_acid(protein_change):
    """
    Extract three-letter alternate amino acid.

    Example:
        p.Asp166Val -> Val
    """

    if pd.isna(protein_change):
        return pd.NA

    protein_change = str(protein_change)

    if protein_change.startswith("p."):
        protein_change = protein_change[2:]

    digits = "".join(
        character
        for character in protein_change
        if character.isdigit()
    )

    if digits == "":
        return pd.NA

    position_index = protein_change.find(digits)

    return protein_change[
        position_index + len(digits):
    ]


# =============================================================================
# AMINO ACID PROPERTY DATABASE
# =============================================================================

AA_PROPERTIES = {

    "Ala": {
        "name": "Alanine",
        "class": "Nonpolar",
        "charge": "Neutral",
        "size": "Small"
    },

    "Arg": {
        "name": "Arginine",
        "class": "Basic",
        "charge": "Positive",
        "size": "Large"
    },

    "Asn": {
        "name": "Asparagine",
        "class": "Polar",
        "charge": "Neutral",
        "size": "Medium"
    },

    "Asp": {
        "name": "Aspartate",
        "class": "Acidic",
        "charge": "Negative",
        "size": "Medium"
    },

    "Cys": {
        "name": "Cysteine",
        "class": "Polar",
        "charge": "Neutral",
        "size": "Small"
    },

    "Gln": {
        "name": "Glutamine",
        "class": "Polar",
        "charge": "Neutral",
        "size": "Medium"
    },

    "Glu": {
        "name": "Glutamate",
        "class": "Acidic",
        "charge": "Negative",
        "size": "Medium"
    },

    "Gly": {
        "name": "Glycine",
        "class": "Nonpolar",
        "charge": "Neutral",
        "size": "Small"
    },

    "His": {
        "name": "Histidine",
        "class": "Basic",
        "charge": "Positive",
        "size": "Medium"
    },

    "Ile": {
        "name": "Isoleucine",
        "class": "Nonpolar",
        "charge": "Neutral",
        "size": "Large"
    },

    "Leu": {
        "name": "Leucine",
        "class": "Nonpolar",
        "charge": "Neutral",
        "size": "Large"
    },

    "Lys": {
        "name": "Lysine",
        "class": "Basic",
        "charge": "Positive",
        "size": "Large"
    },

    "Met": {
        "name": "Methionine",
        "class": "Nonpolar",
        "charge": "Neutral",
        "size": "Large"
    },

    "Phe": {
        "name": "Phenylalanine",
        "class": "Aromatic",
        "charge": "Neutral",
        "size": "Large"
    },

    "Pro": {
        "name": "Proline",
        "class": "Nonpolar",
        "charge": "Neutral",
        "size": "Medium"
    },

    "Ser": {
        "name": "Serine",
        "class": "Polar",
        "charge": "Neutral",
        "size": "Small"
    },

    "Thr": {
        "name": "Threonine",
        "class": "Polar",
        "charge": "Neutral",
        "size": "Medium"
    },

    "Trp": {
        "name": "Tryptophan",
        "class": "Aromatic",
        "charge": "Neutral",
        "size": "Large"
    },

    "Tyr": {
        "name": "Tyrosine",
        "class": "Aromatic",
        "charge": "Neutral",
        "size": "Large"
    },

    "Val": {
        "name": "Valine",
        "class": "Nonpolar",
        "charge": "Neutral",
        "size": "Medium"
    }
}


# =============================================================================
# STRUCTURAL CONTEXT FUNCTIONS
# =============================================================================

def assign_protein_region(position):

    if pd.isna(position):
        return "Unknown"

    position = int(position)

    if position <= 9:
        return "N-terminal region"

    elif PATATIN_DOMAIN_START <= position <= PATATIN_DOMAIN_END:
        return "Patatin-like phospholipase domain"

    elif position <= PNPLA3_PROTEIN_LENGTH:
        return "C-terminal region"

    else:
        return "Outside annotated protein length"


def assign_catalytic_context(position):

    if pd.isna(position):
        return "Unknown"

    position = int(position)

    if position == CATALYTIC_SERINE_POSITION:
        return "Catalytic residue"

    if position == CATALYTIC_ASPARTATE_POSITION:
        return "Catalytic residue"

    distance_ser = abs(
        position - CATALYTIC_SERINE_POSITION
    )

    distance_asp = abs(
        position - CATALYTIC_ASPARTATE_POSITION
    )

    nearest_distance = min(
        distance_ser,
        distance_asp
    )

    if nearest_distance <= 5:
        return "Very close to catalytic residue"

    elif nearest_distance <= 20:
        return "Near catalytic residue"

    elif nearest_distance <= 50:
        return "Within catalytic domain context"

    else:
        return "Not proximal to catalytic residues"


def calculate_nearest_catalytic_residue(position):

    if pd.isna(position):
        return pd.NA

    position = int(position)

    distance_ser = abs(
        position - CATALYTIC_SERINE_POSITION
    )

    distance_asp = abs(
        position - CATALYTIC_ASPARTATE_POSITION
    )

    if distance_ser < distance_asp:

        return (
            f"Ser47 "
            f"(distance={distance_ser})"
        )

    elif distance_asp < distance_ser:

        return (
            f"Asp166 "
            f"(distance={distance_asp})"
        )

    else:

        return (
            f"Ser47/Asp166 "
            f"(equal distance={distance_ser})"
        )


def calculate_property_change(ref_aa, alt_aa):

    if (
        ref_aa not in AA_PROPERTIES
        or alt_aa not in AA_PROPERTIES
    ):
        return "Unknown"

    ref = AA_PROPERTIES[ref_aa]
    alt = AA_PROPERTIES[alt_aa]

    changes = []

    if ref["class"] != alt["class"]:
        changes.append(
            "amino-acid class change"
        )

    if ref["charge"] != alt["charge"]:
        changes.append(
            "charge change"
        )

    if ref["size"] != alt["size"]:
        changes.append(
            "size change"
        )

    if len(changes) == 0:
        return "Conservative physicochemical substitution"

    return "; ".join(changes)


def estimate_substitution_impact(ref_aa, alt_aa):

    if (
        ref_aa not in AA_PROPERTIES
        or alt_aa not in AA_PROPERTIES
    ):
        return "Unknown"

    ref = AA_PROPERTIES[ref_aa]
    alt = AA_PROPERTIES[alt_aa]

    impact_score = 0

    if ref["class"] != alt["class"]:
        impact_score += 1

    if ref["charge"] != alt["charge"]:
        impact_score += 2

    if ref["size"] != alt["size"]:
        impact_score += 1

    # Special structural residues.
    special_residues = [
        "Gly",
        "Pro",
        "Cys"
    ]

    if (
        ref_aa in special_residues
        or alt_aa in special_residues
    ):
        impact_score += 1

    if impact_score >= 4:
        return "High physicochemical disruption"

    elif impact_score >= 2:
        return "Moderate physicochemical disruption"

    else:
        return "Low physicochemical disruption"


# =============================================================================
# INPUT FILE CHECK
# =============================================================================

print("\nChecking input files...")

if not INPUT_FILE.exists():
    fail(
        f"Integrated prioritization file not found:\n"
        f"{INPUT_FILE}"
    )

print("Integrated Step 9D master: PASS")


# =============================================================================
# READ INPUT
# =============================================================================

print("\nReading validated Step 9D master...")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")


# =============================================================================
# BASIC INPUT QC
# =============================================================================

print("\n" + "-" * 78)
print("BASIC INPUT QC")
print("-" * 78)

required_columns = [

    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
    "Candidate_Group",
    "Population_Rarity_Category",
    "AlphaMissense",
    "AlphaMissense_Interpretation",
    "CADD_PHRED",
    "CADD_Interpretation",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
    "Priority_Rank"
]

missing_columns = [

    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    fail(
        "Required columns missing:\n"
        + "\n".join(
            f"  - {column}"
            for column in missing_columns
        )
    )

print("Required columns: PASS")


if len(df) != EXPECTED_TOTAL_VARIANTS:

    fail(
        f"Expected {EXPECTED_TOTAL_VARIANTS} variants "
        f"but found {len(df)}"
    )

print(f"Expected variant count ({EXPECTED_TOTAL_VARIANTS}): PASS")


if df["VariationID"].duplicated().any():

    fail(
        "Duplicate VariationIDs detected."
    )

print("VariationID uniqueness: PASS")


if df["Exact_Variant_Key"].duplicated().any():

    fail(
        "Duplicate genomic identities detected."
    )

print("Exact genomic identity uniqueness: PASS")


# =============================================================================
# EXTRACT HIGH-PRIORITY VARIANTS
# =============================================================================

print("\n" + "-" * 78)
print("EXTRACTING HIGH-PRIORITY VARIANTS")
print("-" * 78)

high_priority = df[
    df["Integrated_Priority_Category"]
    == "High priority"
].copy()


if len(high_priority) != EXPECTED_HIGH_PRIORITY:

    fail(
        f"Expected {EXPECTED_HIGH_PRIORITY} high-priority variants "
        f"but found {len(high_priority)}"
    )


high_priority = high_priority.sort_values(
    by="Priority_Rank",
    ascending=True
).reset_index(drop=True)


print(
    f"High-priority variants identified: "
    f"{len(high_priority)}"
)


# =============================================================================
# PROTEIN POSITION PROCESSING
# =============================================================================

print("\n" + "-" * 78)
print("PROTEIN STRUCTURAL CONTEXT")
print("-" * 78)


high_priority["Protein_Position"] = (
    high_priority["Protein_Change"]
    .apply(extract_protein_position)
)


high_priority["Reference_Amino_Acid"] = (
    high_priority["Protein_Change"]
    .apply(extract_reference_amino_acid)
)


high_priority["Alternate_Amino_Acid"] = (
    high_priority["Protein_Change"]
    .apply(extract_alternate_amino_acid)
)


high_priority["Protein_Region"] = (
    high_priority["Protein_Position"]
    .apply(assign_protein_region)
)


high_priority["Catalytic_Context"] = (
    high_priority["Protein_Position"]
    .apply(assign_catalytic_context)
)


high_priority["Nearest_Catalytic_Residue"] = (
    high_priority["Protein_Position"]
    .apply(calculate_nearest_catalytic_residue)
)


# =============================================================================
# AMINO ACID PHYSICOCHEMICAL ANALYSIS
# =============================================================================

print("\n" + "-" * 78)
print("AMINO-ACID PHYSICOCHEMICAL CONTEXT")
print("-" * 78)


high_priority["Reference_AA_Class"] = (
    high_priority["Reference_Amino_Acid"]
    .map(
        lambda aa:
        AA_PROPERTIES.get(aa, {}).get(
            "class",
            pd.NA
        )
    )
)


high_priority["Alternate_AA_Class"] = (
    high_priority["Alternate_Amino_Acid"]
    .map(
        lambda aa:
        AA_PROPERTIES.get(aa, {}).get(
            "class",
            pd.NA
        )
    )
)


high_priority["Reference_AA_Charge"] = (
    high_priority["Reference_Amino_Acid"]
    .map(
        lambda aa:
        AA_PROPERTIES.get(aa, {}).get(
            "charge",
            pd.NA
        )
    )
)


high_priority["Alternate_AA_Charge"] = (
    high_priority["Alternate_Amino_Acid"]
    .map(
        lambda aa:
        AA_PROPERTIES.get(aa, {}).get(
            "charge",
            pd.NA
        )
    )
)


high_priority["Physicochemical_Change"] = (
    high_priority.apply(
        lambda row:
        calculate_property_change(
            row["Reference_Amino_Acid"],
            row["Alternate_Amino_Acid"]
        ),
        axis=1
    )
)


high_priority["Physicochemical_Impact"] = (
    high_priority.apply(
        lambda row:
        estimate_substitution_impact(
            row["Reference_Amino_Acid"],
            row["Alternate_Amino_Acid"]
        ),
        axis=1
    )
)


print("Protein position extraction: PASS")
print("Protein region assignment: PASS")
print("Catalytic context assignment: PASS")
print("Physicochemical characterization: PASS")


# =============================================================================
# STRUCTURAL RELEVANCE SCORING
# =============================================================================

print("\n" + "-" * 78)
print("STRUCTURAL RELEVANCE SCORING")
print("-" * 78)


def calculate_structural_relevance(row):

    score = 0

    # Catalytic context.
    catalytic_context = row["Catalytic_Context"]

    if catalytic_context == "Catalytic residue":
        score += 4

    elif catalytic_context == "Very close to catalytic residue":
        score += 3

    elif catalytic_context == "Near catalytic residue":
        score += 2

    elif catalytic_context == "Within catalytic domain context":
        score += 1


    # Protein domain.
    if (
        row["Protein_Region"]
        == "Patatin-like phospholipase domain"
    ):
        score += 2


    # Physicochemical disruption.
    impact = row["Physicochemical_Impact"]

    if impact == "High physicochemical disruption":
        score += 3

    elif impact == "Moderate physicochemical disruption":
        score += 2

    elif impact == "Low physicochemical disruption":
        score += 1


    # AlphaMissense.
    alpha_score = row["AlphaMissense"]

    if pd.notna(alpha_score):

        if alpha_score >= 0.90:
            score += 3

        elif alpha_score >= 0.70:
            score += 2

        elif alpha_score >= 0.50:
            score += 1


    # CADD.
    cadd_score = row["CADD_PHRED"]

    if pd.notna(cadd_score):

        if cadd_score >= 25:
            score += 3

        elif cadd_score >= 20:
            score += 2

        elif cadd_score >= 15:
            score += 1


    # Population rarity.
    rarity = row["Population_Rarity_Category"]

    if rarity == "Not observed":
        score += 3

    elif rarity == "Ultra-rare":
        score += 3

    elif rarity == "Very rare":
        score += 2

    elif rarity == "Rare":
        score += 1


    return score


high_priority["Structural_Relevance_Score"] = (
    high_priority.apply(
        calculate_structural_relevance,
        axis=1
    )
)


def assign_structural_relevance_category(score):

    if score >= 13:
        return "Very high structural relevance"

    elif score >= 10:
        return "High structural relevance"

    elif score >= 7:
        return "Moderate structural relevance"

    else:
        return "Lower structural relevance"


high_priority["Structural_Relevance_Category"] = (
    high_priority[
        "Structural_Relevance_Score"
    ].apply(
        assign_structural_relevance_category
    )
)


# =============================================================================
# FINAL CONTEXTUAL PRIORITY RANKING
# =============================================================================

print("\n" + "-" * 78)
print("FINAL CONTEXTUAL PRIORITIZATION")
print("-" * 78)


high_priority = high_priority.sort_values(

    by=[
        "Structural_Relevance_Score",
        "Integrated_Priority_Score",
        "AlphaMissense",
        "CADD_PHRED"
    ],

    ascending=[
        False,
        False,
        False,
        False
    ]

).reset_index(drop=True)


high_priority["Contextual_Priority_Rank"] = (
    range(
        1,
        len(high_priority) + 1
    )
)


# =============================================================================
# CONTEXTUALIZATION SUMMARY
# =============================================================================

print("\n" + "-" * 78)
print("HIGH-PRIORITY VARIANT CONTEXTUALIZATION RESULTS")
print("-" * 78)


display_columns = [

    "Contextual_Priority_Rank",
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Protein_Region",
    "Catalytic_Context",
    "Physicochemical_Impact",
    "AlphaMissense",
    "CADD_PHRED",
    "Population_Rarity_Category",
    "Structural_Relevance_Score",
    "Structural_Relevance_Category"
]


print(
    high_priority[
        display_columns
    ].to_string(
        index=False
    )
)


# =============================================================================
# FINAL QC
# =============================================================================

print("\n" + "=" * 78)
print("FINAL STEP 9E QUALITY CONTROL")
print("=" * 78)


if len(high_priority) != EXPECTED_HIGH_PRIORITY:

    fail(
        "High-priority variant retention failed."
    )


if high_priority["VariationID"].duplicated().any():

    fail(
        "Duplicate VariationIDs detected after contextualization."
    )


if high_priority["Exact_Variant_Key"].duplicated().any():

    fail(
        "Duplicate genomic identities detected after contextualization."
    )


if high_priority["Protein_Position"].isna().any():

    fail(
        "One or more protein positions could not be extracted."
    )


if (
    high_priority[
        "Structural_Relevance_Score"
    ].isna().any()
):

    fail(
        "Structural relevance scores contain missing values."
    )


print(f"High-priority variants retained: {len(high_priority)}/9")
print("VariationID uniqueness retained: PASS")
print("Exact genomic identity uniqueness retained: PASS")
print("Protein position extraction: PASS")
print("Structural relevance scoring: PASS")


# =============================================================================
# WRITE CLEAN CONTEXTUALIZATION TABLE
# =============================================================================

high_priority.to_csv(
    OUTPUT_FILE,
    index=False
)


# =============================================================================
# CREATE UPDATED MASTER FILE
# =============================================================================

df_master = df.copy()


context_columns = [

    "VariationID",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",
    "Protein_Region",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
    "Reference_AA_Class",
    "Alternate_AA_Class",
    "Reference_AA_Charge",
    "Alternate_AA_Charge",
    "Physicochemical_Change",
    "Physicochemical_Impact",
    "Structural_Relevance_Score",
    "Structural_Relevance_Category",
    "Contextual_Priority_Rank"
]


for column in context_columns:

    if column != "VariationID":

        mapping = high_priority.set_index(
            "VariationID"
        )[column]

        df_master[column] = (
            df_master["VariationID"]
            .map(mapping)
        )


df_master.to_csv(
    FINAL_MASTER,
    index=False
)


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print("\n" + "=" * 78)
print("STEP 9E SUCCESSFULLY COMPLETED")
print("=" * 78)

print("\nHigh-priority variants contextualized:")
print(len(high_priority))

print("\nContextualization output:")
print(OUTPUT_FILE)

print("\nUpdated master file:")
print(FINAL_MASTER)

print("\nNext scientific stage:")
print(
    "STEP 9F — Final evidence-based selection of PNPLA3 mutant "
    "candidate(s) for structural analysis."
)

print("\n" + "=" * 78)