import pandas as pd
import numpy as np
from pathlib import Path


# =============================================================================
# STEP 9F — FINAL EVIDENCE-BASED SELECTION OF PNPLA3 MUTANT CANDIDATE(S)
# =============================================================================

print("=" * 78)
print("STEP 9F — FINAL EVIDENCE-BASED SELECTION OF PNPLA3 MUTANT CANDIDATE(S)")
print("=" * 78)


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

ANNOTATION_DIR = PROJECT_DIR / "04_VARIANT ANALYSIS" / "annotations"

INPUT_FILE = ANNOTATION_DIR / "pnpla3_step9e_contextualized_FINAL.csv"

OUTPUT_SELECTION = (
    ANNOTATION_DIR /
    "pnpla3_final_mutant_candidate_selection.csv"
)

OUTPUT_MASTER = (
    ANNOTATION_DIR /
    "pnpla3_step9f_final_candidate_selection_FINAL.csv"
)


# =============================================================================
# 2. CHECK INPUT FILE
# =============================================================================

print("\nChecking input files...")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: Required Step 9E master file not found:\n{INPUT_FILE}"
    )

print("Step 9E contextualized master: PASS")


# =============================================================================
# 3. READ INPUT DATA
# =============================================================================

print("\nReading validated Step 9E master...")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")


# =============================================================================
# 4. BASIC INPUT QC
# =============================================================================

print("\n" + "-" * 78)
print("BASIC INPUT QC")
print("-" * 78)

required_columns = [
    "VariationID",
    "Protein_Change",
    "Exact_Variant_Key",
    "Candidate_Group",
    "AlphaMissense",
    "AlphaMissense_Interpretation",
    "CADD_PHRED",
    "CADD_Interpretation",
    "Population_Rarity_Category",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
    "Structural_Relevance_Score",
    "Structural_Relevance_Category",
    "Protein_Position",
    "Protein_Region",
    "Catalytic_Context",
    "Physicochemical_Impact"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "\nERROR: Required columns missing:\n"
        + "\n".join(f"  - {col}" for col in missing_columns)
    )

print("Required columns: PASS")

if len(df) != 82:
    raise ValueError(
        f"ERROR: Expected 82 variants, found {len(df)}"
    )

print("Expected variant count (82): PASS")

if df["VariationID"].duplicated().any():
    raise ValueError("ERROR: Duplicate VariationID values detected")

print("VariationID uniqueness: PASS")

if df["Exact_Variant_Key"].duplicated().any():
    raise ValueError(
        "ERROR: Duplicate Exact_Variant_Key values detected"
    )

print("Exact genomic identity uniqueness: PASS")


# =============================================================================
# 5. EXTRACT HIGH-PRIORITY STRUCTURAL CANDIDATES
# =============================================================================

print("\n" + "-" * 78)
print("EXTRACTING STRUCTURAL ANALYSIS CANDIDATES")
print("-" * 78)

candidates = df[
    df["Integrated_Priority_Category"] == "High priority"
].copy()

print(f"High-priority candidates identified: {len(candidates)}")

if len(candidates) == 0:
    raise ValueError(
        "ERROR: No high-priority variants available for final selection"
    )

if len(candidates) != 9:
    print(
        f"WARNING: Expected 9 high-priority variants, "
        f"found {len(candidates)}"
    )
else:
    print("Expected high-priority candidate count (9): PASS")


# =============================================================================
# 6. PREPARE NUMERIC VARIABLES
# =============================================================================

print("\n" + "-" * 78)
print("PREPARING FINAL EVIDENCE VARIABLES")
print("-" * 78)

numeric_columns = [
    "AlphaMissense",
    "CADD_PHRED",
    "Integrated_Priority_Score",
    "Structural_Relevance_Score",
    "Protein_Position"
]

for col in numeric_columns:
    candidates[col] = pd.to_numeric(
        candidates[col],
        errors="coerce"
    )

print("Numeric evidence conversion: PASS")

if candidates["AlphaMissense"].isna().any():
    raise ValueError("ERROR: Missing AlphaMissense values detected")

if candidates["CADD_PHRED"].isna().any():
    raise ValueError("ERROR: Missing CADD PHRED values detected")

if candidates["Integrated_Priority_Score"].isna().any():
    raise ValueError(
        "ERROR: Missing Integrated Priority Scores detected"
    )

if candidates["Structural_Relevance_Score"].isna().any():
    raise ValueError(
        "ERROR: Missing Structural Relevance Scores detected"
    )

print("Required computational evidence availability: PASS")


# =============================================================================
# 7. EVIDENCE COMPONENT SCORING
# =============================================================================

print("\n" + "-" * 78)
print("FINAL MULTI-EVIDENCE SCORING")
print("-" * 78)


# -----------------------------------------------------------------------------
# 7A. INTEGRATED PRIORITY SCORE
# -----------------------------------------------------------------------------

def score_integrated_priority(value):

    if pd.isna(value):
        return 0

    if value >= 14:
        return 4
    elif value >= 12:
        return 3
    elif value >= 10:
        return 2
    else:
        return 1


candidates["Final_Integrated_Priority_Component"] = (
    candidates["Integrated_Priority_Score"]
    .apply(score_integrated_priority)
)


# -----------------------------------------------------------------------------
# 7B. STRUCTURAL RELEVANCE SCORE
# -----------------------------------------------------------------------------

def score_structural_relevance(value):

    if pd.isna(value):
        return 0

    if value >= 16:
        return 4
    elif value >= 13:
        return 3
    elif value >= 10:
        return 2
    else:
        return 1


candidates["Final_Structural_Relevance_Component"] = (
    candidates["Structural_Relevance_Score"]
    .apply(score_structural_relevance)
)


# -----------------------------------------------------------------------------
# 7C. ALPHAMISSENSE COMPONENT
# -----------------------------------------------------------------------------

def score_alphamissense(value):

    if pd.isna(value):
        return 0

    if value >= 0.95:
        return 4
    elif value >= 0.80:
        return 3
    elif value >= 0.70:
        return 2
    else:
        return 1


candidates["Final_AlphaMissense_Component"] = (
    candidates["AlphaMissense"]
    .apply(score_alphamissense)
)


# -----------------------------------------------------------------------------
# 7D. CADD COMPONENT
# -----------------------------------------------------------------------------

def score_cadd(value):

    if pd.isna(value):
        return 0

    if value >= 25:
        return 4
    elif value >= 20:
        return 3
    elif value >= 15:
        return 2
    else:
        return 1


candidates["Final_CADD_Component"] = (
    candidates["CADD_PHRED"]
    .apply(score_cadd)
)


# -----------------------------------------------------------------------------
# 7E. POPULATION RARITY COMPONENT
# -----------------------------------------------------------------------------

rarity_score_map = {
    "Not observed": 4,
    "Ultra-rare": 4,
    "Very rare": 3,
    "Rare": 2,
    "Low frequency": 1,
    "Common": 0,
    "Population frequency unavailable": 1
}

candidates["Final_Population_Rarity_Component"] = (
    candidates["Population_Rarity_Category"]
    .map(rarity_score_map)
    .fillna(0)
    .astype(int)
)


# -----------------------------------------------------------------------------
# 7F. CATALYTIC / FUNCTIONAL CONTEXT COMPONENT
# -----------------------------------------------------------------------------

catalytic_score_map = {
    "Catalytic residue": 4,
    "Very close to catalytic residue": 4,
    "Near catalytic residue": 3,
    "Within catalytic domain context": 2,
    "Not proximal to catalytic residues": 1
}

candidates["Final_Catalytic_Context_Component"] = (
    candidates["Catalytic_Context"]
    .map(catalytic_score_map)
    .fillna(0)
    .astype(int)
)


# -----------------------------------------------------------------------------
# 7G. PHYSICOCHEMICAL IMPACT COMPONENT
# -----------------------------------------------------------------------------

physicochemical_score_map = {
    "High physicochemical disruption": 3,
    "Moderate physicochemical disruption": 2,
    "Low physicochemical disruption": 1
}

candidates["Final_Physicochemical_Component"] = (
    candidates["Physicochemical_Impact"]
    .map(physicochemical_score_map)
    .fillna(0)
    .astype(int)
)


print("Integrated priority evidence scoring: PASS")
print("Structural relevance evidence scoring: PASS")
print("AlphaMissense evidence scoring: PASS")
print("CADD evidence scoring: PASS")
print("Population rarity evidence scoring: PASS")
print("Catalytic context evidence scoring: PASS")
print("Physicochemical impact scoring: PASS")


# =============================================================================
# 8. CALCULATE FINAL MUTANT SELECTION SCORE
# =============================================================================

print("\n" + "=" * 78)
print("CALCULATING FINAL MUTANT SELECTION SCORE")
print("=" * 78)


score_columns = [
    "Final_Integrated_Priority_Component",
    "Final_Structural_Relevance_Component",
    "Final_AlphaMissense_Component",
    "Final_CADD_Component",
    "Final_Population_Rarity_Component",
    "Final_Catalytic_Context_Component",
    "Final_Physicochemical_Component"
]

candidates["Final_Mutant_Selection_Score"] = (
    candidates[score_columns].sum(axis=1)
)


# =============================================================================
# 9. ASSIGN EVIDENCE CATEGORY
# =============================================================================

def classify_final_evidence(score):

    if score >= 22:
        return "Exceptional integrated evidence"

    elif score >= 18:
        return "Very strong integrated evidence"

    elif score >= 14:
        return "Strong integrated evidence"

    else:
        return "Moderate integrated evidence"


candidates["Final_Evidence_Category"] = (
    candidates["Final_Mutant_Selection_Score"]
    .apply(classify_final_evidence)
)


# =============================================================================
# 10. HANDLE SAME-POSITION VARIANT REDUNDANCY
# =============================================================================

print("\n" + "-" * 78)
print("POSITION-LEVEL REDUNDANCY ASSESSMENT")
print("-" * 78)


position_counts = (
    candidates["Protein_Position"]
    .value_counts()
)

recurrent_positions = position_counts[
    position_counts > 1
]

if len(recurrent_positions) == 0:

    print("No duplicate protein positions among candidates")

else:

    print("Multiple candidate substitutions detected at:")
    
    for position, count in recurrent_positions.items():
        print(
            f"Protein position {position}: "
            f"{count} candidate substitutions"
        )


# =============================================================================
# 11. SORT FINAL CANDIDATES
# =============================================================================

print("\n" + "-" * 78)
print("RANKING FINAL MUTANT CANDIDATES")
print("-" * 78)


candidates = candidates.sort_values(
    by=[
        "Final_Mutant_Selection_Score",
        "Structural_Relevance_Score",
        "Integrated_Priority_Score",
        "AlphaMissense",
        "CADD_PHRED"
    ],
    ascending=[
        False,
        False,
        False,
        False,
        False
    ]
).reset_index(drop=True)


candidates["Final_Mutant_Rank"] = (
    candidates.index + 1
)


# =============================================================================
# 12. ASSIGN STRUCTURAL ANALYSIS RECOMMENDATION
# =============================================================================

def assign_recommendation(row):

    rank = row["Final_Mutant_Rank"]

    if rank == 1:
        return "Primary mutant candidate"

    elif rank <= 3:
        return "Secondary structural candidate"

    elif rank <= 5:
        return "Comparative structural candidate"

    else:
        return "Reserve candidate"


candidates["Structural_Analysis_Recommendation"] = (
    candidates.apply(
        assign_recommendation,
        axis=1
    )
)


# =============================================================================
# 13. PRIMARY CANDIDATE JUSTIFICATION
# =============================================================================

def generate_selection_basis(row):

    evidence = []

    if row["AlphaMissense"] >= 0.90:
        evidence.append("strong AlphaMissense pathogenicity evidence")

    if row["CADD_PHRED"] >= 20:
        evidence.append("high CADD deleteriousness")

    if row["Structural_Relevance_Score"] >= 16:
        evidence.append("very high structural relevance")

    if row["Population_Rarity_Category"] in [
        "Not observed",
        "Ultra-rare",
        "Very rare"
    ]:
        evidence.append("population rarity")

    if row["Catalytic_Context"] in [
        "Catalytic residue",
        "Very close to catalytic residue",
        "Near catalytic residue"
    ]:
        evidence.append("functional proximity to catalytic context")

    return "; ".join(evidence)


candidates["Final_Selection_Basis"] = (
    candidates.apply(
        generate_selection_basis,
        axis=1
    )
)


# =============================================================================
# 14. PRIMARY / SECONDARY / COMPARATIVE CANDIDATE SUMMARY
# =============================================================================

primary_candidate = candidates.iloc[0]

secondary_candidates = candidates[
    candidates["Structural_Analysis_Recommendation"]
    == "Secondary structural candidate"
]

comparative_candidates = candidates[
    candidates["Structural_Analysis_Recommendation"]
    == "Comparative structural candidate"
]


# =============================================================================
# 15. PRINT FINAL RESULTS
# =============================================================================

print("\n" + "=" * 78)
print("FINAL PNPLA3 MUTANT CANDIDATE SELECTION RESULTS")
print("=" * 78)


display_columns = [
    "Final_Mutant_Rank",
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Candidate_Group",
    "AlphaMissense",
    "CADD_PHRED",
    "Population_Rarity_Category",
    "Catalytic_Context",
    "Structural_Relevance_Score",
    "Final_Mutant_Selection_Score",
    "Final_Evidence_Category",
    "Structural_Analysis_Recommendation"
]

print(
    candidates[display_columns]
    .to_string(index=False)
)


# =============================================================================
# 16. FINAL PRIMARY CANDIDATE REPORT
# =============================================================================

print("\n" + "-" * 78)
print("PRIMARY MUTANT CANDIDATE")
print("-" * 78)

print(f"Rank: {primary_candidate['Final_Mutant_Rank']}")
print(f"VariationID: {primary_candidate['VariationID']}")
print(f"Protein change: {primary_candidate['Protein_Change']}")
print(f"Protein position: {primary_candidate['Protein_Position']}")
print(
    f"Final selection score: "
    f"{primary_candidate['Final_Mutant_Selection_Score']}"
)
print(
    f"Evidence category: "
    f"{primary_candidate['Final_Evidence_Category']}"
)
print(
    f"Selection basis: "
    f"{primary_candidate['Final_Selection_Basis']}"
)


# =============================================================================
# 17. FINAL QC
# =============================================================================

print("\n" + "=" * 78)
print("FINAL STEP 9F QUALITY CONTROL")
print("=" * 78)


if len(candidates) != 9:
    raise ValueError(
        f"ERROR: Expected 9 candidates after selection, "
        f"found {len(candidates)}"
    )

print("9/9 high-priority candidates retained: PASS")


if candidates["VariationID"].duplicated().any():
    raise ValueError(
        "ERROR: Duplicate VariationIDs after final ranking"
    )

print("VariationID uniqueness retained: PASS")


if candidates["Exact_Variant_Key"].duplicated().any():
    raise ValueError(
        "ERROR: Duplicate genomic identities after final ranking"
    )

print("Exact genomic identity uniqueness retained: PASS")


if candidates["Final_Mutant_Selection_Score"].isna().any():
    raise ValueError(
        "ERROR: Missing final mutant selection scores"
    )

print("Final selection scores populated: PASS")


if candidates["Structural_Analysis_Recommendation"].isna().any():
    raise ValueError(
        "ERROR: Missing structural analysis recommendations"
    )

print("Structural recommendations populated: PASS")


if len(candidates[
    candidates["Structural_Analysis_Recommendation"]
    == "Primary mutant candidate"
]) != 1:

    raise ValueError(
        "ERROR: Exactly one primary mutant candidate must be selected"
    )

print("Single primary mutant candidate selected: PASS")


# =============================================================================
# 18. SAVE CLEAN CANDIDATE SELECTION TABLE
# =============================================================================

selection_columns = [
    "Final_Mutant_Rank",
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Candidate_Group",
    "Exact_Variant_Key",
    "AlphaMissense",
    "AlphaMissense_Interpretation",
    "CADD_PHRED",
    "CADD_Interpretation",
    "Population_Rarity_Category",
    "Integrated_Priority_Score",
    "Structural_Relevance_Score",
    "Structural_Relevance_Category",
    "Protein_Region",
    "Catalytic_Context",
    "Physicochemical_Impact",
    "Final_Integrated_Priority_Component",
    "Final_Structural_Relevance_Component",
    "Final_AlphaMissense_Component",
    "Final_CADD_Component",
    "Final_Population_Rarity_Component",
    "Final_Catalytic_Context_Component",
    "Final_Physicochemical_Component",
    "Final_Mutant_Selection_Score",
    "Final_Evidence_Category",
    "Structural_Analysis_Recommendation",
    "Final_Selection_Basis"
]

selection_columns = [
    col for col in selection_columns
    if col in candidates.columns
]

candidates[
    selection_columns
].to_csv(
    OUTPUT_SELECTION,
    index=False
)


# =============================================================================
# 19. UPDATE FULL MASTER DATASET
# =============================================================================

final_columns_to_merge = [
    "VariationID",
    "Final_Mutant_Rank",
    "Final_Integrated_Priority_Component",
    "Final_Structural_Relevance_Component",
    "Final_AlphaMissense_Component",
    "Final_CADD_Component",
    "Final_Population_Rarity_Component",
    "Final_Catalytic_Context_Component",
    "Final_Physicochemical_Component",
    "Final_Mutant_Selection_Score",
    "Final_Evidence_Category",
    "Structural_Analysis_Recommendation",
    "Final_Selection_Basis"
]

df = df.merge(
    candidates[final_columns_to_merge],
    on="VariationID",
    how="left",
    validate="one_to_one"
)


# =============================================================================
# 20. ASSIGN NON-HIGH-PRIORITY VARIANT STATUS
# =============================================================================

df["Structural_Analysis_Recommendation"] = (
    df["Structural_Analysis_Recommendation"]
    .fillna("Not selected for structural analysis")
)

df["Final_Evidence_Category"] = (
    df["Final_Evidence_Category"]
    .fillna("Not evaluated in final structural candidate selection")
)


# =============================================================================
# 21. SAVE FINAL MASTER
# =============================================================================

df.to_csv(
    OUTPUT_MASTER,
    index=False
)


# =============================================================================
# 22. FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 78)
print("STEP 9F FINAL SUMMARY")
print("=" * 78)

print(f"\nPNPLA3 variants analyzed: {len(df)}")
print(f"High-priority variants evaluated: {len(candidates)}")

print(
    "\nPrimary structural candidate: "
    f"{primary_candidate['Protein_Change']}"
)

print(
    "Secondary structural candidates: "
    f"{len(secondary_candidates)}"
)

print(
    "Comparative structural candidates: "
    f"{len(comparative_candidates)}"
)


print("\nFinal recommendation distribution:")

print(
    candidates[
        "Structural_Analysis_Recommendation"
    ].value_counts()
)


print("\nFinal selection score distribution:")

print(
    candidates[
        "Final_Mutant_Selection_Score"
    ].describe()
)


print("\nClean final candidate selection table:")
print(OUTPUT_SELECTION)

print("\nFinal Step 9F master file:")
print(OUTPUT_MASTER)


print("\n" + "=" * 78)
print("STEP 9F SUCCESSFULLY COMPLETED")
print("=" * 78)

print(
    "\nNext scientific stage:\n"
    "STEP 10 — Wild-type and selected PNPLA3 mutant structural preparation "
    "and comparative structural analysis."
)

print("=" * 78)