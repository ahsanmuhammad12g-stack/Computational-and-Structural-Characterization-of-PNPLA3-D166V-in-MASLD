import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================================
# STEP 14 — FINAL VARIANT EVIDENCE MATRIX / FINAL INTEGRATED DATASET
# ============================================================================

print("=" * 78)
print("STEP 14 — FINAL VARIANT EVIDENCE MATRIX")
print("=" * 78)


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

STEP9F = PROJECT_ROOT / (
    r"VARIANT ANALYSIS\annotations"
    r"\pnpla3_step9f_final_candidate_selection_FINAL.csv"
)

STEP10_TABLES = (
    PROJECT_ROOT /
    r"RESULTS\STEP10_STRUCTURAL_ANALYSIS\tables"
)

STEP11_TABLES = (
    PROJECT_ROOT /
    r"RESULTS\STEP11_D166V_CATALYTIC_CONTEXT\tables"
)

STEP13_TABLES = (
    PROJECT_ROOT /
    r"RESULTS\STEP13_COMPARATIVE_STRUCTURAL_ANALYSIS\tables"
)

OUTPUT_DIR = (
    PROJECT_ROOT /
    r"RESULTS\STEP14_FINAL_VARIANT_EVIDENCE_MATRIX"
)

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
QC_DIR = OUTPUT_DIR / "QC"

for directory in [OUTPUT_DIR, TABLE_DIR, FIGURE_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================================
# OUTPUT FILES
# ============================================================================

TABLE1 = TABLE_DIR / "Table_S14_1_final_variant_evidence_matrix.csv"
TABLE2 = TABLE_DIR / "Table_S14_2_top_candidate_evidence.csv"
TABLE3 = TABLE_DIR / "Table_S14_3_structural_comparison_summary.csv"
TABLE4 = TABLE_DIR / "Table_S14_4_downstream_candidate_prioritization.csv"

FIG1 = FIGURE_DIR / "Figure_S14_1_integrated_variant_evidence.png"
FIG2 = FIGURE_DIR / "Figure_S14_2_computational_vs_structural_priority.png"
FIG3 = FIGURE_DIR / "Figure_S14_3_top_candidate_evidence_profile.png"

QC_FILE = QC_DIR / "Step14_final_QC_report.json"
SUMMARY_FILE = OUTPUT_DIR / "Step14_final_summary.txt"


# ============================================================================
# EXPECTED STRUCTURAL CANDIDATES
# ============================================================================

EXPECTED_STRUCTURAL_VARIANTS = [
    "p.Asp166Val",
    "p.Gly49Trp",
    "p.Cys28Arg",
    "p.Pro186Ser",
    "p.Pro186His",
    "p.Leu51Ser",
    "p.Tyr21Cys",
    "p.Arg247Thr",
    "p.Leu142Phe",
]

PRIMARY_VARIANT = "p.Asp166Val"
PRIMARY_VARIATION_ID = 3308171


# ============================================================================
# QC
# ============================================================================

qc = {}


def qc_print(name, status, detail=""):

    qc[name] = {
        "status": bool(status),
        "detail": detail
    }

    label = "PASS" if status else "FAIL"

    print(
        f"{name:<58} "
        f"{label}"
        + (f" — {detail}" if detail else "")
    )


def require_file(path, label):

    exists = path.exists()

    qc_print(
        label,
        exists,
        str(path) if exists else f"Missing: {path}"
    )

    return exists


# ============================================================================
# INPUT FILE CHECK
# ============================================================================

print("\n" + "-" * 78)
print("CHECKING INPUT FILES")
print("-" * 78)

require_file(
    STEP9F,
    "Step9F_exists"
)

require_file(
    STEP10_TABLES / "Table_S10_8_D166V_structural_metrics.csv",
    "Step10_structural_metrics_exists"
)

require_file(
    STEP11_TABLES / "Table_S11_8_D166V_structural_impact_summary.csv",
    "Step11_structural_summary_exists"
)

require_file(
    STEP13_TABLES / "Table_S13_1_all9_structural_comparison.csv",
    "Step13_structural_comparison_exists"
)

require_file(
    STEP13_TABLES / "Table_S13_6_final_structural_ranking.csv",
    "Step13_structural_ranking_exists"
)

require_file(
    STEP13_TABLES / "Table_S13_7_mutant_validation.csv",
    "Step13_mutant_validation_exists"
)


if not STEP9F.exists():
    raise FileNotFoundError(STEP9F)

STEP13_COMPARISON = (
    STEP13_TABLES /
    "Table_S13_1_all9_structural_comparison.csv"
)

STEP13_RANKING = (
    STEP13_TABLES /
    "Table_S13_6_final_structural_ranking.csv"
)

if not STEP13_COMPARISON.exists():
    raise FileNotFoundError(STEP13_COMPARISON)

if not STEP13_RANKING.exists():
    raise FileNotFoundError(STEP13_RANKING)


# ============================================================================
# READ STEP 9F
# ============================================================================

print("\n" + "-" * 78)
print("READING AUTHORITATIVE STEP 9F DATASET")
print("-" * 78)

df9 = pd.read_csv(STEP9F)

print(f"Rows loaded    : {len(df9)}")
print(f"Columns loaded : {len(df9.columns)}")


REQUIRED_STEP9 = [
    "VariationID",
    "Protein_Change",
    "Candidate_Group",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "CADD_PHRED",
    "CADD_Interpretation",
    "Maximum_gnomAD_AF",
    "Population_Rarity_Category",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",
    "Protein_Region",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
    "Physicochemical_Impact",
    "Structural_Relevance_Score",
    "Structural_Relevance_Category",
    "Contextual_Priority_Rank",
    "Final_Mutant_Rank",
    "Final_Mutant_Selection_Score",
    "Final_Evidence_Category",
    "Structural_Analysis_Recommendation",
    "Final_Selection_Basis",
]

missing = [
    c for c in REQUIRED_STEP9
    if c not in df9.columns
]

qc_print(
    "Step9F_required_columns",
    len(missing) == 0,
    "All required columns present"
    if not missing
    else f"Missing: {missing}"
)

qc_print(
    "Step9F_expected_rows",
    len(df9) == 82,
    f"Expected 82, observed {len(df9)}"
)


# ============================================================================
# IDENTIFY 9 STRUCTURAL CANDIDATES
# ============================================================================

print("\n" + "-" * 78)
print("IDENTIFYING STRUCTURAL CANDIDATES FROM STEP 9F")
print("-" * 78)

structural_flag = (
    df9["Structural_Analysis_Candidate"]
    .astype(str)
    .str.upper()
    .isin(["YES", "TRUE", "1"])
)

structural9 = df9.loc[structural_flag].copy()

print(
    f"Structural candidates detected: {len(structural9)}"
)

qc_print(
    "Structural_candidate_count",
    len(structural9) == 9,
    f"Expected 9, observed {len(structural9)}"
)

observed = set(
    structural9["Protein_Change"].astype(str)
)

qc_print(
    "Expected_nine_variants",
    observed == set(EXPECTED_STRUCTURAL_VARIANTS),
    "All expected structural candidates present"
    if observed == set(EXPECTED_STRUCTURAL_VARIANTS)
    else (
        f"Observed={sorted(observed)}"
    )
)


# ============================================================================
# READ STEP 13
# ============================================================================

print("\n" + "-" * 78)
print("READING STEP 13 STRUCTURAL COMPARISON")
print("-" * 78)

df13 = pd.read_csv(STEP13_COMPARISON)

df13_rank = pd.read_csv(STEP13_RANKING)

print(f"Rows loaded    : {len(df13)}")
print(f"Columns loaded : {len(df13.columns)}")

print("\nStep 13 columns:")
for col in df13.columns:
    print(f"  - {col}")


# ============================================================================
# VALIDATE ACTUAL STEP 13 SCHEMA
# ============================================================================

# These are the ACTUAL columns shown in the user's successful Step 13 run.

REQUIRED_STEP13 = [
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "WT_Amino_Acid",
    "Mutant_Amino_Acid",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
    "Nearest_Catalytic_Position",
    "WT_Catalytic_Distance_A",
    "Mutant_Catalytic_Distance_A",
    "Catalytic_Distance_Change_A",
    "WT_Contact_Count",
    "Mutant_Contact_Count",
    "Contacts_Retained",
    "Contacts_Lost",
    "Contacts_Gained",
    "WT_Polar_Contact_Count",
    "Mutant_Polar_Contact_Count",
    "Polar_Contact_Change",
    "Global_CA_RMSD_A",
    "Global_Backbone_RMSD_A",
    "Local_CA_RMSD_A",
    "Candidate_pLDDT",
    "pLDDT_Source",
    "Physicochemical_Impact_Score",
    "Structural_Impact_Index",
    "Mutation_Model_Method",
    "Structural_Model_Type",
    "Energy_Minimization_Performed",
    "Molecular_Dynamics_Performed",
]

missing13 = [
    c for c in REQUIRED_STEP13
    if c not in df13.columns
]

qc_print(
    "Step13_required_columns",
    len(missing13) == 0,
    "All actual Step 13 columns present"
    if not missing13
    else f"Missing: {missing13}"
)

qc_print(
    "Step13_expected_variant_count",
    len(df13) == 9,
    f"Expected 9, observed {len(df13)}"
)


step13_variants = set(
    df13["Protein_Change"].astype(str)
)

qc_print(
    "Step13_all_expected_variants",
    set(EXPECTED_STRUCTURAL_VARIANTS).issubset(step13_variants),
    "All 9 expected variants present"
)


# ============================================================================
# BUILD STRUCTURAL EVIDENCE TABLE
# ============================================================================

print("\n" + "-" * 78)
print("BUILDING STEP 13 STRUCTURAL EVIDENCE LAYER")
print("-" * 78)

structural_columns = [
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "WT_Amino_Acid",
    "Mutant_Amino_Acid",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
    "Nearest_Catalytic_Position",
    "WT_Catalytic_Distance_A",
    "Mutant_Catalytic_Distance_A",
    "Catalytic_Distance_Change_A",
    "WT_Contact_Count",
    "Mutant_Contact_Count",
    "Contacts_Retained",
    "Contacts_Lost",
    "Contacts_Gained",
    "WT_Polar_Contact_Count",
    "Mutant_Polar_Contact_Count",
    "Polar_Contact_Change",
    "Global_CA_RMSD_A",
    "Global_Backbone_RMSD_A",
    "Local_CA_RMSD_A",
    "Candidate_pLDDT",
    "pLDDT_Source",
    "Physicochemical_Impact_Score",
    "Structural_Impact_Index",
    "Mutation_Model_Method",
    "Structural_Model_Type",
    "Energy_Minimization_Performed",
    "Molecular_Dynamics_Performed",
]

structural13 = df13[
    structural_columns
].copy()


# ============================================================================
# READ STEP 13 RANKING
# ============================================================================

print("\n" + "-" * 78)
print("READING STEP 13 FINAL STRUCTURAL RANKING")
print("-" * 78)

print("Step 13 ranking columns:")

for col in df13_rank.columns:
    print(f"  - {col}")


# Identify ranking column
rank_candidates = [
    "Rank",
    "Structural_Rank",
    "Final_Structural_Rank",
    "Ranking",
]

rank_col = None

for c in rank_candidates:
    if c in df13_rank.columns:
        rank_col = c
        break


# If the rank column is not literally called Rank,
# identify it from the known Step 13 ranking structure.

if rank_col is None:

    # Search columns containing "rank"
    for c in df13_rank.columns:
        if "rank" in c.lower():
            rank_col = c
            break


if rank_col is None:

    # Since the Step 13 output is already sorted by structural impact,
    # safely derive the rank from Structural_Impact_Index.
    if "Structural_Impact_Index" in df13_rank.columns:

        df13_rank = df13_rank.sort_values(
            "Structural_Impact_Index",
            ascending=False
        ).copy()

        df13_rank["Step13_Structural_Rank"] = (
            np.arange(len(df13_rank)) + 1
        )

        rank_col = "Step13_Structural_Rank"

    elif "Impact" in df13_rank.columns:

        df13_rank = df13_rank.sort_values(
            "Impact",
            ascending=False
        ).copy()

        df13_rank["Step13_Structural_Rank"] = (
            np.arange(len(df13_rank)) + 1
        )

        rank_col = "Step13_Structural_Rank"

    else:
        raise ValueError(
            "Could not identify or derive Step 13 structural rank."
        )


rank_table = df13_rank[
    ["Protein_Change", rank_col]
].copy()

rank_table = rank_table.rename(
    columns={
        rank_col: "Step13_Structural_Rank"
    }
)

rank_table["Protein_Change"] = (
    rank_table["Protein_Change"]
    .astype(str)
    .str.strip()
)

rank_table["Step13_Structural_Rank"] = pd.to_numeric(
    rank_table["Step13_Structural_Rank"],
    errors="coerce"
)

qc_print(
    "Step13_structural_ranks_complete",
    rank_table["Step13_Structural_Rank"].notna().sum() == 9,
    "All 9 structural ranks available"
)


# ============================================================================
# MERGE RANK
# ============================================================================

structural13 = structural13.merge(
    rank_table,
    on="Protein_Change",
    how="left",
    validate="one_to_one"
)


# ============================================================================
# STRUCTURAL DERIVED METRICS
# ============================================================================

structural13["Total_Contact_Change"] = (
    pd.to_numeric(
        structural13["Mutant_Contact_Count"],
        errors="coerce"
    )
    -
    pd.to_numeric(
        structural13["WT_Contact_Count"],
        errors="coerce"
    )
)

structural13["Polar_Contact_Change_Calculated"] = (
    pd.to_numeric(
        structural13["Mutant_Polar_Contact_Count"],
        errors="coerce"
    )
    -
    pd.to_numeric(
        structural13["WT_Polar_Contact_Count"],
        errors="coerce"
    )
)


# ============================================================================
# CHECK STEP 13 INTERNAL CONSISTENCY
# ============================================================================

polar_consistency = np.isclose(
    pd.to_numeric(
        structural13["Polar_Contact_Change"],
        errors="coerce"
    ),
    pd.to_numeric(
        structural13["Polar_Contact_Change_Calculated"],
        errors="coerce"
    ),
    equal_nan=False
)

qc_print(
    "Step13_polar_contact_change_consistency",
    bool(np.all(polar_consistency)),
    "Reported and calculated polar-contact changes agree"
    if bool(np.all(polar_consistency))
    else "Mismatch detected"
)


contact_consistency = np.isclose(
    (
        pd.to_numeric(
            structural13["Mutant_Contact_Count"],
            errors="coerce"
        )
        -
        pd.to_numeric(
            structural13["WT_Contact_Count"],
            errors="coerce"
        )
    ),
    structural13["Total_Contact_Change"],
    equal_nan=False
)

qc_print(
    "Step13_contact_change_consistency",
    bool(np.all(contact_consistency)),
    "Contact-change calculations are internally consistent"
)


# ============================================================================
# COMPUTATIONAL EVIDENCE LAYER
# ============================================================================

print("\n" + "-" * 78)
print("BUILDING COMPUTATIONAL EVIDENCE LAYER")
print("-" * 78)

computational_columns = [
    "VariationID",
    "Protein_Change",
    "Candidate_Group",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "CADD_PHRED",
    "CADD_Interpretation",
    "Maximum_gnomAD_AF",
    "Population_Rarity_Category",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",
    "Protein_Region",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
    "Physicochemical_Impact",
    "Structural_Relevance_Score",
    "Structural_Relevance_Category",
    "Contextual_Priority_Rank",
    "Final_Mutant_Rank",
    "Final_Mutant_Selection_Score",
    "Final_Evidence_Category",
    "Structural_Analysis_Recommendation",
    "Final_Selection_Basis",
]

computational = df9[
    computational_columns
].copy()


# ============================================================================
# MERGE
# ============================================================================

print("\n" + "-" * 78)
print("MERGING COMPUTATIONAL AND STRUCTURAL EVIDENCE")
print("-" * 78)

final_matrix = computational.merge(
    structural13,
    on=[
        "VariationID",
        "Protein_Change"
    ],
    how="left",
    validate="one_to_one"
)


# ============================================================================
# COVERAGE
# ============================================================================

qc_print(
    "Final_matrix_variant_count",
    len(final_matrix) == 82,
    f"Expected 82, observed {len(final_matrix)}"
)

structural_matrix = final_matrix[
    final_matrix["Protein_Change"].isin(
        EXPECTED_STRUCTURAL_VARIANTS
    )
].copy()

qc_print(
    "Structural_matrix_variant_count",
    len(structural_matrix) == 9,
    f"Expected 9, observed {len(structural_matrix)}"
)


# ============================================================================
# VERIFY ALL STRUCTURAL EVIDENCE
# ============================================================================

structural_evidence_fields = [
    "Global_CA_RMSD_A",
    "Global_Backbone_RMSD_A",
    "Local_CA_RMSD_A",
    "WT_Contact_Count",
    "Mutant_Contact_Count",
    "WT_Polar_Contact_Count",
    "Mutant_Polar_Contact_Count",
    "Candidate_pLDDT",
    "Structural_Impact_Index",
    "Step13_Structural_Rank",
]

missing_structural_values = {}

for field in structural_evidence_fields:

    missing_count = int(
        structural_matrix[field].isna().sum()
    )

    missing_structural_values[field] = missing_count


all_structural_complete = all(
    count == 0
    for count in missing_structural_values.values()
)

qc_print(
    "Structural_evidence_complete",
    all_structural_complete,
    "All 9 structural candidates contain complete Step 13 evidence"
    if all_structural_complete
    else str(missing_structural_values)
)


# ============================================================================
# PRIMARY D166V
# ============================================================================

primary_rows = final_matrix[
    (
        final_matrix["VariationID"]
        .astype(str)
        == str(PRIMARY_VARIATION_ID)
    )
    &
    (
        final_matrix["Protein_Change"]
        == PRIMARY_VARIANT
    )
]

qc_print(
    "Primary_D166V_present",
    len(primary_rows) == 1,
    "Exactly one D166V primary candidate found"
)


# ============================================================================
# FINAL INTEGRATED CATEGORY
# ============================================================================

def classify_final_evidence(row):

    variant = str(row["Protein_Change"])

    if variant == PRIMARY_VARIANT:
        return "Primary integrated structural candidate"

    if variant not in EXPECTED_STRUCTURAL_VARIANTS:
        return "Non-structural candidate"

    rank = pd.to_numeric(
        row["Step13_Structural_Rank"],
        errors="coerce"
    )

    if pd.notna(rank) and rank <= 3:
        return "High structural-impact candidate"

    return "Comparative structural candidate"


final_matrix["Final_Integrated_Evidence_Category"] = (
    final_matrix.apply(
        classify_final_evidence,
        axis=1
    )
)


# ============================================================================
# DOWNSTREAM RECOMMENDATION
# ============================================================================

def downstream_recommendation(row):

    variant = str(row["Protein_Change"])

    if variant == PRIMARY_VARIANT:

        return (
            "Primary downstream candidate — retain for detailed "
            "structural, docking, molecular dynamics, and "
            "mechanistic analysis"
        )

    if variant in EXPECTED_STRUCTURAL_VARIANTS:

        rank = pd.to_numeric(
            row["Step13_Structural_Rank"],
            errors="coerce"
        )

        if pd.notna(rank) and rank <= 3:

            return (
                "High-priority comparative structural candidate — "
                "retain as secondary structural comparator"
            )

        return (
            "Comparative structural candidate — retain for "
            "structural context and sensitivity analysis"
        )

    return (
        "No primary structural modeling recommendation"
    )


final_matrix["Downstream_Recommendation"] = (
    final_matrix.apply(
        downstream_recommendation,
        axis=1
    )
)


# ============================================================================
# EVIDENCE LAYER
# ============================================================================

def evidence_level(row):

    computational_present = (
        pd.notna(row["AlphaMissense"])
        or pd.notna(row["CADD_PHRED"])
        or pd.notna(row["Integrated_Priority_Score"])
    )

    structural_present = (
        pd.notna(row["Structural_Impact_Index"])
    )

    if computational_present and structural_present:
        return "Computational + structural"

    if computational_present:
        return "Computational only"

    if structural_present:
        return "Structural only"

    return "Insufficient"


final_matrix["Evidence_Layer_Availability"] = (
    final_matrix.apply(
        evidence_level,
        axis=1
    )
)


# ============================================================================
# STRUCTURAL INTERPRETATION
# ============================================================================

def structural_interpretation(row):

    variant = str(row["Protein_Change"])

    if variant not in EXPECTED_STRUCTURAL_VARIANTS:
        return "Not included in Step 13 structural comparison"

    polar_change = pd.to_numeric(
        row["Polar_Contact_Change"],
        errors="coerce"
    )

    contact_change = pd.to_numeric(
        row["Total_Contact_Change"],
        errors="coerce"
    )

    statements = []

    if pd.notna(polar_change):

        if polar_change < 0:
            statements.append(
                "reduced candidate polar contacts"
            )

        elif polar_change > 0:
            statements.append(
                "increased candidate polar contacts"
            )

        else:
            statements.append(
                "no net candidate polar-contact change"
            )

    if pd.notna(contact_change):

        if contact_change < 0:
            statements.append(
                "reduced local heavy-atom contacts"
            )

        elif contact_change > 0:
            statements.append(
                "increased local heavy-atom contacts"
            )

        else:
            statements.append(
                "no net local heavy-atom contact change"
            )

    return "; ".join(statements)


final_matrix["Structural_Evidence_Interpretation"] = (
    final_matrix.apply(
        structural_interpretation,
        axis=1
    )
)


# ============================================================================
# INTERPRETATION NOTE
# ============================================================================

final_matrix["Interpretation_Note"] = (
    "Structural models were generated by side-chain remodeling "
    "on the validated WT AlphaFold backbone. Step 13 structural "
    "impact represents comparative modeled structural effects and "
    "is not a pathogenicity probability. No energy minimization "
    "or molecular dynamics was performed."
)


# ============================================================================
# COLUMN ORDER
# ============================================================================

preferred_columns = [

    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",

    "Candidate_Group",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "CADD_PHRED",
    "CADD_Interpretation",
    "Maximum_gnomAD_AF",
    "Population_Rarity_Category",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
    "Final_Mutant_Rank",

    "Protein_Region",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
    "Physicochemical_Impact",

    "Structural_Relevance_Score",
    "Structural_Relevance_Category",
    "Contextual_Priority_Rank",
    "Final_Mutant_Selection_Score",
    "Final_Evidence_Category",

    "Step13_Structural_Rank",
    "Global_CA_RMSD_A",
    "Global_Backbone_RMSD_A",
    "Local_CA_RMSD_A",

    "WT_Contact_Count",
    "Mutant_Contact_Count",
    "Contacts_Retained",
    "Contacts_Lost",
    "Contacts_Gained",
    "Total_Contact_Change",

    "WT_Polar_Contact_Count",
    "Mutant_Polar_Contact_Count",
    "Polar_Contact_Change",

    "Candidate_pLDDT",
    "pLDDT_Source",

    "WT_Catalytic_Distance_A",
    "Mutant_Catalytic_Distance_A",
    "Catalytic_Distance_Change_A",

    "Physicochemical_Impact_Score",
    "Structural_Impact_Index",

    "Mutation_Model_Method",
    "Structural_Model_Type",
    "Energy_Minimization_Performed",
    "Molecular_Dynamics_Performed",

    "Evidence_Layer_Availability",
    "Final_Integrated_Evidence_Category",
    "Structural_Evidence_Interpretation",
    "Downstream_Recommendation",

    "Structural_Analysis_Recommendation",
    "Final_Selection_Basis",
    "Interpretation_Note",
]

available_columns = [
    c for c in preferred_columns
    if c in final_matrix.columns
]

final_matrix = final_matrix[
    available_columns
].copy()


# ============================================================================
# TABLE 1
# ============================================================================

final_matrix.to_csv(
    TABLE1,
    index=False
)

print(
    f"\nTable_S14_1 created:\n{TABLE1}"
)


# ============================================================================
# TABLE 2 — TOP CANDIDATES
# ============================================================================

top_candidates = final_matrix[
    final_matrix["Protein_Change"].isin(
        EXPECTED_STRUCTURAL_VARIANTS
    )
].copy()

top_candidates = top_candidates.sort_values(
    [
        "Final_Integrated_Evidence_Category",
        "Step13_Structural_Rank"
    ],
    ascending=[
        True,
        True
    ]
)

# Force D166V to top
top_candidates["Primary_Order"] = (
    top_candidates["Protein_Change"]
    == PRIMARY_VARIANT
).astype(int)

top_candidates = top_candidates.sort_values(
    [
        "Primary_Order",
        "Step13_Structural_Rank"
    ],
    ascending=[
        False,
        True
    ]
)

top_candidates.drop(
    columns=["Primary_Order"],
    inplace=True
)

top_candidates.to_csv(
    TABLE2,
    index=False
)

print(
    f"Table_S14_2 created:\n{TABLE2}"
)


# ============================================================================
# TABLE 3 — STRUCTURAL SUMMARY
# ============================================================================

structural_summary_columns = [
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
    "Global_CA_RMSD_A",
    "Global_Backbone_RMSD_A",
    "Local_CA_RMSD_A",
    "WT_Contact_Count",
    "Mutant_Contact_Count",
    "Contacts_Retained",
    "Contacts_Lost",
    "Contacts_Gained",
    "WT_Polar_Contact_Count",
    "Mutant_Polar_Contact_Count",
    "Polar_Contact_Change",
    "Candidate_pLDDT",
    "WT_Catalytic_Distance_A",
    "Mutant_Catalytic_Distance_A",
    "Catalytic_Distance_Change_A",
    "Physicochemical_Impact_Score",
    "Structural_Impact_Index",
    "Step13_Structural_Rank",
]

structural_summary = final_matrix[
    final_matrix["Protein_Change"].isin(
        EXPECTED_STRUCTURAL_VARIANTS
    )
][
    [
        c for c in structural_summary_columns
        if c in final_matrix.columns
    ]
].copy()

structural_summary = structural_summary.sort_values(
    "Step13_Structural_Rank"
)

structural_summary.to_csv(
    TABLE3,
    index=False
)

print(
    f"Table_S14_3 created:\n{TABLE3}"
)


# ============================================================================
# TABLE 4 — DOWNSTREAM PRIORITIZATION
# ============================================================================

downstream_columns = [
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Candidate_Group",
    "AlphaMissense",
    "AlphaMissense_Classification",
    "CADD_PHRED",
    "CADD_Interpretation",
    "Maximum_gnomAD_AF",
    "Population_Rarity_Category",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
    "Structural_Relevance_Score",
    "Structural_Relevance_Category",
    "Structural_Impact_Index",
    "Step13_Structural_Rank",
    "Final_Integrated_Evidence_Category",
    "Downstream_Recommendation",
]

downstream = final_matrix[
    final_matrix["Protein_Change"].isin(
        EXPECTED_STRUCTURAL_VARIANTS
    )
][
    [
        c for c in downstream_columns
        if c in final_matrix.columns
    ]
].copy()

downstream["Primary_Order"] = (
    downstream["Protein_Change"]
    == PRIMARY_VARIANT
).astype(int)

downstream = downstream.sort_values(
    [
        "Primary_Order",
        "Step13_Structural_Rank"
    ],
    ascending=[
        False,
        True
    ]
)

downstream.drop(
    columns=["Primary_Order"],
    inplace=True
)

downstream.to_csv(
    TABLE4,
    index=False
)

print(
    f"Table_S14_4 created:\n{TABLE4}"
)


# ============================================================================
# FIGURE 1 — COMPUTATIONAL VS STRUCTURAL EVIDENCE
# ============================================================================

print("\n" + "-" * 78)
print("GENERATING FIGURES")
print("-" * 78)

plot_df = final_matrix[
    final_matrix["Protein_Change"].isin(
        EXPECTED_STRUCTURAL_VARIANTS
    )
].copy()

plot_df = plot_df.sort_values(
    "Step13_Structural_Rank"
)

x = np.arange(len(plot_df))

computational_score = pd.to_numeric(
    plot_df["Integrated_Priority_Score"],
    errors="coerce"
)

structural_score = pd.to_numeric(
    plot_df["Structural_Impact_Index"],
    errors="coerce"
)

fig, ax = plt.subplots(
    figsize=(12, 7)
)

width = 0.36

ax.bar(
    x - width / 2,
    computational_score,
    width,
    label="Step 9F Integrated Priority"
)

ax.bar(
    x + width / 2,
    structural_score,
    width,
    label="Step 13 Structural Impact"
)

ax.set_xticks(x)

ax.set_xticklabels(
    plot_df["Protein_Change"],
    rotation=45,
    ha="right"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "Integrated Computational and Structural Evidence"
)

ax.legend()

plt.tight_layout()

fig.savefig(
    FIG1,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================================
# FIGURE 2 — COMPUTATIONAL VS STRUCTURAL PRIORITY
# ============================================================================

fig, ax = plt.subplots(
    figsize=(9, 7)
)

x_values = pd.to_numeric(
    plot_df["Integrated_Priority_Score"],
    errors="coerce"
)

y_values = pd.to_numeric(
    plot_df["Structural_Impact_Index"],
    errors="coerce"
)

ax.scatter(
    x_values,
    y_values,
    s=80
)

for _, row in plot_df.iterrows():

    xv = pd.to_numeric(
        row["Integrated_Priority_Score"],
        errors="coerce"
    )

    yv = pd.to_numeric(
        row["Structural_Impact_Index"],
        errors="coerce"
    )

    if pd.notna(xv) and pd.notna(yv):

        ax.annotate(
            row["Protein_Change"],
            (xv, yv),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )

ax.set_xlabel(
    "Step 9F Integrated Priority Score"
)

ax.set_ylabel(
    "Step 13 Structural Impact Index"
)

ax.set_title(
    "Computational Priority versus Structural Impact"
)

plt.tight_layout()

fig.savefig(
    FIG2,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================================
# FIGURE 3 — D166V EVIDENCE PROFILE
# ============================================================================

primary = final_matrix[
    final_matrix["Protein_Change"]
    == PRIMARY_VARIANT
].iloc[0]

profile_labels = [
    "AlphaMissense",
    "CADD PHRED",
    "Step 9F Priority",
    "Structural Relevance",
    "Step 13 Impact",
]

profile_values = [
    float(primary["AlphaMissense"]),
    float(primary["CADD_PHRED"]),
    float(primary["Integrated_Priority_Score"]),
    float(primary["Structural_Relevance_Score"]),
    float(primary["Structural_Impact_Index"]),
]

fig, ax = plt.subplots(
    figsize=(10, 6)
)

positions = np.arange(
    len(profile_labels)
)

ax.bar(
    positions,
    profile_values
)

ax.set_xticks(
    positions
)

ax.set_xticklabels(
    profile_labels,
    rotation=25,
    ha="right"
)

ax.set_ylabel(
    "Score"
)

ax.set_title(
    "p.Asp166Val Integrated Evidence Profile"
)

plt.tight_layout()

fig.savefig(
    FIG3,
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================================
# FINAL QC
# ============================================================================

print("\n" + "-" * 78)
print("FINAL STEP 14 QUALITY CONTROL")
print("-" * 78)

qc_print(
    "Final_matrix_created",
    TABLE1.exists(),
    str(TABLE1)
)

qc_print(
    "Top_candidate_table_created",
    TABLE2.exists(),
    str(TABLE2)
)

qc_print(
    "Structural_summary_created",
    TABLE3.exists(),
    str(TABLE3)
)

qc_print(
    "Downstream_prioritization_created",
    TABLE4.exists(),
    str(TABLE4)
)

qc_print(
    "Figure_1_created",
    FIG1.exists(),
    str(FIG1)
)

qc_print(
    "Figure_2_created",
    FIG2.exists(),
    str(FIG2)
)

qc_print(
    "Figure_3_created",
    FIG3.exists(),
    str(FIG3)
)


# ============================================================================
# READBACK
# ============================================================================

final_readback = pd.read_csv(
    TABLE1
)

qc_print(
    "Final_matrix_82_rows",
    len(final_readback) == 82,
    f"Expected 82, observed {len(final_readback)}"
)

structural_readback = final_readback[
    final_readback["Protein_Change"].isin(
        EXPECTED_STRUCTURAL_VARIANTS
    )
]

qc_print(
    "Final_matrix_9_structural_candidates",
    len(structural_readback) == 9,
    f"Expected 9, observed {len(structural_readback)}"
)


d166v_check = final_readback[
    final_readback["Protein_Change"]
    == PRIMARY_VARIANT
]

qc_print(
    "Final_D166V_present",
    len(d166v_check) == 1,
    "D166V present exactly once"
)


impact_complete = (
    structural_readback[
        "Structural_Impact_Index"
    ]
    .notna()
    .sum()
    == 9
)

qc_print(
    "All_9_structural_impacts_present",
    impact_complete,
    "All 9 structural impact values present"
    if impact_complete
    else "One or more structural impact values missing"
)


rank_complete = (
    structural_readback[
        "Step13_Structural_Rank"
    ]
    .notna()
    .sum()
    == 9
)

qc_print(
    "All_9_structural_ranks_present",
    rank_complete,
    "All 9 structural ranks present"
    if rank_complete
    else "One or more structural ranks missing"
)


# ============================================================================
# VERIFY EXPECTED STRUCTURAL VALUES
# ============================================================================

expected_step13_impacts = {
    "p.Arg247Thr": 16.787,
    "p.Tyr21Cys": 13.024,
    "p.Asp166Val": 13.022,
    "p.Pro186His": 11.816,
    "p.Gly49Trp": 10.193,
    "p.Pro186Ser": 7.068,
    "p.Leu51Ser": 7.023,
    "p.Cys28Arg": 7.018,
    "p.Leu142Phe": 5.716,
}

impact_values_correct = True

for variant, expected_value in expected_step13_impacts.items():

    rows = structural_readback[
        structural_readback["Protein_Change"]
        == variant
    ]

    if len(rows) != 1:
        impact_values_correct = False
        continue

    observed_value = float(
        rows.iloc[0]["Structural_Impact_Index"]
    )

    if not np.isclose(
        observed_value,
        expected_value,
        atol=0.001
    ):
        impact_values_correct = False

qc_print(
    "Step13_structural_values_preserved",
    impact_values_correct,
    "All 9 Step 13 structural-impact values preserved"
    if impact_values_correct
    else "One or more Step 13 structural-impact values changed"
)


# ============================================================================
# VERIFY D166V SPECIFIC VALUES
# ============================================================================

d166v = structural_readback[
    structural_readback["Protein_Change"]
    == PRIMARY_VARIANT
].iloc[0]

d166v_checks = {

    "D166V_Global_CA_RMSD":
        np.isclose(
            float(d166v["Global_CA_RMSD_A"]),
            0.0010,
            atol=0.0001
        ),

    "D166V_Local_CA_RMSD":
        np.isclose(
            float(d166v["Local_CA_RMSD_A"]),
            0.0045,
            atol=0.0001
        ),

    "D166V_WT_polar_contacts":
        int(d166v["WT_Polar_Contact_Count"]) == 13,

    "D166V_mutant_polar_contacts":
        int(d166v["Mutant_Polar_Contact_Count"]) == 9,

    "D166V_structural_impact":
        np.isclose(
            float(d166v["Structural_Impact_Index"]),
            13.022,
            atol=0.001
        ),
}

for name, status in d166v_checks.items():

    qc_print(
        name,
        status
    )


# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "-" * 78)
print("GENERATING FINAL STEP 14 SUMMARY")
print("-" * 78)

summary_lines = []

summary_lines.append(
    "STEP 14 — FINAL VARIANT EVIDENCE MATRIX"
)

summary_lines.append(
    "=" * 60
)

summary_lines.append("")

summary_lines.append(
    f"Total variants in Step 9F: {len(df9)}"
)

summary_lines.append(
    f"Structural candidates integrated: "
    f"{len(structural_readback)}"
)

summary_lines.append("")

summary_lines.append(
    "Primary downstream candidate:"
)

summary_lines.append(
    "  p.Asp166Val"
)

summary_lines.append(
    f"  VariationID: {PRIMARY_VARIATION_ID}"
)

summary_lines.append(
    "  Protein position: 166"
)

summary_lines.append(
    "  Reference residue: Asp"
)

summary_lines.append(
    "  Alternate residue: Val"
)

summary_lines.append(
    f"  AlphaMissense: {primary['AlphaMissense']}"
)

summary_lines.append(
    f"  CADD PHRED: {primary['CADD_PHRED']}"
)

summary_lines.append(
    f"  Maximum gnomAD AF: "
    f"{primary['Maximum_gnomAD_AF']}"
)

summary_lines.append(
    f"  Step 9F integrated priority score: "
    f"{primary['Integrated_Priority_Score']}"
)

summary_lines.append(
    f"  Step 9F structural relevance score: "
    f"{primary['Structural_Relevance_Score']}"
)

summary_lines.append(
    f"  Step 13 structural impact: "
    f"{primary['Structural_Impact_Index']}"
)

summary_lines.append("")

summary_lines.append(
    "Step 13 comparative structural ranking:"
)

ranking_for_summary = structural_readback.sort_values(
    "Step13_Structural_Rank"
)

for _, row in ranking_for_summary.iterrows():

    rank_value = pd.to_numeric(
        row["Step13_Structural_Rank"],
        errors="coerce"
    )

    impact_value = pd.to_numeric(
        row["Structural_Impact_Index"],
        errors="coerce"
    )

    if pd.notna(rank_value) and pd.notna(impact_value):

        summary_lines.append(
            f"  Rank {int(rank_value)}: "
            f"{row['Protein_Change']} "
            f"(Impact={impact_value:.3f})"
        )


summary_lines.append("")

summary_lines.append(
    "Scientific interpretation:"
)

summary_lines.append(
    "Step 9F remains the authoritative computational and "
    "clinical prioritization layer."
)

summary_lines.append(
    "Step 13 provides a comparative structural-impact layer "
    "and is not a pathogenicity probability."
)

summary_lines.append(
    "p.Asp166Val remains the primary downstream candidate "
    "because its integrated computational evidence, rarity, "
    "structural relevance, catalytic-context location, and "
    "local physicochemical alteration jointly support "
    "continued mechanistic investigation."
)

summary_lines.append(
    "The Step 13 structural ranking is retained as a separate "
    "comparative structural evidence layer."
)

summary_lines.append("")

summary_lines.append(
    "Important limitation:"
)

summary_lines.append(
    "Step 13 mutant structures were generated by side-chain "
    "remodeling on the validated WT AlphaFold backbone."
)

summary_lines.append(
    "No energy minimization or molecular dynamics was performed "
    "in Step 13."
)

summary_lines.append(
    "Candidate polar contacts are heavy-atom proximity measurements "
    "and should not be described as experimentally confirmed "
    "hydrogen bonds."
)

summary_lines.append("")

summary_lines.append(
    "Recommended next stage:"
)

summary_lines.append(
    "Step 15 — Therapeutic Target / Compound Prioritization."
)

summary_lines.append(
    "D166V should remain the primary mechanistic variant for "
    "downstream ligand-binding, docking, molecular dynamics, "
    "and mechanistic analysis."
)

SUMMARY_FILE.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


# ============================================================================
# FINAL QC STATUS
# ============================================================================

overall_pass = all(
    item["status"]
    for item in qc.values()
)

qc_report = {
    "step": 14,
    "step_name": "Final Variant Evidence Matrix",
    "overall_status": (
        "PASS"
        if overall_pass
        else "FAIL"
    ),
    "project_root": str(PROJECT_ROOT),
    "step9f_rows": int(len(df9)),
    "structural_candidates": int(
        len(structural_readback)
    ),
    "primary_variant": PRIMARY_VARIANT,
    "primary_variation_id": PRIMARY_VARIATION_ID,
    "qc_checks": qc,
    "outputs": {
        "table_1": str(TABLE1),
        "table_2": str(TABLE2),
        "table_3": str(TABLE3),
        "table_4": str(TABLE4),
        "figure_1": str(FIG1),
        "figure_2": str(FIG2),
        "figure_3": str(FIG3),
        "summary": str(SUMMARY_FILE),
    },
}

QC_FILE.write_text(
    json.dumps(
        qc_report,
        indent=4,
        default=str
    ),
    encoding="utf-8"
)


# ============================================================================
# FINAL OUTPUT
# ============================================================================

print("\n" + "=" * 78)
print("STEP 14 COMPLETE")
print("=" * 78)

print(
    f"\nFINAL STEP 14 QC: "
    f"{'PASS' if overall_pass else 'FAIL'}"
)

print(
    "\nFinal integrated dataset:"
)

print(
    f"    {TABLE1}"
)

print(
    "\nPrimary downstream candidate:"
)

print(
    "    p.Asp166Val"
)

print(
    "    VariationID=3308171"
)

print(
    "\nStep 13 structural ranking preserved:"
)

for _, row in ranking_for_summary.iterrows():

    rank_value = int(
        row["Step13_Structural_Rank"]
    )

    impact_value = float(
        row["Structural_Impact_Index"]
    )

    print(
        f"    Rank {rank_value}: "
        f"{row['Protein_Change']} "
        f"(Impact={impact_value:.3f})"
    )

print(
    "\nQC report:"
)

print(
    f"    {QC_FILE}"
)

print(
    "\nFinal summary:"
)

print(
    f"    {SUMMARY_FILE}"
)

print(
    "\nStep 14 outputs are ready for downstream therapeutic analysis."
)

print(
    "\n" + "=" * 78
)