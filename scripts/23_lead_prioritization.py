from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# STEP 23 — LEAD PRIORITIZATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# VERIFIED INPUT FILES
STEP19_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP19_DRUG_LIKENESS_FILTERING"
    / "tables"
    / "STEP19_MASTER_DRUG_LIKENESS_TABLE.csv"
)

STEP20_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP20_PHARMACOPHORE_SHAPE_SCREENING"
    / "tables"
    / "STEP20_PHARMACOPHORE_SHAPE_SCREENING.csv"
)

STEP21_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
    / "tables"
    / "STEP21_BEST_DOCKING_RESULTS.csv"
)

STEP22_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
    / "tables"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING.csv"
)

# OUTPUT
OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP23_LEAD_PRIORITIZATION"
)

TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
QC_DIR = OUT_DIR / "qc"

for folder in [OUT_DIR, TABLE_DIR, FIGURE_DIR, QC_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HELPERS
# =============================================================================

def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} file not found:\n{path}"
        )


def require_columns(df, columns, label):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            f"{label} is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n{list(df.columns)}"
        )


def minmax(series, reverse=False):
    x = pd.to_numeric(series, errors="coerce")

    lo = x.min()
    hi = x.max()

    if pd.isna(lo) or pd.isna(hi):
        return pd.Series(np.nan, index=x.index)

    if np.isclose(lo, hi):
        result = pd.Series(1.0, index=x.index)
    else:
        result = (x - lo) / (hi - lo)

    if reverse:
        result = 1.0 - result

    return result


# =============================================================================
# HEADER
# =============================================================================

print()
print("=" * 78)
print("STEP 23 — LEAD PRIORITIZATION")
print("=" * 78)
print(f"Project root : {PROJECT_ROOT}")
print()


# =============================================================================
# VERIFY INPUT FILES
# =============================================================================

require_file(STEP19_FILE, "Step 19")
require_file(STEP20_FILE, "Step 20")
require_file(STEP21_FILE, "Step 21")
require_file(STEP22_FILE, "Step 22")

print("Verified input files:")
print(f"  Step 19 : {STEP19_FILE}")
print(f"  Step 20 : {STEP20_FILE}")
print(f"  Step 21 : {STEP21_FILE}")
print(f"  Step 22 : {STEP22_FILE}")
print()


# =============================================================================
# READ INPUTS
# =============================================================================

df19 = pd.read_csv(STEP19_FILE)
df20 = pd.read_csv(STEP20_FILE)
df21 = pd.read_csv(STEP21_FILE)
df22 = pd.read_csv(STEP22_FILE)

print(f"Step 19 rows : {len(df19)}")
print(f"Step 20 rows : {len(df20)}")
print(f"Step 21 rows : {len(df21)}")
print(f"Step 22 rows : {len(df22)}")


# =============================================================================
# VERIFY EXACT COLUMN STRUCTURES
# =============================================================================

require_columns(
    df19,
    [
        "CID",
        "IUPACName",
        "MolecularFormula",
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "Drug_Likeness_Class",
        "Retain_for_Step20",
        "Chemical_Filter_Score",
    ],
    "Step 19"
)

require_columns(
    df20,
    [
        "CID",
        "Step20_Class",
        "Combined_Screening_Score",
        "Matched_Features",
        "Ligand_Features",
        "Feature_Match_Percent",
        "Feature_Types_Matched",
        "Shape_Compatibility",
        "Shape_Score",
        "Volume_Ratio",
    ],
    "Step 20"
)

require_columns(
    df21,
    [
        "CID",
    ],
    "Step 21"
)

require_columns(
    df22,
    [
        "CID",
        "D166V_Best_Vina_Score_kcal_mol",
        "WT_Best_Vina_Score_kcal_mol",
        "Delta_Vina_D166V_minus_WT_kcal_mol",
    ],
    "Step 22"
)

print("Column validation : PASS")


# =============================================================================
# STEP 19
# =============================================================================

d19 = df19[
    [
        "CID",
        "IUPACName",
        "MolecularFormula",
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "Drug_Likeness_Class",
        "Retain_for_Step20",
        "Chemical_Filter_Score",
    ]
].copy()

d19.rename(
    columns={
        "IUPACName": "IUPAC_Name",
        "MolecularFormula": "Molecular_Formula",
        "MolecularWeight": "MW",
        "HBondDonorCount": "HBD",
        "HBondAcceptorCount": "HBA",
        "RotatableBondCount": "RotB",
    },
    inplace=True
)


# =============================================================================
# STEP 20
# =============================================================================
#
# VERIFIED:
# Combined_Screening_Score
# Step20_Class
# =============================================================================

d20 = df20[
    [
        "CID",
        "Step20_Class",
        "Combined_Screening_Score",
        "Matched_Features",
        "Ligand_Features",
        "Feature_Match_Percent",
        "Feature_Types_Matched",
        "Shape_Compatibility",
        "Shape_Score",
        "Volume_Ratio",
        "Ligand_Volume_Estimate",
        "Conformer_ID",
    ]
].copy()

d20.rename(
    columns={
        "Combined_Screening_Score": "Step20_Score"
    },
    inplace=True
)


# =============================================================================
# STEP 21
# =============================================================================
#
# We need to identify the actual docking-score column from the verified
# Step 21 table header.
# =============================================================================

step21_vina_candidates = [
    "Best_Vina_Score_kcal_mol",
    "Best_Vina_Score",
    "Best_Docking_Score_kcal_mol",
    "Docking_Score",
]

step21_vina = None

for col in step21_vina_candidates:
    if col in df21.columns:
        step21_vina = col
        break

if step21_vina is None:
    raise KeyError(
        "Could not identify the Step 21 docking score column.\n\n"
        f"Available Step 21 columns:\n{list(df21.columns)}"
    )

d21 = df21[
    [
        "CID",
        step21_vina
    ]
].copy()

d21.rename(
    columns={
        step21_vina: "Step21_D166V_Vina"
    },
    inplace=True
)


# =============================================================================
# STEP 22
# =============================================================================

d22 = df22[
    [
        "CID",
        "D166V_Best_Vina_Score_kcal_mol",
        "WT_Best_Vina_Score_kcal_mol",
        "Delta_Vina_D166V_minus_WT_kcal_mol",
    ]
].copy()

d22.rename(
    columns={
        "D166V_Best_Vina_Score_kcal_mol": "D166V_Vina",
        "WT_Best_Vina_Score_kcal_mol": "WT_Vina",
        "Delta_Vina_D166V_minus_WT_kcal_mol":
            "Delta_D166V_minus_WT",
    },
    inplace=True
)

if "Differential_Docking_Class" in df22.columns:
    d22["Differential_Docking_Class"] = (
        df22["Differential_Docking_Class"]
    )


# =============================================================================
# CID CLEANUP
# =============================================================================

for dataframe in [d19, d20, d21, d22]:

    dataframe["CID"] = pd.to_numeric(
        dataframe["CID"],
        errors="coerce"
    )

    dataframe.dropna(
        subset=["CID"],
        inplace=True
    )

    dataframe["CID"] = (
        dataframe["CID"]
        .astype(int)
    )


# =============================================================================
# MERGE
# =============================================================================

df = (
    d19
    .merge(d20, on="CID", how="inner")
    .merge(d21, on="CID", how="inner")
    .merge(d22, on="CID", how="inner")
)

print()
print(f"Integrated compounds : {len(df)}")

if len(df) != 10:
    print(
        "WARNING: Integrated compound count is not 10."
    )


if len(df) == 0:
    raise RuntimeError(
        "Step 23 integration produced zero compounds."
    )


# =============================================================================
# NUMERIC CONVERSION
# =============================================================================

numeric_columns = [
    "MW",
    "XLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",
    "Chemical_Filter_Score",
    "Step20_Score",
    "Step21_D166V_Vina",
    "D166V_Vina",
    "WT_Vina",
    "Delta_D166V_minus_WT",
]

for column in numeric_columns:

    if column in df.columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


# =============================================================================
# DOCKING CONSISTENCY
# =============================================================================

df["Docking_Consistency"] = np.isclose(
    df["Step21_D166V_Vina"],
    df["D166V_Vina"],
    atol=0.001,
    equal_nan=False
)


# =============================================================================
# 1. DOCKING COMPONENT
# =============================================================================
#
# More negative Vina score = more favorable.
#
# Relative normalization only.
# =============================================================================

df["Docking_Normalized"] = minmax(
    df["D166V_Vina"],
    reverse=True
)


# =============================================================================
# 2. STEP 20 COMPONENT
# =============================================================================
#
# VERIFIED SOURCE:
# Combined_Screening_Score
#
# Higher = better.
# =============================================================================

df["Pharmacophore_Normalized"] = minmax(
    df["Step20_Score"],
    reverse=False
)


# =============================================================================
# 3. D166V PREFERENCE COMPONENT
# =============================================================================
#
# Delta = D166V - WT
#
# Negative:
# D166V has the more favorable Vina score.
#
# Positive:
# WT has the more favorable Vina score.
#
# The contribution is deliberately modest.
# =============================================================================

df["Mutation_Preference_Normalized"] = np.clip(
    0.5
    - (
        df["Delta_D166V_minus_WT"]
        / 4.0
    ),
    0.0,
    1.0
)


# =============================================================================
# 4. DRUG-LIKENESS COMPONENT
# =============================================================================
#
# Step 19 formally filtered the compounds.
#
# This additional score is only a relative quality component.
# =============================================================================

drug_score = pd.Series(
    1.0,
    index=df.index
)

drug_score -= (
    np.clip(
        (df["MW"] - 400.0) / 250.0,
        0,
        1
    )
    * 0.18
)

drug_score -= (
    np.clip(
        (df["XLogP"] - 3.0) / 4.0,
        0,
        1
    )
    * 0.15
)

drug_score -= (
    np.clip(
        (df["TPSA"] - 100.0) / 100.0,
        0,
        1
    )
    * 0.15
)

drug_score -= (
    np.clip(
        (df["HBA"] - 8.0) / 8.0,
        0,
        1
    )
    * 0.15
)

drug_score -= (
    np.clip(
        (df["RotB"] - 6.0) / 8.0,
        0,
        1
    )
    * 0.15
)

drug_score -= (
    np.clip(
        (df["HBD"] - 3.0) / 5.0,
        0,
        1
    )
    * 0.10
)

drug_score = np.clip(
    drug_score,
    0,
    1
)

chemical_component = (
    df["Chemical_Filter_Score"]
    .clip(
        lower=0,
        upper=100
    )
    / 100.0
)

df["Drug_Likeness_Normalized"] = (
    0.90 * drug_score
    + 0.10 * chemical_component
)

df["Drug_Likeness_Normalized"] = np.clip(
    df["Drug_Likeness_Normalized"],
    0,
    1
)


# =============================================================================
# INTEGRATED LEAD-PRIORITY SCORE
# =============================================================================
#
# Docking                         40%
# Step 20 screening               25%
# D166V-vs-WT preference          15%
# Drug-likeness                   20%
# =============================================================================

df["Lead_Priority_Score"] = (
    0.40 * df["Docking_Normalized"]
    + 0.25 * df["Pharmacophore_Normalized"]
    + 0.15 * df["Mutation_Preference_Normalized"]
    + 0.20 * df["Drug_Likeness_Normalized"]
)

df["Lead_Priority_Percent"] = (
    df["Lead_Priority_Score"]
    * 100.0
)


# =============================================================================
# LEAD CLASS
# =============================================================================

def classify_lead(score):

    if score >= 0.70:
        return "TOP_LEAD"

    if score >= 0.55:
        return "PRIORITY_LEAD"

    return "SECONDARY"


df["Lead_Class"] = (
    df["Lead_Priority_Score"]
    .apply(classify_lead)
)


# =============================================================================
# DOWNSTREAM TIER
# =============================================================================

def downstream_tier(lead_class):

    if lead_class == "TOP_LEAD":
        return "MD_AND_INTERACTION_ANALYSIS"

    if lead_class == "PRIORITY_LEAD":
        return "INTERACTION_ANALYSIS"

    return "BACKUP"


df["Downstream_Tier"] = (
    df["Lead_Class"]
    .apply(downstream_tier)
)


# =============================================================================
# FINAL RANKING
# =============================================================================

df.sort_values(
    by=[
        "Lead_Priority_Score",
        "D166V_Vina",
    ],
    ascending=[
        False,
        True,
    ],
    inplace=True
)

df.reset_index(
    drop=True,
    inplace=True
)

df["Lead_Rank"] = (
    np.arange(
        1,
        len(df) + 1
    )
)


# =============================================================================
# OUTPUT COLUMN ORDER
# =============================================================================

column_order = [
    "Lead_Rank",
    "CID",
    "Lead_Class",
    "Downstream_Tier",
    "Lead_Priority_Score",
    "Lead_Priority_Percent",

    "IUPAC_Name",
    "Molecular_Formula",

    "MW",
    "XLogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotB",

    "Drug_Likeness_Class",
    "Chemical_Filter_Score",

    "Step20_Class",
    "Step20_Score",
    "Matched_Features",
    "Ligand_Features",
    "Feature_Match_Percent",
    "Feature_Types_Matched",
    "Shape_Compatibility",
    "Shape_Score",
    "Volume_Ratio",

    "Step21_D166V_Vina",

    "D166V_Vina",
    "WT_Vina",
    "Delta_D166V_minus_WT",

    "Differential_Docking_Class",

    "Docking_Normalized",
    "Pharmacophore_Normalized",
    "Mutation_Preference_Normalized",
    "Drug_Likeness_Normalized",

    "Docking_Consistency",
]

column_order = [
    c for c in column_order
    if c in df.columns
]

remaining = [
    c
    for c in df.columns
    if c not in column_order
]

final_df = df[
    column_order + remaining
]


# =============================================================================
# SAVE MAIN TABLE
# =============================================================================

main_table = (
    TABLE_DIR
    / "STEP23_LEAD_PRIORITIZATION.csv"
)

final_df.to_csv(
    main_table,
    index=False
)


# =============================================================================
# TOP FIVE
# =============================================================================

top5_table = (
    TABLE_DIR
    / "STEP23_TOP_LEADS.csv"
)

final_df.head(5).to_csv(
    top5_table,
    index=False
)


# =============================================================================
# COMPONENT TABLE
# =============================================================================

component_table = (
    TABLE_DIR
    / "STEP23_EVIDENCE_COMPONENT_SCORES.csv"
)

final_df[
    [
        "CID",
        "Lead_Rank",
        "Lead_Priority_Percent",
        "Docking_Normalized",
        "Pharmacophore_Normalized",
        "Mutation_Preference_Normalized",
        "Drug_Likeness_Normalized",
    ]
].to_csv(
    component_table,
    index=False
)


# =============================================================================
# FIGURE 23.01 — INTEGRATED LEAD RANKING
# =============================================================================

plot_df = final_df.sort_values(
    "Lead_Priority_Percent",
    ascending=True
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    plot_df["CID"].astype(str),
    plot_df["Lead_Priority_Percent"]
)

plt.xlabel(
    "Integrated lead-priority score (%)"
)

plt.ylabel(
    "PubChem CID"
)

plt.title(
    "Step 23 — Integrated Lead Prioritization"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_23_01_Lead_Prioritization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# FIGURE 23.02 — STEP 20 VS D166V DOCKING
# =============================================================================

plt.figure(
    figsize=(8, 6)
)

plt.scatter(
    final_df["Step20_Score"],
    final_df["D166V_Vina"],
    s=80
)

for _, row in final_df.iterrows():

    plt.annotate(
        str(int(row["CID"])),
        (
            row["Step20_Score"],
            row["D166V_Vina"]
        ),
        xytext=(4, 4),
        textcoords="offset points",
        fontsize=8
    )

plt.xlabel(
    "Step 20 combined screening score"
)

plt.ylabel(
    "D166V Vina score (kcal/mol)"
)

plt.title(
    "Step 23 — Step 20 Screening vs D166V Docking"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_23_02_Step20_vs_Docking.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# FIGURE 23.03 — WT VS D166V
# =============================================================================

x = np.arange(
    len(final_df)
)

width = 0.38

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width / 2,
    final_df["WT_Vina"],
    width,
    label="WT"
)

plt.bar(
    x + width / 2,
    final_df["D166V_Vina"],
    width,
    label="D166V"
)

plt.xticks(
    x,
    final_df["CID"].astype(str),
    rotation=45,
    ha="right"
)

plt.xlabel(
    "PubChem CID"
)

plt.ylabel(
    "Vina score (kcal/mol)"
)

plt.title(
    "Step 23 — WT vs D166V Docking"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_23_03_WT_vs_D166V.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# QC
# =============================================================================

qc = {
    "step": 23,

    "inputs": {
        "step19": str(STEP19_FILE),
        "step20": str(STEP20_FILE),
        "step21": str(STEP21_FILE),
        "step22": str(STEP22_FILE),
    },

    "row_counts": {
        "step19": int(len(df19)),
        "step20": int(len(df20)),
        "step21": int(len(df21)),
        "step22": int(len(df22)),
        "integrated": int(len(final_df)),
    },

    "lead_classes": {
        "TOP_LEAD": int(
            (
                final_df["Lead_Class"]
                == "TOP_LEAD"
            ).sum()
        ),
        "PRIORITY_LEAD": int(
            (
                final_df["Lead_Class"]
                == "PRIORITY_LEAD"
            ).sum()
        ),
        "SECONDARY": int(
            (
                final_df["Lead_Class"]
                == "SECONDARY"
            ).sum()
        ),
    },

    "docking_consistency": {
        "consistent": int(
            final_df["Docking_Consistency"].sum()
        ),
        "total": int(len(final_df)),
    },

    "weights": {
        "docking": 0.40,
        "step20_screening": 0.25,
        "mutation_preference": 0.15,
        "drug_likeness": 0.20,
    },

    "scientific_caution": (
        "Step 23 generates a computational lead-priority ranking. "
        "The resulting score does not establish experimental binding "
        "affinity, inhibitory activity, selectivity, or efficacy."
    ),
}

with open(
    QC_DIR / "STEP23_QC.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        qc,
        handle,
        indent=2
    )


# =============================================================================
# INTERPRETATION
# =============================================================================

interpretation = [
    "STEP 23 — LEAD PRIORITIZATION",
    "",
    f"Integrated compounds: {len(final_df)}",
    "",
    "Evidence weighting:",
    "Docking = 40%",
    "Step 20 combined screening score = 25%",
    "D166V-vs-WT docking preference = 15%",
    "Drug-likeness = 20%",
    "",
    "Top five computationally prioritized compounds:",
]

for _, row in final_df.head(5).iterrows():

    interpretation.append(
        f"Rank {int(row['Lead_Rank'])}: "
        f"CID {int(row['CID'])} | "
        f"Priority {row['Lead_Priority_Percent']:.2f}% | "
        f"D166V {row['D166V_Vina']:.3f} kcal/mol | "
        f"WT {row['WT_Vina']:.3f} kcal/mol | "
        f"Delta {row['Delta_D166V_minus_WT']:.3f} kcal/mol | "
        f"{row['Lead_Class']}"
    )

interpretation.extend(
    [
        "",
        "Scientific interpretation:",
        "Step 23 integrates the existing computational evidence",
        "to prioritize compounds for downstream interaction analysis",
        "and molecular dynamics.",
        "",
        "The integrated score is a ranking metric only.",
        "It does not establish experimental binding affinity,",
        "inhibitory activity, selectivity, or therapeutic efficacy.",
        "",
        "The D166V-versus-WT docking component is treated as",
        "comparative supporting evidence rather than proof",
        "of mutation-specific pharmacological activity.",
    ]
)

with open(
    QC_DIR / "STEP23_INTERPRETATION.txt",
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "\n".join(interpretation)
    )


# =============================================================================
# FINAL CONSOLE
# =============================================================================

print()
print("=" * 78)
print("STEP 23 COMPLETE")
print("=" * 78)

print(
    f"Integrated compounds : {len(final_df)}"
)

print(
    "TOP_LEAD             : "
    f"{(final_df['Lead_Class'] == 'TOP_LEAD').sum()}"
)

print(
    "PRIORITY_LEAD        : "
    f"{(final_df['Lead_Class'] == 'PRIORITY_LEAD').sum()}"
)

print(
    "SECONDARY            : "
    f"{(final_df['Lead_Class'] == 'SECONDARY').sum()}"
)

print(
    "Docking consistency   : "
    f"{final_df['Docking_Consistency'].sum()}/"
    f"{len(final_df)}"
)

print()
print("Final ranking:")
print(
    final_df[
        [
            "Lead_Rank",
            "CID",
            "Lead_Priority_Percent",
            "D166V_Vina",
            "WT_Vina",
            "Delta_D166V_minus_WT",
            "Lead_Class",
        ]
    ].to_string(index=False)
)

print()
print("Results:")
print(OUT_DIR)

print()
print("Figures created:")
print(
    FIGURE_DIR
    / "Figure_23_01_Lead_Prioritization.png"
)
print(
    FIGURE_DIR
    / "Figure_23_02_Step20_vs_Docking.png"
)
print(
    FIGURE_DIR
    / "Figure_23_03_WT_vs_D166V.png"
)