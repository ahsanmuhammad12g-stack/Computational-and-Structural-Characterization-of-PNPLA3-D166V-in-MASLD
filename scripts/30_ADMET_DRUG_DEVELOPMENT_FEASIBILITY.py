from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# STEP 30 — ADMET-ORIENTED DRUG-DEVELOPMENT FEASIBILITY
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP19_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP19_DRUG_LIKENESS_FILTERING"
    / "tables"
    / "STEP19_MASTER_DRUG_LIKENESS_TABLE.csv"
)

STEP23_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP23_LEAD_PRIORITIZATION"
    / "tables"
    / "STEP23_LEAD_PRIORITIZATION.csv"
)

STEP28_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP28_MUTATION_SPECIFIC_THERAPEUTIC_VULNERABILITY"
    / "tables"
    / "STEP28_FULL_THERAPEUTIC_VULNERABILITY_ANALYSIS.csv"
)

STEP29_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP29_LEAD_VALIDATION_RESCORING"
    / "tables"
    / "STEP29_FULL_LEAD_VALIDATION.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP30_ADMET_DRUG_DEVELOPMENT_FEASIBILITY"
)

TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
QC_DIR = OUT_DIR / "QC"

for folder in [
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# DEVELOPMENT THRESHOLDS
# =============================================================================

# These are screening/developability thresholds, not clinical cutoffs.

MW_SOFT_LIMIT = 500.0
XLOGP_SOFT_LIMIT = 5.0
TPSA_SOFT_LIMIT = 140.0
HBD_SOFT_LIMIT = 5.0
HBA_SOFT_LIMIT = 10.0
ROTB_SOFT_LIMIT = 10.0

# Stronger preferred zone used for relative ranking.
PREFERRED_MW = 450.0
PREFERRED_XLOGP_LOW = -1.0
PREFERRED_XLOGP_HIGH = 4.0
PREFERRED_TPSA_LOW = 40.0
PREFERRED_TPSA_HIGH = 120.0

# Penalty values for physicochemical liabilities.
PENALTY_MW = 20
PENALTY_XLOGP = 20
PENALTY_TPSA = 15
PENALTY_HBD = 10
PENALTY_HBA = 10
PENALTY_ROTB = 10
PENALTY_BORDERLINE = 10


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):
    print()
    print("=" * 78)
    print("STEP 30 FAILED")
    print("=" * 78)
    print(message)
    print()
    sys.exit(1)


def read_csv(path):
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(
            f"Warning: could not read {path}: {exc}"
        )
        return None


def normalize_cid(value):
    try:
        return str(int(float(value)))
    except Exception:
        return str(value).strip()


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


def find_column(df, candidates):

    if df is None:
        return None

    lookup = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()
        if key in lookup:
            return lookup[key]

    return None


def minmax(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    valid = values.dropna()

    if valid.empty:
        return pd.Series(
            0.0,
            index=series.index,
        )

    low = valid.min()
    high = valid.max()

    if np.isclose(low, high):
        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        (values - low)
        /
        (high - low)
    ).fillna(0.0)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_step19():

    df = read_csv(
        STEP19_FILE
    )

    if df is None or df.empty:
        fail(
            "Step 19 master drug-likeness table is missing or empty."
        )

    if "CID" not in df.columns:
        fail(
            "CID column not found in Step 19."
        )

    result = df.copy()

    result["CID"] = (
        result["CID"]
        .map(normalize_cid)
    )

    return result


def load_step23():

    df = read_csv(
        STEP23_FILE
    )

    if df is None or df.empty:
        fail(
            "Step 23 lead-prioritization table is missing or empty."
        )

    if "CID" not in df.columns:
        fail(
            "CID column not found in Step 23."
        )

    result = df.copy()

    result["CID"] = (
        result["CID"]
        .map(normalize_cid)
    )

    return result


def load_optional(path):

    df = read_csv(path)

    if df is None or df.empty:
        return pd.DataFrame()

    cid_column = find_column(
        df,
        [
            "CID",
            "CID_Normalized",
        ],
    )

    if cid_column is None:
        return pd.DataFrame()

    result = df.copy()

    result["CID"] = (
        result[cid_column]
        .map(normalize_cid)
    )

    return result


# =============================================================================
# PROPERTY IDENTIFICATION
# =============================================================================

def property_column(df, candidates):

    column = find_column(
        df,
        candidates,
    )

    if column is None:
        fail(
            f"Could not identify required property column from: "
            f"{candidates}"
        )

    return column


# =============================================================================
# ADMET-ORIENTED SCORING
# =============================================================================

def calculate_feasibility(df):

    mw_col = property_column(
        df,
        [
            "MolecularWeight",
            "Molecular_Weight",
            "MW",
        ],
    )

    logp_col = property_column(
        df,
        [
            "XLogP",
            "XLogP3",
            "LogP",
        ],
    )

    tpsa_col = property_column(
        df,
        [
            "TPSA",
            "TopologicalPolarSurfaceArea",
        ],
    )

    hbd_col = property_column(
        df,
        [
            "HBondDonorCount",
            "HBD",
        ],
    )

    hba_col = property_column(
        df,
        [
            "HBondAcceptorCount",
            "HBA",
        ],
    )

    rotb_col = property_column(
        df,
        [
            "RotatableBondCount",
            "Rotatable_Bonds",
            "RotB",
        ],
    )

    result = pd.DataFrame()

    result["CID"] = (
        df["CID"]
        .astype(str)
    )

    result["MolecularWeight"] = pd.to_numeric(
        df[mw_col],
        errors="coerce",
    )

    result["XLogP"] = pd.to_numeric(
        df[logp_col],
        errors="coerce",
    )

    result["TPSA"] = pd.to_numeric(
        df[tpsa_col],
        errors="coerce",
    )

    result["HBD"] = pd.to_numeric(
        df[hbd_col],
        errors="coerce",
    )

    result["HBA"] = pd.to_numeric(
        df[hba_col],
        errors="coerce",
    )

    result["Rotatable_Bonds"] = pd.to_numeric(
        df[rotb_col],
        errors="coerce",
    )

    # -------------------------------------------------------------------------
    # Base pass/fail criteria
    # -------------------------------------------------------------------------

    result["MW_Pass"] = (
        result["MolecularWeight"]
        <= MW_SOFT_LIMIT
    )

    result["XLogP_Pass"] = (
        result["XLogP"]
        <= XLOGP_SOFT_LIMIT
    )

    result["TPSA_Pass"] = (
        result["TPSA"]
        <= TPSA_SOFT_LIMIT
    )

    result["HBD_Pass"] = (
        result["HBD"]
        <= HBD_SOFT_LIMIT
    )

    result["HBA_Pass"] = (
        result["HBA"]
        <= HBA_SOFT_LIMIT
    )

    result["RotB_Pass"] = (
        result["Rotatable_Bonds"]
        <= ROTB_SOFT_LIMIT
    )

    result["Physicochemical_Pass_Count"] = (
        result[
            [
                "MW_Pass",
                "XLogP_Pass",
                "TPSA_Pass",
                "HBD_Pass",
                "HBA_Pass",
                "RotB_Pass",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )

    # -------------------------------------------------------------------------
    # Preferred-zone descriptors
    # -------------------------------------------------------------------------

    result["MW_Preferred"] = (
        result["MolecularWeight"]
        <= PREFERRED_MW
    )

    result["XLogP_Preferred"] = (
        (
            result["XLogP"]
            >= PREFERRED_XLOGP_LOW
        )
        &
        (
            result["XLogP"]
            <= PREFERRED_XLOGP_HIGH
        )
    )

    result["TPSA_Preferred"] = (
        (
            result["TPSA"]
            >= PREFERRED_TPSA_LOW
        )
        &
        (
            result["TPSA"]
            <= PREFERRED_TPSA_HIGH
        )
    )

    result["HBD_Preferred"] = (
        result["HBD"]
        <= 3
    )

    result["HBA_Preferred"] = (
        result["HBA"]
        <= 8
    )

    result["RotB_Preferred"] = (
        result["Rotatable_Bonds"]
        <= 7
    )

    result["Preferred_Property_Count"] = (
        result[
            [
                "MW_Preferred",
                "XLogP_Preferred",
                "TPSA_Preferred",
                "HBD_Preferred",
                "HBA_Preferred",
                "RotB_Preferred",
            ]
        ]
        .astype(int)
        .sum(axis=1)
    )

    # -------------------------------------------------------------------------
    # Liability score
    # -------------------------------------------------------------------------

    result["ADMET_Liability_Penalty"] = 0.0

    result.loc[
        ~result["MW_Pass"],
        "ADMET_Liability_Penalty",
    ] += PENALTY_MW

    result.loc[
        ~result["XLogP_Pass"],
        "ADMET_Liability_Penalty",
    ] += PENALTY_XLOGP

    result.loc[
        ~result["TPSA_Pass"],
        "ADMET_Liability_Penalty",
    ] += PENALTY_TPSA

    result.loc[
        ~result["HBD_Pass"],
        "ADMET_Liability_Penalty",
    ] += PENALTY_HBD

    result.loc[
        ~result["HBA_Pass"],
        "ADMET_Liability_Penalty",
    ] += PENALTY_HBA

    result.loc[
        ~result["RotB_Pass"],
        "ADMET_Liability_Penalty",
    ] += PENALTY_ROTB

    # -------------------------------------------------------------------------
    # Feasibility score
    # -------------------------------------------------------------------------

    result["Property_Fitness_Score"] = (
        100
        *
        (
            result["Physicochemical_Pass_Count"]
            /
            6.0
        )
    )

    result["Preferred_Property_Score"] = (
        100
        *
        (
            result["Preferred_Property_Count"]
            /
            6.0
        )
    )

    result["ADMET_Development_Score"] = (
        0.70
        * result["Property_Fitness_Score"]
        +
        0.30
        * result["Preferred_Property_Score"]
        -
        result["ADMET_Liability_Penalty"]
    ).clip(
        0,
        100,
    )

    # -------------------------------------------------------------------------
    # Development class
    # -------------------------------------------------------------------------

    classes = []

    for _, row in result.iterrows():

        score = safe_float(
            row[
                "ADMET_Development_Score"
            ]
        )

        pass_count = int(
            row[
                "Physicochemical_Pass_Count"
            ]
        )

        preferred_count = int(
            row[
                "Preferred_Property_Count"
            ]
        )

        if (
            pass_count == 6
            and preferred_count >= 5
            and score >= 80
        ):

            classes.append(
                "HIGH_DEVELOPABILITY"
            )

        elif (
            pass_count >= 5
            and score >= 65
        ):

            classes.append(
                "GOOD_DEVELOPABILITY"
            )

        elif (
            pass_count >= 4
            and score >= 45
        ):

            classes.append(
                "MODERATE_DEVELOPABILITY"
            )

        elif pass_count >= 3:

            classes.append(
                "BORDERLINE_DEVELOPABILITY"
            )

        else:

            classes.append(
                "POOR_DEVELOPABILITY"
            )

    result[
        "Development_Class"
    ] = classes

    return result


# =============================================================================
# MERGE LEAD CONTEXT
# =============================================================================

def merge_context(
    properties,
    step23,
    step28,
    step29,
):

    result = properties.copy()

    # -------------------------------------------------------------------------
    # Step 23
    # -------------------------------------------------------------------------

    score_col = find_column(
        step23,
        [
            "Integrated_Score",
            "Step23_Integrated_Score",
            "Priority_Score",
        ],
    )

    rank_col = find_column(
        step23,
        [
            "Step23_Rank",
            "Rank",
        ],
    )

    class_col = find_column(
        step23,
        [
            "Lead_Class",
            "Priority",
            "Class",
        ],
    )

    keep = ["CID"]

    if score_col:
        keep.append(score_col)

    if rank_col:
        keep.append(rank_col)

    if class_col:
        keep.append(class_col)

    step23_sub = step23[
        keep
    ].copy()

    rename = {}

    if score_col:
        rename[
            score_col
        ] = "Step23_Integrated_Score"

    if rank_col:
        rename[
            rank_col
        ] = "Step23_Rank"

    if class_col:
        rename[
            class_col
        ] = "Step23_Lead_Class"

    step23_sub = step23_sub.rename(
        columns=rename
    )

    result = result.merge(
        step23_sub,
        on="CID",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Step 28
    # -------------------------------------------------------------------------

    if (
        step28 is not None
        and not step28.empty
    ):

        step28_columns = [
            "CID",
            "Ligand_Specific_Vulnerability_Score",
            "Therapeutic_Vulnerability_Score",
            "Therapeutic_Interpretation",
            "Docking_Preference_Class",
        ]

        step28_columns = [
            column
            for column in step28_columns
            if column in step28.columns
        ]

        result = result.merge(
            step28[
                step28_columns
            ],
            on="CID",
            how="left",
            suffixes=(
                "",
                "_Step28",
            ),
        )

    # -------------------------------------------------------------------------
    # Step 29
    # -------------------------------------------------------------------------

    if (
        step29 is not None
        and not step29.empty
    ):

        step29_columns = [
            "CID",
            "Step29_Validation_Score",
            "Step29_Validation_Class",
        ]

        step29_columns = [
            column
            for column in step29_columns
            if column in step29.columns
        ]

        result = result.merge(
            step29[
                step29_columns
            ],
            on="CID",
            how="left",
        )

    return result


# =============================================================================
# FINAL COMPOSITE DEVELOPMENT PRIORITY
# =============================================================================

def calculate_final_priority(df):

    df["Lead_Priority_Normalized"] = minmax(
        df[
            "Step23_Integrated_Score"
        ]
    )

    df[
        "Step28_Vulnerability_Normalized"
    ] = pd.to_numeric(
        df[
            "Ligand_Specific_Vulnerability_Score"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    df[
        "Step29_Validation_Normalized"
    ] = pd.to_numeric(
        df[
            "Step29_Validation_Score"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    df[
        "ADMET_Development_Normalized"
    ] = (
        df[
            "ADMET_Development_Score"
        ]
        /
        100.0
    )

    # Step 30 is deliberately not allowed to dominate the
    # biological evidence already accumulated.
    df[
        "Step30_Development_Priority_Score"
    ] = (
        0.20
        * df[
            "Lead_Priority_Normalized"
        ]
        +
        0.25
        * df[
            "Step28_Vulnerability_Normalized"
        ]
        +
        0.20
        * df[
            "Step29_Validation_Normalized"
        ]
        +
        0.35
        * df[
            "ADMET_Development_Normalized"
        ]
    ) * 100.0

    df = df.sort_values(
        [
            "Step30_Development_Priority_Score",
            "ADMET_Development_Score",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    df[
        "Step30_Rank"
    ] = np.arange(
        1,
        len(df) + 1,
    )

    return df


# =============================================================================
# FIGURE 1 — DEVELOPMENT SCORE
# =============================================================================

def plot_development_score(df):

    plot_df = df.sort_values(
        "Step30_Development_Priority_Score",
        ascending=False,
    )

    labels = (
        "CID "
        + plot_df[
            "CID"
        ].astype(str)
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        plot_df[
            "Step30_Development_Priority_Score"
        ],
    )

    plt.ylabel(
        "Step 30 development-priority score"
    )

    plt.xlabel(
        "Compound"
    )

    plt.title(
        "Step 30 — Development Priority"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()


# =============================================================================
# FIGURE 2 — PROPERTY PROFILE
# =============================================================================

def plot_property_profile(df):

    top = df.head(
        min(
            6,
            len(df),
        )
    ).copy()

    labels = (
        "CID "
        + top[
            "CID"
        ].astype(str)
    )

    values = top[
        "ADMET_Development_Score"
    ]

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "ADMET-oriented development score"
    )

    plt.xlabel(
        "Compound"
    )

    plt.title(
        "Step 30 — Physicochemical Development Feasibility"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 30 — ADMET-ORIENTED DRUG-DEVELOPMENT FEASIBILITY"
    )
    print("=" * 78)

    print(
        f"Step 19 : {STEP19_FILE}"
    )

    print(
        f"Step 23 : {STEP23_FILE}"
    )

    print(
        f"Step 28 : {STEP28_FILE}"
    )

    print(
        f"Step 29 : {STEP29_FILE}"
    )

    print()

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    print(
        "Loading Step 19 physicochemical data..."
    )

    step19 = load_step19()

    print(
        f"Step 19 records: {len(step19)}"
    )

    print(
        "Loading Step 23 lead data..."
    )

    step23 = load_step23()

    print(
        f"Step 23 records: {len(step23)}"
    )

    print(
        "Loading Step 28 vulnerability data..."
    )

    step28 = load_optional(
        STEP28_FILE
    )

    print(
        f"Step 28 records: {len(step28)}"
    )

    print(
        "Loading Step 29 validation data..."
    )

    step29 = load_optional(
        STEP29_FILE
    )

    print(
        f"Step 29 records: {len(step29)}"
    )

    print()

    # -------------------------------------------------------------------------
    # Restrict to Step 23 prioritized compounds
    # -------------------------------------------------------------------------

    prioritized_cids = set(
        step23[
            "CID"
        ].astype(str)
    )

    properties = step19[
        step19[
            "CID"
        ].astype(str)
        .isin(
            prioritized_cids
        )
    ].copy()

    if properties.empty:

        fail(
            "No Step 23 prioritized compounds were found in Step 19."
        )

    print(
        f"Prioritized compounds evaluated: "
        f"{len(properties)}"
    )

    # -------------------------------------------------------------------------
    # Calculate ADMET-oriented metrics
    # -------------------------------------------------------------------------

    print(
        "Calculating physicochemical development metrics..."
    )

    scored = calculate_feasibility(
        properties
    )

    # -------------------------------------------------------------------------
    # Merge biological/lead context
    # -------------------------------------------------------------------------

    print(
        "Integrating Step 23, Step 28 and Step 29 context..."
    )

    integrated = merge_context(
        scored,
        step23,
        step28,
        step29,
    )

    # Missing optional values are explicitly retained as missing/zero.
    for column in [
        "Step23_Integrated_Score",
        "Ligand_Specific_Vulnerability_Score",
        "Step29_Validation_Score",
    ]:

        if column not in integrated.columns:

            integrated[
                column
            ] = 0.0

    integrated = calculate_final_priority(
        integrated
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    class_counts = (
        integrated[
            "Development_Class"
        ]
        .value_counts()
        .rename_axis(
            "Development_Class"
        )
        .reset_index(
            name="Count"
        )
    )

    # -------------------------------------------------------------------------
    # Final interpretation
    # -------------------------------------------------------------------------

    top = integrated.iloc[0]

    high = int(
        (
            integrated[
                "Development_Class"
            ]
            == "HIGH_DEVELOPABILITY"
        ).sum()
    )

    good = int(
        (
            integrated[
                "Development_Class"
            ]
            == "GOOD_DEVELOPABILITY"
        ).sum()
    )

    moderate = int(
        (
            integrated[
                "Development_Class"
            ]
            == "MODERATE_DEVELOPABILITY"
        ).sum()
    )

    borderline = int(
        (
            integrated[
                "Development_Class"
            ]
            == "BORDERLINE_DEVELOPABILITY"
        ).sum()
    )

    poor = int(
        (
            integrated[
                "Development_Class"
            ]
            == "POOR_DEVELOPABILITY"
        ).sum()
    )

    final_statement = (
        f"Step 30 evaluated the physicochemical drug-development "
        f"feasibility of the ten Step 23 prioritized compounds using "
        f"molecular weight, XLogP, TPSA, hydrogen-bonding capacity and "
        f"rotatable-bond count. CID {top['CID']} obtained the highest "
        f"combined Step 30 development-priority score "
        f"({safe_float(top['Step30_Development_Priority_Score']):.2f}/100) "
        f"with an ADMET-oriented physicochemical development score of "
        f"{safe_float(top['ADMET_Development_Score']):.2f}/100. "
        f"The dataset contains {high} high-developability, {good} "
        f"good-developability, {moderate} moderate-developability, "
        f"{borderline} borderline-developability and {poor} "
        f"poor-developability compounds. These are physicochemical "
        f"development classifications rather than experimental ADMET "
        f"predictions. They do not establish absorption, distribution, "
        f"metabolism, excretion, toxicity, pharmacokinetics or clinical "
        f"safety."
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    integrated.to_csv(
        TABLE_DIR
        / "STEP30_FULL_ADMET_DEVELOPMENT_ANALYSIS.csv",
        index=False,
    )

    summary_columns = [
        "Step30_Rank",
        "CID",
        "MolecularWeight",
        "XLogP",
        "TPSA",
        "HBD",
        "HBA",
        "Rotatable_Bonds",
        "Physicochemical_Pass_Count",
        "Preferred_Property_Count",
        "Property_Fitness_Score",
        "Preferred_Property_Score",
        "ADMET_Liability_Penalty",
        "ADMET_Development_Score",
        "Development_Class",
        "Step23_Integrated_Score",
        "Step28_Vulnerability_Normalized",
        "Step29_Validation_Normalized",
        "Step30_Development_Priority_Score",
    ]

    summary_columns = [
        column
        for column in summary_columns
        if column in integrated.columns
    ]

    integrated[
        summary_columns
    ].to_csv(
        TABLE_DIR
        / "STEP30_ADMET_DEVELOPMENT_SUMMARY.csv",
        index=False,
    )

    class_counts.to_csv(
        TABLE_DIR
        / "STEP30_DEVELOPMENT_CLASS_COUNTS.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Input provenance
    # -------------------------------------------------------------------------

    provenance = {
        "Step19_file":
            str(STEP19_FILE),
        "Step23_file":
            str(STEP23_FILE),
        "Step28_file":
            str(STEP28_FILE),
        "Step29_file":
            str(STEP29_FILE),
        "Step29_interaction_consistency_limitation":
            (
                "Step 29 reported Jaccard 0.000 because explicit "
                "Step 24 WT/D166V interaction reconstruction was not "
                "available. This is not interpreted as biological absence "
                "of shared interactions."
            ),
        "External_ADMET_server_used":
            False,
        "Experimental_ADMET_data_used":
            False,
    }

    with open(
        QC_DIR
        / "STEP30_PROVENANCE.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            provenance,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Interpretation file
    # -------------------------------------------------------------------------

    with open(
        QC_DIR
        / "STEP30_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            final_statement
        )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    print(
        "Generating figures..."
    )

    plot_development_score(
        integrated
    )

    plt.savefig(
        FIGURE_DIR
        / "Figure_30_01_Development_Priority.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plot_property_profile(
        integrated
    )

    plt.savefig(
        FIGURE_DIR
        / "Figure_30_02_ADMET_Development_Profile.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc = {
        "Step":
            30,
        "Step19_loaded":
            not step19.empty,
        "Step23_loaded":
            not step23.empty,
        "Step28_loaded":
            not step28.empty,
        "Step29_loaded":
            not step29.empty,
        "Compounds_evaluated":
            len(integrated),
        "High_developability_count":
            high,
        "Good_developability_count":
            good,
        "Moderate_developability_count":
            moderate,
        "Borderline_developability_count":
            borderline,
        "Poor_developability_count":
            poor,
        "External_ADMET_server_used":
            False,
        "Experimental_ADMET_used":
            False,
        "Status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP30_QC.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final console
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STEP 30 RESULTS")
    print("=" * 78)

    print(
        f"Compounds evaluated              : "
        f"{len(integrated)}"
    )

    print(
        f"High developability              : "
        f"{high}"
    )

    print(
        f"Good developability              : "
        f"{good}"
    )

    print(
        f"Moderate developability          : "
        f"{moderate}"
    )

    print(
        f"Borderline developability        : "
        f"{borderline}"
    )

    print(
        f"Poor developability              : "
        f"{poor}"
    )

    print()

    print(
        f"Top Step 30 CID                  : "
        f"{top['CID']}"
    )

    print(
        f"Top ADMET-oriented score         : "
        f"{safe_float(top['ADMET_Development_Score']):.2f}"
    )

    print(
        f"Top Step 30 priority score       : "
        f"{safe_float(top['Step30_Development_Priority_Score']):.2f}"
    )

    print(
        f"Top development class            : "
        f"{top['Development_Class']}"
    )

    print()

    print(
        "Final interpretation:"
    )

    print(
        final_statement
    )

    print()

    print(
        "Step 30 completed successfully."
    )

    print(
        f"Summary table : "
        f"{TABLE_DIR / 'STEP30_ADMET_DEVELOPMENT_SUMMARY.csv'}"
    )

    print(
        f"Full analysis : "
        f"{TABLE_DIR / 'STEP30_FULL_ADMET_DEVELOPMENT_ANALYSIS.csv'}"
    )

    print()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStep 30 interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 30 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise