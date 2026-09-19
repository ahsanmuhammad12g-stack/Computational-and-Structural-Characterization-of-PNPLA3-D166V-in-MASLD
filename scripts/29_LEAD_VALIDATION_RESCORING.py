from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# STEP 29 — LEAD VALIDATION, RESCORING & INTERACTION CONSISTENCY
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP22_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
    / "tables"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING.csv"
)

STEP23_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP23_LEAD_PRIORITIZATION"
    / "tables"
    / "STEP23_LEAD_PRIORITIZATION.csv"
)

STEP24_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP24_INTERACTION_FINGERPRINTING"
    / "tables"
)

STEP28_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP28_MUTATION_SPECIFIC_THERAPEUTIC_VULNERABILITY"
    / "tables"
    / "STEP28_FULL_THERAPEUTIC_VULNERABILITY_ANALYSIS.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP29_LEAD_VALIDATION_RESCORING"
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
# PARAMETERS
# =============================================================================

TARGET_RESIDUE = 166

MUTANT_PREFERENCE_THRESHOLD = -0.50

HIGH_INTERACTION_OVERLAP = 0.70
MODERATE_INTERACTION_OVERLAP = 0.40

WEIGHT_DOCKING = 0.30
WEIGHT_INTERACTION_CONSISTENCY = 0.30
WEIGHT_RESIDUE_166 = 0.20
WEIGHT_LEAD_PRIORITY = 0.20


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):

    print()
    print("=" * 78)
    print("STEP 29 FAILED")
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
        return str(
            int(float(value))
        )

    except Exception:

        return str(
            value
        ).strip()


def safe_float(value):

    try:
        return float(value)

    except Exception:

        return np.nan


def find_column(
    df,
    candidates,
):

    if df is None:
        return None

    normalized = {
        str(column).strip().lower():
            column
        for column in df.columns
    }

    # Exact matching first.
    for candidate in candidates:

        key = str(
            candidate
        ).strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def find_column_contains(
    df,
    candidate_fragments,
):

    if df is None:
        return None

    for column in df.columns:

        lower = str(
            column
        ).lower()

        for fragment in candidate_fragments:

            if fragment.lower() in lower:
                return column

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

    minimum = valid.min()
    maximum = valid.max()

    if np.isclose(
        minimum,
        maximum,
    ):

        return pd.Series(
            1.0,
            index=series.index,
        )

    return (
        (
            values - minimum
        )
        /
        (
            maximum - minimum
        )
    ).fillna(0.0)


def parse_residue_set(value):

    if pd.isna(value):
        return set()

    text = str(
        value
    ).strip()

    if not text:
        return set()

    for separator in [
        ",",
        ";",
        "|",
        "/",
    ]:

        text = text.replace(
            separator,
            " ",
        )

    residues = set()

    for token in text.split():

        try:

            residues.add(
                int(
                    float(
                        token
                    )
                )
            )

        except Exception:

            continue

    return residues


def jaccard(
    set_a,
    set_b,
):

    union = (
        set_a
        |
        set_b
    )

    if not union:
        return 0.0

    return (
        len(
            set_a
            &
            set_b
        )
        /
        len(union)
    )


# =============================================================================
# STEP 22 — EXACT CURRENT SCHEMA
# =============================================================================

def load_step22():

    df = read_csv(
        STEP22_FILE
    )

    if df is None or df.empty:

        fail(
            "Step 22 differential-docking table is missing or empty."
        )

    required = [
        "CID",
        "WT_Best_Vina_Score_kcal_mol",
        "D166V_Best_Vina_Score_kcal_mol",
        "Delta_Vina_D166V_minus_WT_kcal_mol",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        fail(
            "The verified Step 22 schema is missing required "
            f"column(s): {missing}"
        )

    result = pd.DataFrame()

    result["CID"] = (
        df["CID"]
        .map(normalize_cid)
    )

    result["WT_Docking_Score"] = pd.to_numeric(
        df[
            "WT_Best_Vina_Score_kcal_mol"
        ],
        errors="coerce",
    )

    result["D166V_Docking_Score"] = pd.to_numeric(
        df[
            "D166V_Best_Vina_Score_kcal_mol"
        ],
        errors="coerce",
    )

    result["Delta_D166V_minus_WT"] = pd.to_numeric(
        df[
            "Delta_Vina_D166V_minus_WT_kcal_mol"
        ],
        errors="coerce",
    )

    # Carry forward existing Step 22 descriptors.
    for column in [
        "Differential_Rank",
        "Docking_Rank",
        "Step20_Screening_Score",
        "Drug_Likeness_Class",
        "Chemical_Filter_Score",
        "Step19_Rank",
        "Differential_Docking_Class",
        "Docking_Status",
        "WT_Status",
        "D166V_Best_Pose",
        "WT_Best_Pose",
        "WT_Number_of_Poses",
        "Number_of_Poses",
    ]:

        if column in df.columns:

            result[column] = df[
                column
            ]

    return result


# =============================================================================
# STEP 23
# =============================================================================

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

    result = pd.DataFrame()

    result["CID"] = (
        df["CID"]
        .map(normalize_cid)
    )

    score_column = find_column(
        df,
        [
            "Integrated_Score",
            "Step23_Integrated_Score",
            "Priority_Score",
        ],
    )

    rank_column = find_column(
        df,
        [
            "Step23_Rank",
            "Rank",
        ],
    )

    class_column = find_column(
        df,
        [
            "Lead_Class",
            "Priority",
            "Class",
        ],
    )

    if score_column:

        result["Step23_Integrated_Score"] = pd.to_numeric(
            df[
                score_column
            ],
            errors="coerce",
        )

    else:

        result["Step23_Integrated_Score"] = 0.0

    if rank_column:

        result["Step23_Rank"] = pd.to_numeric(
            df[
                rank_column
            ],
            errors="coerce",
        )

    else:

        result["Step23_Rank"] = np.nan

    if class_column:

        result["Step23_Lead_Class"] = (
            df[
                class_column
            ]
            .astype(str)
        )

    else:

        result["Step23_Lead_Class"] = (
            "UNSPECIFIED"
        )

    return result


# =============================================================================
# STEP 28
# =============================================================================

def load_step28():

    df = read_csv(
        STEP28_FILE
    )

    if df is None or df.empty:

        print(
            "Warning: Step 28 table not available."
        )

        return pd.DataFrame(
            columns=[
                "CID",
            ]
        )

    cid_column = find_column(
        df,
        [
            "CID",
            "CID_Normalized",
        ],
    )

    if cid_column is None:

        return pd.DataFrame(
            columns=[
                "CID",
            ]
        )

    result = pd.DataFrame()

    result["CID"] = (
        df[
            cid_column
        ]
        .map(normalize_cid)
    )

    for column in [
        "Ligand_Specific_Vulnerability_Score",
        "Therapeutic_Vulnerability_Score",
        "Therapeutic_Interpretation",
        "Docking_Preference_Class",
        "Residue_166_Interaction_Count",
    ]:

        if column in df.columns:

            result[
                column
            ] = df[
                column
            ]

    return result


# =============================================================================
# STEP 24 — INTERACTION FOOTPRINT EXTRACTION
# =============================================================================

def inspect_step24_files():

    if not STEP24_DIR.exists():

        print(
            "Warning: Step 24 tables directory does not exist."
        )

        return []

    files = sorted(
        STEP24_DIR.glob(
            "*.csv"
        )
    )

    if not files:

        print(
            "Warning: no Step 24 CSV files found."
        )

    return files


def find_cid_column(df):

    return find_column(
        df,
        [
            "CID",
            "Compound_ID",
            "CompoundID",
            "PubChem_CID",
        ],
    )


def find_residue_column(df):

    return find_column(
        df,
        [
            "Residue_Number",
            "Residue_ID",
            "Residue",
            "Interaction_Residue",
            "Target_Residue",
        ],
    )


def find_interaction_type_column(df):

    return find_column(
        df,
        [
            "Interaction_Type",
            "Interaction",
            "Type",
            "Interaction_Class",
        ],
    )


def extract_residue_level_footprints():

    files = inspect_step24_files()

    output = {}

    for path in files:

        df = read_csv(
            path
        )

        if df is None or df.empty:
            continue

        cid_column = find_cid_column(
            df
        )

        residue_column = find_residue_column(
            df
        )

        if (
            cid_column is None
            or residue_column is None
        ):
            continue

        print(
            f"  Using Step 24 residue table: "
            f"{path.name}"
        )

        for _, row in df.iterrows():

            cid = normalize_cid(
                row[
                    cid_column
                ]
            )

            residue_value = row[
                residue_column
            ]

            try:

                residue = int(
                    float(
                        residue_value
                    )
                )

            except Exception:

                continue

            if cid not in output:

                output[cid] = {
                    "all_residues":
                        set(),
                    "residue166":
                        False,
                    "interaction_types":
                        set(),
                    "sources":
                        set(),
                }

            output[cid][
                "all_residues"
            ].add(
                residue
            )

            if residue == TARGET_RESIDUE:

                output[cid][
                    "residue166"
                ] = True

            type_column = (
                find_interaction_type_column(
                    df
                )
            )

            if type_column:

                value = str(
                    row[
                        type_column
                    ]
                )

                if value.lower() != "nan":

                    output[cid][
                        "interaction_types"
                    ].add(
                        value
                    )

            output[cid][
                "sources"
            ].add(
                path.name
            )

    return output


def extract_wt_mutant_fingerprints():

    files = inspect_step24_files()

    output = {}

    wt_candidates = [
        "WT_Interacting_Residues",
        "WT_Residues",
        "WT_Interaction_Residues",
        "WT_Residue_Fingerprint",
    ]

    mutant_candidates = [
        "D166V_Interacting_Residues",
        "D166V_Residues",
        "D166V_Interaction_Residues",
        "D166V_Residue_Fingerprint",
        "Mutant_Residues",
    ]

    for path in files:

        df = read_csv(
            path
        )

        if df is None or df.empty:
            continue

        cid_column = find_cid_column(
            df
        )

        if cid_column is None:
            continue

        wt_column = find_column(
            df,
            wt_candidates,
        )

        mutant_column = find_column(
            df,
            mutant_candidates,
        )

        if (
            wt_column is None
            and mutant_column is None
        ):
            continue

        print(
            f"  Using Step 24 WT/D166V fingerprint table: "
            f"{path.name}"
        )

        for _, row in df.iterrows():

            cid = normalize_cid(
                row[
                    cid_column
                ]
            )

            if cid not in output:

                output[cid] = {
                    "WT":
                        set(),
                    "D166V":
                        set(),
                    "sources":
                        set(),
                }

            if wt_column:

                output[cid][
                    "WT"
                ].update(
                    parse_residue_set(
                        row[
                            wt_column
                        ]
                    )
                )

            if mutant_column:

                output[cid][
                    "D166V"
                ].update(
                    parse_residue_set(
                        row[
                            mutant_column
                        ]
                    )
                )

            output[cid][
                "sources"
            ].add(
                path.name
            )

    return output


def build_interaction_table():

    residue_level = (
        extract_residue_level_footprints()
    )

    fingerprint_level = (
        extract_wt_mutant_fingerprints()
    )

    all_cids = (
        set(
            residue_level
        )
        |
        set(
            fingerprint_level
        )
    )

    rows = []

    for cid in sorted(
        all_cids
    ):

        wt_residues = set()
        mutant_residues = set()
        sources = set()

        residue166_from_table = False

        if cid in fingerprint_level:

            wt_residues.update(
                fingerprint_level[
                    cid
                ][
                    "WT"
                ]
            )

            mutant_residues.update(
                fingerprint_level[
                    cid
                ][
                    "D166V"
                ]
            )

            sources.update(
                fingerprint_level[
                    cid
                ][
                    "sources"
                ]
            )

        if cid in residue_level:

            footprint = residue_level[
                cid
            ][
                "all_residues"
            ]

            sources.update(
                residue_level[
                    cid
                ][
                    "sources"
                ]
            )

            residue166_from_table = (
                residue_level[
                    cid
                ][
                    "residue166"
                ]
            )

            # A residue-level Step 24 table does not necessarily encode
            # separate WT and D166V footprints. Therefore we use it only
            # as a shared interaction footprint when no explicit WT/D166V
            # fingerprint is available.
            if not wt_residues:
                wt_residues.update(
                    footprint
                )

            if not mutant_residues:
                mutant_residues.update(
                    footprint
                )

        conserved = (
            wt_residues
            &
            mutant_residues
        )

        wt_only = (
            wt_residues
            -
            mutant_residues
        )

        mutant_only = (
            mutant_residues
            -
            wt_residues
        )

        wt_residue166 = (
            TARGET_RESIDUE
            in wt_residues
        )

        mutant_residue166 = (
            TARGET_RESIDUE
            in mutant_residues
        )

        if residue166_from_table:

            wt_residue166 = True
            mutant_residue166 = True

        rows.append(
            {
                "CID":
                    cid,
                "WT_Residue_Count":
                    len(wt_residues),
                "D166V_Residue_Count":
                    len(mutant_residues),
                "Conserved_Residue_Count":
                    len(conserved),
                "WT_Only_Residue_Count":
                    len(wt_only),
                "D166V_Only_Residue_Count":
                    len(mutant_only),
                "Interaction_Jaccard":
                    jaccard(
                        wt_residues,
                        mutant_residues,
                    ),
                "WT_Residue_166_Contact":
                    wt_residue166,
                "D166V_Residue_166_Contact":
                    mutant_residue166,
                "Step24_Source_Files":
                    "; ".join(
                        sorted(
                            sources
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# RESCORING / VALIDATION
# =============================================================================

def perform_validation(
    docking,
    leads,
    step28,
    interactions,
):

    result = docking.merge(
        leads,
        on="CID",
        how="left",
    )

    if (
        step28 is not None
        and not step28.empty
    ):

        result = result.merge(
            step28,
            on="CID",
            how="left",
            suffixes=(
                "",
                "_Step28",
            ),
        )

    if (
        interactions is not None
        and not interactions.empty
    ):

        result = result.merge(
            interactions,
            on="CID",
            how="left",
        )

    else:

        for column in [
            "WT_Residue_Count",
            "D166V_Residue_Count",
            "Conserved_Residue_Count",
            "WT_Only_Residue_Count",
            "D166V_Only_Residue_Count",
        ]:

            result[
                column
            ] = 0

        result[
            "Interaction_Jaccard"
        ] = 0.0

        result[
            "WT_Residue_166_Contact"
        ] = False

        result[
            "D166V_Residue_166_Contact"
        ] = False

    # -------------------------------------------------------------------------
    # Docking strength
    # -------------------------------------------------------------------------

    result[
        "Docking_Strength_Normalized"
    ] = minmax(
        -result[
            "D166V_Docking_Score"
        ]
    )

    # -------------------------------------------------------------------------
    # Mutation preference
    # -------------------------------------------------------------------------

    result[
        "Mutation_Preference_Component"
    ] = (
        -result[
            "Delta_D166V_minus_WT"
        ]
    )

    result[
        "Mutation_Preference_Normalized"
    ] = minmax(
        result[
            "Mutation_Preference_Component"
        ]
    )

    # -------------------------------------------------------------------------
    # Interaction consistency
    # -------------------------------------------------------------------------

    result[
        "Interaction_Jaccard"
    ] = pd.to_numeric(
        result[
            "Interaction_Jaccard"
        ],
        errors="coerce",
    ).fillna(
        0.0
    )

    result[
        "Interaction_Consistency_Score"
    ] = result[
        "Interaction_Jaccard"
    ].clip(
        0.0,
        1.0,
    )

    # -------------------------------------------------------------------------
    # Residue 166 involvement
    # -------------------------------------------------------------------------

    result[
        "D166V_Residue166_Signal"
    ] = (
        result[
            "D166V_Residue_166_Contact"
        ]
        .astype(bool)
        .astype(float)
    )

    # -------------------------------------------------------------------------
    # Lead priority
    # -------------------------------------------------------------------------

    result[
        "Lead_Priority_Normalized"
    ] = minmax(
        result[
            "Step23_Integrated_Score"
        ]
    )

    # -------------------------------------------------------------------------
    # Final validation score
    # -------------------------------------------------------------------------

    result[
        "Step29_Validation_Score"
    ] = (
        WEIGHT_DOCKING
        * result[
            "Docking_Strength_Normalized"
        ]
        +
        WEIGHT_INTERACTION_CONSISTENCY
        * result[
            "Interaction_Consistency_Score"
        ]
        +
        WEIGHT_RESIDUE_166
        * result[
            "D166V_Residue166_Signal"
        ]
        +
        WEIGHT_LEAD_PRIORITY
        * result[
            "Lead_Priority_Normalized"
        ]
    )

    # -------------------------------------------------------------------------
    # Validation class
    # -------------------------------------------------------------------------

    classes = []

    for _, row in result.iterrows():

        delta = safe_float(
            row[
                "Delta_D166V_minus_WT"
            ]
        )

        jaccard_value = safe_float(
            row[
                "Interaction_Jaccard"
            ]
        )

        residue166 = bool(
            row[
                "D166V_Residue166_Signal"
            ]
        )

        validation_score = safe_float(
            row[
                "Step29_Validation_Score"
            ]
        )

        if (
            not pd.isna(delta)
            and delta <= MUTANT_PREFERENCE_THRESHOLD
            and residue166
            and jaccard_value >= HIGH_INTERACTION_OVERLAP
            and validation_score >= 0.60
        ):

            classes.append(
                "ROBUST_MUTATION_CONSISTENT"
            )

        elif (
            not pd.isna(delta)
            and delta <= MUTANT_PREFERENCE_THRESHOLD
            and validation_score >= 0.50
        ):

            classes.append(
                "MUTATION_CONSISTENT"
            )

        elif (
            jaccard_value >= MODERATE_INTERACTION_OVERLAP
            and validation_score >= 0.50
        ):

            classes.append(
                "INTERACTION_CONSISTENT"
            )

        elif (
            not pd.isna(delta)
            and delta >= 1.0
        ):

            classes.append(
                "WT_PREFERRED"
            )

        else:

            classes.append(
                "LIMITED_VALIDATION"
            )

    result[
        "Step29_Validation_Class"
    ] = classes

    # More negative D166V-WT is more mutation-preferred.
    result = result.sort_values(
        [
            "Step29_Validation_Score",
            "Mutation_Preference_Normalized",
        ],
        ascending=False,
    ).reset_index(
        drop=True
    )

    result[
        "Step29_Rank"
    ] = np.arange(
        1,
        len(result) + 1,
    )

    return result


# =============================================================================
# FIGURES
# =============================================================================

def plot_validation_scores(
    result
):

    plot_df = result.sort_values(
        "Step29_Validation_Score",
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
            "Step29_Validation_Score"
        ],
    )

    plt.ylabel(
        "Step 29 validation score"
    )

    plt.xlabel(
        "Compound"
    )

    plt.title(
        "Step 29 — Lead Validation Score"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.tight_layout()


def plot_interaction_consistency(
    result
):

    plot_df = result.sort_values(
        "Interaction_Jaccard",
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
            "Interaction_Jaccard"
        ],
    )

    plt.axhline(
        HIGH_INTERACTION_OVERLAP,
        linewidth=1,
        linestyle="--",
        label="High-overlap threshold",
    )

    plt.axhline(
        MODERATE_INTERACTION_OVERLAP,
        linewidth=1,
        linestyle=":",
        label="Moderate-overlap threshold",
    )

    plt.ylabel(
        "WT ↔ D166V interaction Jaccard"
    )

    plt.xlabel(
        "Compound"
    )

    plt.title(
        "Step 29 — Interaction Consistency"
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.legend()

    plt.tight_layout()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 29 — LEAD VALIDATION, RESCORING & INTERACTION CONSISTENCY"
    )
    print("=" * 78)

    print(
        f"Step 22 : {STEP22_FILE}"
    )

    print(
        f"Step 23 : {STEP23_FILE}"
    )

    print(
        f"Step 24 : {STEP24_DIR}"
    )

    print()

    # -------------------------------------------------------------------------
    # Load Step 22
    # -------------------------------------------------------------------------

    print(
        "Loading Step 22 differential docking..."
    )

    docking = load_step22()

    print(
        f"Step 22 compounds loaded: {len(docking)}"
    )

    # -------------------------------------------------------------------------
    # Load Step 23
    # -------------------------------------------------------------------------

    print(
        "Loading Step 23 lead prioritization..."
    )

    leads = load_step23()

    print(
        f"Step 23 records loaded: {len(leads)}"
    )

    # -------------------------------------------------------------------------
    # Load Step 28
    # -------------------------------------------------------------------------

    print(
        "Loading Step 28 therapeutic vulnerability..."
    )

    step28 = load_step28()

    if step28.empty:

        print(
            "Step 28 carry-forward not available."
        )

    else:

        print(
            f"Step 28 records loaded: {len(step28)}"
        )

    print()

    # -------------------------------------------------------------------------
    # Step 24
    # -------------------------------------------------------------------------

    print(
        "Inspecting Step 24 interaction fingerprints..."
    )

    interactions = build_interaction_table()

    if interactions.empty:

        print(
            "No explicit Step 24 interaction table could be reconstructed."
        )

    else:

        print(
            f"Step 24 interaction records reconstructed: "
            f"{len(interactions)}"
        )

    print()

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    print(
        "Calculating Step 29 validation scores..."
    )

    result = perform_validation(
        docking,
        leads,
        step28,
        interactions,
    )

    # -------------------------------------------------------------------------
    # Save complete analysis
    # -------------------------------------------------------------------------

    result.to_csv(
        TABLE_DIR
        / "STEP29_FULL_LEAD_VALIDATION.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Top compounds
    # -------------------------------------------------------------------------

    top_columns = [
        "Step29_Rank",
        "CID",
        "D166V_Docking_Score",
        "WT_Docking_Score",
        "Delta_D166V_minus_WT",
        "Mutation_Preference_Normalized",
        "WT_Residue_Count",
        "D166V_Residue_Count",
        "Conserved_Residue_Count",
        "WT_Only_Residue_Count",
        "D166V_Only_Residue_Count",
        "Interaction_Jaccard",
        "WT_Residue_166_Contact",
        "D166V_Residue_166_Contact",
        "Step23_Integrated_Score",
        "Step23_Rank",
        "Step23_Lead_Class",
        "Step29_Validation_Score",
        "Step29_Validation_Class",
    ]

    top_columns = [
        column
        for column in top_columns
        if column in result.columns
    ]

    result[
        top_columns
    ].head(
        6
    ).to_csv(
        TABLE_DIR
        / "STEP29_TOP_LEAD_VALIDATION.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Class counts
    # -------------------------------------------------------------------------

    class_counts = (
        result[
            "Step29_Validation_Class"
        ]
        .value_counts()
        .rename_axis(
            "Validation_Class"
        )
        .reset_index(
            name="Count"
        )
    )

    class_counts.to_csv(
        TABLE_DIR
        / "STEP29_VALIDATION_CLASS_COUNTS.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Interaction table
    # -------------------------------------------------------------------------

    interaction_columns = [
        "CID",
        "WT_Residue_Count",
        "D166V_Residue_Count",
        "Conserved_Residue_Count",
        "WT_Only_Residue_Count",
        "D166V_Only_Residue_Count",
        "Interaction_Jaccard",
        "WT_Residue_166_Contact",
        "D166V_Residue_166_Contact",
        "Step24_Source_Files",
    ]

    interaction_columns = [
        column
        for column in interaction_columns
        if column in result.columns
    ]

    result[
        interaction_columns
    ].to_csv(
        TABLE_DIR
        / "STEP29_INTERACTION_CONSISTENCY.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Top result
    # -------------------------------------------------------------------------

    top = result.iloc[0]

    top_cid = str(
        top[
            "CID"
        ]
    )

    top_validation_score = safe_float(
        top[
            "Step29_Validation_Score"
        ]
    )

    top_delta = safe_float(
        top[
            "Delta_D166V_minus_WT"
        ]
    )

    top_jaccard = safe_float(
        top[
            "Interaction_Jaccard"
        ]
    )

    top_class = str(
        top[
            "Step29_Validation_Class"
        ]
    )

    robust_count = int(
        (
            result[
                "Step29_Validation_Class"
            ]
            ==
            "ROBUST_MUTATION_CONSISTENT"
        ).sum()
    )

    mutation_consistent_count = int(
        (
            result[
                "Step29_Validation_Class"
            ]
            ==
            "MUTATION_CONSISTENT"
        ).sum()
    )

    interaction_consistent_count = int(
        (
            result[
                "Step29_Validation_Class"
            ]
            ==
            "INTERACTION_CONSISTENT"
        ).sum()
    )

    wt_preferred_count = int(
        (
            result[
                "Step29_Validation_Class"
            ]
            ==
            "WT_PREFERRED"
        ).sum()
    )

    limited_count = int(
        (
            result[
                "Step29_Validation_Class"
            ]
            ==
            "LIMITED_VALIDATION"
        ).sum()
    )

    # -------------------------------------------------------------------------
    # Scientific interpretation
    # -------------------------------------------------------------------------

    final_interpretation = (
        f"Step 29 re-evaluated the existing differential docking "
        f"and interaction evidence without performing another docking "
        f"simulation. CID {top_cid} obtained the highest integrated "
        f"Step 29 validation score ({top_validation_score:.3f}). "
        f"Its D166V-minus-WT docking difference is "
        f"{top_delta:.3f} kcal/mol and its reconstructed "
        f"WT↔D166V interaction Jaccard similarity is "
        f"{top_jaccard:.3f}. The analysis identified "
        f"{robust_count} robust mutation-consistent lead(s), "
        f"{mutation_consistent_count} mutation-consistent lead(s), "
        f"{interaction_consistent_count} interaction-consistent "
        f"lead(s), {wt_preferred_count} WT-preferred lead(s), "
        f"and {limited_count} lead(s) with limited validation. "
        f"These classifications measure consistency among the "
        f"existing computational analyses. They do not establish "
        f"experimental binding affinity, biochemical inhibition, "
        f"functional rescue, or therapeutic efficacy."
    )

    with open(
        QC_DIR
        / "STEP29_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            final_interpretation
        )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    print(
        "Generating figures..."
    )

    plot_validation_scores(
        result
    )

    plt.savefig(
        FIGURE_DIR
        / "Figure_29_01_Lead_Validation_Scores.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    plot_interaction_consistency(
        result
    )

    plt.savefig(
        FIGURE_DIR
        / "Figure_29_02_Interaction_Consistency.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc = {
        "Step":
            29,
        "Step22_loaded":
            not docking.empty,
        "Step23_loaded":
            not leads.empty,
        "Step28_loaded":
            not step28.empty,
        "Step24_interaction_data_available":
            not interactions.empty,
        "Compounds_analyzed":
            len(result),
        "Top_CID":
            top_cid,
        "Top_Step29_Validation_Score":
            top_validation_score,
        "Top_D166V_minus_WT_kcal_mol":
            top_delta,
        "Top_Interaction_Jaccard":
            top_jaccard,
        "Robust_mutation_consistent_count":
            robust_count,
        "Mutation_consistent_count":
            mutation_consistent_count,
        "Interaction_consistent_count":
            interaction_consistent_count,
        "WT_preferred_count":
            wt_preferred_count,
        "Limited_validation_count":
            limited_count,
        "Status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP29_QC.json",
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
    print("STEP 29 RESULTS")
    print("=" * 78)

    print(
        f"Compounds analyzed              : "
        f"{len(result)}"
    )

    print(
        f"Top validation CID              : "
        f"{top_cid}"
    )

    print(
        f"Top Step 29 validation score    : "
        f"{top_validation_score:.3f}"
    )

    print(
        f"Top D166V − WT docking delta    : "
        f"{top_delta:.3f} kcal/mol"
    )

    print(
        f"Top interaction Jaccard         : "
        f"{top_jaccard:.3f}"
    )

    print(
        f"Top validation class            : "
        f"{top_class}"
    )

    print()

    print(
        f"Robust mutation-consistent      : "
        f"{robust_count}"
    )

    print(
        f"Mutation-consistent             : "
        f"{mutation_consistent_count}"
    )

    print(
        f"Interaction-consistent          : "
        f"{interaction_consistent_count}"
    )

    print(
        f"WT-preferred                    : "
        f"{wt_preferred_count}"
    )

    print(
        f"Limited validation              : "
        f"{limited_count}"
    )

    print()

    print(
        "Final interpretation:"
    )

    print(
        final_interpretation
    )

    print()

    print(
        "Step 29 completed successfully."
    )

    print(
        f"Full validation : "
        f"{TABLE_DIR / 'STEP29_FULL_LEAD_VALIDATION.csv'}"
    )

    print(
        f"Top leads       : "
        f"{TABLE_DIR / 'STEP29_TOP_LEAD_VALIDATION.csv'}"
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
            "\nStep 29 interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 29 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise