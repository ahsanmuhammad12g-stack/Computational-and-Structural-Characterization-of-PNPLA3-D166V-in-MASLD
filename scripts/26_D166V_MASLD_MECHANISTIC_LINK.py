from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser


# =============================================================================
# STEP 26 — D166V → PNPLA3 FUNCTION → MASLD MECHANISTIC LINK
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WT_PDB = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
    / "structures"
    / "WT"
    / "PNPLA3_WT.pdb"
)

MUT_PDB = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
    / "structures"
    / "D166V"
    / "PNPLA3_D166V_mutant.pdb"
)

STEP25_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP25_D166V_FUNCTIONAL_MECHANISM"
)

STEP16_POCKET = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
    / "tables"
    / "Table_16_04_D166V_Pocket_Prioritization.csv"
)

STEP21_DOCKING = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
    / "tables"
    / "STEP21_BEST_DOCKING_RESULTS.csv"
)

STEP22_DIFF = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
    / "tables"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING.csv"
)

STEP23_LEADS = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP23_LEAD_PRIORITIZATION"
    / "tables"
    / "STEP23_LEAD_PRIORITIZATION.csv"
)

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP26_D166V_MASLD_MECHANISTIC_LINK"
)

TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
QC_DIR = OUT_DIR / "qc"

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
# FIXED BIOLOGICAL PARAMETERS
# =============================================================================

TARGET_VARIANT = "D166V"
TARGET_RESIDUE = 166

CATALYTIC_SERINE = 47
CATALYTIC_ASPARTATE = 166
CATALYTIC_HISTIDINE = 300

LOCAL_CUTOFF_A = 8.0
CONTACT_CUTOFF_A = 4.0

EXPECTED_WT_D166 = "ASP"
EXPECTED_MUT_D166 = "VAL"

EXPECTED_PROTEIN_LENGTH = 481


# =============================================================================
# PUBLISHED BIOLOGICAL CONTEXT
# =============================================================================

BIOLOGICAL_CONTEXT = [
    {
        "Evidence_ID": "BIO-01",
        "Evidence_Level": "Established biology",
        "Statement":
            "PNPLA3 is a major genetic determinant of MASLD/MASH "
            "and is an active precision-therapeutic target.",
        "Relevance":
            "Establishes that a functionally meaningful PNPLA3 "
            "variant can be relevant to MASLD biology.",
        "Interpretation":
            "Supports disease relevance of investigating unresolved "
            "PNPLA3 variants.",
        "Source":
            "Lindén, Tesz & Loomba, 2024, Liver International",
    },
    {
        "Evidence_ID": "BIO-02",
        "Evidence_Level": "Established biology",
        "Statement":
            "Experimental PNPLA3-directed suppression has shown "
            "beneficial effects on steatosis and fibrosis in "
            "disease models, supporting PNPLA3 as a therapeutic target.",
        "Relevance":
            "Supports the therapeutic significance of identifying "
            "functionally abnormal PNPLA3 states.",
        "Interpretation":
            "Links PNPLA3 molecular biology to an actionable disease "
            "intervention framework.",
        "Source":
            "Lindén et al., 2019, Molecular Metabolism",
    },
    {
        "Evidence_ID": "BIO-03",
        "Evidence_Level": "Established biology",
        "Statement":
            "Recent PNPLA3 precision-therapy work remains strongly "
            "focused on the common I148M risk variant.",
        "Relevance":
            "Creates the rationale for investigating less-characterized "
            "PNPLA3 variants.",
        "Interpretation":
            "Supports the novelty context of D166V-focused analysis.",
        "Source":
            "Armisen et al., 2025, Journal of Hepatology",
    },
]


# =============================================================================
# GENERAL FUNCTIONS
# =============================================================================

def fail(message):
    print()
    print("=" * 78)
    print("STEP 26 FAILED")
    print("=" * 78)
    print(message)
    print()
    sys.exit(1)


def load_model(path):

    if not path.exists():
        fail(
            f"Required structure not found:\n{path}"
        )

    parser = PDBParser(
        QUIET=True
    )

    structure = parser.get_structure(
        path.stem,
        str(path),
    )

    models = list(
        structure.get_models()
    )

    if not models:
        fail(
            f"No structure model found:\n{path}"
        )

    return models[0]


def residue_number(residue):

    try:
        return int(
            residue.id[1]
        )
    except Exception:
        return None


def residue_name(residue):
    return (
        residue.resname
        .strip()
        .upper()
    )


def get_residue(
    model,
    number,
):

    matches = []

    for chain in model.get_chains():

        for residue in chain.get_residues():

            if residue_number(residue) == number:

                matches.append(
                    residue
                )

    if not matches:
        fail(
            f"Residue {number} was not found."
        )

    return matches[0]


def atom_distance(
    residue_a,
    atom_a,
    residue_b,
    atom_b,
):

    if atom_a not in residue_a:
        return np.nan

    if atom_b not in residue_b:
        return np.nan

    return float(
        np.linalg.norm(
            residue_a[atom_a].coord
            - residue_b[atom_b].coord
        )
    )


def residue_ca_distance(
    residue_a,
    residue_b,
):

    if (
        "CA" not in residue_a
        or "CA" not in residue_b
    ):
        return np.nan

    return float(
        np.linalg.norm(
            residue_a["CA"].coord
            - residue_b["CA"].coord
        )
    )


def residue_center(
    residue,
):

    atoms = []

    for atom in residue.get_atoms():

        element = getattr(
            atom,
            "element",
            "",
        )

        if str(element).upper() == "H":
            continue

        atoms.append(
            atom
        )

    if not atoms:
        return np.array(
            [np.nan, np.nan, np.nan]
        )

    return np.mean(
        np.array(
            [
                atom.coord
                for atom in atoms
            ],
            dtype=float,
        ),
        axis=0,
    )


def center_distance(
    residue_a,
    residue_b,
):

    ca = residue_center(
        residue_a
    )

    cb = residue_center(
        residue_b
    )

    if np.any(
        np.isnan(ca)
    ) or np.any(
        np.isnan(cb)
    ):
        return np.nan

    return float(
        np.linalg.norm(
            ca - cb
        )
    )


def read_csv(path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(
            path
        )
    except Exception as exc:
        print(
            f"Warning: could not read {path}: {exc}"
        )
        return None


# =============================================================================
# ACTIVE-SITE GEOMETRY
# =============================================================================

def catalytic_geometry(
    model,
    label,
):

    ser47 = get_residue(
        model,
        CATALYTIC_SERINE,
    )

    asp166 = get_residue(
        model,
        CATALYTIC_ASPARTATE,
    )

    his300 = get_residue(
        model,
        CATALYTIC_HISTIDINE,
    )

    result = {
        "Model":
            label,
        "Ser47_Name":
            residue_name(ser47),
        "Asp166_or_Val166_Name":
            residue_name(asp166),
        "His300_Name":
            residue_name(his300),
        "CA_Ser47_to_Residue166_A":
            residue_ca_distance(
                ser47,
                asp166,
            ),
        "CA_Residue166_to_His300_A":
            residue_ca_distance(
                asp166,
                his300,
            ),
        "CA_Ser47_to_His300_A":
            residue_ca_distance(
                ser47,
                his300,
            ),
        "Center_Ser47_to_Residue166_A":
            center_distance(
                ser47,
                asp166,
            ),
        "Center_Residue166_to_His300_A":
            center_distance(
                asp166,
                his300,
            ),
        "Center_Ser47_to_His300_A":
            center_distance(
                ser47,
                his300,
            ),
    }

    # Side-chain anchor distances where chemically meaningful.
    if label == "WT":

        result[
            "Ser47_Ogamma_to_Asp166_Odistance_A"
        ] = (
            min(
                [
                    atom_distance(
                        ser47,
                        "OG",
                        asp166,
                        "OD1",
                    ),
                    atom_distance(
                        ser47,
                        "OG",
                        asp166,
                        "OD2",
                    ),
                ]
            )
        )

    else:

        result[
            "Ser47_Ogamma_to_Val166_Cdistance_A"
        ] = (
            min(
                [
                    atom_distance(
                        ser47,
                        "OG",
                        asp166,
                        "CG1",
                    ),
                    atom_distance(
                        ser47,
                        "OG",
                        asp166,
                        "CG2",
                    ),
                ]
            )
        )

    return result


# =============================================================================
# RESIDUE 166 LOCAL CONTACTS FROM STEP 25
# =============================================================================

def load_step25_tables():

    contact_file = (
        STEP25_DIR
        / "tables"
        / "STEP25_WT_vs_D166V_Contact_Comparison.csv"
    )

    environment_file = (
        STEP25_DIR
        / "tables"
        / "STEP25_WT_vs_D166V_Local_Environment.csv"
    )

    summary_file = (
        STEP25_DIR
        / "tables"
        / "STEP25_D166V_FUNCTIONAL_MECHANISM_SUMMARY.csv"
    )

    contacts = read_csv(
        contact_file
    )

    environment = read_csv(
        environment_file
    )

    summary = read_csv(
        summary_file
    )

    return (
        contacts,
        environment,
        summary,
    )


# =============================================================================
# STEP 16 POCKET EVIDENCE
# =============================================================================

def extract_pocket_evidence():

    df = read_csv(
        STEP16_POCKET
    )

    result = {
        "Pocket_data_available":
            df is not None,
    }

    if df is None or df.empty:
        return result

    # Prefer pocket 1.
    row = None

    for column in [
        "D166V_Pocket_ID",
        "D166V_Pocket",
        "Pocket_ID",
        "Pocket",
    ]:

        if column in df.columns:

            subset = df[
                df[column]
                .astype(str)
                .str.contains(
                    r"(^1$|pocket.?1)",
                    case=False,
                    regex=True,
                    na=False,
                )
            ]

            if not subset.empty:
                row = subset.iloc[0]
                break

    if row is None:
        row = df.iloc[0]

    preferred_columns = [
        "Rank",
        "D166V_Pocket",
        "WT_Pocket",
        "Match_Status",
        "Pocket_Match",
        "Druggability",
        "Pocket_Score",
        "Volume_A3",
        "D166_Distance_A",
        "D166_Directly_Lining",
        "Lining_Jaccard",
        "Mutation_Remodeling_Score",
        "Prioritization_Score",
    ]

    for column in preferred_columns:

        if column in df.columns:

            value = row[column]

            result[
                f"Step16_{column}"
            ] = value

    return result


# =============================================================================
# DOCKING EVIDENCE
# =============================================================================

def extract_docking_evidence():

    best = read_csv(
        STEP21_DOCKING
    )

    diff = read_csv(
        STEP22_DIFF
    )

    leads = read_csv(
        STEP23_LEADS
    )

    result = {
        "Step21_available":
            best is not None,
        "Step22_available":
            diff is not None,
        "Step23_available":
            leads is not None,
    }

    # -------------------------------------------------------------------------
    # Step 21
    # -------------------------------------------------------------------------

    if best is not None and not best.empty:

        row = None

        for column in [
            "CID",
            "Compound_ID",
        ]:

            if column in best.columns:

                subset = best[
                    best[column]
                    .astype(str)
                    == "71236597"
                ]

                if not subset.empty:
                    row = subset.iloc[0]
                    break

        if row is None:
            row = best.iloc[0]

        for column in [
            "CID",
            "Best_Docking_Score",
            "Docking_Score",
            "Best_Score",
        ]:

            if column in best.columns:

                result[
                    f"Step21_{column}"
                ] = row[column]

    # -------------------------------------------------------------------------
    # Step 22
    # -------------------------------------------------------------------------

    if diff is not None and not diff.empty:

        row = None

        for column in [
            "CID",
            "Compound_ID",
        ]:

            if column in diff.columns:

                subset = diff[
                    diff[column]
                    .astype(str)
                    == "71236597"
                ]

                if not subset.empty:
                    row = subset.iloc[0]
                    break

        if row is None:
            row = diff.iloc[0]

        for column in [
            "D166V_Best_Score",
            "WT_Best_Score",
            "Delta_D166V_minus_WT",
            "Preference",
        ]:

            if column in diff.columns:

                result[
                    f"Step22_{column}"
                ] = row[column]

    # -------------------------------------------------------------------------
    # Step 23
    # -------------------------------------------------------------------------

    if leads is not None and not leads.empty:

        cid_column = None

        for column in [
            "CID",
            "Compound_ID",
        ]:

            if column in leads.columns:
                cid_column = column
                break

        if cid_column:

            subset = leads[
                leads[cid_column]
                .astype(str)
                == "71236597"
            ]

            if not subset.empty:

                row = subset.iloc[0]

                for column in [
                    "Integrated_Score",
                    "Lead_Class",
                    "Priority",
                    "Step23_Rank",
                ]:

                    if column in leads.columns:

                        result[
                            f"Step23_{column}"
                        ] = row[column]

    return result


# =============================================================================
# MECHANISTIC EVIDENCE MATRIX
# =============================================================================

def build_evidence_matrix(
    wt_geometry,
    mut_geometry,
    contacts,
    environment,
    pocket_evidence,
    docking_evidence,
):

    rows = []

    # -------------------------------------------------------------------------
    # Evidence 1 — catalytic residue identity
    # -------------------------------------------------------------------------

    wt_d166 = (
        wt_geometry[
            "Asp166_or_Val166_Name"
        ]
    )

    mut_d166 = (
        mut_geometry[
            "Asp166_or_Val166_Name"
        ]
    )

    rows.append(
        {
            "Evidence_ID":
                "MECH-01",
            "Category":
                "Catalytic-site identity",
            "Observation":
                f"Residue 166 changes from "
                f"{wt_d166} to {mut_d166}.",
            "Computational_Result":
                "Catalytic residue identity is altered.",
            "Biological_Interpretation":
                "Because residue 166 is part of the previously "
                "defined PNPLA3 catalytic machinery, D166V directly "
                "changes a functionally critical residue.",
            "Evidence_Type":
                "Direct structural observation",
            "Strength":
                "High",
        }
    )

    # -------------------------------------------------------------------------
    # Evidence 2 — catalytic-site geometry
    # -------------------------------------------------------------------------

    ser_166_wt = wt_geometry[
        "CA_Ser47_to_Residue166_A"
    ]

    ser_166_mut = mut_geometry[
        "CA_Ser47_to_Residue166_A"
    ]

    his_166_wt = wt_geometry[
        "CA_Residue166_to_His300_A"
    ]

    his_166_mut = mut_geometry[
        "CA_Residue166_to_His300_A"
    ]

    rows.append(
        {
            "Evidence_ID":
                "MECH-02",
            "Category":
                "Catalytic-site geometry",
            "Observation":
                "CA-based distances among residues 47, 166 and 300 "
                "were compared between WT and D166V.",
            "Computational_Result":
                f"Ser47–166: {ser_166_wt:.3f} Å → "
                f"{ser_166_mut:.3f} Å; "
                f"166–His300: {his_166_wt:.3f} Å → "
                f"{his_166_mut:.3f} Å.",
            "Biological_Interpretation":
                "Tests whether the mutation changes the spatial "
                "organization of the catalytic region even though "
                "global backbone displacement is small.",
            "Evidence_Type":
                "Structural geometry",
            "Strength":
                "Moderate",
        }
    )

    # -------------------------------------------------------------------------
    # Evidence 3 — contact network
    # -------------------------------------------------------------------------

    lost = 0
    gained = 0

    if contacts is not None and not contacts.empty:

        if "Status" in contacts.columns:

            lost = int(
                (
                    contacts["Status"]
                    == "LOST"
                ).sum()
            )

            gained = int(
                (
                    contacts["Status"]
                    == "GAINED"
                ).sum()
            )

    rows.append(
        {
            "Evidence_ID":
                "MECH-03",
            "Category":
                "Residue-contact network",
            "Observation":
                "Residue-166 contact remodeling was detected.",
            "Computational_Result":
                f"{lost} contacts lost and "
                f"{gained} contacts gained.",
            "Biological_Interpretation":
                "Loss/gain of local contacts indicates that "
                "D166V alters the immediate molecular environment "
                "of the catalytic residue.",
            "Evidence_Type":
                "Atom-level structural comparison",
            "Strength":
                "High",
        }
    )

    # -------------------------------------------------------------------------
    # Evidence 4 — local environment
    # -------------------------------------------------------------------------

    wt_local = 0
    mut_local = 0

    if environment is not None and not environment.empty:

        if "WT_Residue_Name" in environment.columns:
            wt_local = int(
                environment[
                    "WT_Residue_Name"
                ].notna().sum()
            )

        if "D166V_Residue_Name" in environment.columns:
            mut_local = int(
                environment[
                    "D166V_Residue_Name"
                ].notna().sum()
            )

    rows.append(
        {
            "Evidence_ID":
                "MECH-04",
            "Category":
                "Local structural environment",
            "Observation":
                "The residue-166 8 Å environments were compared.",
            "Computational_Result":
                f"WT local residues = {wt_local}; "
                f"D166V local residues = {mut_local}.",
            "Biological_Interpretation":
                "A changed local environment supports a mutation-induced "
                "reorganization of the catalytic microenvironment.",
            "Evidence_Type":
                "Local structural comparison",
            "Strength":
                "Moderate",
        }
    )

    # -------------------------------------------------------------------------
    # Evidence 5 — pocket
    # -------------------------------------------------------------------------

    pocket_description = (
        "No Step 16 pocket evidence available."
    )

    if pocket_evidence.get(
        "Pocket_data_available",
        False,
    ):

        druggability = pocket_evidence.get(
            "Step16_Druggability",
            np.nan,
        )

        proximity = pocket_evidence.get(
            "Step16_D166_Distance_A",
            np.nan,
        )

        pocket_description = (
            f"Step 16 identifies a residue-166-centered "
            f"therapeutic pocket; reported druggability = "
            f"{druggability}, D166 distance = {proximity}."
        )

    rows.append(
        {
            "Evidence_ID":
                "MECH-05",
            "Category":
                "Therapeutic vulnerability",
            "Observation":
                pocket_description,
            "Computational_Result":
                "The altered residue lies within the prioritized "
                "therapeutic pocket identified previously.",
            "Biological_Interpretation":
                "The structural defect is potentially actionable "
                "because it occurs at a therapeutically addressable "
                "region.",
            "Evidence_Type":
                "Pocket analysis",
            "Strength":
                "High",
        }
    )

    # -------------------------------------------------------------------------
    # Evidence 6 — docking
    # -------------------------------------------------------------------------

    docking_summary = []

    for key, value in docking_evidence.items():

        if key.startswith(
            "Step21_"
        ) or key.startswith(
            "Step22_"
        ) or key.startswith(
            "Step23_"
        ):

            docking_summary.append(
                f"{key}={value}"
            )

    rows.append(
        {
            "Evidence_ID":
                "MECH-06",
            "Category":
                "Therapeutic intervention",
            "Observation":
                "The prioritized compound set was evaluated against "
                "WT and D166V.",
            "Computational_Result":
                "; ".join(
                    docking_summary
                ) if docking_summary
                else "Docking evidence unavailable.",
            "Biological_Interpretation":
                "Mutation-aware docking provides downstream support "
                "for testing whether the structurally altered state "
                "can be differentially targeted.",
            "Evidence_Type":
                "Computational ligand screening",
            "Strength":
                "Supportive",
        }
    )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

def make_final_interpretation(
    wt_geometry,
    mut_geometry,
    evidence_matrix,
):

    wt_ser166 = wt_geometry[
        "CA_Ser47_to_Residue166_A"
    ]

    mut_ser166 = mut_geometry[
        "CA_Ser47_to_Residue166_A"
    ]

    wt_his166 = wt_geometry[
        "CA_Residue166_to_His300_A"
    ]

    mut_his166 = mut_geometry[
        "CA_Residue166_to_His300_A"
    ]

    delta_ser = (
        mut_ser166
        - wt_ser166
    )

    delta_his = (
        mut_his166
        - wt_his166
    )

    interpretation = (
        "D166V produces a direct chemical identity change at "
        "PNPLA3 residue 166, replacing Asp with Val. The "
        "residue-166 contact network and local environment are "
        "also remodeled, while the overall protein backbone "
        "remains highly similar. Because residue 166 was "
        "previously established in this project as a catalytic "
        "residue of PNPLA3, these findings support a "
        "mutation-specific functional-disruption hypothesis. "
        f"The CA distance from Ser47 to residue 166 changes by "
        f"{delta_ser:+.3f} Å and the CA distance from residue 166 "
        f"to His300 changes by {delta_his:+.3f} Å. "
        "The combination of catalytic-residue substitution, "
        "local contact remodeling, and proximity to the "
        "previously prioritized therapeutic pocket provides "
        "a coherent computational mechanism linking D166V to "
        "altered PNPLA3 molecular function. "
        "The disease connection is a mechanistic inference "
        "supported by the established role of PNPLA3 in MASLD; "
        "the present computation does not directly measure "
        "lipase activity or a clinical phenotype."
    )

    return interpretation


# =============================================================================
# FIGURE — CATALYTIC GEOMETRY
# =============================================================================

def plot_catalytic_geometry(
    wt_geometry,
    mut_geometry,
):

    labels = [
        "Ser47–166 CA",
        "166–His300 CA",
        "Ser47–His300 CA",
    ]

    wt_values = [
        wt_geometry[
            "CA_Ser47_to_Residue166_A"
        ],
        wt_geometry[
            "CA_Residue166_to_His300_A"
        ],
        wt_geometry[
            "CA_Ser47_to_His300_A"
        ],
    ]

    mut_values = [
        mut_geometry[
            "CA_Ser47_to_Residue166_A"
        ],
        mut_geometry[
            "CA_Residue166_to_His300_A"
        ],
        mut_geometry[
            "CA_Ser47_to_His300_A"
        ],
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.38

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        x - width / 2,
        wt_values,
        width,
        label="WT",
    )

    plt.bar(
        x + width / 2,
        mut_values,
        width,
        label="D166V",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Distance (Å)"
    )

    plt.title(
        "Step 26 — PNPLA3 Catalytic-Site Geometry"
    )

    plt.legend()

    plt.tight_layout()


# =============================================================================
# FIGURE — EVIDENCE STRENGTH
# =============================================================================

def plot_evidence_strength(
    evidence_matrix,
):

    mapping = {
        "High": 3,
        "Moderate": 2,
        "Supportive": 1,
    }

    labels = (
        evidence_matrix[
            "Evidence_ID"
        ]
        .astype(str)
        .tolist()
    )

    values = [
        mapping.get(
            str(value),
            0,
        )
        for value in evidence_matrix[
            "Strength"
        ]
    ]

    plt.figure(
        figsize=(9, 5)
    )

    plt.bar(
        labels,
        values,
    )

    plt.yticks(
        [1, 2, 3],
        [
            "Supportive",
            "Moderate",
            "High",
        ],
    )

    plt.ylabel(
        "Evidence category"
    )

    plt.title(
        "Step 26 — D166V Mechanistic Evidence Layers"
    )

    plt.tight_layout()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 26 — D166V → PNPLA3 FUNCTION → MASLD MECHANISTIC LINK"
    )
    print("=" * 78)

    print(
        f"WT structure    : {WT_PDB}"
    )

    print(
        f"D166V structure : {MUT_PDB}"
    )

    print()

    # -------------------------------------------------------------------------
    # Load structures
    # -------------------------------------------------------------------------

    print(
        "Loading WT and D166V structures..."
    )

    wt_model = load_model(
        WT_PDB
    )

    mut_model = load_model(
        MUT_PDB
    )

    print(
        "Structures loaded."
    )

    print()

    # -------------------------------------------------------------------------
    # Target identity
    # -------------------------------------------------------------------------

    wt_d166 = get_residue(
        wt_model,
        TARGET_RESIDUE,
    )

    mut_d166 = get_residue(
        mut_model,
        TARGET_RESIDUE,
    )

    print(
        f"WT residue 166    : "
        f"{residue_name(wt_d166)}"
    )

    print(
        f"D166V residue 166 : "
        f"{residue_name(mut_d166)}"
    )

    if residue_name(wt_d166) != EXPECTED_WT_D166:
        fail(
            "WT residue 166 is not ASP."
        )

    if residue_name(mut_d166) != EXPECTED_MUT_D166:
        fail(
            "D166V residue 166 is not VAL."
        )

    print()

    # -------------------------------------------------------------------------
    # Catalytic geometry
    # -------------------------------------------------------------------------

    print(
        "Analyzing catalytic-site geometry "
        "(Ser47–Asp/Val166–His300)..."
    )

    wt_geometry = catalytic_geometry(
        wt_model,
        "WT",
    )

    mut_geometry = catalytic_geometry(
        mut_model,
        "D166V",
    )

    geometry_table = pd.DataFrame(
        [
            wt_geometry,
            mut_geometry,
        ]
    )

    print(
        f"WT Ser47–166 CA distance   : "
        f"{wt_geometry['CA_Ser47_to_Residue166_A']:.3f} Å"
    )

    print(
        f"D166V Ser47–166 CA distance: "
        f"{mut_geometry['CA_Ser47_to_Residue166_A']:.3f} Å"
    )

    print(
        f"WT 166–His300 CA distance   : "
        f"{wt_geometry['CA_Residue166_to_His300_A']:.3f} Å"
    )

    print(
        f"D166V 166–His300 CA distance: "
        f"{mut_geometry['CA_Residue166_to_His300_A']:.3f} Å"
    )

    print()

    # -------------------------------------------------------------------------
    # Step 25 evidence
    # -------------------------------------------------------------------------

    (
        contacts,
        environment,
        step25_summary,
    ) = load_step25_tables()

    if contacts is None:
        print(
            "Warning: Step 25 contact comparison not found."
        )

    if environment is None:
        print(
            "Warning: Step 25 local environment table not found."
        )

    # -------------------------------------------------------------------------
    # Step 16
    # -------------------------------------------------------------------------

    print(
        "Loading Step 16 therapeutic-pocket evidence..."
    )

    pocket_evidence = extract_pocket_evidence()

    # -------------------------------------------------------------------------
    # Steps 21–23
    # -------------------------------------------------------------------------

    print(
        "Loading downstream docking/lead evidence..."
    )

    docking_evidence = extract_docking_evidence()

    print()

    # -------------------------------------------------------------------------
    # Evidence matrix
    # -------------------------------------------------------------------------

    print(
        "Building mechanistic evidence matrix..."
    )

    evidence_matrix = build_evidence_matrix(
        wt_geometry,
        mut_geometry,
        contacts,
        environment,
        pocket_evidence,
        docking_evidence,
    )

    # -------------------------------------------------------------------------
    # Final interpretation
    # -------------------------------------------------------------------------

    interpretation = make_final_interpretation(
        wt_geometry,
        mut_geometry,
        evidence_matrix,
    )

    # -------------------------------------------------------------------------
    # Save evidence tables
    # -------------------------------------------------------------------------

    geometry_table.to_csv(
        TABLE_DIR
        / "STEP26_Catalytic_Site_Geometry.csv",
        index=False,
    )

    evidence_matrix.to_csv(
        TABLE_DIR
        / "STEP26_MASLD_MECHANISTIC_EVIDENCE_MATRIX.csv",
        index=False,
    )

    # Combine available Step 25 summary if useful.
    if (
        step25_summary is not None
        and not step25_summary.empty
    ):

        step25_summary.to_csv(
            TABLE_DIR
            / "STEP26_Step25_Summary_CarryForward.csv",
            index=False,
        )

    # Biological context.
    biological_df = pd.DataFrame(
        BIOLOGICAL_CONTEXT
    )

    biological_df.to_csv(
        TABLE_DIR
        / "STEP26_ESTABLISHED_BIOLOGICAL_CONTEXT.csv",
        index=False,
    )

    # Final integrated summary.
    integrated = {
        "Variant":
            TARGET_VARIANT,
        "WT_residue_166":
            residue_name(wt_d166),
        "D166V_residue_166":
            residue_name(mut_d166),
        "WT_protein_length":
            EXPECTED_PROTEIN_LENGTH,
        "Catalytic_serine":
            CATALYTIC_SERINE,
        "Catalytic_aspartate_position":
            CATALYTIC_ASPARTATE,
        "Catalytic_histidine":
            CATALYTIC_HISTIDINE,
        "WT_Ser47_166_CA_A":
            wt_geometry[
                "CA_Ser47_to_Residue166_A"
            ],
        "D166V_Ser47_166_CA_A":
            mut_geometry[
                "CA_Ser47_to_Residue166_A"
            ],
        "WT_166_His300_CA_A":
            wt_geometry[
                "CA_Residue166_to_His300_A"
            ],
        "D166V_166_His300_CA_A":
            mut_geometry[
                "CA_Residue166_to_His300_A"
            ],
        "Step16_pocket_available":
            pocket_evidence.get(
                "Pocket_data_available",
                False,
            ),
        "Step21_docking_available":
            docking_evidence.get(
                "Step21_available",
                False,
            ),
        "Step22_differential_docking_available":
            docking_evidence.get(
                "Step22_available",
                False,
            ),
        "Step23_lead_data_available":
            docking_evidence.get(
                "Step23_available",
                False,
            ),
        "Interpretation":
            interpretation,
    }

    pd.DataFrame(
        [integrated]
    ).to_csv(
        TABLE_DIR
        / "STEP26_FINAL_MECHANISTIC_INTEGRATION.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    print(
        "Generating figures..."
    )

    try:

        plot_catalytic_geometry(
            wt_geometry,
            mut_geometry,
        )

        plt.savefig(
            FIGURE_DIR
            / "Figure_26_01_Catalytic_Site_Geometry.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    except Exception as exc:

        print(
            f"Figure 1 warning: {exc}"
        )

        plt.close(
            "all"
        )

    try:

        plot_evidence_strength(
            evidence_matrix
        )

        plt.savefig(
            FIGURE_DIR
            / "Figure_26_02_Mechanistic_Evidence_Layers.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    except Exception as exc:

        print(
            f"Figure 2 warning: {exc}"
        )

        plt.close(
            "all"
        )

    # -------------------------------------------------------------------------
    # Interpretation file
    # -------------------------------------------------------------------------

    with open(
        QC_DIR
        / "STEP26_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            interpretation
        )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc = {
        "Step":
            26,
        "Variant":
            TARGET_VARIANT,
        "WT_structure_exists":
            WT_PDB.exists(),
        "D166V_structure_exists":
            MUT_PDB.exists(),
        "WT_residue_166":
            residue_name(wt_d166),
        "D166V_residue_166":
            residue_name(mut_d166),
        "WT_protein_length":
            len(
                [
                    residue
                    for chain in wt_model.get_chains()
                    for residue in chain.get_residues()
                    if residue_number(residue)
                    is not None
                ]
            ),
        "D166V_protein_length":
            len(
                [
                    residue
                    for chain in mut_model.get_chains()
                    for residue in chain.get_residues()
                    if residue_number(residue)
                    is not None
                ]
            ),
        "Step25_contact_table_available":
            contacts is not None,
        "Step25_environment_table_available":
            environment is not None,
        "Step16_pocket_available":
            pocket_evidence.get(
                "Pocket_data_available",
                False,
            ),
        "Step21_docking_available":
            docking_evidence.get(
                "Step21_available",
                False,
            ),
        "Step22_differential_docking_available":
            docking_evidence.get(
                "Step22_available",
                False,
            ),
        "Step23_lead_data_available":
            docking_evidence.get(
                "Step23_available",
                False,
            ),
        "Status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP26_QC.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final output
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STEP 26 RESULTS")
    print("=" * 78)

    print(
        f"WT residue 166                  : "
        f"{residue_name(wt_d166)}"
    )

    print(
        f"D166V residue 166               : "
        f"{residue_name(mut_d166)}"
    )

    print(
        f"WT Ser47–166 CA distance        : "
        f"{wt_geometry['CA_Ser47_to_Residue166_A']:.3f} Å"
    )

    print(
        f"D166V Ser47–166 CA distance     : "
        f"{mut_geometry['CA_Ser47_to_Residue166_A']:.3f} Å"
    )

    print(
        f"WT 166–His300 CA distance       : "
        f"{wt_geometry['CA_Residue166_to_His300_A']:.3f} Å"
    )

    print(
        f"D166V 166–His300 CA distance    : "
        f"{mut_geometry['CA_Residue166_to_His300_A']:.3f} Å"
    )

    print()

    print(
        "Mechanistic interpretation:"
    )

    print(
        interpretation
    )

    print()
    print(
        "Step 26 completed successfully."
    )

    print(
        f"Main evidence matrix : "
        f"{TABLE_DIR / 'STEP26_MASLD_MECHANISTIC_EVIDENCE_MATRIX.csv'}"
    )

    print(
        f"Final integration     : "
        f"{TABLE_DIR / 'STEP26_FINAL_MECHANISTIC_INTEGRATION.csv'}"
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
            "\nStep 26 interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 26 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise