from pathlib import Path
import json
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser


# =============================================================================
# STEP 25 — D166V FUNCTIONAL / STRUCTURAL MECHANISM ANALYSIS
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

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP25_D166V_FUNCTIONAL_MECHANISM"
)

TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
QC_DIR = OUT_DIR / "qc"

for folder in [
    OUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# STUDY SETTINGS
# =============================================================================

TARGET_RESIDUE = 166

LOCAL_CUTOFF_A = 8.0
CONTACT_CUTOFF_A = 4.0

EXPECTED_WT_RESIDUE = "ASP"
EXPECTED_MUTANT_RESIDUE = "VAL"

EXPECTED_PROTEIN_LENGTH = 481

VARIANT = "D166V"
VARIANT_DESCRIPTION = "p.Asp166Val"

# Kyte-Doolittle hydropathy scale.
HYDROPATHY = {
    "ALA": 1.8,
    "ARG": -4.5,
    "ASN": -3.5,
    "ASP": -3.5,
    "CYS": 2.5,
    "GLN": -3.5,
    "GLU": -3.5,
    "GLY": -0.4,
    "HIS": -3.2,
    "ILE": 4.5,
    "LEU": 3.8,
    "LYS": -3.9,
    "MET": 1.9,
    "PHE": 2.8,
    "PRO": -1.6,
    "SER": -0.8,
    "THR": -0.7,
    "TRP": -0.9,
    "TYR": -1.3,
    "VAL": 4.2,
}

SIDECHAIN_CHARGE = {
    "ASP": -1.0,
    "GLU": -1.0,
    "ARG": 1.0,
    "LYS": 1.0,
    "HIS": 0.0,
}

BACKBONE_ATOMS = {
    "N",
    "CA",
    "C",
    "O",
}


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def stop_with_error(message):
    print()
    print("=" * 78)
    print("STEP 25 FAILED")
    print("=" * 78)
    print(message)
    print()
    sys.exit(1)


def atom_element(atom):
    element = getattr(
        atom,
        "element",
        "",
    )

    if element:
        return str(element).upper()

    name = atom.get_name().strip().upper()

    if name.startswith("CL"):
        return "CL"

    if name.startswith("BR"):
        return "BR"

    if not name:
        return ""

    return name[0]


def is_hydrogen(atom):
    return atom_element(atom) == "H"


def is_heavy_atom(atom):
    return not is_hydrogen(atom)


def is_standard_protein_residue(residue):
    return (
        residue.resname.strip().upper()
        in {
            "ALA", "ARG", "ASN", "ASP", "CYS",
            "GLN", "GLU", "GLY", "HIS", "ILE",
            "LEU", "LYS", "MET", "PHE", "PRO",
            "SER", "THR", "TRP", "TYR", "VAL",
        }
    )


def residue_number(residue):
    try:
        return int(residue.id[1])
    except Exception:
        return None


def heavy_atoms(residue):
    return [
        atom
        for atom in residue.get_atoms()
        if is_heavy_atom(atom)
    ]


def sidechain_heavy_atoms(residue):
    return [
        atom
        for atom in residue.get_atoms()
        if (
            is_heavy_atom(atom)
            and atom.get_name().strip()
            not in BACKBONE_ATOMS
        )
    ]


def euclidean_distance(coord_a, coord_b):
    return float(
        np.linalg.norm(
            np.asarray(coord_a, dtype=float)
            - np.asarray(coord_b, dtype=float)
        )
    )


def residue_label(residue):
    number = residue_number(residue)

    return (
        f"{residue.resname.strip().upper()}{number}"
    )


def safe_float(value):
    try:
        return float(value)
    except Exception:
        return np.nan


# =============================================================================
# LOAD STRUCTURES
# =============================================================================

def load_first_model(path):

    if not path.exists():
        stop_with_error(
            f"Required structure was not found:\n{path}"
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

    if len(models) == 0:
        stop_with_error(
            f"No model was found in:\n{path}"
        )

    model = models[0]

    return structure, model


def get_protein_residues(model):

    residues = []

    for chain in model.get_chains():

        for residue in chain.get_residues():

            if not is_standard_protein_residue(
                residue
            ):
                continue

            residues.append(
                residue
            )

    return residues


# =============================================================================
# FIND TARGET RESIDUE
# =============================================================================

def find_target(model, expected_name):

    candidates = []

    for chain in model.get_chains():

        for residue in chain.get_residues():

            if not is_standard_protein_residue(
                residue
            ):
                continue

            number = residue_number(
                residue
            )

            if number != TARGET_RESIDUE:
                continue

            candidates.append(
                residue
            )

    if len(candidates) == 0:
        stop_with_error(
            f"Residue {TARGET_RESIDUE} was not found."
        )

    matching = [
        residue
        for residue in candidates
        if residue.resname.strip().upper()
        == expected_name
    ]

    if matching:
        return matching[0]

    return candidates[0]


# =============================================================================
# TARGET RESIDUE METRICS
# =============================================================================

def get_target_metrics(residue):

    name = (
        residue.resname
        .strip()
        .upper()
    )

    atoms = sidechain_heavy_atoms(
        residue
    )

    if atoms:

        coords = np.array(
            [
                atom.coord
                for atom in atoms
            ],
            dtype=float,
        )

        center = coords.mean(
            axis=0
        )

        radius = float(
            np.mean(
                np.linalg.norm(
                    coords - center,
                    axis=1,
                )
            )
        )

        min_x = float(coords[:, 0].min())
        max_x = float(coords[:, 0].max())
        min_y = float(coords[:, 1].min())
        max_y = float(coords[:, 1].max())
        min_z = float(coords[:, 2].min())
        max_z = float(coords[:, 2].max())

        span_x = max_x - min_x
        span_y = max_y - min_y
        span_z = max_z - min_z

    else:

        center = np.array(
            [np.nan, np.nan, np.nan]
        )

        radius = np.nan
        span_x = np.nan
        span_y = np.nan
        span_z = np.nan

    return {
        "Residue": f"{name}{TARGET_RESIDUE}",
        "Residue_Name": name,
        "Sidechain_Heavy_Atoms": len(atoms),
        "Sidechain_Center_X_A":
            float(center[0]),
        "Sidechain_Center_Y_A":
            float(center[1]),
        "Sidechain_Center_Z_A":
            float(center[2]),
        "Sidechain_Mean_Radius_A":
            radius,
        "Sidechain_Span_X_A":
            span_x,
        "Sidechain_Span_Y_A":
            span_y,
        "Sidechain_Span_Z_A":
            span_z,
        "Hydropathy":
            HYDROPATHY.get(
                name,
                np.nan,
            ),
        "Approximate_Sidechain_Charge":
            SIDECHAIN_CHARGE.get(
                name,
                0.0,
            ),
    }


# =============================================================================
# LOCAL RESIDUE ENVIRONMENT
# =============================================================================

def calculate_local_environment(
    model,
    target_residue,
):

    target_atoms = heavy_atoms(
        target_residue
    )

    rows = []

    for chain in model.get_chains():

        for residue in chain.get_residues():

            if not is_standard_protein_residue(
                residue
            ):
                continue

            if residue == target_residue:
                continue

            residue_atoms = heavy_atoms(
                residue
            )

            if not residue_atoms:
                continue

            minimum_distance = min(
                euclidean_distance(
                    target_atom.coord,
                    neighbor_atom.coord,
                )
                for target_atom in target_atoms
                for neighbor_atom in residue_atoms
            )

            if minimum_distance <= LOCAL_CUTOFF_A:

                rows.append(
                    {
                        "Chain":
                            str(chain.id),
                        "Residue_Number":
                            residue_number(residue),
                        "Residue_Name":
                            residue.resname
                            .strip()
                            .upper(),
                        "Residue":
                            residue_label(residue),
                        "Minimum_Distance_A":
                            minimum_distance,
                        "Hydropathy":
                            HYDROPATHY.get(
                                residue.resname
                                .strip()
                                .upper(),
                                np.nan,
                            ),
                        "Approximate_Charge":
                            SIDECHAIN_CHARGE.get(
                                residue.resname
                                .strip()
                                .upper(),
                                0.0,
                            ),
                    }
                )

    if not rows:

        return pd.DataFrame(
            columns=[
                "Chain",
                "Residue_Number",
                "Residue_Name",
                "Residue",
                "Minimum_Distance_A",
                "Hydropathy",
                "Approximate_Charge",
            ]
        )

    df = pd.DataFrame(
        rows
    )

    return df.sort_values(
        "Minimum_Distance_A"
    ).reset_index(
        drop=True
    )


# =============================================================================
# TARGET CONTACTS
# =============================================================================

def calculate_target_contacts(
    model,
    target_residue,
):

    target_atoms = heavy_atoms(
        target_residue
    )

    rows = []

    for chain in model.get_chains():

        for residue in chain.get_residues():

            if not is_standard_protein_residue(
                residue
            ):
                continue

            if residue == target_residue:
                continue

            for target_atom in target_atoms:

                for neighbor_atom in heavy_atoms(
                    residue
                ):

                    d = euclidean_distance(
                        target_atom.coord,
                        neighbor_atom.coord,
                    )

                    if d <= CONTACT_CUTOFF_A:

                        rows.append(
                            {
                                "Chain":
                                    str(chain.id),
                                "Residue_Number":
                                    residue_number(residue),
                                "Residue_Name":
                                    residue.resname
                                    .strip()
                                    .upper(),
                                "Residue":
                                    residue_label(residue),
                                "Target_Atom":
                                    target_atom
                                    .get_name()
                                    .strip(),
                                "Neighbor_Atom":
                                    neighbor_atom
                                    .get_name()
                                    .strip(),
                                "Target_Element":
                                    atom_element(
                                        target_atom
                                    ),
                                "Neighbor_Element":
                                    atom_element(
                                        neighbor_atom
                                    ),
                                "Distance_A":
                                    d,
                            }
                        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "Chain",
                "Residue_Number",
                "Residue_Name",
                "Residue",
                "Target_Atom",
                "Neighbor_Atom",
                "Target_Element",
                "Neighbor_Element",
                "Distance_A",
            ]
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "Distance_A"
    ).reset_index(
        drop=True
    )


# =============================================================================
# RESIDUE-LEVEL CONTACT SUMMARY
# =============================================================================

def summarize_contacts(
    contacts
):

    if contacts.empty:

        return pd.DataFrame(
            columns=[
                "Residue",
                "Contact_Count",
                "Closest_Contact_A",
            ]
        )

    summary = (
        contacts
        .groupby(
            "Residue",
            as_index=False,
        )
        .agg(
            Contact_Count=(
                "Residue",
                "size",
            ),
            Closest_Contact_A=(
                "Distance_A",
                "min",
            ),
        )
    )

    return summary.sort_values(
        "Closest_Contact_A"
    ).reset_index(
        drop=True
    )


# =============================================================================
# COMPARE CONTACTS
# =============================================================================

def build_contact_comparison(
    wt_contacts,
    mut_contacts,
):

    def signature(df):

        sig = {}

        for _, row in df.iterrows():

            key = (
                row["Residue"],
                row["Target_Atom"],
                row["Neighbor_Atom"],
            )

            sig[key] = float(
                row["Distance_A"]
            )

        return sig

    wt_sig = signature(
        wt_contacts
    )

    mut_sig = signature(
        mut_contacts
    )

    keys = sorted(
        set(wt_sig.keys())
        | set(mut_sig.keys())
    )

    rows = []

    for key in keys:

        residue, target_atom, neighbor_atom = key

        wt_d = wt_sig.get(
            key,
            np.nan,
        )

        mut_d = mut_sig.get(
            key,
            np.nan,
        )

        if np.isnan(wt_d) and not np.isnan(mut_d):

            status = "GAINED"

        elif not np.isnan(wt_d) and np.isnan(mut_d):

            status = "LOST"

        elif not np.isnan(wt_d) and not np.isnan(mut_d):

            delta = mut_d - wt_d

            if abs(delta) <= 0.25:
                status = "STABLE"
            elif delta < 0:
                status = "CLOSER"
            else:
                status = "FARTHER"

        else:

            status = "UNDEFINED"

        if (
            not np.isnan(wt_d)
            and not np.isnan(mut_d)
        ):

            delta = mut_d - wt_d

        else:

            delta = np.nan

        rows.append(
            {
                "Residue":
                    residue,
                "Target_Atom":
                    target_atom,
                "Neighbor_Atom":
                    neighbor_atom,
                "WT_Distance_A":
                    wt_d,
                "D166V_Distance_A":
                    mut_d,
                "Delta_Distance_A":
                    delta,
                "Status":
                    status,
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "Residue",
                "Target_Atom",
                "Neighbor_Atom",
                "WT_Distance_A",
                "D166V_Distance_A",
                "Delta_Distance_A",
                "Status",
            ]
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MATCH WT / MUTANT LOCAL ENVIRONMENT
# =============================================================================

def compare_local_environments(
    wt_environment,
    mut_environment,
):

    wt = wt_environment[
        [
            "Residue",
            "Residue_Name",
            "Minimum_Distance_A",
            "Hydropathy",
            "Approximate_Charge",
        ]
    ].copy()

    mut = mut_environment[
        [
            "Residue",
            "Residue_Name",
            "Minimum_Distance_A",
            "Hydropathy",
            "Approximate_Charge",
        ]
    ].copy()

    wt.columns = [
        "Residue",
        "WT_Residue_Name",
        "WT_Minimum_Distance_A",
        "WT_Hydropathy",
        "WT_Approximate_Charge",
    ]

    mut.columns = [
        "Residue",
        "D166V_Residue_Name",
        "D166V_Minimum_Distance_A",
        "D166V_Hydropathy",
        "D166V_Approximate_Charge",
    ]

    merged = pd.merge(
        wt,
        mut,
        on="Residue",
        how="outer",
    )

    merged[
        "Delta_Minimum_Distance_A"
    ] = (
        merged["D166V_Minimum_Distance_A"]
        - merged["WT_Minimum_Distance_A"]
    )

    merged[
        "Environment_Status"
    ] = "SHARED"

    merged.loc[
        merged["WT_Residue_Name"].isna(),
        "Environment_Status"
    ] = "D166V_ONLY"

    merged.loc[
        merged["D166V_Residue_Name"].isna(),
        "Environment_Status"
    ] = "WT_ONLY"

    return merged.sort_values(
        [
            "Environment_Status",
            "WT_Minimum_Distance_A",
            "D166V_Minimum_Distance_A",
        ],
        na_position="last",
    ).reset_index(
        drop=True
    )


# =============================================================================
# STRUCTURAL BACKBONE COMPARISON
# =============================================================================

def common_ca_coordinates(
    wt_model,
    mut_model,
):

    wt_atoms = {}
    mut_atoms = {}

    for chain in wt_model.get_chains():

        for residue in chain.get_residues():

            if not is_standard_protein_residue(
                residue
            ):
                continue

            number = residue_number(
                residue
            )

            if number is None:
                continue

            if "CA" in residue:

                wt_atoms[
                    (
                        str(chain.id),
                        number,
                    )
                ] = np.array(
                    residue["CA"].coord,
                    dtype=float,
                )

    for chain in mut_model.get_chains():

        for residue in chain.get_residues():

            if not is_standard_protein_residue(
                residue
            ):
                continue

            number = residue_number(
                residue
            )

            if number is None:
                continue

            if "CA" in residue:

                mut_atoms[
                    (
                        str(chain.id),
                        number,
                    )
                ] = np.array(
                    residue["CA"].coord,
                    dtype=float,
                )

    keys = sorted(
        set(wt_atoms.keys())
        & set(mut_atoms.keys())
    )

    if not keys:

        return (
            np.empty((0, 3)),
            np.empty((0, 3)),
            [],
        )

    wt_xyz = np.array(
        [
            wt_atoms[key]
            for key in keys
        ],
        dtype=float,
    )

    mut_xyz = np.array(
        [
            mut_atoms[key]
            for key in keys
        ],
        dtype=float,
    )

    return (
        wt_xyz,
        mut_xyz,
        keys,
    )


def rmsd_without_superposition(
    wt_xyz,
    mut_xyz,
):

    if len(wt_xyz) == 0:
        return np.nan

    diff = (
        mut_xyz - wt_xyz
    )

    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    diff * diff,
                    axis=1,
                )
            )
        )
    )


def local_ca_rmsd(
    wt_xyz,
    mut_xyz,
    keys,
):

    selected = []

    for i, key in enumerate(keys):

        chain_id, number = key

        if number == TARGET_RESIDUE:
            selected.append(i)
            continue

        # Residues near 166 are selected later using actual coordinates.
        # This initial list only ensures residue-number identity is available.
        distance_from_target = abs(
            number - TARGET_RESIDUE
        )

        if distance_from_target <= 20:
            selected.append(i)

    if not selected:
        return np.nan, 0

    wt_local = wt_xyz[
        selected,
        :
    ]

    mut_local = mut_xyz[
        selected,
        :
    ]

    diff = (
        mut_local - wt_local
    )

    rmsd = float(
        np.sqrt(
            np.mean(
                np.sum(
                    diff * diff,
                    axis=1,
                )
            )
        )
    )

    return (
        rmsd,
        len(selected),
    )


# =============================================================================
# SIDE-CHAIN CENTER DISPLACEMENT
# =============================================================================

def sidechain_center_displacement(
    wt_target,
    mut_target,
):

    wt_atoms = sidechain_heavy_atoms(
        wt_target
    )

    mut_atoms = sidechain_heavy_atoms(
        mut_target
    )

    if not wt_atoms or not mut_atoms:
        return np.nan

    wt_center = np.mean(
        np.array(
            [
                atom.coord
                for atom in wt_atoms
            ],
            dtype=float,
        ),
        axis=0,
    )

    mut_center = np.mean(
        np.array(
            [
                atom.coord
                for atom in mut_atoms
            ],
            dtype=float,
        ),
        axis=0,
    )

    return euclidean_distance(
        wt_center,
        mut_center,
    )


# =============================================================================
# SAFE FIGURE WRITER
# =============================================================================

def safe_plot(
    plot_function,
    filename,
):

    try:

        plot_function()

        plt.savefig(
            FIGURE_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

        return True

    except Exception as exc:

        plt.close(
            "all"
        )

        print(
            f"Figure warning ({filename}): {exc}"
        )

        return False


# =============================================================================
# FIGURE 1 — LOCAL DISTANCES
# =============================================================================

def make_local_distance_plot(
    wt_environment,
    mut_environment,
):

    combined = pd.merge(
        wt_environment[
            [
                "Residue",
                "Minimum_Distance_A",
            ]
        ].rename(
            columns={
                "Minimum_Distance_A":
                    "WT"
            }
        ),
        mut_environment[
            [
                "Residue",
                "Minimum_Distance_A",
            ]
        ].rename(
            columns={
                "Minimum_Distance_A":
                    "D166V"
            }
        ),
        on="Residue",
        how="outer",
    )

    if combined.empty:
        return

    wt_values = (
        combined["WT"]
        .fillna(
            LOCAL_CUTOFF_A
        )
        .to_numpy()
    )

    mut_values = (
        combined["D166V"]
        .fillna(
            LOCAL_CUTOFF_A
        )
        .to_numpy()
    )

    labels = (
        combined["Residue"]
        .astype(str)
        .tolist()
    )

    order = np.argsort(
        np.minimum(
            wt_values,
            mut_values,
        )
    )

    wt_values = wt_values[
        order
    ]

    mut_values = mut_values[
        order
    ]

    labels = [
        labels[i]
        for i in order
    ]

    x = np.arange(
        len(labels)
    )

    width = 0.38

    plt.figure(
        figsize=(
            max(
                10,
                len(labels) * 0.40,
            ),
            6,
        )
    )

    plt.bar(
        x - width / 2,
        wt_values,
        width=width,
        label="WT",
    )

    plt.bar(
        x + width / 2,
        mut_values,
        width=width,
        label="D166V",
    )

    plt.xticks(
        x,
        labels,
        rotation=75,
        ha="right",
    )

    plt.ylabel(
        "Minimum distance to residue 166 (Å)"
    )

    plt.title(
        "D166V Local Residue Environment"
    )

    plt.legend()

    plt.tight_layout()


# =============================================================================
# FIGURE 2 — CONTACT REMODELING
# =============================================================================

def make_contact_status_plot(
    contact_comparison,
):

    if contact_comparison.empty:
        return

    categories = [
        "GAINED",
        "LOST",
        "CLOSER",
        "FARTHER",
        "STABLE",
    ]

    counts = [
        int(
            (
                contact_comparison["Status"]
                == category
            ).sum()
        )
        for category in categories
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        categories,
        counts,
    )

    plt.ylabel(
        "Atom-level contacts"
    )

    plt.title(
        "WT vs D166V Contact Remodeling"
    )

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()


# =============================================================================
# FIGURE 3 — PHYSICOCHEMICAL CHANGE
# =============================================================================

def make_physicochemical_plot(
    wt_metrics,
    mut_metrics,
):

    labels = [
        "Hydropathy",
        "Side-chain charge",
        "Heavy atoms",
    ]

    wt_values = [
        wt_metrics[
            "Hydropathy"
        ],
        wt_metrics[
            "Approximate_Sidechain_Charge"
        ],
        wt_metrics[
            "Sidechain_Heavy_Atoms"
        ],
    ]

    mut_values = [
        mut_metrics[
            "Hydropathy"
        ],
        mut_metrics[
            "Approximate_Sidechain_Charge"
        ],
        mut_metrics[
            "Sidechain_Heavy_Atoms"
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
        width=width,
        label="WT ASP166",
    )

    plt.bar(
        x + width / 2,
        mut_values,
        width=width,
        label="D166V VAL166",
    )

    plt.xticks(
        x,
        labels,
    )

    plt.ylabel(
        "Descriptor value"
    )

    plt.title(
        "D166V Physicochemical Change"
    )

    plt.legend()

    plt.tight_layout()


# =============================================================================
# FINAL SUMMARY TABLE
# =============================================================================

def make_summary(
    wt_structure,
    mut_structure,
    wt_model,
    mut_model,
    wt_residues,
    mut_residues,
    wt_target,
    mut_target,
    wt_metrics,
    mut_metrics,
    wt_environment,
    mut_environment,
    wt_contacts,
    mut_contacts,
    contact_comparison,
    backbone_rmsd,
    local_rmsd,
    local_ca_count,
    target_sidechain_displacement,
):

    gained = int(
        (
            contact_comparison["Status"]
            == "GAINED"
        ).sum()
    )

    lost = int(
        (
            contact_comparison["Status"]
            == "LOST"
        ).sum()
    )

    closer = int(
        (
            contact_comparison["Status"]
            == "CLOSER"
        ).sum()
    )

    farther = int(
        (
            contact_comparison["Status"]
            == "FARTHER"
        ).sum()
    )

    stable = int(
        (
            contact_comparison["Status"]
            == "STABLE"
        ).sum()
    )

    wt_env_residues = set(
        wt_environment[
            "Residue"
        ].astype(str)
    )

    mut_env_residues = set(
        mut_environment[
            "Residue"
        ].astype(str)
    )

    shared = (
        wt_env_residues
        & mut_env_residues
    )

    wt_only = (
        wt_env_residues
        - mut_env_residues
    )

    mut_only = (
        mut_env_residues
        - wt_env_residues
    )

    hydropathy_change = (
        mut_metrics[
            "Hydropathy"
        ]
        - wt_metrics[
            "Hydropathy"
        ]
    )

    charge_change = (
        mut_metrics[
            "Approximate_Sidechain_Charge"
        ]
        - wt_metrics[
            "Approximate_Sidechain_Charge"
        ]
    )

    heavy_atom_change = (
        mut_metrics[
            "Sidechain_Heavy_Atoms"
        ]
        - wt_metrics[
            "Sidechain_Heavy_Atoms"
        ]
    )

    summary = {
        "Variant":
            VARIANT,
        "Description":
            VARIANT_DESCRIPTION,
        "WT_residue_166":
            wt_target.resname.strip().upper(),
        "Mutant_residue_166":
            mut_target.resname.strip().upper(),
        "WT_protein_residues":
            len(wt_residues),
        "D166V_protein_residues":
            len(mut_residues),
        "WT_local_residue_count":
            len(wt_env_residues),
        "D166V_local_residue_count":
            len(mut_env_residues),
        "Shared_local_residue_count":
            len(shared),
        "WT_only_local_residue_count":
            len(wt_only),
        "D166V_only_local_residue_count":
            len(mut_only),
        "WT_contact_count":
            len(wt_contacts),
        "D166V_contact_count":
            len(mut_contacts),
        "Contact_gained_count":
            gained,
        "Contact_lost_count":
            lost,
        "Contact_closer_count":
            closer,
        "Contact_farther_count":
            farther,
        "Contact_stable_count":
            stable,
        "WT_hydropathy":
            wt_metrics[
                "Hydropathy"
            ],
        "D166V_hydropathy":
            mut_metrics[
                "Hydropathy"
            ],
        "Hydropathy_change":
            hydropathy_change,
        "WT_approximate_charge":
            wt_metrics[
                "Approximate_Sidechain_Charge"
            ],
        "D166V_approximate_charge":
            mut_metrics[
                "Approximate_Sidechain_Charge"
            ],
        "Charge_change":
            charge_change,
        "WT_sidechain_heavy_atoms":
            wt_metrics[
                "Sidechain_Heavy_Atoms"
            ],
        "D166V_sidechain_heavy_atoms":
            mut_metrics[
                "Sidechain_Heavy_Atoms"
            ],
        "Sidechain_heavy_atom_change":
            heavy_atom_change,
        "Target_sidechain_center_displacement_A":
            target_sidechain_displacement,
        "Global_CA_RMSD_without_superposition_A":
            backbone_rmsd,
        "Local_CA_RMSD_without_superposition_A":
            local_rmsd,
        "Local_CA_residue_count":
            local_ca_count,
        "Conclusion":
            (
                "D166V replaces negatively charged Asp166 "
                "with neutral, more hydrophobic Val166. "
                "The mechanistic significance is determined "
                "from local-environment remodeling, contact "
                "changes, and the physicochemical change at "
                "the functionally important residue. The "
                "result is a computational mechanism hypothesis "
                "and does not constitute experimental proof of "
                "disease causality."
            ),
    }

    return summary


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 25 — D166V FUNCTIONAL / STRUCTURAL MECHANISM ANALYSIS"
    )
    print("=" * 78)

    print(
        f"WT structure : {WT_PDB}"
    )

    print(
        f"Mutant       : {MUT_PDB}"
    )

    print(
        f"Target       : residue {TARGET_RESIDUE} ({VARIANT})"
    )

    print()

    # -------------------------------------------------------------------------
    # Load structures
    # -------------------------------------------------------------------------

    print("Loading WT structure...")

    wt_structure, wt_model = load_first_model(
        WT_PDB
    )

    print("Loading D166V structure...")

    mut_structure, mut_model = load_first_model(
        MUT_PDB
    )

    # -------------------------------------------------------------------------
    # Protein residues
    # -------------------------------------------------------------------------

    wt_residues = get_protein_residues(
        wt_model
    )

    mut_residues = get_protein_residues(
        mut_model
    )

    print(
        f"WT protein residues     : {len(wt_residues)}"
    )

    print(
        f"D166V protein residues  : {len(mut_residues)}"
    )

    if len(wt_residues) != EXPECTED_PROTEIN_LENGTH:
        print(
            f"WARNING: WT residue count differs from expected "
            f"{EXPECTED_PROTEIN_LENGTH}."
        )

    if len(mut_residues) != EXPECTED_PROTEIN_LENGTH:
        print(
            f"WARNING: D166V residue count differs from expected "
            f"{EXPECTED_PROTEIN_LENGTH}."
        )

    print()

    # -------------------------------------------------------------------------
    # Target residue
    # -------------------------------------------------------------------------

    wt_target = find_target(
        wt_model,
        EXPECTED_WT_RESIDUE,
    )

    mut_target = find_target(
        mut_model,
        EXPECTED_MUTANT_RESIDUE,
    )

    print(
        f"WT residue 166     : "
        f"{wt_target.resname.strip().upper()}"
    )

    print(
        f"D166V residue 166  : "
        f"{mut_target.resname.strip().upper()}"
    )

    if (
        wt_target.resname
        .strip()
        .upper()
        != EXPECTED_WT_RESIDUE
    ):
        stop_with_error(
            "WT residue 166 is not ASP."
        )

    if (
        mut_target.resname
        .strip()
        .upper()
        != EXPECTED_MUTANT_RESIDUE
    ):
        stop_with_error(
            "D166V residue 166 is not VAL."
        )

    print()

    # -------------------------------------------------------------------------
    # Target physicochemical metrics
    # -------------------------------------------------------------------------

    print(
        "Calculating residue-166 physicochemical metrics..."
    )

    wt_metrics = get_target_metrics(
        wt_target
    )

    mut_metrics = get_target_metrics(
        mut_target
    )

    target_metrics = pd.DataFrame(
        [
            {
                "Model": "WT",
                **wt_metrics,
            },
            {
                "Model": "D166V",
                **mut_metrics,
            },
        ]
    )

    print()

    # -------------------------------------------------------------------------
    # Local environments
    # -------------------------------------------------------------------------

    print(
        f"Analyzing residues within {LOCAL_CUTOFF_A:.1f} Å..."
    )

    wt_environment = calculate_local_environment(
        wt_model,
        wt_target,
    )

    mut_environment = calculate_local_environment(
        mut_model,
        mut_target,
    )

    print(
        f"WT local residues    : {len(wt_environment)}"
    )

    print(
        f"D166V local residues : {len(mut_environment)}"
    )

    print()

    # -------------------------------------------------------------------------
    # Target contacts
    # -------------------------------------------------------------------------

    print(
        "Calculating WT residue-166 contacts..."
    )

    wt_contacts = calculate_target_contacts(
        wt_model,
        wt_target,
    )

    print(
        f"WT contacts          : {len(wt_contacts)}"
    )

    print(
        "Calculating D166V residue-166 contacts..."
    )

    mut_contacts = calculate_target_contacts(
        mut_model,
        mut_target,
    )

    print(
        f"D166V contacts       : {len(mut_contacts)}"
    )

    print()

    # -------------------------------------------------------------------------
    # Contact comparison
    # -------------------------------------------------------------------------

    print(
        "Comparing WT vs D166V contacts..."
    )

    contact_comparison = build_contact_comparison(
        wt_contacts,
        mut_contacts,
    )

    print()

    # -------------------------------------------------------------------------
    # Environment comparison
    # -------------------------------------------------------------------------

    environment_comparison = compare_local_environments(
        wt_environment,
        mut_environment,
    )

    # -------------------------------------------------------------------------
    # Contact summaries
    # -------------------------------------------------------------------------

    wt_contact_summary = summarize_contacts(
        wt_contacts
    )

    mut_contact_summary = summarize_contacts(
        mut_contacts
    )

    # -------------------------------------------------------------------------
    # Backbone / CA comparison
    # -------------------------------------------------------------------------

    print(
        "Comparing common Cα coordinates..."
    )

    (
        wt_ca,
        mut_ca,
        common_keys,
    ) = common_ca_coordinates(
        wt_model,
        mut_model,
    )

    global_ca_rmsd = (
        rmsd_without_superposition(
            wt_ca,
            mut_ca,
        )
    )

    (
        local_ca_rmsd_value,
        local_ca_count,
    ) = local_ca_rmsd(
        wt_ca,
        mut_ca,
        common_keys,
    )

    print(
        f"Common Cα residues : {len(common_keys)}"
    )

    print(
        f"Global Cα RMSD      : "
        f"{global_ca_rmsd:.6f} Å"
    )

    print(
        f"Local Cα RMSD       : "
        f"{local_ca_rmsd_value:.6f} Å"
    )

    print()

    # -------------------------------------------------------------------------
    # Side-chain center displacement
    # -------------------------------------------------------------------------

    sidechain_displacement = (
        sidechain_center_displacement(
            wt_target,
            mut_target,
        )
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    summary = make_summary(
        wt_structure,
        mut_structure,
        wt_model,
        mut_model,
        wt_residues,
        mut_residues,
        wt_target,
        mut_target,
        wt_metrics,
        mut_metrics,
        wt_environment,
        mut_environment,
        wt_contacts,
        mut_contacts,
        contact_comparison,
        global_ca_rmsd,
        local_ca_rmsd_value,
        local_ca_count,
        sidechain_displacement,
    )

    summary_df = pd.DataFrame(
        [summary]
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    print(
        "Writing Step 25 tables..."
    )

    wt_environment.to_csv(
        TABLE_DIR
        / "STEP25_WT_D166_Local_Environment.csv",
        index=False,
    )

    mut_environment.to_csv(
        TABLE_DIR
        / "STEP25_D166V_Local_Environment.csv",
        index=False,
    )

    environment_comparison.to_csv(
        TABLE_DIR
        / "STEP25_WT_vs_D166V_Local_Environment.csv",
        index=False,
    )

    target_metrics.to_csv(
        TABLE_DIR
        / "STEP25_D166V_Target_Residue_Metrics.csv",
        index=False,
    )

    wt_contacts.to_csv(
        TABLE_DIR
        / "STEP25_WT_D166_Contacts.csv",
        index=False,
    )

    mut_contacts.to_csv(
        TABLE_DIR
        / "STEP25_D166V_Contacts.csv",
        index=False,
    )

    wt_contact_summary.to_csv(
        TABLE_DIR
        / "STEP25_WT_Residue_166_Contact_Summary.csv",
        index=False,
    )

    mut_contact_summary.to_csv(
        TABLE_DIR
        / "STEP25_D166V_Residue_166_Contact_Summary.csv",
        index=False,
    )

    contact_comparison.to_csv(
        TABLE_DIR
        / "STEP25_WT_vs_D166V_Contact_Comparison.csv",
        index=False,
    )

    summary_df.to_csv(
        TABLE_DIR
        / "STEP25_D166V_FUNCTIONAL_MECHANISM_SUMMARY.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    print(
        "Generating Step 25 figures..."
    )

    figure_status = {}

    figure_status[
        "Figure_25_01_Local_Environment_Distance.png"
    ] = safe_plot(
        lambda: make_local_distance_plot(
            wt_environment,
            mut_environment,
        ),
        "Figure_25_01_Local_Environment_Distance.png",
    )

    figure_status[
        "Figure_25_02_Contact_Remodeling.png"
    ] = safe_plot(
        lambda: make_contact_status_plot(
            contact_comparison,
        ),
        "Figure_25_02_Contact_Remodeling.png",
    )

    figure_status[
        "Figure_25_03_D166V_Physicochemical_Change.png"
    ] = safe_plot(
        lambda: make_physicochemical_plot(
            wt_metrics,
            mut_metrics,
        ),
        "Figure_25_03_D166V_Physicochemical_Change.png",
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    gained = int(
        (
            contact_comparison["Status"]
            == "GAINED"
        ).sum()
    )

    lost = int(
        (
            contact_comparison["Status"]
            == "LOST"
        ).sum()
    )

    qc = {
        "Step": 25,
        "Variant": VARIANT,
        "WT_structure_exists": WT_PDB.exists(),
        "D166V_structure_exists": MUT_PDB.exists(),
        "WT_residue_166": (
            wt_target.resname.strip().upper()
        ),
        "D166V_residue_166": (
            mut_target.resname.strip().upper()
        ),
        "WT_protein_residue_count":
            len(wt_residues),
        "D166V_protein_residue_count":
            len(mut_residues),
        "Expected_protein_residue_count":
            EXPECTED_PROTEIN_LENGTH,
        "Common_CA_count":
            len(common_keys),
        "WT_local_residue_count":
            len(wt_environment),
        "D166V_local_residue_count":
            len(mut_environment),
        "WT_contact_count":
            len(wt_contacts),
        "D166V_contact_count":
            len(mut_contacts),
        "Gained_contact_count":
            gained,
        "Lost_contact_count":
            lost,
        "Global_CA_RMSD_A":
            global_ca_rmsd,
        "Local_CA_RMSD_A":
            local_ca_rmsd_value,
        "Target_sidechain_center_displacement_A":
            sidechain_displacement,
        "Figure_generation":
            figure_status,
        "status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP25_QC.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    interpretation = (
        "Step 25 evaluates the molecular consequence of "
        "PNPLA3 D166V by comparing independent WT and "
        "D166V structures. The analysis measures the "
        "physicochemical change from Asp166 to Val166, "
        "the local residue environment, residue-166 atom-level "
        "contacts, and common Cα structural differences. "
        "D166V changes a negatively charged aspartate to a "
        "neutral, more hydrophobic valine. Contact gains/losses "
        "and local structural differences are used to formulate "
        "a computationally testable mechanism hypothesis. "
        "These observations do not constitute experimental "
        "proof of disease causality."
    )

    with open(
        QC_DIR
        / "STEP25_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            interpretation
        )

    # -------------------------------------------------------------------------
    # Final printed results
    # -------------------------------------------------------------------------

    hydropathy_change = (
        mut_metrics["Hydropathy"]
        - wt_metrics["Hydropathy"]
    )

    charge_change = (
        mut_metrics[
            "Approximate_Sidechain_Charge"
        ]
        - wt_metrics[
            "Approximate_Sidechain_Charge"
        ]
    )

    print()
    print("=" * 78)
    print("STEP 25 RESULTS")
    print("=" * 78)

    print(
        f"WT residue 166                  : "
        f"{wt_target.resname.strip().upper()}"
    )

    print(
        f"D166V residue 166               : "
        f"{mut_target.resname.strip().upper()}"
    )

    print(
        f"WT local residues               : "
        f"{len(wt_environment)}"
    )

    print(
        f"D166V local residues            : "
        f"{len(mut_environment)}"
    )

    print(
        f"WT residue-166 contacts         : "
        f"{len(wt_contacts)}"
    )

    print(
        f"D166V residue-166 contacts      : "
        f"{len(mut_contacts)}"
    )

    print(
        f"Contacts gained                 : "
        f"{gained}"
    )

    print(
        f"Contacts lost                   : "
        f"{lost}"
    )

    print(
        f"WT hydropathy                   : "
        f"{wt_metrics['Hydropathy']:.3f}"
    )

    print(
        f"D166V hydropathy                : "
        f"{mut_metrics['Hydropathy']:.3f}"
    )

    print(
        f"Hydropathy change               : "
        f"{hydropathy_change:+.3f}"
    )

    print(
        f"WT approximate side-chain charge: "
        f"{wt_metrics['Approximate_Sidechain_Charge']:+.1f}"
    )

    print(
        f"D166V approximate side-chain charge: "
        f"{mut_metrics['Approximate_Sidechain_Charge']:+.1f}"
    )

    print(
        f"Charge change                   : "
        f"{charge_change:+.1f}"
    )

    print(
        f"Side-chain center displacement  : "
        f"{sidechain_displacement:.6f} Å"
    )

    print(
        f"Global Cα RMSD                  : "
        f"{global_ca_rmsd:.6f} Å"
    )

    print(
        f"Local Cα RMSD                   : "
        f"{local_ca_rmsd_value:.6f} Å"
    )

    print(
        f"Common Cα residues              : "
        f"{len(common_keys)}"
    )

    print()
    print(
        "Primary computational interpretation:"
    )

    print(
        "D166V replaces negatively charged Asp166 with "
        "neutral, more hydrophobic Val166. The next "
        "mechanistic assessment should determine whether "
        "the altered residue environment and contact network "
        "provide a biologically meaningful functional disruption."
    )

    print()
    print(
        "Step 25 completed successfully."
    )

    print(
        f"Summary table : "
        f"{TABLE_DIR / 'STEP25_D166V_FUNCTIONAL_MECHANISM_SUMMARY.csv'}"
    )

    print(
        f"QC file      : "
        f"{QC_DIR / 'STEP25_QC.json'}"
    )

    print()


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Step 25 interrupted by user."
        )
        sys.exit(130)

    except SystemExit:

        raise

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 25 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        traceback.print_exc()
        sys.exit(1)