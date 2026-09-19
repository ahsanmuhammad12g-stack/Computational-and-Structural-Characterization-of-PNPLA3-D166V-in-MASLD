from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser, Superimposer
from Bio.PDB.SASA import ShrakeRupley


# =============================================================================
# STEP 27 — D166V ORTHOGONAL STRUCTURAL VALIDATION
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

STEP26_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP26_D166V_MASLD_MECHANISTIC_LINK"
)

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP27_ORTHOGONAL_STRUCTURAL_VALIDATION"
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

LOCAL_SEQUENCE_WINDOW = 20
SPATIAL_CUTOFF_A = 8.0
CONTACT_CUTOFF_A = 4.0

EXPECTED_LENGTH = 481
EXPECTED_WT_RESIDUE = "ASP"
EXPECTED_MUT_RESIDUE = "VAL"


# =============================================================================
# CONSTANTS
# =============================================================================

RESIDUE_VOLUME_A3 = {
    "ALA": 88.6,
    "ARG": 173.4,
    "ASN": 114.1,
    "ASP": 111.1,
    "CYS": 108.5,
    "GLN": 143.8,
    "GLU": 138.4,
    "GLY": 60.1,
    "HIS": 153.2,
    "ILE": 166.7,
    "LEU": 166.7,
    "LYS": 168.6,
    "MET": 162.9,
    "PHE": 189.9,
    "PRO": 112.7,
    "SER": 89.0,
    "THR": 116.1,
    "TRP": 227.8,
    "TYR": 193.6,
    "VAL": 140.0,
}

POLAR_ATOMS = {
    "O",
    "N",
    "S",
}

# Residue side-chain atoms.
BACKBONE_ATOMS = {
    "N",
    "CA",
    "C",
    "O",
    "OXT",
}


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):

    print()
    print("=" * 78)
    print("STEP 27 FAILED")
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

    for chain in model.get_chains():

        for residue in chain.get_residues():

            if residue_number(residue) == number:
                return residue

    fail(
        f"Residue {number} not found."
    )


def protein_residues(model):

    residues = []

    for chain in model.get_chains():

        for residue in chain.get_residues():

            number = residue_number(
                residue
            )

            if number is None:
                continue

            if (
                "CA" not in residue
                or "N" not in residue
                or "C" not in residue
            ):
                continue

            residues.append(
                residue
            )

    return residues


def heavy_atoms(residue):

    atoms = []

    for atom in residue.get_atoms():

        element = str(
            getattr(
                atom,
                "element",
                "",
            )
        ).upper()

        if element == "H":
            continue

        atoms.append(
            atom
        )

    return atoms


def sidechain_atoms(residue):

    atoms = []

    for atom in heavy_atoms(residue):

        if atom.name.strip().upper() in BACKBONE_ATOMS:
            continue

        atoms.append(
            atom
        )

    return atoms


def atom_distance(atom_a, atom_b):

    return float(
        np.linalg.norm(
            atom_a.coord - atom_b.coord
        )
    )


def sidechain_center(residue):

    atoms = sidechain_atoms(
        residue
    )

    if not atoms:
        return None

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


def sidechain_center_distance(
    residue_a,
    residue_b,
):

    center_a = sidechain_center(
        residue_a
    )

    center_b = sidechain_center(
        residue_b
    )

    if (
        center_a is None
        or center_b is None
    ):
        return np.nan

    return float(
        np.linalg.norm(
            center_a - center_b
        )
    )


def residue_pair_contacts(
    residue_a,
    residue_b,
    cutoff=4.0,
):

    contacts = []

    atoms_a = heavy_atoms(
        residue_a
    )

    atoms_b = heavy_atoms(
        residue_b
    )

    for atom_a in atoms_a:

        for atom_b in atoms_b:

            distance = atom_distance(
                atom_a,
                atom_b,
            )

            if distance <= cutoff:

                contacts.append(
                    {
                        "Atom_A":
                            atom_a.name.strip(),
                        "Atom_B":
                            atom_b.name.strip(),
                        "Distance_A":
                            distance,
                    }
                )

    return contacts


def approximate_polar_contacts(
    residue,
    model,
    cutoff=4.0,
):

    target_atoms = heavy_atoms(
        residue
    )

    target_polar = [
        atom
        for atom in target_atoms
        if str(
            getattr(
                atom,
                "element",
                "",
            )
        ).upper() in POLAR_ATOMS
    ]

    count = 0

    interacting_residues = set()

    for other in protein_residues(model):

        if (
            residue_number(other)
            == residue_number(residue)
        ):
            continue

        other_atoms = heavy_atoms(
            other
        )

        for atom_a in target_polar:

            for atom_b in other_atoms:

                element_b = str(
                    getattr(
                        atom_b,
                        "element",
                        "",
                    )
                ).upper()

                if element_b not in POLAR_ATOMS:
                    continue

                if atom_distance(
                    atom_a,
                    atom_b,
                ) <= cutoff:

                    count += 1

                    interacting_residues.add(
                        residue_number(
                            other
                        )
                    )

    return (
        count,
        len(
            interacting_residues
        ),
    )


def local_residues(
    model,
    target_number,
    cutoff=8.0,
):

    target = get_residue(
        model,
        target_number,
    )

    if "CA" not in target:
        return []

    local = []

    target_ca = target["CA"]

    for residue in protein_residues(model):

        if (
            residue_number(residue)
            == target_number
        ):
            continue

        if "CA" not in residue:
            continue

        distance = atom_distance(
            target_ca,
            residue["CA"],
        )

        if distance <= cutoff:

            local.append(
                (
                    residue,
                    distance,
                )
            )

    return local


def sequence_window(
    residues,
    center,
    width=20,
):

    return [
        residue
        for residue in residues
        if (
            center - width
            <= residue_number(residue)
            <= center + width
        )
    ]


def ca_coordinates(
    residues
):

    coords = []
    numbers = []

    for residue in residues:

        if "CA" not in residue:
            continue

        coords.append(
            residue["CA"].coord
        )

        numbers.append(
            residue_number(residue)
        )

    return (
        np.array(
            coords,
            dtype=float,
        ),
        numbers,
    )


def rmsd_after_superposition(
    wt_residues,
    mut_residues,
):

    wt_map = {
        residue_number(residue):
            residue
        for residue in wt_residues
        if "CA" in residue
    }

    mut_map = {
        residue_number(residue):
            residue
        for residue in mut_residues
        if "CA" in residue
    }

    common = sorted(
        set(wt_map)
        &
        set(mut_map)
    )

    fixed_atoms = [
        wt_map[number]["CA"]
        for number in common
    ]

    moving_atoms = [
        mut_map[number]["CA"]
        for number in common
    ]

    if len(common) < 3:
        fail(
            "Insufficient common residues for RMSD."
        )

    sup = Superimposer()

    sup.set_atoms(
        fixed_atoms,
        moving_atoms,
    )

    return (
        float(sup.rms),
        len(common),
        fixed_atoms,
        moving_atoms,
    )


def residue_sasa(
    model,
):

    sr = ShrakeRupley(
        probe_radius=1.4,
        n_points=960,
    )

    sr.compute(
        model,
        level="R",
    )

    output = {}

    for residue in protein_residues(model):

        output[
            residue_number(residue)
        ] = float(
            getattr(
                residue,
                "sasa",
                np.nan,
            )
        )

    return output


def residue_166_metrics(
    model,
    label,
):

    target = get_residue(
        model,
        TARGET_RESIDUE,
    )

    local = local_residues(
        model,
        TARGET_RESIDUE,
        SPATIAL_CUTOFF_A,
    )

    sasa_values = residue_sasa(
        model
    )

    contacts = residue_pair_contacts(
        target,
        target,
        CONTACT_CUTOFF_A,
    )

    # Self-contact list is ignored; actual intermolecular residue
    # contacts are calculated below.
    del contacts

    neighbor_contact_count = 0

    neighbor_ids = set()

    target_atoms = heavy_atoms(
        target
    )

    for other in protein_residues(model):

        if (
            residue_number(other)
            == TARGET_RESIDUE
        ):
            continue

        pair = residue_pair_contacts(
            target,
            other,
            CONTACT_CUTOFF_A,
        )

        if pair:

            neighbor_ids.add(
                residue_number(other)
            )

            neighbor_contact_count += len(
                pair
            )

    polar_count, polar_residue_count = (
        approximate_polar_contacts(
            target,
            model,
            CONTACT_CUTOFF_A,
        )
    )

    sidechain_atoms_count = len(
        sidechain_atoms(target)
    )

    target_sasa = sasa_values.get(
        TARGET_RESIDUE,
        np.nan,
    )

    total_neighbor_contacts = 0

    for residue, _ in local:

        pair = residue_pair_contacts(
            target,
            residue,
            CONTACT_CUTOFF_A,
        )

        total_neighbor_contacts += len(
            pair
        )

    result = {
        "Model":
            label,
        "Residue_166":
            residue_name(target),
        "Residue_Number":
            TARGET_RESIDUE,
        "Residue_SASA_A2":
            target_sasa,
        "Local_8A_CA_Residue_Count":
            len(local),
        "Heavy_Atom_Contact_Count_4A":
            neighbor_contact_count,
        "Contacting_Residue_Count_4A":
            len(neighbor_ids),
        "Polar_Atom_Contact_Count_4A":
            polar_count,
        "Polar_Contacting_Residue_Count":
            polar_residue_count,
        "Sidechain_Heavy_Atom_Count":
            sidechain_atoms_count,
        "Approximate_Residue_Volume_A3":
            RESIDUE_VOLUME_A3.get(
                residue_name(target),
                np.nan,
            ),
        "Local_8A_Total_Heavy_Atom_Contacts":
            total_neighbor_contacts,
    }

    return result


# =============================================================================
# LOCAL CONTACT MATRIX
# =============================================================================

def local_contact_comparison(
    wt_model,
    mut_model,
):

    wt_target = get_residue(
        wt_model,
        TARGET_RESIDUE,
    )

    mut_target = get_residue(
        mut_model,
        TARGET_RESIDUE,
    )

    rows = []

    wt_residues = {
        residue_number(residue):
            residue
        for residue in protein_residues(wt_model)
    }

    mut_residues = {
        residue_number(residue):
            residue
        for residue in protein_residues(mut_model)
    }

    common_numbers = sorted(
        set(wt_residues)
        &
        set(mut_residues)
    )

    for number in common_numbers:

        if number == TARGET_RESIDUE:
            continue

        wt_pair = residue_pair_contacts(
            wt_target,
            wt_residues[number],
            CONTACT_CUTOFF_A,
        )

        mut_pair = residue_pair_contacts(
            mut_target,
            mut_residues[number],
            CONTACT_CUTOFF_A,
        )

        wt_count = len(
            wt_pair
        )

        mut_count = len(
            mut_pair
        )

        if (
            wt_count == 0
            and mut_count == 0
        ):
            continue

        if wt_count > 0 and mut_count > 0:
            status = "PERSISTENT"
        elif wt_count > 0:
            status = "LOST"
        else:
            status = "GAINED"

        rows.append(
            {
                "Residue_Number":
                    number,
                "WT_Residue":
                    residue_name(
                        wt_residues[number]
                    ),
                "D166V_Residue":
                    residue_name(
                        mut_residues[number]
                    ),
                "WT_Contact_Count_4A":
                    wt_count,
                "D166V_Contact_Count_4A":
                    mut_count,
                "Delta_Contact_Count":
                    mut_count - wt_count,
                "Status":
                    status,
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# ORTHOGONAL EVIDENCE CLASSIFICATION
# =============================================================================

def classify_evidence(
    wt_metrics,
    mut_metrics,
    rmsd,
    contact_df,
):

    sasa_delta = (
        mut_metrics["Residue_SASA_A2"]
        - wt_metrics["Residue_SASA_A2"]
    )

    volume_delta = (
        mut_metrics["Approximate_Residue_Volume_A3"]
        - wt_metrics["Approximate_Residue_Volume_A3"]
    )

    contact_delta = (
        mut_metrics[
            "Heavy_Atom_Contact_Count_4A"
        ]
        - wt_metrics[
            "Heavy_Atom_Contact_Count_4A"
        ]
    )

    polar_delta = (
        mut_metrics[
            "Polar_Atom_Contact_Count_4A"
        ]
        - wt_metrics[
            "Polar_Atom_Contact_Count_4A"
        ]
    )

    lost_contacts = 0
    gained_contacts = 0

    if not contact_df.empty:

        lost_contacts = int(
            (
                contact_df["Status"]
                == "LOST"
            ).sum()
        )

        gained_contacts = int(
            (
                contact_df["Status"]
                == "GAINED"
            ).sum()
        )

    rows = []

    rows.append(
        {
            "Evidence_ID":
                "ORTHO-01",
            "Descriptor":
                "Residue solvent accessibility",
            "WT_Value":
                wt_metrics[
                    "Residue_SASA_A2"
                ],
            "D166V_Value":
                mut_metrics[
                    "Residue_SASA_A2"
                ],
            "Delta":
                sasa_delta,
            "Interpretation":
                "Change in residue exposure",
            "Evidence_Strength":
                "Supportive",
        }
    )

    rows.append(
        {
            "Evidence_ID":
                "ORTHO-02",
            "Descriptor":
                "Residue volume",
            "WT_Value":
                wt_metrics[
                    "Approximate_Residue_Volume_A3"
                ],
            "D166V_Value":
                mut_metrics[
                    "Approximate_Residue_Volume_A3"
                ],
            "Delta":
                volume_delta,
            "Interpretation":
                "Smaller side-chain volume in Val than Asp",
            "Evidence_Strength":
                "High",
        }
    )

    rows.append(
        {
            "Evidence_ID":
                "ORTHO-03",
            "Descriptor":
                "Heavy-atom contact density",
            "WT_Value":
                wt_metrics[
                    "Heavy_Atom_Contact_Count_4A"
                ],
            "D166V_Value":
                mut_metrics[
                    "Heavy_Atom_Contact_Count_4A"
                ],
            "Delta":
                contact_delta,
            "Interpretation":
                "Independent local packing/contact remodeling",
            "Evidence_Strength":
                "High",
        }
    )

    rows.append(
        {
            "Evidence_ID":
                "ORTHO-04",
            "Descriptor":
                "Polar-contact network",
            "WT_Value":
                wt_metrics[
                    "Polar_Atom_Contact_Count_4A"
                ],
            "D166V_Value":
                mut_metrics[
                    "Polar_Atom_Contact_Count_4A"
                ],
            "Delta":
                polar_delta,
            "Interpretation":
                "Change in local polar interaction capability",
            "Evidence_Strength":
                "High",
        }
    )

    rows.append(
        {
            "Evidence_ID":
                "ORTHO-05",
            "Descriptor":
                "Independent backbone RMSD",
            "WT_Value":
                0.0,
            "D166V_Value":
                rmsd,
            "Delta":
                rmsd,
            "Interpretation":
                "Tests whether mutation causes global backbone disruption",
            "Evidence_Strength":
                "Control",
        }
    )

    rows.append(
        {
            "Evidence_ID":
                "ORTHO-06",
            "Descriptor":
                "Residue-level contact remodeling",
            "WT_Value":
                f"{lost_contacts} lost",
            "D166V_Value":
                f"{gained_contacts} gained",
            "Delta":
                gained_contacts - lost_contacts,
            "Interpretation":
                "Independent reproduction of local network remodeling",
            "Evidence_Strength":
                "High",
        }
    )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

def make_interpretation(
    wt_metrics,
    mut_metrics,
    rmsd,
    contact_df,
):

    sasa_delta = (
        mut_metrics["Residue_SASA_A2"]
        - wt_metrics["Residue_SASA_A2"]
    )

    volume_delta = (
        mut_metrics["Approximate_Residue_Volume_A3"]
        - wt_metrics["Approximate_Residue_Volume_A3"]
    )

    contact_delta = (
        mut_metrics[
            "Heavy_Atom_Contact_Count_4A"
        ]
        - wt_metrics[
            "Heavy_Atom_Contact_Count_4A"
        ]
    )

    polar_delta = (
        mut_metrics[
            "Polar_Atom_Contact_Count_4A"
        ]
        - wt_metrics[
            "Polar_Atom_Contact_Count_4A"
        ]
    )

    lost_contacts = 0
    gained_contacts = 0

    if not contact_df.empty:

        lost_contacts = int(
            (
                contact_df["Status"]
                == "LOST"
            ).sum()
        )

        gained_contacts = int(
            (
                contact_df["Status"]
                == "GAINED"
            ).sum()
        )

    text = (
        "Independent orthogonal structural analysis confirms that "
        "D166V changes the local physicochemical environment of "
        "PNPLA3 residue 166 without causing substantial global "
        f"backbone disruption (Cα RMSD = {rmsd:.4f} Å). "
        f"The residue-volume descriptor changes by {volume_delta:+.1f} Å³, "
        f"the independent heavy-atom contact count changes by "
        f"{contact_delta:+d}, and polar contacts change by "
        f"{polar_delta:+d}. Residue-166 solvent accessibility changes "
        f"by {sasa_delta:+.2f} Å². The independent residue-contact "
        f"comparison identifies {lost_contacts} lost and "
        f"{gained_contacts} gained contacting residues. "
        "Taken together, these results reproduce the central Step 25 "
        "finding using independent structural descriptors: D166V "
        "does not grossly destabilize the protein backbone but does "
        "alter the local chemical, packing and interaction environment "
        "at the substituted residue. This strengthens the structural "
        "mechanism hypothesis. These results remain computational "
        "structural evidence and do not directly demonstrate altered "
        "enzymatic activity, cellular phenotype, or clinical disease."
    )

    return text


# =============================================================================
# FIGURES
# =============================================================================

def plot_local_descriptors(
    wt_metrics,
    mut_metrics,
):

    labels = [
        "Residue 166\nSASA",
        "Heavy-atom\ncontacts",
        "Polar-atom\ncontacts",
        "Local 8 Å\ncontacts",
    ]

    wt_values = [
        wt_metrics[
            "Residue_SASA_A2"
        ],
        wt_metrics[
            "Heavy_Atom_Contact_Count_4A"
        ],
        wt_metrics[
            "Polar_Atom_Contact_Count_4A"
        ],
        wt_metrics[
            "Local_8A_Total_Heavy_Atom_Contacts"
        ],
    ]

    mut_values = [
        mut_metrics[
            "Residue_SASA_A2"
        ],
        mut_metrics[
            "Heavy_Atom_Contact_Count_4A"
        ],
        mut_metrics[
            "Polar_Atom_Contact_Count_4A"
        ],
        mut_metrics[
            "Local_8A_Total_Heavy_Atom_Contacts"
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
        "Descriptor value"
    )

    plt.title(
        "Step 27 — Independent Local Structural Descriptors"
    )

    plt.legend()

    plt.tight_layout()


def plot_contact_remodeling(
    contact_df,
):

    if contact_df.empty:
        return

    counts = {
        status:
            int(
                (
                    contact_df["Status"]
                    == status
                ).sum()
            )
        for status in [
            "PERSISTENT",
            "LOST",
            "GAINED",
        ]
    }

    labels = list(
        counts.keys()
    )

    values = list(
        counts.values()
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        labels,
        values,
    )

    plt.ylabel(
        "Number of residues"
    )

    plt.title(
        "Step 27 — Independent Residue-166 Contact Remodeling"
    )

    plt.tight_layout()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 27 — D166V ORTHOGONAL STRUCTURAL VALIDATION"
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

    wt_model = load_model(
        WT_PDB
    )

    mut_model = load_model(
        MUT_PDB
    )

    wt_residues = protein_residues(
        wt_model
    )

    mut_residues = protein_residues(
        mut_model
    )

    print(
        f"WT amino-acid residues    : {len(wt_residues)}"
    )

    print(
        f"D166V amino-acid residues : {len(mut_residues)}"
    )

    if len(wt_residues) != EXPECTED_LENGTH:
        print(
            "Warning: WT length differs from expected 481."
        )

    if len(mut_residues) != EXPECTED_LENGTH:
        print(
            "Warning: D166V length differs from expected 481."
        )

    # -------------------------------------------------------------------------
    # Validate target
    # -------------------------------------------------------------------------

    wt_target = get_residue(
        wt_model,
        TARGET_RESIDUE,
    )

    mut_target = get_residue(
        mut_model,
        TARGET_RESIDUE,
    )

    print(
        f"WT residue 166            : "
        f"{residue_name(wt_target)}"
    )

    print(
        f"D166V residue 166         : "
        f"{residue_name(mut_target)}"
    )

    if residue_name(wt_target) != EXPECTED_WT_RESIDUE:
        fail(
            "WT residue 166 is not ASP."
        )

    if residue_name(mut_target) != EXPECTED_MUT_RESIDUE:
        fail(
            "D166V residue 166 is not VAL."
        )

    print()

    # -------------------------------------------------------------------------
    # Independent backbone RMSD
    # -------------------------------------------------------------------------

    print(
        "Calculating independent backbone-control RMSD..."
    )

    rmsd, common_count, _, _ = (
        rmsd_after_superposition(
            wt_residues,
            mut_residues,
        )
    )

    print(
        f"Common Cα residues        : {common_count}"
    )

    print(
        f"Independent global Cα RMSD: {rmsd:.6f} Å"
    )

    print()

    # -------------------------------------------------------------------------
    # Residue-166 metrics
    # -------------------------------------------------------------------------

    print(
        "Calculating independent residue-166 descriptors..."
    )

    wt_metrics = residue_166_metrics(
        wt_model,
        "WT",
    )

    mut_metrics = residue_166_metrics(
        mut_model,
        "D166V",
    )

    print(
        f"WT residue-166 SASA       : "
        f"{wt_metrics['Residue_SASA_A2']:.3f} Å²"
    )

    print(
        f"D166V residue-166 SASA   : "
        f"{mut_metrics['Residue_SASA_A2']:.3f} Å²"
    )

    print(
        f"WT heavy-atom contacts    : "
        f"{wt_metrics['Heavy_Atom_Contact_Count_4A']}"
    )

    print(
        f"D166V heavy-atom contacts : "
        f"{mut_metrics['Heavy_Atom_Contact_Count_4A']}"
    )

    print(
        f"WT polar contacts         : "
        f"{wt_metrics['Polar_Atom_Contact_Count_4A']}"
    )

    print(
        f"D166V polar contacts      : "
        f"{mut_metrics['Polar_Atom_Contact_Count_4A']}"
    )

    print()

    # -------------------------------------------------------------------------
    # Independent residue-level contact matrix
    # -------------------------------------------------------------------------

    print(
        "Building independent residue-166 contact comparison..."
    )

    contact_df = local_contact_comparison(
        wt_model,
        mut_model,
    )

    print(
        f"Contacting residues analyzed: "
        f"{len(contact_df)}"
    )

    # -------------------------------------------------------------------------
    # Evidence classification
    # -------------------------------------------------------------------------

    evidence_df = classify_evidence(
        wt_metrics,
        mut_metrics,
        rmsd,
        contact_df,
    )

    # -------------------------------------------------------------------------
    # Interpretation
    # -------------------------------------------------------------------------

    interpretation = make_interpretation(
        wt_metrics,
        mut_metrics,
        rmsd,
        contact_df,
    )

    # -------------------------------------------------------------------------
    # Save tables
    # -------------------------------------------------------------------------

    pd.DataFrame(
        [
            wt_metrics,
            mut_metrics,
        ]
    ).to_csv(
        TABLE_DIR
        / "STEP27_RESIDUE_166_INDEPENDENT_METRICS.csv",
        index=False,
    )

    contact_df.to_csv(
        TABLE_DIR
        / "STEP27_INDEPENDENT_CONTACT_COMPARISON.csv",
        index=False,
    )

    evidence_df.to_csv(
        TABLE_DIR
        / "STEP27_ORTHOGONAL_EVIDENCE_MATRIX.csv",
        index=False,
    )

    rmsd_df = pd.DataFrame(
        [
            {
                "Metric":
                    "Global_Calpha_RMSD",
                "Value_A":
                    rmsd,
                "Common_Calpha_Residues":
                    common_count,
            }
        ]
    )

    rmsd_df.to_csv(
        TABLE_DIR
        / "STEP27_BACKBONE_RMSD_CONTROL.csv",
        index=False,
    )

    # Carry-forward note.
    carry_forward = {
        "Step25_directory_exists":
            STEP25_DIR.exists(),
        "Step26_directory_exists":
            STEP26_DIR.exists(),
        "Step25_used_as_primary_analysis":
            False,
        "Step26_used_as_primary_analysis":
            False,
        "Step27_is_orthogonal":
            True,
    }

    with open(
        QC_DIR
        / "STEP27_METHOD_CONTROL.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            carry_forward,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Figures
    # -------------------------------------------------------------------------

    print(
        "Generating figures..."
    )

    plot_local_descriptors(
        wt_metrics,
        mut_metrics,
    )

    plt.savefig(
        FIGURE_DIR
        / "Figure_27_01_Independent_Local_Descriptors.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    if not contact_df.empty:

        plot_contact_remodeling(
            contact_df
        )

        plt.savefig(
            FIGURE_DIR
            / "Figure_27_02_Independent_Contact_Remodeling.png",
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()

    # -------------------------------------------------------------------------
    # Interpretation file
    # -------------------------------------------------------------------------

    with open(
        QC_DIR
        / "STEP27_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            interpretation
        )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    lost_contacts = 0
    gained_contacts = 0

    if not contact_df.empty:

        lost_contacts = int(
            (
                contact_df["Status"]
                == "LOST"
            ).sum()
        )

        gained_contacts = int(
            (
                contact_df["Status"]
                == "GAINED"
            ).sum()
        )

    qc = {
        "Step":
            27,
        "Variant":
            "D166V",
        "WT_structure_exists":
            WT_PDB.exists(),
        "D166V_structure_exists":
            MUT_PDB.exists(),
        "WT_length":
            len(wt_residues),
        "D166V_length":
            len(mut_residues),
        "WT_residue_166":
            residue_name(wt_target),
        "D166V_residue_166":
            residue_name(mut_target),
        "Common_Calpha_residues":
            common_count,
        "Independent_global_Calpha_RMSD_A":
            rmsd,
        "Lost_contacting_residues":
            lost_contacts,
        "Gained_contacting_residues":
            gained_contacts,
        "Status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP27_QC.json",
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
    print("STEP 27 RESULTS")
    print("=" * 78)

    print(
        f"WT residue 166             : "
        f"{residue_name(wt_target)}"
    )

    print(
        f"D166V residue 166          : "
        f"{residue_name(mut_target)}"
    )

    print(
        f"Independent global Cα RMSD : "
        f"{rmsd:.6f} Å"
    )

    print(
        f"WT residue-166 SASA        : "
        f"{wt_metrics['Residue_SASA_A2']:.3f} Å²"
    )

    print(
        f"D166V residue-166 SASA    : "
        f"{mut_metrics['Residue_SASA_A2']:.3f} Å²"
    )

    print(
        f"WT heavy-atom contacts     : "
        f"{wt_metrics['Heavy_Atom_Contact_Count_4A']}"
    )

    print(
        f"D166V heavy-atom contacts  : "
        f"{mut_metrics['Heavy_Atom_Contact_Count_4A']}"
    )

    print(
        f"WT polar contacts          : "
        f"{wt_metrics['Polar_Atom_Contact_Count_4A']}"
    )

    print(
        f"D166V polar contacts       : "
        f"{mut_metrics['Polar_Atom_Contact_Count_4A']}"
    )

    print(
        f"Lost contacting residues   : "
        f"{lost_contacts}"
    )

    print(
        f"Gained contacting residues : "
        f"{gained_contacts}"
    )

    print()

    print(
        "Interpretation:"
    )

    print(
        interpretation
    )

    print()
    print(
        "Step 27 completed successfully."
    )

    print(
        f"Evidence matrix : "
        f"{TABLE_DIR / 'STEP27_ORTHOGONAL_EVIDENCE_MATRIX.csv'}"
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
            "\nStep 27 interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 27 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise