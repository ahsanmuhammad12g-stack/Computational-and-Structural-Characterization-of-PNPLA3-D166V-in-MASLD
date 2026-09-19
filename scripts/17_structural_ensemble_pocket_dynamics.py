# =============================================================================
# STEP 17 — FINAL STRUCTURAL REMODELING & POCKET SENSITIVITY ANALYSIS
# =============================================================================
#
# PURPOSE
# -------
# Finalize Step 17 using the experimentally/model-derived WT and D166V PDB
# structures already prepared in Step 10.
#
# This step deliberately does NOT:
#   - perform molecular dynamics
#   - generate artificial random coordinate noise
#   - claim time-dependent dynamics
#   - rerun fpocket
#   - invent mutation-induced structural differences
#
# Instead, it evaluates:
#   1. Global WT vs D166V backbone similarity
#   2. Local structural remodeling around residue 166
#   3. Asp166 -> Val166 side-chain remodeling
#   4. Neighboring-residue displacement
#   5. Pocket center and compactness
#   6. Pocket-lining residue conservation
#   7. Residue-166 contact changes
#   8. Overall structural consequence of D166V
#
# The analysis is intentionally conservative:
#   - ~0.02 Å backbone differences are interpreted as negligible remodeling.
#   - Side-chain chemical changes are analyzed separately.
#   - No dynamic claims are made.
#
# REQUIREMENTS
# ------------
# pip install biopython pandas numpy matplotlib
#
# =============================================================================

from pathlib import Path
import json
import math
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser, Superimposer


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

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

STEP16_TABLE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
    / "tables"
    / "Table_16_04_D166V_Pocket_Prioritization.csv"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP17_FINAL_STRUCTURAL_REMODELING"
)

TABLE_DIR = OUT_ROOT / "tables"
FIG_DIR = OUT_ROOT / "figures"
QC_DIR = OUT_ROOT / "QC"

for directory in [TABLE_DIR, FIG_DIR, QC_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

TARGET_RESIDUE = 166

# Step 16 Pocket 1 was identified as the highest-priority therapeutic pocket.
POCKET_ID = 1

# Structural definition used when an explicit residue list is unavailable.
POCKET_CA_CUTOFF = 10.0

# Local remodeling neighborhood.
LOCAL_CA_CUTOFF = 10.0

# Contact analysis around residue 166.
CONTACT_CUTOFF = 5.0

# Heavy atom definition.
BACKBONE_ATOMS = ["N", "CA", "C", "O", "CB"]


# =============================================================================
# 3. UTILITY FUNCTIONS
# =============================================================================

def fail(message):
    print("\nERROR:")
    print(message)
    print()
    sys.exit(1)


def atom_is_hydrogen(atom):
    element = getattr(atom, "element", "")
    if element:
        return str(element).upper() == "H"

    name = atom.get_name().strip().upper()
    return name.startswith("H")


def residue_label(residue):
    het, resseq, icode = residue.id

    if icode and str(icode).strip():
        return f"{residue.resname}{resseq}{icode.strip()}"

    return f"{residue.resname}{resseq}"


def residue_key(residue):
    het, resseq, icode = residue.id
    return int(resseq), str(icode).strip()


def get_chain(structure, preferred_chain="A"):
    if preferred_chain in structure[0]:
        return structure[0][preferred_chain]

    chains = list(structure[0].get_chains())

    if not chains:
        fail("No protein chain was found in the PDB structure.")

    return chains[0]


def get_residue(chain, residue_number):
    for residue in chain:
        het, resseq, icode = residue.id

        if het.strip() == "" and int(resseq) == residue_number:
            return residue

    return None


def get_ca(residue):
    return residue["CA"] if "CA" in residue else None


def distance(coord1, coord2):
    return float(np.linalg.norm(np.asarray(coord1) - np.asarray(coord2)))


def heavy_atoms(residue):
    return [
        atom
        for atom in residue.get_atoms()
        if not atom_is_hydrogen(atom)
    ]


def residue_distance_by_ca(res1, res2):
    ca1 = get_ca(res1)
    ca2 = get_ca(res2)

    if ca1 is None or ca2 is None:
        return np.nan

    return distance(ca1.coord, ca2.coord)


def get_common_backbone_atoms(wt_residue, mut_residue):
    common = []

    for atom_name in BACKBONE_ATOMS:
        if atom_name in wt_residue and atom_name in mut_residue:
            common.append(
                (
                    atom_name,
                    wt_residue[atom_name],
                    mut_residue[atom_name],
                )
            )

    return common


# =============================================================================
# 4. LOAD INPUTS
# =============================================================================

print("=" * 78)
print("STEP 17 — FINAL STRUCTURAL REMODELING & POCKET SENSITIVITY")
print("=" * 78)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nWT structure:")
print(WT_PDB)

print("\nD166V structure:")
print(MUT_PDB)

print("\nStep 16 pocket table:")
print(STEP16_TABLE)


if not WT_PDB.exists():
    fail(
        "WT PDB was not found.\n\n"
        f"Expected:\n{WT_PDB}\n\n"
        "Do not download another structure yet. "
        "Check the Step 10 output."
    )

if not MUT_PDB.exists():
    fail(
        "D166V PDB was not found.\n\n"
        f"Expected:\n{MUT_PDB}\n\n"
        "Do not download another structure yet. "
        "Check the Step 10 output."
    )

if not STEP16_TABLE.exists():
    fail(
        "Step 16 pocket prioritization table was not found.\n\n"
        f"Expected:\n{STEP16_TABLE}"
    )


# =============================================================================
# 5. LOAD STRUCTURES
# =============================================================================

parser = PDBParser(QUIET=True)

wt_structure = parser.get_structure("PNPLA3_WT", str(WT_PDB))
mut_structure = parser.get_structure("PNPLA3_D166V", str(MUT_PDB))

wt_chain = get_chain(wt_structure, "A")
mut_chain = get_chain(mut_structure, "A")

wt_166 = get_residue(wt_chain, TARGET_RESIDUE)
mut_166 = get_residue(mut_chain, TARGET_RESIDUE)

if wt_166 is None:
    fail("Residue 166 was not found in the WT structure.")

if mut_166 is None:
    fail("Residue 166 was not found in the D166V structure.")


print("\nWT chain:", wt_chain.id)
print("D166V chain:", mut_chain.id)

print("WT residue 166:", residue_label(wt_166))
print("D166V residue 166:", residue_label(mut_166))


# =============================================================================
# 6. VALIDATE MUTATION IDENTITY
# =============================================================================

wt_resname = wt_166.resname.strip()
mut_resname = mut_166.resname.strip()

mutation_identity = {
    "WT_residue": wt_resname,
    "Mutant_residue": mut_resname,
    "Expected_WT": "ASP",
    "Expected_mutant": "VAL",
    "Identity_valid": (
        wt_resname == "ASP" and mut_resname == "VAL"
    ),
}

if not mutation_identity["Identity_valid"]:
    fail(
        "Residue 166 does not match the expected Asp166 -> Val166 mutation.\n"
        f"Observed WT: {wt_resname}\n"
        f"Observed mutant: {mut_resname}"
    )


# =============================================================================
# 7. LOAD STEP 16 POCKET INFORMATION
# =============================================================================

step16 = pd.read_csv(STEP16_TABLE)

required_columns = [
    "D166V_Pocket_ID",
    "WT_Pocket_ID",
    "Match_Status",
    "Druggability",
    "Volume_A3",
    "D166_Distance_A",
    "D166_Directly_Lining",
    "Prioritization_Score",
]

missing = [c for c in required_columns if c not in step16.columns]

if missing:
    fail(
        "The Step 16 table is missing required columns:\n"
        + "\n".join(missing)
    )


pocket1 = step16[
    step16["D166V_Pocket_ID"].astype(str) == str(POCKET_ID)
]

if pocket1.empty:
    fail("Step 16 Pocket 1 was not found.")

pocket1 = pocket1.iloc[0]

print("\n" + "-" * 78)
print("STEP 16 THERAPEUTIC POCKET VALIDATION")
print("-" * 78)

print("Pocket ID:", pocket1["D166V_Pocket_ID"])
print("WT pocket:", pocket1["WT_Pocket_ID"])
print("Match status:", pocket1["Match_Status"])
print("Druggability:", pocket1["Druggability"])
print("Volume:", pocket1["Volume_A3"], "Å³")
print("D166 distance:", pocket1["D166_Distance_A"], "Å")
print("D166 directly lining:", pocket1["D166_Directly_Lining"])
print("Prioritization score:", pocket1["Prioritization_Score"])


# =============================================================================
# 8. IDENTIFY COMMON RESIDUES
# =============================================================================

wt_residues = {
    residue_key(res): res
    for res in wt_chain
    if res.id[0].strip() == "" and get_ca(res) is not None
}

mut_residues = {
    residue_key(res): res
    for res in mut_chain
    if res.id[0].strip() == "" and get_ca(res) is not None
}

common_keys = sorted(set(wt_residues) & set(mut_residues))

print("\nCommon CA residues:", len(common_keys))


# =============================================================================
# 9. GLOBAL Cα RMSD
# =============================================================================

wt_atoms_global = []
mut_atoms_global = []

for key in common_keys:
    wt_res = wt_residues[key]
    mut_res = mut_residues[key]

    wt_ca = get_ca(wt_res)
    mut_ca = get_ca(mut_res)

    if wt_ca is not None and mut_ca is not None:
        wt_atoms_global.append(wt_ca)
        mut_atoms_global.append(mut_ca)


global_superimposer = Superimposer()
global_superimposer.set_atoms(
    wt_atoms_global,
    mut_atoms_global
)

global_rmsd = float(global_superimposer.rms)


# =============================================================================
# 10. LOCAL 10 Å RESIDUE-166 SET
# =============================================================================

wt166_ca = get_ca(wt_166)
mut166_ca = get_ca(mut_166)

if wt166_ca is None or mut166_ca is None:
    fail("CA atom missing at residue 166.")

local_keys = []

for key in common_keys:
    wt_res = wt_residues[key]

    d = residue_distance_by_ca(wt_res, wt_166)

    if not np.isnan(d) and d <= LOCAL_CA_CUTOFF:
        local_keys.append(key)

print("Local residues within", LOCAL_CA_CUTOFF, "Å of WT residue 166:", len(local_keys))


# =============================================================================
# 11. LOCAL Cα RMSD AFTER GLOBAL SUPERPOSITION
# =============================================================================

# Apply global transformation to WT coordinates.
rotation, translation = global_superimposer.rotran

local_wt_coords = []
local_mut_coords = []

local_rows = []

for key in local_keys:

    wt_res = wt_residues[key]
    mut_res = mut_residues[key]

    wt_ca = get_ca(wt_res)
    mut_ca = get_ca(mut_res)

    wt_transformed = np.dot(wt_ca.coord, rotation) + translation

    d = distance(wt_transformed, mut_ca.coord)

    local_wt_coords.append(wt_transformed)
    local_mut_coords.append(mut_ca.coord)

    local_rows.append(
        {
            "Residue_Number": key[0],
            "Insertion_Code": key[1],
            "WT_Residue": wt_res.resname,
            "Mutant_Residue": mut_res.resname,
            "WT_Label": residue_label(wt_res),
            "Mutant_Label": residue_label(mut_res),
            "CA_Displacement_A": d,
            "Within_10A_of_D166": True,
        }
    )

local_rmsd = float(
    np.sqrt(
        np.mean(
            [
                np.sum(
                    (np.asarray(a) - np.asarray(b)) ** 2
                )
                for a, b in zip(local_wt_coords, local_mut_coords)
            ]
        )
    )
)


# =============================================================================
# 12. BACKBONE / CB COMPARISON AT RESIDUE 166
# =============================================================================

residue166_atom_rows = []

for atom_name, wt_atom, mut_atom in get_common_backbone_atoms(
    wt_166,
    mut_166
):

    d = distance(wt_atom.coord, mut_atom.coord)

    residue166_atom_rows.append(
        {
            "Atom": atom_name,
            "WT_Residue": wt_resname,
            "Mutant_Residue": mut_resname,
            "WT_X": float(wt_atom.coord[0]),
            "WT_Y": float(wt_atom.coord[1]),
            "WT_Z": float(wt_atom.coord[2]),
            "Mutant_X": float(mut_atom.coord[0]),
            "Mutant_Y": float(mut_atom.coord[1]),
            "Mutant_Z": float(mut_atom.coord[2]),
            "Absolute_Difference_A": d,
        }
    )

residue166_atom_df = pd.DataFrame(residue166_atom_rows)


# =============================================================================
# 13. SIDE-CHAIN ATOM INVENTORY
# =============================================================================

wt166_atoms = {
    atom.get_name().strip(): atom
    for atom in wt_166.get_atoms()
    if not atom_is_hydrogen(atom)
}

mut166_atoms = {
    atom.get_name().strip(): atom
    for atom in mut_166.get_atoms()
    if not atom_is_hydrogen(atom)
}

common_sidechain_atom_names = sorted(
    set(wt166_atoms) & set(mut166_atoms)
    - {"N", "CA", "C", "O", "CB"}
)

sidechain_rows = []

for name in common_sidechain_atom_names:

    d = distance(
        wt166_atoms[name].coord,
        mut166_atoms[name].coord
    )

    sidechain_rows.append(
        {
            "Atom": name,
            "WT_Atom": name,
            "Mutant_Atom": name,
            "Coordinate_Difference_A": d,
        }
    )

sidechain_df = pd.DataFrame(sidechain_rows)


# =============================================================================
# 14. POCKET DEFINITION
# =============================================================================
#
# Step 16 identified Pocket 1 as:
#   - matched WT/D166V pocket
#   - druggability 0.851
#   - volume 602.118 Å³
#   - D166 directly lining
#
# Because the Step 16 table does not contain an explicit residue list,
# Pocket 1 is represented here by the residues whose CA atoms fall within
# 10 Å of residue 166.
#
# This is a transparent structural proxy and is NOT a re-run of fpocket.
# =============================================================================

def get_pocket_residues(chain, center_residue, cutoff):
    center_ca = get_ca(center_residue)

    result = []

    for residue in chain:
        if residue.id[0].strip() != "":
            continue

        ca = get_ca(residue)

        if ca is None:
            continue

        d = distance(ca.coord, center_ca.coord)

        if d <= cutoff:
            result.append(residue)

    return result


wt_pocket_residues = get_pocket_residues(
    wt_chain,
    wt_166,
    POCKET_CA_CUTOFF
)

mut_pocket_residues = get_pocket_residues(
    mut_chain,
    mut_166,
    POCKET_CA_CUTOFF
)

wt_pocket_keys = {
    residue_key(r): r
    for r in wt_pocket_residues
}

mut_pocket_keys = {
    residue_key(r): r
    for r in mut_pocket_residues
}

common_pocket_keys = sorted(
    set(wt_pocket_keys) & set(mut_pocket_keys)
)

wt_only_pocket = sorted(
    set(wt_pocket_keys) - set(mut_pocket_keys)
)

mut_only_pocket = sorted(
    set(mut_pocket_keys) - set(wt_pocket_keys)
)


# =============================================================================
# 15. POCKET CENTER
# =============================================================================

def pocket_center(residues):
    coords = np.array(
        [get_ca(r).coord for r in residues],
        dtype=float
    )

    return np.mean(coords, axis=0)


wt_pocket_center = pocket_center(wt_pocket_residues)
mut_pocket_center = pocket_center(mut_pocket_residues)

pocket_center_displacement = distance(
    wt_pocket_center,
    mut_pocket_center
)


# =============================================================================
# 16. POCKET RADIUS OF GYRATION
# =============================================================================

def radius_of_gyration(residues):
    coords = np.array(
        [get_ca(r).coord for r in residues],
        dtype=float
    )

    center = np.mean(coords, axis=0)

    squared_distances = np.sum(
        (coords - center) ** 2,
        axis=1
    )

    return float(np.sqrt(np.mean(squared_distances)))


wt_rg = radius_of_gyration(wt_pocket_residues)
mut_rg = radius_of_gyration(mut_pocket_residues)


# =============================================================================
# 17. LOCAL RESIDUE DISPLACEMENT TABLE
# =============================================================================

local_df = pd.DataFrame(local_rows)

if not local_df.empty:

    local_df["Structural_Category"] = np.where(
        local_df["CA_Displacement_A"] < 0.25,
        "Minimal",
        np.where(
            local_df["CA_Displacement_A"] < 0.75,
            "Small",
            np.where(
                local_df["CA_Displacement_A"] < 1.5,
                "Moderate",
                "Large"
            )
        )
    )

    local_df = local_df.sort_values(
        "CA_Displacement_A",
        ascending=False
    )


# =============================================================================
# 18. RESIDUE-166 NEIGHBOR CONTACT ANALYSIS
# =============================================================================

def residue166_contacts(
    target_residue,
    chain,
    cutoff=5.0
):

    rows = []

    target_atoms = heavy_atoms(target_residue)

    for residue in chain:

        if residue.id[0].strip() != "":
            continue

        if residue_key(residue) == residue_key(target_residue):
            continue

        neighbor_min_distance = np.inf
        contact_atom_pairs = 0

        for atom1 in target_atoms:

            for atom2 in heavy_atoms(residue):

                d = distance(atom1.coord, atom2.coord)

                if d <= cutoff:

                    contact_atom_pairs += 1

                    if d < neighbor_min_distance:
                        neighbor_min_distance = d

        if contact_atom_pairs > 0:

            rows.append(
                {
                    "Residue_Number": residue_key(residue)[0],
                    "Residue": residue.resname,
                    "Label": residue_label(residue),
                    "Minimum_HeavyAtom_Distance_A": float(
                        neighbor_min_distance
                    ),
                    "Contact_Atom_Pairs": int(contact_atom_pairs),
                }
            )

    return pd.DataFrame(rows)


wt_contacts = residue166_contacts(
    wt_166,
    wt_chain,
    CONTACT_CUTOFF
)

mut_contacts = residue166_contacts(
    mut_166,
    mut_chain,
    CONTACT_CUTOFF
)

if not wt_contacts.empty:
    wt_contacts["Structure"] = "WT"

if not mut_contacts.empty:
    mut_contacts["Structure"] = "D166V"


# =============================================================================
# 19. CONTACT DIFFERENCE TABLE
# =============================================================================

wt_contact_dict = {}

for _, row in wt_contacts.iterrows():
    wt_contact_dict[int(row["Residue_Number"])] = row

mut_contact_dict = {}

for _, row in mut_contacts.iterrows():
    mut_contact_dict[int(row["Residue_Number"])] = row

all_contact_residues = sorted(
    set(wt_contact_dict) | set(mut_contact_dict)
)

contact_difference_rows = []

for resnum in all_contact_residues:

    wt_row = wt_contact_dict.get(resnum)
    mut_row = mut_contact_dict.get(resnum)

    wt_distance = (
        float(wt_row["Minimum_HeavyAtom_Distance_A"])
        if wt_row is not None
        else np.nan
    )

    mut_distance = (
        float(mut_row["Minimum_HeavyAtom_Distance_A"])
        if mut_row is not None
        else np.nan
    )

    wt_pairs = (
        int(wt_row["Contact_Atom_Pairs"])
        if wt_row is not None
        else 0
    )

    mut_pairs = (
        int(mut_row["Contact_Atom_Pairs"])
        if mut_row is not None
        else 0
    )

    if wt_row is not None and mut_row is not None:
        status = "Retained"
    elif wt_row is not None:
        status = "Lost_in_D166V"
    else:
        status = "Gained_in_D166V"

    residue_name = (
        wt_row["Residue"]
        if wt_row is not None
        else mut_row["Residue"]
    )

    contact_difference_rows.append(
        {
            "Residue_Number": resnum,
            "Residue": residue_name,
            "WT_Min_Distance_A": wt_distance,
            "D166V_Min_Distance_A": mut_distance,
            "WT_Contact_Pairs": wt_pairs,
            "D166V_Contact_Pairs": mut_pairs,
            "Contact_Status": status,
        }
    )

contact_difference_df = pd.DataFrame(
    contact_difference_rows
)


# =============================================================================
# 20. POCKET RESIDUE COMPARISON
# =============================================================================

pocket_rows = []

for key in sorted(
    set(wt_pocket_keys) | set(mut_pocket_keys)
):

    wt_res = wt_pocket_keys.get(key)
    mut_res = mut_pocket_keys.get(key)

    pocket_rows.append(
        {
            "Residue_Number": key[0],
            "WT_Residue": (
                wt_res.resname if wt_res is not None else ""
            ),
            "D166V_Residue": (
                mut_res.resname if mut_res is not None else ""
            ),
            "WT_Present": wt_res is not None,
            "D166V_Present": mut_res is not None,
            "Residue_Identity_Changed": (
                wt_res is not None
                and mut_res is not None
                and wt_res.resname != mut_res.resname
            ),
        }
    )

pocket_residue_df = pd.DataFrame(pocket_rows)


# =============================================================================
# 21. STRUCTURAL SUMMARY
# =============================================================================

mean_local_displacement = (
    float(local_df["CA_Displacement_A"].mean())
    if not local_df.empty
    else np.nan
)

max_local_displacement = (
    float(local_df["CA_Displacement_A"].max())
    if not local_df.empty
    else np.nan
)

pocket_lining_changed = int(
    pocket_residue_df["Residue_Identity_Changed"].sum()
)

wt_pocket_count = len(wt_pocket_residues)
mut_pocket_count = len(mut_pocket_residues)

pocket_rmsd = np.nan

if common_pocket_keys:

    pocket_wt = []
    pocket_mut = []

    for key in common_pocket_keys:

        wt_ca = get_ca(wt_pocket_keys[key])
        mut_ca = get_ca(mut_pocket_keys[key])

        wt_transformed = (
            np.dot(wt_ca.coord, rotation) + translation
        )

        pocket_wt.append(wt_transformed)
        pocket_mut.append(mut_ca.coord)

    pocket_wt = np.asarray(pocket_wt)
    pocket_mut = np.asarray(pocket_mut)

    pocket_rmsd = float(
        np.sqrt(
            np.mean(
                np.sum(
                    (pocket_wt - pocket_mut) ** 2,
                    axis=1
                )
            )
        )
    )


# =============================================================================
# 22. INTERPRETATION LOGIC
# =============================================================================

if global_rmsd < 0.10:
    global_category = "Very small global structural difference"
elif global_rmsd < 0.50:
    global_category = "Small global structural difference"
elif global_rmsd < 1.00:
    global_category = "Moderate global structural difference"
else:
    global_category = "Substantial global structural difference"


if local_rmsd < 0.10:
    local_category = "Minimal local backbone remodeling"
elif local_rmsd < 0.50:
    local_category = "Small local backbone remodeling"
elif local_rmsd < 1.00:
    local_category = "Moderate local remodeling"
else:
    local_category = "Substantial local remodeling"


if pocket_center_displacement < 0.25:
    pocket_center_category = "Minimal pocket-center displacement"
elif pocket_center_displacement < 0.75:
    pocket_center_category = "Small pocket-center displacement"
else:
    pocket_center_category = "Appreciable pocket-center displacement"


if abs(mut_rg - wt_rg) < 0.05:
    pocket_compactness_category = (
        "Essentially unchanged pocket compactness"
    )
else:
    pocket_compactness_category = (
        "Altered pocket compactness"
    )


if (
    global_rmsd < 0.10
    and local_rmsd < 0.10
    and pocket_center_displacement < 0.25
    and abs(mut_rg - wt_rg) < 0.05
):
    remodeling_conclusion = (
        "The D166V model shows no appreciable backbone or pocket-level "
        "geometric remodeling relative to WT. The modeled structural "
        "difference is therefore primarily localized to the Asp166-to-Val166 "
        "side-chain and chemical identity change."
    )
else:
    remodeling_conclusion = (
        "The D166V model shows measurable structural differences relative "
        "to WT. These differences should be interpreted as static model "
        "remodeling and not as evidence of time-dependent molecular dynamics."
    )


# =============================================================================
# 23. SUMMARY TABLE
# =============================================================================

summary_rows = [
    {
        "Metric": "Global C-alpha RMSD",
        "WT": "Reference",
        "D166V": "Compared with WT",
        "Value": global_rmsd,
        "Unit": "Å",
        "Interpretation": global_category,
    },
    {
        "Metric": "Local C-alpha RMSD around D166",
        "WT": "Reference",
        "D166V": "Compared with WT",
        "Value": local_rmsd,
        "Unit": "Å",
        "Interpretation": local_category,
    },
    {
        "Metric": "Maximum local C-alpha displacement",
        "WT": "Reference",
        "D166V": "Compared with WT",
        "Value": max_local_displacement,
        "Unit": "Å",
        "Interpretation": "Largest local backbone displacement",
    },
    {
        "Metric": "Pocket-center displacement",
        "WT": "Reference",
        "D166V": "Compared with WT",
        "Value": pocket_center_displacement,
        "Unit": "Å",
        "Interpretation": pocket_center_category,
    },
    {
        "Metric": "WT pocket radius of gyration",
        "WT": wt_rg,
        "D166V": np.nan,
        "Value": wt_rg,
        "Unit": "Å",
        "Interpretation": "Pocket compactness",
    },
    {
        "Metric": "D166V pocket radius of gyration",
        "WT": np.nan,
        "D166V": mut_rg,
        "Value": mut_rg,
        "Unit": "Å",
        "Interpretation": "Pocket compactness",
    },
    {
        "Metric": "Pocket Rg difference",
        "WT": wt_rg,
        "D166V": mut_rg,
        "Value": mut_rg - wt_rg,
        "Unit": "Å",
        "Interpretation": pocket_compactness_category,
    },
    {
        "Metric": "Pocket RMSD",
        "WT": "Reference",
        "D166V": "Compared with WT",
        "Value": pocket_rmsd,
        "Unit": "Å",
        "Interpretation": "Pocket-region structural difference",
    },
    {
        "Metric": "WT pocket residue count",
        "WT": wt_pocket_count,
        "D166V": mut_pocket_count,
        "Value": wt_pocket_count,
        "Unit": "residues",
        "Interpretation": "10 Å CA-defined pocket proxy",
    },
    {
        "Metric": "Pocket residue identities changed",
        "WT": wt_pocket_count,
        "D166V": mut_pocket_count,
        "Value": pocket_lining_changed,
        "Unit": "residues",
        "Interpretation": "Includes the D166V substitution",
    },
]


summary_df = pd.DataFrame(summary_rows)


# =============================================================================
# 24. SAVE TABLES
# =============================================================================

summary_path = TABLE_DIR / "Table_17_01_Final_Structural_Summary.csv"
atom_path = TABLE_DIR / "Table_17_02_Residue166_Atom_Comparison.csv"
local_path = TABLE_DIR / "Table_17_03_Local_Residue_Displacement.csv"
pocket_path = TABLE_DIR / "Table_17_04_Pocket_Residue_Comparison.csv"
contact_path = TABLE_DIR / "Table_17_05_D166_Contact_Comparison.csv"
sidechain_path = TABLE_DIR / "Table_17_06_Residue166_Sidechain_Comparison.csv"

summary_df.to_csv(summary_path, index=False)
residue166_atom_df.to_csv(atom_path, index=False)
local_df.to_csv(local_path, index=False)
pocket_residue_df.to_csv(pocket_path, index=False)
contact_difference_df.to_csv(contact_path, index=False)
sidechain_df.to_csv(sidechain_path, index=False)


# =============================================================================
# 25. FIGURE 1 — LOCAL RESIDUE DISPLACEMENT
# =============================================================================

if not local_df.empty:

    plot_df = local_df.sort_values(
        "Residue_Number"
    )

    plt.figure(figsize=(12, 6))

    plt.bar(
        plot_df["Residue_Number"].astype(str),
        plot_df["CA_Displacement_A"]
    )

    plt.axhline(
        0.25,
        linestyle="--",
        linewidth=1
    )

    plt.xlabel("Residue")
    plt.ylabel("Cα displacement after global superposition (Å)")
    plt.title("Local Structural Displacement Around Residue 166")
    plt.xticks(rotation=90)
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "Figure_17_01_Local_Residue_Displacement.png",
        dpi=300
    )

    plt.close()


# =============================================================================
# 26. FIGURE 2 — POCKET COMPACTNESS
# =============================================================================

plt.figure(figsize=(7, 6))

plt.bar(
    ["WT", "D166V"],
    [wt_rg, mut_rg]
)

plt.ylabel("Pocket radius of gyration (Å)")
plt.title("Pocket Compactness: WT vs D166V")
plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_17_02_Pocket_Compactness.png",
    dpi=300
)

plt.close()


# =============================================================================
# 27. FIGURE 3 — RESIDUE 166 COMMON ATOM DISPLACEMENT
# =============================================================================

if not residue166_atom_df.empty:

    plt.figure(figsize=(8, 6))

    plt.bar(
        residue166_atom_df["Atom"],
        residue166_atom_df["Absolute_Difference_A"]
    )

    plt.ylabel("Coordinate difference (Å)")
    plt.xlabel("Atom")
    plt.title("WT Asp166 vs D166V Val166: Common Atom Displacement")
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / "Figure_17_03_Residue166_Atom_Displacement.png",
        dpi=300
    )

    plt.close()


# =============================================================================
# 28. FIGURE 4 — GLOBAL VS LOCAL RMSD
# =============================================================================

plt.figure(figsize=(7, 6))

plt.bar(
    ["Global Cα RMSD", "Local Cα RMSD", "Pocket RMSD"],
    [
        global_rmsd,
        local_rmsd,
        pocket_rmsd if not np.isnan(pocket_rmsd) else 0
    ]
)

plt.ylabel("RMSD (Å)")
plt.title("WT vs D166V Structural Difference")
plt.xticks(rotation=20)
plt.tight_layout()

plt.savefig(
    FIG_DIR / "Figure_17_04_Global_Local_Pocket_RMSD.png",
    dpi=300
)

plt.close()


# =============================================================================
# 29. QC
# =============================================================================

qc = {
    "step": 17,
    "status": "COMPLETED",
    "analysis_type": (
        "Static structural remodeling and structure-derived "
        "pocket sensitivity analysis"
    ),
    "molecular_dynamics_performed": False,
    "random_coordinate_noise_used": False,
    "fpocket_rerun": False,
    "prody_used": False,
    "gromacs_used": False,
    "WT_structure_exists": WT_PDB.exists(),
    "D166V_structure_exists": MUT_PDB.exists(),
    "mutation_identity": mutation_identity,
    "step16_pocket_id": POCKET_ID,
    "step16_druggability": float(pocket1["Druggability"]),
    "step16_volume_A3": float(pocket1["Volume_A3"]),
    "step16_D166_directly_lining": bool(
        pocket1["D166_Directly_Lining"]
    ),
    "global_ca_rmsd_A": global_rmsd,
    "local_ca_rmsd_A": local_rmsd,
    "max_local_ca_displacement_A": max_local_displacement,
    "pocket_center_displacement_A": pocket_center_displacement,
    "wt_pocket_Rg_A": wt_rg,
    "d166v_pocket_Rg_A": mut_rg,
    "pocket_Rg_difference_A": mut_rg - wt_rg,
    "pocket_RMSD_A": (
        None if np.isnan(pocket_rmsd)
        else float(pocket_rmsd)
    ),
    "local_residue_count": len(local_df),
    "wt_pocket_residue_count": wt_pocket_count,
    "d166v_pocket_residue_count": mut_pocket_count,
    "pocket_residue_identity_changes": pocket_lining_changed,
    "interpretation": remodeling_conclusion,
}

with open(
    QC_DIR / "Step17_Final_QC_Summary.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        qc,
        f,
        indent=4
    )


# =============================================================================
# 30. FINAL INTERPRETATION REPORT
# =============================================================================

report_path = OUT_ROOT / "STEP17_FINAL_INTERPRETATION.txt"

report = f"""
===============================================================================
STEP 17 — FINAL STRUCTURAL REMODELING & POCKET SENSITIVITY
===============================================================================

Project:
MASLD–PNPLA3 computational variant prioritization

Mutation:
Asp166Val (D166V)

Step 16 therapeutic pocket:
Pocket 1

Step 16 druggability:
{float(pocket1["Druggability"]):.3f}

Step 16 pocket volume:
{float(pocket1["Volume_A3"]):.3f} Å³

Step 16 D166 directly lining:
{pocket1["D166_Directly_Lining"]}


------------------------------------------------------------------------------
STRUCTURAL COMPARISON
------------------------------------------------------------------------------

Global Cα RMSD:
{global_rmsd:.4f} Å

Local Cα RMSD around residue 166:
{local_rmsd:.4f} Å

Maximum local Cα displacement:
{max_local_displacement:.4f} Å

Pocket RMSD:
{pocket_rmsd:.4f} Å


------------------------------------------------------------------------------
POCKET GEOMETRY
------------------------------------------------------------------------------

WT pocket radius of gyration:
{wt_rg:.4f} Å

D166V pocket radius of gyration:
{mut_rg:.4f} Å

Pocket Rg difference:
{mut_rg - wt_rg:.4f} Å

Pocket-center displacement:
{pocket_center_displacement:.4f} Å


------------------------------------------------------------------------------
RESIDUE 166
------------------------------------------------------------------------------

WT residue:
{wt_resname}166

D166V residue:
{mut_resname}166

Common atom comparison:
{residue166_atom_df.to_string(index=False)}


------------------------------------------------------------------------------
SCIENTIFIC INTERPRETATION
------------------------------------------------------------------------------

{remodeling_conclusion}


The residue-level coordinate comparison should be interpreted separately from
the chemical mutation itself. Asp166 is replaced by Val166, producing a change
from an acidic, negatively charged side chain to a neutral hydrophobic,
branched side chain. The absence of substantial backbone displacement does not
mean that the mutation is chemically insignificant; rather, it indicates that
the current structural model represents the mutation primarily through local
side-chain/chemical remodeling rather than backbone rearrangement.


------------------------------------------------------------------------------
IMPORTANT LIMITATION
------------------------------------------------------------------------------

This analysis is a static structural comparison and structure-derived pocket
sensitivity assessment.

It is NOT molecular dynamics.

Therefore, this step does not establish:
- time-dependent conformational dynamics
- altered binding free energy
- altered ligand residence time
- dynamic druggability
- experimentally observed conformational changes

Those questions belong to the later molecular-dynamics and energetic stages of
the pipeline.


------------------------------------------------------------------------------
PIPELINE DECISION
------------------------------------------------------------------------------

Step 16 identified Pocket 1 as the highest-priority therapeutic pocket and
showed that D166 directly lines this pocket.

Step 17 evaluates whether the available WT and D166V structural models exhibit
appreciable geometric remodeling.

The interpretation should follow the numerical results above and should not
be modified to artificially increase the apparent mutation effect.

If the calculated RMSD and local displacement values remain very small, the
appropriate conclusion is that D166V causes limited static backbone/pocket
remodeling in the current model, while producing a chemically important
Asp-to-Val substitution at the pocket-lining catalytic residue.

This result is suitable for proceeding to compound-library and screening
stages, provided that later docking and molecular-dynamics analyses continue
to distinguish static structural evidence from dynamic or energetic evidence.

===============================================================================
END OF STEP 17
===============================================================================
"""

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(report)


# =============================================================================
# 31. FINAL CONSOLE OUTPUT
# =============================================================================

print("\n" + "=" * 78)
print("STEP 17 COMPLETED")
print("=" * 78)

print("\nMutation:")
print(f"  {wt_resname}166 -> {mut_resname}166")

print("\nGlobal Cα RMSD:")
print(f"  {global_rmsd:.4f} Å")

print("\nLocal Cα RMSD around residue 166:")
print(f"  {local_rmsd:.4f} Å")

print("\nMaximum local Cα displacement:")
print(f"  {max_local_displacement:.4f} Å")

print("\nPocket center displacement:")
print(f"  {pocket_center_displacement:.4f} Å")

print("\nPocket Rg:")
print(f"  WT    : {wt_rg:.4f} Å")
print(f"  D166V : {mut_rg:.4f} Å")
print(f"  ΔRg   : {mut_rg - wt_rg:.4f} Å")

print("\nPocket RMSD:")
print(
    f"  {pocket_rmsd:.4f} Å"
    if not np.isnan(pocket_rmsd)
    else "  Not available"
)

print("\nInterpretation:")
print(" ", remodeling_conclusion)

print("\nOutputs:")
print("  Tables  :", TABLE_DIR)
print("  Figures :", FIG_DIR)
print("  QC      :", QC_DIR)
print("  Report  :", report_path)

print("\nIMPORTANT:")
print("  This is NOT molecular dynamics.")
print("  No artificial coordinate noise was introduced.")
print("  No dynamic claims are made.")
print("  No differences are artificially amplified.")

print("\n" + "=" * 78)
print("STEP 17 IS READY FOR FINAL QC")
print("=" * 78)