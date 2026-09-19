# =============================================================================
# STEP 11 — D166V CATALYTIC-CONTEXT AND STRUCTURAL-IMPACT ANALYSIS
# =============================================================================
#
# SCRIPT NAME:
#   11_D166V_catalytic_context.py
#
# PURPOSE:
#   1. Load and validate the authoritative Step 9F candidate dataset
#   2. Identify the primary PNPLA3 structural candidate D166V
#   3. Load the validated Step 10 WT and D166V structures
#   4. Validate residue 166 remodeling
#   5. Analyze Ser47–D166 and Pro186–D166 catalytic/structural context
#   6. Compare local WT vs D166V residue distances
#   7. Compare local heavy-atom contacts
#   8. Identify retained, lost, and gained contacts
#   9. Compare candidate polar heavy-atom contacts
#  10. Characterize the Asp166 -> Val166 physicochemical change
#  11. Generate publication-quality CSV tables and PNG figures
#  12. Generate final QC report and summary
#
# IMPORTANT:
#   Step 9F uses:
#       UniProt_ID
#       Protein_Position
#       Reference_Amino_Acid
#       Alternate_Amino_Acid
#
#   Position_GRCh38 is genomic and MUST NOT be used as protein position.
#
#   Candidate polar contacts are NOT automatically hydrogen bonds.
#   They are heavy-atom N/O/S proximity observations only.
#
# =============================================================================


# =============================================================================
# IMPORTS
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

STEP9F_FILE = (
    PROJECT_ROOT
    / "VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_step9f_final_candidate_selection_FINAL.csv"
)

STEP10_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
)

STEP11_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP11_D166V_CATALYTIC_CONTEXT"
)

WT_PDB = (
    STEP10_DIR
    / "structures"
    / "WT"
    / "PNPLA3_WT.pdb"
)

MUTANT_PDB = (
    STEP10_DIR
    / "structures"
    / "D166V"
    / "PNPLA3_D166V_mutant.pdb"
)

TABLE_DIR = STEP11_DIR / "tables"
FIGURE_DIR = STEP11_DIR / "figures"
QC_DIR = STEP11_DIR / "QC"

for directory in [
    STEP11_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# EXPECTED PRIMARY CANDIDATE
# =============================================================================

EXPECTED_VARIATION_ID = 3308171
EXPECTED_PROTEIN_CHANGE = "p.Asp166Val"
EXPECTED_UNIPROT_ID = "Q9NST1"
EXPECTED_PROTEIN_POSITION = 166
EXPECTED_REFERENCE_AA = "Asp"
EXPECTED_ALTERNATE_AA = "Val"

# Catalytic / structural context
SER47_POSITION = 47
D166_POSITION = 166
PRO186_POSITION = 186

# Local analysis window
LOCAL_RADIUS = 10

# Heavy-atom contact cutoff
CONTACT_CUTOFF_A = 4.5

# Candidate polar heavy-atom cutoff
POLAR_CONTACT_CUTOFF_A = 3.5

# Protein chain
CHAIN_ID = "A"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def print_subsection(title):
    print()
    print("-" * 78)
    print(title)
    print("-" * 78)


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


# =============================================================================
# PDB PARSER
# =============================================================================

def parse_pdb(pdb_file):
    """
    Parse standard ATOM records from a PDB file.

    Returns a DataFrame containing:
        chain
        resnum
        resname
        atom
        x
        y
        z
        element
    """

    records = []

    if not pdb_file.exists():
        raise FileNotFoundError(
            f"PDB file not found: {pdb_file}"
        )

    with open(
        pdb_file,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for line in handle:

            if not line.startswith("ATOM"):
                continue

            atom_name = line[12:16].strip()
            residue_name = line[17:20].strip().upper()
            chain = line[21:22].strip()
            residue_number_text = line[22:26].strip()

            if not residue_number_text:
                continue

            try:
                residue_number = int(
                    residue_number_text
                )
            except ValueError:
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            element = line[76:78].strip().upper()

            # Fallback if element column is absent
            if not element:
                element = atom_name[0].upper()

            records.append(
                {
                    "chain": chain,
                    "resnum": residue_number,
                    "resname": residue_name,
                    "atom": atom_name,
                    "x": x,
                    "y": y,
                    "z": z,
                    "element": element,
                }
            )

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(
            f"No ATOM records could be parsed from {pdb_file}"
        )

    return df


# =============================================================================
# RESIDUE FUNCTIONS
# =============================================================================

def get_residue_atoms(
    pdb_df,
    position,
    chain=CHAIN_ID,
):
    return pdb_df[
        (pdb_df["chain"] == chain)
        & (pdb_df["resnum"] == int(position))
    ].copy()


def get_residue_name(
    pdb_df,
    position,
    chain=CHAIN_ID,
):
    atoms = get_residue_atoms(
        pdb_df,
        position,
        chain,
    )

    if atoms.empty:
        return None

    return str(
        atoms["resname"].iloc[0]
    ).upper()


def get_heavy_atoms(
    residue_df,
):
    return residue_df[
        residue_df["element"] != "H"
    ].copy()


def calculate_atom_distance(atom1, atom2):
    vector = np.array(
        [
            atom1["x"],
            atom1["y"],
            atom1["z"],
        ],
        dtype=float,
    )

    vector2 = np.array(
        [
            atom2["x"],
            atom2["y"],
            atom2["z"],
        ],
        dtype=float,
    )

    return float(
        np.linalg.norm(vector - vector2)
    )


# =============================================================================
# MINIMUM RESIDUE DISTANCE
# =============================================================================

def minimum_residue_distance(
    pdb_df,
    position1,
    position2,
    chain=CHAIN_ID,
):
    """
    Calculate the minimum heavy-atom distance between two residues.

    Returns:
        distance_A
        atom_from_residue_1
        atom_from_residue_2

    For the same residue, the function excludes identical atom pairs.
    """

    residue1 = get_heavy_atoms(
        get_residue_atoms(
            pdb_df,
            position1,
            chain,
        )
    )

    residue2 = get_heavy_atoms(
        get_residue_atoms(
            pdb_df,
            position2,
            chain,
        )
    )

    if residue1.empty or residue2.empty:
        return np.nan, None, None

    best_distance = np.inf
    best_atom1 = None
    best_atom2 = None

    for _, atom1 in residue1.iterrows():

        for _, atom2 in residue2.iterrows():

            if (
                int(position1) == int(position2)
                and atom1["atom"] == atom2["atom"]
            ):
                continue

            distance = calculate_atom_distance(
                atom1,
                atom2,
            )

            if distance < best_distance:
                best_distance = distance
                best_atom1 = atom1["atom"]
                best_atom2 = atom2["atom"]

    if not np.isfinite(best_distance):
        return np.nan, None, None

    return (
        float(best_distance),
        best_atom1,
        best_atom2,
    )


# =============================================================================
# HEAVY-ATOM CONTACT ANALYSIS
# =============================================================================

def heavy_atom_contacts(
    pdb_df,
    candidate_position,
    cutoff=CONTACT_CUTOFF_A,
    chain=CHAIN_ID,
):
    """
    Calculate residue-level local heavy-atom contacts.

    A partner residue is considered a contact when at least one
    heavy-atom pair is <= cutoff Å.

    For each partner residue, only the closest atom pair is retained.

    IMPORTANT:
        The dictionary key is 'distance_A'.
        This avoids the previous KeyError caused by using
        'distance' instead of 'distance_A'.
    """

    candidate_atoms = get_heavy_atoms(
        get_residue_atoms(
            pdb_df,
            candidate_position,
            chain,
        )
    )

    if candidate_atoms.empty:
        return pd.DataFrame(
            columns=[
                "candidate_position",
                "candidate_residue",
                "candidate_atom",
                "partner_position",
                "partner_residue",
                "partner_atom",
                "distance_A",
            ]
        )

    candidate_residue_name = (
        candidate_atoms["resname"].iloc[0]
    )

    contacts = []

    partner_positions = sorted(
        pdb_df[
            pdb_df["chain"] == chain
        ]["resnum"].unique()
    )

    for partner_position in partner_positions:

        if int(partner_position) == int(candidate_position):
            continue

        partner_atoms = get_heavy_atoms(
            get_residue_atoms(
                pdb_df,
                partner_position,
                chain,
            )
        )

        if partner_atoms.empty:
            continue

        best = None

        for _, atom1 in candidate_atoms.iterrows():

            for _, atom2 in partner_atoms.iterrows():

                distance = calculate_atom_distance(
                    atom1,
                    atom2,
                )

                if distance <= cutoff:

                    # CORRECTED KEY:
                    # best["distance_A"]
                    if (
                        best is None
                        or distance < best["distance_A"]
                    ):

                        best = {
                            "candidate_position": int(
                                candidate_position
                            ),
                            "candidate_residue": str(
                                candidate_residue_name
                            ),
                            "candidate_atom": str(
                                atom1["atom"]
                            ),
                            "partner_position": int(
                                partner_position
                            ),
                            "partner_residue": str(
                                partner_atoms[
                                    "resname"
                                ].iloc[0]
                            ),
                            "partner_atom": str(
                                atom2["atom"]
                            ),
                            "distance_A": float(
                                distance
                            ),
                        }

        if best is not None:
            contacts.append(best)

    return pd.DataFrame(
        contacts,
        columns=[
            "candidate_position",
            "candidate_residue",
            "candidate_atom",
            "partner_position",
            "partner_residue",
            "partner_atom",
            "distance_A",
        ],
    )


# =============================================================================
# POLAR HEAVY-ATOM CONTACT ANALYSIS
# =============================================================================

def polar_heavy_atom_contacts(
    pdb_df,
    candidate_position,
    cutoff=POLAR_CONTACT_CUTOFF_A,
    chain=CHAIN_ID,
):
    """
    Identify N/O/S heavy-atom contacts <= cutoff Å.

    These are candidate polar contacts only.

    No hydrogen-bond donor/acceptor geometry or angle criteria
    are evaluated.
    """

    polar_elements = {
        "N",
        "O",
        "S",
    }

    candidate_atoms = get_residue_atoms(
        pdb_df,
        candidate_position,
        chain,
    )

    candidate_atoms = candidate_atoms[
        candidate_atoms["element"].isin(
            polar_elements
        )
    ]

    if candidate_atoms.empty:
        return pd.DataFrame(
            columns=[
                "candidate_position",
                "candidate_residue",
                "candidate_atom",
                "partner_position",
                "partner_residue",
                "partner_atom",
                "distance_A",
            ]
        )

    candidate_residue_name = (
        candidate_atoms["resname"].iloc[0]
    )

    records = []

    partner_positions = sorted(
        pdb_df[
            pdb_df["chain"] == chain
        ]["resnum"].unique()
    )

    for partner_position in partner_positions:

        if int(partner_position) == int(candidate_position):
            continue

        partner_atoms = get_residue_atoms(
            pdb_df,
            partner_position,
            chain,
        )

        partner_atoms = partner_atoms[
            partner_atoms["element"].isin(
                polar_elements
            )
        ]

        if partner_atoms.empty:
            continue

        for _, atom1 in candidate_atoms.iterrows():

            for _, atom2 in partner_atoms.iterrows():

                distance = calculate_atom_distance(
                    atom1,
                    atom2,
                )

                if distance <= cutoff:

                    records.append(
                        {
                            "candidate_position": int(
                                candidate_position
                            ),
                            "candidate_residue": str(
                                candidate_residue_name
                            ),
                            "candidate_atom": str(
                                atom1["atom"]
                            ),
                            "partner_position": int(
                                partner_position
                            ),
                            "partner_residue": str(
                                partner_atoms[
                                    "resname"
                                ].iloc[0]
                            ),
                            "partner_atom": str(
                                atom2["atom"]
                            ),
                            "distance_A": float(
                                distance
                            ),
                        }
                    )

    result = pd.DataFrame(
        records,
        columns=[
            "candidate_position",
            "candidate_residue",
            "candidate_atom",
            "partner_position",
            "partner_residue",
            "partner_atom",
            "distance_A",
        ],
    )

    if not result.empty:
        result = result.sort_values(
            by=[
                "distance_A",
                "partner_position",
            ]
        ).reset_index(drop=True)

    return result


# =============================================================================
# START
# =============================================================================

print_section(
    "STEP 11 — D166V CATALYTIC-CONTEXT AND STRUCTURAL-IMPACT ANALYSIS"
)

print(
    f"Project root: {PROJECT_ROOT}"
)

print(
    f"Step 9F file: {STEP9F_FILE}"
)

print(
    f"Step 10 directory: {STEP10_DIR}"
)

print(
    f"Step 11 output: {STEP11_DIR}"
)


# =============================================================================
# INPUT FILE CHECK
# =============================================================================

print_section("CHECKING INPUT FILES")

input_checks = {
    "Step9F_exists": STEP9F_FILE.exists(),
    "WT_PDB_exists": WT_PDB.exists(),
    "D166V_PDB_exists": MUTANT_PDB.exists(),
}

for name, result in input_checks.items():
    print(
        f"{name}: "
        f"{'PASS' if result else 'FAIL'}"
    )

if not all(input_checks.values()):
    raise FileNotFoundError(
        "Required Step 9F or Step 10 input file is missing."
    )


# =============================================================================
# LOAD STEP 9F
# =============================================================================

print_section(
    "READING AUTHORITATIVE STEP 9F DATASET"
)

step9f = pd.read_csv(
    STEP9F_FILE
)

print(
    f"Rows loaded: {len(step9f)}"
)

print(
    f"Columns loaded: {len(step9f.columns)}"
)


# =============================================================================
# EXACT COLUMN VALIDATION
# =============================================================================

print_subsection(
    "REQUIRED COLUMN VALIDATION"
)

required_columns = [
    "VariationID",
    "Protein_Change",
    "UniProt_ID",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",
    "Catalytic_Context",
    "Nearest_Catalytic_Residue",
]

missing_columns = [
    column
    for column in required_columns
    if column not in step9f.columns
]

if missing_columns:

    print(
        "Missing columns: "
        + ", ".join(missing_columns)
    )

    raise KeyError(
        "Required Step 9F columns are missing: "
        + ", ".join(missing_columns)
    )

print(
    "Required Step 9F columns: PASS"
)

if "Position_GRCh38" in step9f.columns:

    print(
        "Position_GRCh38 detected: PASS "
        "(genomic coordinate retained separately)"
    )

print(
    "Protein residue position source: "
    "Protein_Position (authoritative)"
)


# =============================================================================
# IDENTIFY PRIMARY D166V
# =============================================================================

print_section(
    "IDENTIFYING PRIMARY PNPLA3 STRUCTURAL CANDIDATE"
)

candidate_matches = step9f[
    step9f["VariationID"].astype(str).str.strip()
    == str(EXPECTED_VARIATION_ID)
].copy()

# Robust fallback if VariationID representation differs
if candidate_matches.empty:

    candidate_matches = step9f[
        step9f["Protein_Change"]
        .astype(str)
        .str.strip()
        == EXPECTED_PROTEIN_CHANGE
    ].copy()

if candidate_matches.empty:
    raise ValueError(
        "D166V was not found in the Step 9F dataset."
    )

if len(candidate_matches) != 1:
    raise ValueError(
        f"Expected exactly one D166V record, "
        f"but found {len(candidate_matches)}."
    )

candidate = candidate_matches.iloc[0]

print(
    f"VariationID: {candidate['VariationID']}"
)

print(
    f"Protein change: {candidate['Protein_Change']}"
)

print(
    f"UniProt ID: {candidate['UniProt_ID']}"
)

print(
    f"Protein position: {candidate['Protein_Position']}"
)

print(
    f"Reference amino acid: "
    f"{candidate['Reference_Amino_Acid']}"
)

print(
    f"Alternate amino acid: "
    f"{candidate['Alternate_Amino_Acid']}"
)

print(
    f"Catalytic context: "
    f"{candidate['Catalytic_Context']}"
)

print(
    f"Nearest catalytic residue: "
    f"{candidate['Nearest_Catalytic_Residue']}"
)


# =============================================================================
# PRIMARY CANDIDATE VALIDATION
# =============================================================================

print_section(
    "VALIDATING PRIMARY CANDIDATE IDENTITY"
)

protein_position = int(
    float(
        candidate["Protein_Position"]
    )
)

candidate_checks = {
    "VariationID": (
        str(candidate["VariationID"]).strip()
        == str(EXPECTED_VARIATION_ID)
    ),

    "Protein_Change": (
        normalize_text(
            candidate["Protein_Change"]
        )
        == EXPECTED_PROTEIN_CHANGE
    ),

    "UniProt_ID": (
        normalize_text(
            candidate["UniProt_ID"]
        )
        == EXPECTED_UNIPROT_ID
    ),

    "Protein_Position": (
        protein_position
        == EXPECTED_PROTEIN_POSITION
    ),

    "Reference_Amino_Acid": (
        normalize_text(
            candidate["Reference_Amino_Acid"]
        ).lower()
        == EXPECTED_REFERENCE_AA.lower()
    ),

    "Alternate_Amino_Acid": (
        normalize_text(
            candidate["Alternate_Amino_Acid"]
        ).lower()
        == EXPECTED_ALTERNATE_AA.lower()
    ),
}

for name, result in candidate_checks.items():

    print(
        f"{name}: "
        f"{'PASS' if result else 'FAIL'}"
    )

if not all(candidate_checks.values()):
    raise ValueError(
        "Primary D166V candidate validation FAILED."
    )

print(
    "PRIMARY CANDIDATE VALIDATION: PASS"
)


# =============================================================================
# LOAD WT AND MUTANT STRUCTURES
# =============================================================================

print_section(
    "LOADING STEP 10 STRUCTURES"
)

wt = parse_pdb(
    WT_PDB
)

mutant = parse_pdb(
    MUTANT_PDB
)

print(
    f"WT atoms loaded: {len(wt)}"
)

print(
    f"D166V atoms loaded: {len(mutant)}"
)


# =============================================================================
# STRUCTURAL RESIDUE VALIDATION
# =============================================================================

print_section(
    "VALIDATING RESIDUE 166 IN WT AND D166V"
)

wt_166_name = get_residue_name(
    wt,
    D166_POSITION
)

mutant_166_name = get_residue_name(
    mutant,
    D166_POSITION
)

print(
    f"WT residue 166: {wt_166_name}"
)

print(
    f"D166V residue 166: {mutant_166_name}"
)

wt_166_atoms = get_residue_atoms(
    wt,
    D166_POSITION
)

mutant_166_atoms = get_residue_atoms(
    mutant,
    D166_POSITION
)

structure_checks = {
    "WT_residue_166_ASP": (
        wt_166_name == "ASP"
    ),

    "D166V_residue_166_VAL": (
        mutant_166_name == "VAL"
    ),

    "WT_residue_present": (
        not wt_166_atoms.empty
    ),

    "D166V_residue_present": (
        not mutant_166_atoms.empty
    ),

    "WT_OD1_present": (
        "OD1"
        in wt_166_atoms["atom"].values
    ),

    "WT_OD2_present": (
        "OD2"
        in wt_166_atoms["atom"].values
    ),

    "D166V_CG1_present": (
        "CG1"
        in mutant_166_atoms["atom"].values
    ),

    "D166V_CG2_present": (
        "CG2"
        in mutant_166_atoms["atom"].values
    ),

    "D166V_OD1_absent": (
        "OD1"
        not in mutant_166_atoms["atom"].values
    ),

    "D166V_OD2_absent": (
        "OD2"
        not in mutant_166_atoms["atom"].values
    ),
}

for name, result in structure_checks.items():

    print(
        f"{name}: "
        f"{'PASS' if result else 'FAIL'}"
    )

if not all(structure_checks.values()):
    raise ValueError(
        "Structural validation of residue 166 FAILED."
    )

print(
    "RESIDUE 166 STRUCTURAL VALIDATION: PASS"
)


# =============================================================================
# CATALYTIC / STRUCTURAL CONTEXT
# =============================================================================

print_section(
    "ANALYZING D166V CATALYTIC CONTEXT"
)

context_records = []

# -------------------------------------------------------------------------
# Ser47
# -------------------------------------------------------------------------

ser47_wt_distance, ser47_wt_atom1, ser47_wt_atom2 = (
    minimum_residue_distance(
        wt,
        D166_POSITION,
        SER47_POSITION,
    )
)

ser47_mut_distance, ser47_mut_atom1, ser47_mut_atom2 = (
    minimum_residue_distance(
        mutant,
        D166_POSITION,
        SER47_POSITION,
    )
)

context_records.append(
    {
        "Residue_Position": SER47_POSITION,
        "Context_Label": (
            "Ser47 — catalytic-context residue"
        ),
        "WT_Residue": get_residue_name(
            wt,
            SER47_POSITION
        ),
        "D166V_Residue": get_residue_name(
            mutant,
            SER47_POSITION
        ),
        "WT_Distance_to_D166_A": (
            ser47_wt_distance
        ),
        "D166V_Distance_to_Residue166_A": (
            ser47_mut_distance
        ),
        "Distance_Change_A": (
            ser47_mut_distance
            - ser47_wt_distance
        ),
        "WT_Closest_Atom_1": (
            ser47_wt_atom1
        ),
        "WT_Closest_Atom_2": (
            ser47_wt_atom2
        ),
        "D166V_Closest_Atom_1": (
            ser47_mut_atom1
        ),
        "D166V_Closest_Atom_2": (
            ser47_mut_atom2
        ),
    }
)

# -------------------------------------------------------------------------
# Asp166 / Val166 itself
# -------------------------------------------------------------------------

context_records.append(
    {
        "Residue_Position": D166_POSITION,
        "Context_Label": (
            "Asp166/Val166 — candidate residue"
        ),
        "WT_Residue": get_residue_name(
            wt,
            D166_POSITION
        ),
        "D166V_Residue": get_residue_name(
            mutant,
            D166_POSITION
        ),
        "WT_Distance_to_D166_A": 0.0,
        "D166V_Distance_to_Residue166_A": 0.0,
        "Distance_Change_A": 0.0,
        "WT_Closest_Atom_1": "",
        "WT_Closest_Atom_2": "",
        "D166V_Closest_Atom_1": "",
        "D166V_Closest_Atom_2": "",
    }
)

# -------------------------------------------------------------------------
# Pro186
# -------------------------------------------------------------------------

pro186_wt_distance, pro186_wt_atom1, pro186_wt_atom2 = (
    minimum_residue_distance(
        wt,
        D166_POSITION,
        PRO186_POSITION,
    )
)

pro186_mut_distance, pro186_mut_atom1, pro186_mut_atom2 = (
    minimum_residue_distance(
        mutant,
        D166_POSITION,
        PRO186_POSITION,
    )
)

context_records.append(
    {
        "Residue_Position": PRO186_POSITION,
        "Context_Label": (
            "Pro186 — structural-context residue"
        ),
        "WT_Residue": get_residue_name(
            wt,
            PRO186_POSITION
        ),
        "D166V_Residue": get_residue_name(
            mutant,
            PRO186_POSITION
        ),
        "WT_Distance_to_D166_A": (
            pro186_wt_distance
        ),
        "D166V_Distance_to_Residue166_A": (
            pro186_mut_distance
        ),
        "Distance_Change_A": (
            pro186_mut_distance
            - pro186_wt_distance
        ),
        "WT_Closest_Atom_1": (
            pro186_wt_atom1
        ),
        "WT_Closest_Atom_2": (
            pro186_wt_atom2
        ),
        "D166V_Closest_Atom_1": (
            pro186_mut_atom1
        ),
        "D166V_Closest_Atom_2": (
            pro186_mut_atom2
        ),
    }
)

context_df = pd.DataFrame(
    context_records
)

print(
    context_df.to_string(
        index=False
    )
)

context_df.to_csv(
    TABLE_DIR
    / "Table_S11_1_D166V_catalytic_context.csv",
    index=False,
)


# =============================================================================
# LOCAL DISTANCE ANALYSIS
# =============================================================================

print_section(
    "LOCAL DISTANCE ANALYSIS AROUND D166"
)

local_start = (
    D166_POSITION
    - LOCAL_RADIUS
)

local_end = (
    D166_POSITION
    + LOCAL_RADIUS
)

wt_local_positions = set(
    wt[
        (wt["chain"] == CHAIN_ID)
        & (wt["resnum"] >= local_start)
        & (wt["resnum"] <= local_end)
    ]["resnum"].unique()
)

mutant_local_positions = set(
    mutant[
        (mutant["chain"] == CHAIN_ID)
        & (mutant["resnum"] >= local_start)
        & (mutant["resnum"] <= local_end)
    ]["resnum"].unique()
)

local_positions = sorted(
    wt_local_positions.intersection(
        mutant_local_positions
    )
)

distance_records = []

for position in local_positions:

    wt_residue = get_residue_name(
        wt,
        position
    )

    mutant_residue = get_residue_name(
        mutant,
        position
    )

    if position == D166_POSITION:

        wt_distance = 0.0
        mutant_distance = 0.0

    else:

        wt_distance, _, _ = (
            minimum_residue_distance(
                wt,
                D166_POSITION,
                position,
            )
        )

        mutant_distance, _, _ = (
            minimum_residue_distance(
                mutant,
                D166_POSITION,
                position,
            )
        )

    if (
        pd.isna(wt_distance)
        or pd.isna(mutant_distance)
    ):
        distance_change = np.nan
    else:
        distance_change = (
            mutant_distance
            - wt_distance
        )

    distance_records.append(
        {
            "Residue_Position": int(
                position
            ),
            "WT_Residue": wt_residue,
            "D166V_Residue": mutant_residue,
            "WT_Min_Distance_A": (
                wt_distance
            ),
            "D166V_Min_Distance_A": (
                mutant_distance
            ),
            "Distance_Change_A": (
                distance_change
            ),
        }
    )

distance_df = pd.DataFrame(
    distance_records
)

print(
    distance_df.to_string(
        index=False
    )
)

distance_df.to_csv(
    TABLE_DIR
    / "Table_S11_3_residue_distance_changes.csv",
    index=False,
)


# =============================================================================
# LOCAL HEAVY-ATOM CONTACTS
# =============================================================================

print_section(
    "LOCAL HEAVY-ATOM CONTACT ANALYSIS"
)

wt_contacts = heavy_atom_contacts(
    wt,
    D166_POSITION,
    cutoff=CONTACT_CUTOFF_A,
)

mutant_contacts = heavy_atom_contacts(
    mutant,
    D166_POSITION,
    cutoff=CONTACT_CUTOFF_A,
)

print(
    f"WT contacts <= {CONTACT_CUTOFF_A:.1f} Å: "
    f"{len(wt_contacts)}"
)

print(
    f"D166V contacts <= {CONTACT_CUTOFF_A:.1f} Å: "
    f"{len(mutant_contacts)}"
)


# =============================================================================
# RESIDUE-LEVEL CONTACT COMPARISON
# =============================================================================

def build_contact_key(row):
    return (
        int(row["partner_position"]),
        str(row["partner_residue"]),
    )


wt_contact_keys = {
    build_contact_key(row)
    for _, row in wt_contacts.iterrows()
}

mutant_contact_keys = {
    build_contact_key(row)
    for _, row in mutant_contacts.iterrows()
}

retained_contact_keys = (
    wt_contact_keys.intersection(
        mutant_contact_keys
    )
)

lost_contact_keys = (
    wt_contact_keys
    - mutant_contact_keys
)

gained_contact_keys = (
    mutant_contact_keys
    - wt_contact_keys
)

contact_records = []

for position, residue in sorted(
    retained_contact_keys
):

    contact_records.append(
        {
            "Partner_Position": position,
            "Partner_Residue": residue,
            "Contact_Status": "Retained",
        }
    )

for position, residue in sorted(
    lost_contact_keys
):

    contact_records.append(
        {
            "Partner_Position": position,
            "Partner_Residue": residue,
            "Contact_Status": "Lost",
        }
    )

for position, residue in sorted(
    gained_contact_keys
):

    contact_records.append(
        {
            "Partner_Position": position,
            "Partner_Residue": residue,
            "Contact_Status": "Gained",
        }
    )

contact_comparison_df = pd.DataFrame(
    contact_records,
    columns=[
        "Partner_Position",
        "Partner_Residue",
        "Contact_Status",
    ],
)

contact_comparison_df.to_csv(
    TABLE_DIR
    / "Table_S11_2_WT_vs_D166V_local_interactions.csv",
    index=False,
)

print(
    f"Retained contacts: "
    f"{len(retained_contact_keys)}"
)

print(
    f"Lost contacts: "
    f"{len(lost_contact_keys)}"
)

print(
    f"Gained contacts: "
    f"{len(gained_contact_keys)}"
)


# =============================================================================
# CANDIDATE POLAR HEAVY-ATOM CONTACTS
# =============================================================================

print_section(
    "CANDIDATE POLAR HEAVY-ATOM CONTACT ANALYSIS"
)

wt_polar_contacts = (
    polar_heavy_atom_contacts(
        wt,
        D166_POSITION,
        cutoff=POLAR_CONTACT_CUTOFF_A,
    )
)

mutant_polar_contacts = (
    polar_heavy_atom_contacts(
        mutant,
        D166_POSITION,
        cutoff=POLAR_CONTACT_CUTOFF_A,
    )
)

wt_polar_contacts.to_csv(
    TABLE_DIR
    / "Table_S11_5_WT_D166_candidate_polar_contacts.csv",
    index=False,
)

mutant_polar_contacts.to_csv(
    TABLE_DIR
    / "Table_S11_6_D166V_candidate_polar_contacts.csv",
    index=False,
)

polar_summary_df = pd.DataFrame(
    [
        {
            "Structure": "WT_D166",
            "Candidate_Polar_Heavy_Atom_Contacts": (
                len(wt_polar_contacts)
            ),
            "Distance_Cutoff_A": (
                POLAR_CONTACT_CUTOFF_A
            ),
        },
        {
            "Structure": "D166V",
            "Candidate_Polar_Heavy_Atom_Contacts": (
                len(mutant_polar_contacts)
            ),
            "Distance_Cutoff_A": (
                POLAR_CONTACT_CUTOFF_A
            ),
        },
    ]
)

polar_summary_df.to_csv(
    TABLE_DIR
    / "Table_S11_7_candidate_polar_contact_summary.csv",
    index=False,
)

print(
    f"WT candidate polar contacts: "
    f"{len(wt_polar_contacts)}"
)

print(
    f"D166V candidate polar contacts: "
    f"{len(mutant_polar_contacts)}"
)

print()
print(
    "NOTE: Candidate polar contacts are defined "
    "by N/O/S heavy-atom proximity <= "
    f"{POLAR_CONTACT_CUTOFF_A:.1f} Å."
)

print(
    "They are NOT automatically classified as "
    "hydrogen bonds."
)


# =============================================================================
# PHYSICOCHEMICAL CHANGE
# =============================================================================

print_section(
    "D166V PHYSICOCHEMICAL CHANGE"
)

physicochemical_df = pd.DataFrame(
    [
        {
            "Property": "Residue identity",
            "WT": "Aspartate (Asp/D)",
            "D166V": "Valine (Val/V)",
            "Interpretation": (
                "Residue substitution"
            ),
        },
        {
            "Property": "Side-chain functional group",
            "WT": "Carboxylate",
            "D166V": "Hydrocarbon side chain",
            "Interpretation": (
                "Loss of carboxylate functionality"
            ),
        },
        {
            "Property": "Charge character",
            "WT": "Negative",
            "D166V": "Neutral",
            "Interpretation": (
                "Negative to neutral"
            ),
        },
        {
            "Property": "Polarity",
            "WT": "Polar",
            "D166V": "Non-polar",
            "Interpretation": (
                "Marked reduction in side-chain polarity"
            ),
        },
        {
            "Property": "Hydrophobic character",
            "WT": "Lower",
            "D166V": "Higher",
            "Interpretation": (
                "Increased hydrophobic character"
            ),
        },
        {
            "Property": "Candidate polar contacts",
            "WT": str(
                len(wt_polar_contacts)
            ),
            "D166V": str(
                len(mutant_polar_contacts)
            ),
            "Interpretation": (
                "Change in local polar-contact environment"
            ),
        },
    ]
)

physicochemical_df.to_csv(
    TABLE_DIR
    / "Table_S11_4_D166V_physicochemical_change.csv",
    index=False,
)


# =============================================================================
# STRUCTURAL IMPACT SUMMARY
# =============================================================================

print_section(
    "STRUCTURAL IMPACT SUMMARY"
)

non_self_distance_df = distance_df[
    distance_df["Residue_Position"]
    != D166_POSITION
].copy()

if not non_self_distance_df.empty:

    max_abs_distance_change = (
        non_self_distance_df[
            "Distance_Change_A"
        ]
        .abs()
        .max()
    )

    mean_abs_distance_change = (
        non_self_distance_df[
            "Distance_Change_A"
        ]
        .abs()
        .mean()
    )

else:

    max_abs_distance_change = np.nan
    mean_abs_distance_change = np.nan


structural_summary_df = pd.DataFrame(
    [
        {
            "Metric": "Candidate",
            "Value": "D166V",
            "Interpretation": (
                "Primary PNPLA3 structural candidate"
            ),
        },
        {
            "Metric": "Protein position",
            "Value": D166_POSITION,
            "Interpretation": (
                "Authoritative Step 9F protein position"
            ),
        },
        {
            "Metric": "WT residue",
            "Value": "ASP",
            "Interpretation": (
                "Reference residue"
            ),
        },
        {
            "Metric": "D166V residue",
            "Value": "VAL",
            "Interpretation": (
                "Modeled mutant residue"
            ),
        },
        {
            "Metric": "WT local heavy-atom contacts",
            "Value": len(wt_contacts),
            "Interpretation": (
                f"Residue contacts <= "
                f"{CONTACT_CUTOFF_A:.1f} Å"
            ),
        },
        {
            "Metric": "D166V local heavy-atom contacts",
            "Value": len(mutant_contacts),
            "Interpretation": (
                f"Residue contacts <= "
                f"{CONTACT_CUTOFF_A:.1f} Å"
            ),
        },
        {
            "Metric": "Retained contacts",
            "Value": len(retained_contact_keys),
            "Interpretation": (
                "Present in both structures"
            ),
        },
        {
            "Metric": "Lost contacts",
            "Value": len(lost_contact_keys),
            "Interpretation": (
                "Present in WT but absent in D166V"
            ),
        },
        {
            "Metric": "Gained contacts",
            "Value": len(gained_contact_keys),
            "Interpretation": (
                "Absent in WT but present in D166V"
            ),
        },
        {
            "Metric": "WT candidate polar contacts",
            "Value": len(wt_polar_contacts),
            "Interpretation": (
                f"N/O/S contacts <= "
                f"{POLAR_CONTACT_CUTOFF_A:.1f} Å"
            ),
        },
        {
            "Metric": "D166V candidate polar contacts",
            "Value": len(mutant_polar_contacts),
            "Interpretation": (
                f"N/O/S contacts <= "
                f"{POLAR_CONTACT_CUTOFF_A:.1f} Å"
            ),
        },
        {
            "Metric": "Maximum absolute local distance change (Å)",
            "Value": (
                max_abs_distance_change
            ),
            "Interpretation": (
                "Residue-level local distance comparison"
            ),
        },
        {
            "Metric": "Mean absolute local distance change (Å)",
            "Value": (
                mean_abs_distance_change
            ),
            "Interpretation": (
                "Residue-level local distance comparison"
            ),
        },
    ]
)

structural_summary_df.to_csv(
    TABLE_DIR
    / "Table_S11_8_D166V_structural_impact_summary.csv",
    index=False,
)


# =============================================================================
# FIGURE 1 — CATALYTIC CONTEXT
# =============================================================================

print_section(
    "GENERATING PUBLICATION-QUALITY FIGURES"
)

context_plot_df = context_df[
    context_df["Residue_Position"]
    .isin(
        [
            SER47_POSITION,
            PRO186_POSITION,
        ]
    )
].copy()

x_values = np.arange(
    len(context_plot_df)
)

plt.figure(
    figsize=(9, 6)
)

plt.plot(
    x_values,
    context_plot_df[
        "WT_Distance_to_D166_A"
    ],
    marker="o",
    linewidth=2,
    label="WT",
)

plt.plot(
    x_values,
    context_plot_df[
        "D166V_Distance_to_Residue166_A"
    ],
    marker="s",
    linewidth=2,
    label="D166V",
)

plt.xticks(
    x_values,
    [
        f"{int(row['Residue_Position'])}\n"
        f"{row['WT_Residue']}"
        for _, row in context_plot_df.iterrows()
    ],
)

plt.xlabel(
    "Context residue"
)

plt.ylabel(
    "Minimum heavy-atom distance to residue 166 (Å)"
)

plt.title(
    "D166V Catalytic-Context Distance Comparison"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_S11_1_catalytic_context_distance_comparison.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# =============================================================================
# FIGURE 2 — LOCAL DISTANCES
# =============================================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    distance_df[
        "Residue_Position"
    ],
    distance_df[
        "WT_Min_Distance_A"
    ],
    marker="o",
    linewidth=2,
    label="WT",
)

plt.plot(
    distance_df[
        "Residue_Position"
    ],
    distance_df[
        "D166V_Min_Distance_A"
    ],
    marker="s",
    linewidth=2,
    label="D166V",
)

plt.axvline(
    D166_POSITION,
    linestyle="--",
    linewidth=1.5,
    label="Residue 166",
)

plt.xlabel(
    "Residue position"
)

plt.ylabel(
    "Minimum heavy-atom distance to residue 166 (Å)"
)

plt.title(
    "Local Structural Distance Comparison Around D166"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_S11_2_local_distance_comparison.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# =============================================================================
# FIGURE 3 — CONTACT STATUS
# =============================================================================

status_labels = [
    "Retained",
    "Lost",
    "Gained",
]

status_values = [
    len(retained_contact_keys),
    len(lost_contact_keys),
    len(gained_contact_keys),
]

plt.figure(
    figsize=(8, 6)
)

bars = plt.bar(
    status_labels,
    status_values,
)

plt.xlabel(
    "Contact status"
)

plt.ylabel(
    "Number of residue contacts"
)

plt.title(
    "D166 Local Contact Status: WT vs D166V"
)

for bar, value in zip(
    bars,
    status_values,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        str(value),
        ha="center",
        va="bottom",
    )

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_S11_3_local_contact_status.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# =============================================================================
# FIGURE 4 — POLAR CONTACT COMPARISON
# =============================================================================

polar_labels = [
    "WT D166",
    "D166V",
]

polar_values = [
    len(wt_polar_contacts),
    len(mutant_polar_contacts),
]

plt.figure(
    figsize=(8, 6)
)

bars = plt.bar(
    polar_labels,
    polar_values,
)

plt.ylabel(
    "Candidate polar heavy-atom contacts"
)

plt.title(
    "Candidate Polar Heavy-Atom Contacts Around D166"
)

for bar, value in zip(
    bars,
    polar_values,
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        str(value),
        ha="center",
        va="bottom",
    )

plt.grid(
    axis="y",
    alpha=0.25
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_S11_4_D166V_polar_contact_comparison.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# =============================================================================
# FIGURE 5 — QUALITATIVE PHYSICOCHEMICAL CHANGE
# =============================================================================

properties = [
    "Charge",
    "Polarity",
    "Hydrophobicity",
]

# Qualitative indicator only.
# 1 = characteristic represented in the corresponding category.
# This is NOT a numerical amino-acid property score.

wt_indicator = [
    1,  # negative charge
    1,  # polar
    0,  # lower hydrophobic character
]

mutant_indicator = [
    0,  # neutral
    0,  # non-polar
    1,  # higher hydrophobic character
]

x = np.arange(
    len(properties)
)

width = 0.35

plt.figure(
    figsize=(10, 6)
)

plt.bar(
    x - width / 2,
    wt_indicator,
    width,
    label="WT Asp",
)

plt.bar(
    x + width / 2,
    mutant_indicator,
    width,
    label="D166V Val",
)

plt.xticks(
    x,
    properties,
)

plt.yticks([])

plt.ylabel(
    "Qualitative property category"
)

plt.title(
    "Qualitative Physicochemical Character of D166 and D166V"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.20
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR
    / "Figure_S11_5_D166V_physicochemical_change.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

print_section(
    "VALIDATING GENERATED OUTPUTS"
)

expected_tables = [
    TABLE_DIR
    / "Table_S11_1_D166V_catalytic_context.csv",

    TABLE_DIR
    / "Table_S11_2_WT_vs_D166V_local_interactions.csv",

    TABLE_DIR
    / "Table_S11_3_residue_distance_changes.csv",

    TABLE_DIR
    / "Table_S11_4_D166V_physicochemical_change.csv",

    TABLE_DIR
    / "Table_S11_5_WT_D166_candidate_polar_contacts.csv",

    TABLE_DIR
    / "Table_S11_6_D166V_candidate_polar_contacts.csv",

    TABLE_DIR
    / "Table_S11_7_candidate_polar_contact_summary.csv",

    TABLE_DIR
    / "Table_S11_8_D166V_structural_impact_summary.csv",
]

expected_figures = [
    FIGURE_DIR
    / "Figure_S11_1_catalytic_context_distance_comparison.png",

    FIGURE_DIR
    / "Figure_S11_2_local_distance_comparison.png",

    FIGURE_DIR
    / "Figure_S11_3_local_contact_status.png",

    FIGURE_DIR
    / "Figure_S11_4_D166V_polar_contact_comparison.png",

    FIGURE_DIR
    / "Figure_S11_5_D166V_physicochemical_change.png",
]

for table_file in expected_tables:

    print(
        f"{table_file.name}: "
        f"{'PASS' if table_file.exists() else 'FAIL'}"
    )

for figure_file in expected_figures:

    print(
        f"{figure_file.name}: "
        f"{'PASS' if figure_file.exists() else 'FAIL'}"
    )


# =============================================================================
# FINAL QC
# =============================================================================

print_section(
    "FINAL STEP 11 QUALITY CONTROL"
)

qc = {}

qc["Step9F_file_exists"] = (
    STEP9F_FILE.exists()
)

qc["WT_PDB_exists"] = (
    WT_PDB.exists()
)

qc["D166V_PDB_exists"] = (
    MUTANT_PDB.exists()
)

qc["Step9F_required_columns"] = (
    len(missing_columns) == 0
)

qc["D166V_candidate_identity"] = (
    all(candidate_checks.values())
)

qc["WT_residue_166_is_ASP"] = (
    wt_166_name == "ASP"
)

qc["D166V_residue_166_is_VAL"] = (
    mutant_166_name == "VAL"
)

qc["WT_OD1_present"] = (
    "OD1"
    in wt_166_atoms["atom"].values
)

qc["WT_OD2_present"] = (
    "OD2"
    in wt_166_atoms["atom"].values
)

qc["D166V_CG1_present"] = (
    "CG1"
    in mutant_166_atoms["atom"].values
)

qc["D166V_CG2_present"] = (
    "CG2"
    in mutant_166_atoms["atom"].values
)

qc["D166V_OD1_absent"] = (
    "OD1"
    not in mutant_166_atoms["atom"].values
)

qc["D166V_OD2_absent"] = (
    "OD2"
    not in mutant_166_atoms["atom"].values
)

qc["Ser47_present_WT"] = (
    get_residue_name(
        wt,
        SER47_POSITION
    )
    == "SER"
)

qc["Ser47_present_D166V"] = (
    get_residue_name(
        mutant,
        SER47_POSITION
    )
    == "SER"
)

qc["Pro186_present_WT"] = (
    get_residue_name(
        wt,
        PRO186_POSITION
    )
    == "PRO"
)

qc["Pro186_present_D166V"] = (
    get_residue_name(
        mutant,
        PRO186_POSITION
    )
    == "PRO"
)

qc["Local_distance_analysis_completed"] = (
    not distance_df.empty
)

qc["WT_contact_analysis_completed"] = True

qc["D166V_contact_analysis_completed"] = True

qc["WT_polar_contact_analysis_completed"] = True

qc["D166V_polar_contact_analysis_completed"] = True

qc["All_expected_tables_created"] = all(
    file.exists()
    for file in expected_tables
)

qc["All_expected_figures_created"] = all(
    file.exists()
    for file in expected_figures
)

overall_pass = all(
    qc.values()
)

qc["OVERALL_STEP11_QC"] = (
    "PASS"
    if overall_pass
    else "FAIL"
)

qc_file = (
    QC_DIR
    / "Step11_final_QC_report.json"
)

with open(
    qc_file,
    "w",
    encoding="utf-8",
) as handle:

    json.dump(
        qc,
        handle,
        indent=4,
        ensure_ascii=False,
    )

for name, result in qc.items():

    if name == "OVERALL_STEP11_QC":
        continue

    print(
        f"{name}: "
        f"{'PASS' if result else 'FAIL'}"
    )

print()
print(
    "OVERALL STEP 11 QC: "
    + (
        "PASS"
        if overall_pass
        else "FAIL"
    )
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_section(
    "WRITING STEP 11 FINAL SUMMARY"
)

summary_lines = [
    "STEP 11 — D166V CATALYTIC-CONTEXT AND STRUCTURAL-IMPACT ANALYSIS",
    "=" * 78,
    "",
    "PRIMARY CANDIDATE",
    "-" * 78,
    f"VariationID: {candidate['VariationID']}",
    f"Protein change: {candidate['Protein_Change']}",
    f"UniProt ID: {candidate['UniProt_ID']}",
    f"Protein position: {candidate['Protein_Position']}",
    f"Reference amino acid: {candidate['Reference_Amino_Acid']}",
    f"Alternate amino acid: {candidate['Alternate_Amino_Acid']}",
    "",
    "STRUCTURAL VALIDATION",
    "-" * 78,
    f"WT residue 166: {wt_166_name}",
    f"D166V residue 166: {mutant_166_name}",
    "WT Asp166 OD1/OD2: present",
    "D166V Val166 CG1/CG2: present",
    "D166V Val166 OD1/OD2: absent",
    "",
    "CATALYTIC / STRUCTURAL CONTEXT",
    "-" * 78,
    (
        f"Ser47 WT minimum heavy-atom distance to D166: "
        f"{ser47_wt_distance:.3f} Å"
    ),
    (
        f"Ser47 D166V minimum heavy-atom distance to residue 166: "
        f"{ser47_mut_distance:.3f} Å"
    ),
    (
        f"Ser47 distance change: "
        f"{ser47_mut_distance - ser47_wt_distance:+.3f} Å"
    ),
    (
        f"Pro186 WT minimum heavy-atom distance to D166: "
        f"{pro186_wt_distance:.3f} Å"
    ),
    (
        f"Pro186 D166V minimum heavy-atom distance to residue 166: "
        f"{pro186_mut_distance:.3f} Å"
    ),
    (
        f"Pro186 distance change: "
        f"{pro186_mut_distance - pro186_wt_distance:+.3f} Å"
    ),
    "",
    "LOCAL CONTACT ANALYSIS",
    "-" * 78,
    (
        f"WT heavy-atom contacts <= "
        f"{CONTACT_CUTOFF_A:.1f} Å: "
        f"{len(wt_contacts)}"
    ),
    (
        f"D166V heavy-atom contacts <= "
        f"{CONTACT_CUTOFF_A:.1f} Å: "
        f"{len(mutant_contacts)}"
    ),
    f"Retained contacts: {len(retained_contact_keys)}",
    f"Lost contacts: {len(lost_contact_keys)}",
    f"Gained contacts: {len(gained_contact_keys)}",
    "",
    "CANDIDATE POLAR CONTACT ANALYSIS",
    "-" * 78,
    (
        f"WT candidate polar contacts <= "
        f"{POLAR_CONTACT_CUTOFF_A:.1f} Å: "
        f"{len(wt_polar_contacts)}"
    ),
    (
        f"D166V candidate polar contacts <= "
        f"{POLAR_CONTACT_CUTOFF_A:.1f} Å: "
        f"{len(mutant_polar_contacts)}"
    ),
    "",
    "PHYSICOCHEMICAL INTERPRETATION",
    "-" * 78,
    (
        "Aspartate -> valine replaces a negatively charged, "
        "polar, carboxylate-containing side chain with a "
        "neutral, non-polar, hydrocarbon side chain."
    ),
    (
        "The substitution therefore produces a substantial "
        "change in local residue chemistry even when the "
        "overall modeled backbone remains highly similar."
    ),
    "",
    "STRUCTURAL INTERPRETATION",
    "-" * 78,
    (
        "Local distance changes are interpreted as modeled "
        "structural differences rather than direct evidence "
        "of experimental functional loss."
    ),
    (
        "Changes in candidate polar contacts provide a "
        "mechanistic structural hypothesis for downstream "
        "interaction, docking, and molecular-dynamics analyses."
    ),
    "",
    "IMPORTANT LIMITATIONS",
    "-" * 78,
    (
        "Candidate polar contacts are based on heavy-atom "
        "proximity and are not automatically confirmed "
        "hydrogen bonds."
    ),
    (
        "The structural model does not by itself establish "
        "loss of PNPLA3 enzymatic activity."
    ),
    (
        "Energy minimization and molecular dynamics are not "
        "performed in Step 11."
    ),
    "",
    "FINAL QC",
    "-" * 78,
    (
        "Overall Step 11 QC: "
        + (
            "PASS"
            if overall_pass
            else "FAIL"
        )
    ),
]

summary_file = (
    STEP11_DIR
    / "Step11_final_summary.txt"
)

with open(
    summary_file,
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        "\n".join(summary_lines)
    )

print(
    f"Summary written to: {summary_file}"
)

print(
    f"QC report written to: {qc_file}"
)


# =============================================================================
# FINAL STATUS
# =============================================================================

print_section(
    "STEP 11 COMPLETE"
)

print(
    f"Output directory: {STEP11_DIR}"
)

print(
    f"Tables directory: {TABLE_DIR}"
)

print(
    f"Figures directory: {FIGURE_DIR}"
)

print(
    f"QC report: {qc_file}"
)

print(
    f"Final summary: {summary_file}"
)

print()

print(
    "FINAL STATUS: "
    + (
        "PASS"
        if overall_pass
        else "FAIL"
    )
)

if overall_pass:

    print()
    print(
        "Step 11 completed successfully."
    )

else:

    print()
    print(
        "Step 11 FAILED QC."
    )

    print(
        "Review Step11_final_QC_report.json "
        "before proceeding."
    )