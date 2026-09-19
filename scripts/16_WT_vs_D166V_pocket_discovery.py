# ============================================================
# STEP 16 — WT vs D166V POCKET DISCOVERY
# PNPLA3 — UniProt Q9NST1
# ============================================================
#
# PURPOSE
# -------
# Independently characterize fpocket-discovered pockets in:
#
#   1. PNPLA3 WT
#   2. PNPLA3 D166V
#
# The analysis:
#   - reuses existing fpocket outputs
#   - maps pockets to residue 166
#   - compares WT and D166V pocket geometry
#   - compares pocket lining residues
#   - identifies D166V-specific / WT-specific pockets
#   - evaluates D166 proximity
#   - creates a transparent computational prioritization score
#   - produces tables and publication-quality figures
#
# IMPORTANT
# ---------
# This script DOES NOT:
#   - rerun fpocket
#   - perform docking
#   - perform molecular dynamics
#   - perform energy minimization
#   - calculate binding energies
#
# fpocket scores and prioritization scores are computational
# descriptors and NOT experimental evidence of therapeutic binding.
#
# ============================================================


import re
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

UNIPROT_ID = "Q9NST1"

EXPECTED_PROTEIN_LENGTH = 481

PRIMARY_VARIANT = "p.Asp166Val"
PRIMARY_POSITION = 166
PRIMARY_VARIATION_ID = 3308171

EXPECTED_POCKET_COUNT = 35


# ============================================================
# EXISTING FPOCKET OUTPUT LOCATIONS
# ============================================================
#
# IMPORTANT:
# The actual WSL folder is:
#
# PNPLA3_FP0CKET_TEST
#           ^
#           ZERO (0), NOT LETTER O
#
# ============================================================

WT_FPOCKET = Path(
    r"\\wsl.localhost\Ubuntu-24.04\home\asif_computer"
    r"\PNPLA3_FP0CKET_TEST\WT\PNPLA3_WT_out"
)

D166V_FPOCKET = Path(
    r"\\wsl.localhost\Ubuntu-24.04\home\asif_computer"
    r"\PNPLA3_FP0CKET_TEST\D166V\PNPLA3_D166V_mutant_out"
)


# ============================================================
# STRUCTURAL INPUT FILES
# ============================================================

WT_PDB = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
    / "structures"
    / "WT"
    / "PNPLA3_WT.pdb"
)

D166V_PDB = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
    / "structures"
    / "D166V"
    / "PNPLA3_D166V_mutant.pdb"
)


# ============================================================
# STEP 9F CANDIDATE FILE
# ============================================================

STEP9F = (
    PROJECT_ROOT
    / "VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_step9f_final_candidate_selection_FINAL.csv"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
)

POCKET_DIR = OUTPUT_DIR / "pockets"
WT_POCKET_DIR = POCKET_DIR / "WT"
D166V_POCKET_DIR = POCKET_DIR / "D166V"

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
QC_DIR = OUTPUT_DIR / "QC"


# ============================================================
# POCKET MATCHING PARAMETERS
# ============================================================

CENTER_MATCH_THRESHOLD_A = 8.0
LINING_JACCARD_THRESHOLD = 0.20


# ============================================================
# D166 PROXIMITY CATEGORIES
# ============================================================

VERY_CLOSE_A = 5.0
CLOSE_A = 10.0
MODERATE_A = 15.0


# ============================================================
# PRIORITIZATION WEIGHTS
# ============================================================

W_DRUGGABILITY = 0.30
W_POCKET_SCORE = 0.15
W_VOLUME = 0.10
W_D166_PROXIMITY = 0.20
W_MUTATION_REMODELING = 0.25


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(text):
    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


def print_pass(text):
    print(f"[PASS] {text}")


def print_warn(text):
    print(f"[WARN] {text}")


def print_fail(text):
    print(f"[FAIL] {text}")


def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.strip()

        if value == "":
            return default

        result = float(value)

        if math.isnan(result):
            return default

        return result

    except Exception:
        return default


def make_json_safe(obj):

    if isinstance(obj, dict):
        return {
            str(k): make_json_safe(v)
            for k, v in obj.items()
        }

    if isinstance(obj, list):
        return [
            make_json_safe(v)
            for v in obj
        ]

    if isinstance(obj, tuple):
        return [
            make_json_safe(v)
            for v in obj
        ]

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)

        if not np.isfinite(value):
            return None

        return value

    if isinstance(obj, float):

        if not np.isfinite(obj):
            return None

        return obj

    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    return obj


def normalize_position(value):

    try:
        return int(
            float(
                str(value)
                .replace("p.", "")
                .strip()
            )
        )

    except Exception:
        return None


def jaccard_similarity(set_a, set_b):

    if not set_a and not set_b:
        return 1.0

    union = set_a | set_b

    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


def min_distance_to_coordinates(coordinates_a, coordinates_b):

    if (
        coordinates_a is None
        or coordinates_b is None
        or len(coordinates_a) == 0
        or len(coordinates_b) == 0
    ):
        return np.nan

    a = np.asarray(
        coordinates_a,
        dtype=float
    )

    b = np.asarray(
        coordinates_b,
        dtype=float
    )

    distances = np.sqrt(
        np.sum(
            (
                a[:, None, :]
                - b[None, :, :]
            ) ** 2,
            axis=2
        )
    )

    return float(np.min(distances))


def residue_atom_coordinates(structure, residue_number):

    coordinates = []

    for model in structure:

        for chain in model:

            for residue in chain:

                if residue.id[0] != " ":
                    continue

                if residue.id[1] != residue_number:
                    continue

                for atom in residue:

                    element = getattr(
                        atom,
                        "element",
                        ""
                    )

                    if str(element).upper() == "H":
                        continue

                    coordinates.append(
                        atom.coord
                    )

    if not coordinates:

        return np.empty(
            (0, 3),
            dtype=float
        )

    return np.asarray(
        coordinates,
        dtype=float
    )


def parse_pqr_coordinates(pqr_file):

    coordinates = []

    if not pqr_file.exists():

        return np.empty(
            (0, 3),
            dtype=float
        )

    with open(
        pqr_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as handle:

        for line in handle:

            if not line.startswith(
                ("ATOM", "HETATM")
            ):
                continue

            try:

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                coordinates.append(
                    [x, y, z]
                )

                continue

            except Exception:
                pass

            tokens = line.split()

            numeric_triplets = []

            for i in range(len(tokens) - 2):

                try:

                    x = float(tokens[i])
                    y = float(tokens[i + 1])
                    z = float(tokens[i + 2])

                    numeric_triplets.append(
                        [x, y, z]
                    )

                except Exception:
                    continue

            if numeric_triplets:

                coordinates.append(
                    numeric_triplets[-1]
                )

    if not coordinates:

        return np.empty(
            (0, 3),
            dtype=float
        )

    return np.asarray(
        coordinates,
        dtype=float
    )


def parse_pocket_atoms(pdb_file):

    records = []

    if not pdb_file.exists():
        return records

    with open(
        pdb_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as handle:

        for line in handle:

            if not line.startswith(
                ("ATOM", "HETATM")
            ):
                continue

            try:

                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                chain_id = line[21].strip()

                residue_number = int(
                    line[22:26].strip()
                )

                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                element = line[76:78].strip()

                if not element:
                    element = atom_name[:1]

                if str(element).upper() == "H":
                    continue

                records.append(
                    {
                        "Atom_Name": atom_name,
                        "Residue_Name": residue_name,
                        "Chain": chain_id,
                        "Residue_Number": residue_number,
                        "X": x,
                        "Y": y,
                        "Z": z
                    }
                )

            except Exception:
                continue

    return records


def normalize_residue_key(
    residue_name,
    residue_number,
    chain
):

    return (
        f"{chain}:"
        f"{residue_name}:"
        f"{int(residue_number)}"
    )


def lining_position_set(records):

    positions = set()

    for record in records:

        positions.add(
            int(record["Residue_Number"])
        )

    return positions


def lining_identity_map(records):

    mapping = {}

    for record in records:

        key = normalize_residue_key(
            record["Residue_Name"],
            record["Residue_Number"],
            record["Chain"]
        )

        mapping[key] = (
            record["Residue_Name"],
            int(record["Residue_Number"])
        )

    return mapping


def mutation_proximity_category(distance):

    if pd.isna(distance):
        return "Unavailable"

    if distance <= VERY_CLOSE_A:
        return "Very close (≤5 Å)"

    if distance <= CLOSE_A:
        return "Close (5–10 Å)"

    if distance <= MODERATE_A:
        return "Moderate (10–15 Å)"

    return "Remote (>15 Å)"


def proximity_score(distance):

    if pd.isna(distance):
        return 0.0

    if distance <= 0:
        return 1.0

    if distance >= CLOSE_A:
        return 0.0

    return float(
        1.0 - (distance / CLOSE_A)
    )


def mutation_remodeling_score(
    remodeling_category,
    lining_jaccard,
    pocket_score_change,
    druggability_change,
    volume_change
):

    score = 0.0

    category = str(
        remodeling_category
    ).lower()

    if "specific" in category:

        score += 0.50

    elif "strong" in category:

        score += 0.40

    elif "moderate" in category:

        score += 0.25

    elif "minor" in category:

        score += 0.10

    if not pd.isna(lining_jaccard):

        score += (
            1.0 - float(lining_jaccard)
        ) * 0.25

    if not pd.isna(pocket_score_change):

        score += min(
            abs(float(pocket_score_change)),
            1.0
        ) * 0.125

    if not pd.isna(druggability_change):

        score += min(
            abs(float(druggability_change)),
            1.0
        ) * 0.125

    return float(
        min(score, 1.0)
    )


def minmax_normalize(series):

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    if values.notna().sum() == 0:

        return pd.Series(
            0.0,
            index=series.index
        )

    minimum = values.min()
    maximum = values.max()

    if pd.isna(minimum) or pd.isna(maximum):

        return pd.Series(
            0.0,
            index=series.index
        )

    if maximum == minimum:

        return pd.Series(
            1.0,
            index=series.index
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


def volume_suitability(volume):

    if pd.isna(volume):
        return 0.0

    volume = float(volume)

    if volume < 150:

        return max(
            0.0,
            volume / 150.0
        )

    if volume <= 1000:
        return 1.0

    if volume >= 2000:
        return 0.0

    return (
        1.0
        - (
            volume - 1000
        ) / 1000.0
    )


def extract_pocket_number(filename):

    match = re.search(
        r"pocket[_\-]?(\d+)",
        filename.lower()
    )

    if match:
        return int(match.group(1))

    return None


# ============================================================
# FPOCKET INFO PARSER
# ============================================================

def parse_fpocket_info(info_file):

    results = {}

    if not info_file.exists():

        print_warn(
            f"fpocket info file not found: {info_file}"
        )

        return results

    with open(
        info_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as handle:

        lines = handle.readlines()

    pocket_starts = []

    for i, line in enumerate(lines):

        match = re.search(
            r"Pocket\s+(\d+)",
            line,
            flags=re.I
        )

        if match:

            pocket_starts.append(
                (
                    i,
                    int(match.group(1))
                )
            )

    patterns = {

        "Pocket_Score": [
            r"Pocket\s*Score\s*[:=]\s*([-+0-9.eE]+)",
            r"Score\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Druggability": [
            r"Druggability\s*Score\s*[:=]\s*([-+0-9.eE]+)",
            r"Drug\s*Score\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Volume_A3": [
            r"Volume\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Alpha_Sphere_Count": [
            r"Alpha\s*Sphere\s*Count\s*[:=]\s*([-+0-9.eE]+)",
            r"Number\s*of\s*alpha\s*spheres\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Alpha_Sphere_Radius": [
            r"Alpha\s*Sphere\s*Radius\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Hydrophobicity": [
            r"Hydrophobicity\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Polarity": [
            r"Polarity\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Charge": [
            r"Charge\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "B_Factor": [
            r"B[-\s]*factor\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Local_Hydrophobic_Density": [
            r"Local\s*hydrophobic\s*density\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Apolar_Sphere_Count": [
            r"Apolar\s*Sphere\s*Count\s*[:=]\s*([-+0-9.eE]+)"
        ],

        "Apolar_Sphere_Proportion": [
            r"Apolar\s*Sphere\s*Proportion\s*[:=]\s*([-+0-9.eE]+)"
        ]
    }

    for idx, (start, pocket_number) in enumerate(pocket_starts):

        if idx + 1 < len(pocket_starts):

            end = pocket_starts[idx + 1][0]

        else:

            end = len(lines)

        block = "\n".join(
            lines[start:end]
        )

        entry = {}

        for field, regex_list in patterns.items():

            for pattern in regex_list:

                match = re.search(
                    pattern,
                    block,
                    flags=re.I
                )

                if match:

                    entry[field] = safe_float(
                        match.group(1)
                    )

                    break

        results[pocket_number] = entry

    return results


# ============================================================
# COPY EXISTING FPOCKET RESULTS
# ============================================================

def copy_existing_fpocket_results(
    source_dir,
    destination_dir
):

    if destination_dir.exists():

        shutil.rmtree(
            destination_dir
        )

    destination_dir.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    shutil.copytree(
        source_dir,
        destination_dir
    )


# ============================================================
# LOAD STRUCTURE
# ============================================================

def load_structure(
    pdb_file,
    structure_id
):

    parser = PDBParser(
        QUIET=True
    )

    return parser.get_structure(
        structure_id,
        str(pdb_file)
    )


# ============================================================
# FIND POCKET FILES
# ============================================================

def get_pocket_files(pocket_directory):

    atom_files = {}
    pqr_files = {}

    for file in pocket_directory.rglob("*"):

        if not file.is_file():
            continue

        name = file.name.lower()

        pocket_number = extract_pocket_number(
            file.name
        )

        if pocket_number is None:
            continue

        if name.endswith("_atm.pdb"):

            atom_files[pocket_number] = file

        elif name.endswith("_vert.pqr"):

            pqr_files[pocket_number] = file

    return atom_files, pqr_files


# ============================================================
# BUILD POCKET TABLE
# ============================================================

def build_pocket_table(
    atom_files,
    pqr_files,
    info_records,
    residue_166_coordinates,
    structure_label
):

    records = []

    for pocket_id in sorted(atom_files.keys()):

        atom_file = atom_files[pocket_id]

        atom_records = parse_pocket_atoms(
            atom_file
        )

        if atom_records:

            coordinates = np.asarray(
                [
                    [
                        record["X"],
                        record["Y"],
                        record["Z"]
                    ]
                    for record in atom_records
                ],
                dtype=float
            )

        else:

            coordinates = np.empty(
                (0, 3),
                dtype=float
            )

        # ----------------------------------------------------
        # Pocket center
        # ----------------------------------------------------

        if len(coordinates) > 0:

            center = coordinates.mean(axis=0)

            center_x = float(center[0])
            center_y = float(center[1])
            center_z = float(center[2])

            center_distance = (
                min_distance_to_coordinates(
                    [center],
                    residue_166_coordinates
                )
            )

            atom_distance = (
                min_distance_to_coordinates(
                    coordinates,
                    residue_166_coordinates
                )
            )

        else:

            center_x = np.nan
            center_y = np.nan
            center_z = np.nan

            center_distance = np.nan
            atom_distance = np.nan

        # ----------------------------------------------------
        # Alpha sphere coordinates
        # ----------------------------------------------------

        pqr_coordinates = np.empty(
            (0, 3),
            dtype=float
        )

        if pocket_id in pqr_files:

            pqr_coordinates = parse_pqr_coordinates(
                pqr_files[pocket_id]
            )

        # ----------------------------------------------------
        # fpocket information
        # ----------------------------------------------------

        info = info_records.get(
            pocket_id,
            {}
        )

        # ----------------------------------------------------
        # Lining residues
        # ----------------------------------------------------

        lining_positions = lining_position_set(
            atom_records
        )

        lining_map = lining_identity_map(
            atom_records
        )

        direct_lining = (
            PRIMARY_POSITION
            in lining_positions
        )

        records.append(
            {
                "Structure": structure_label,

                "Pocket_ID": pocket_id,

                "Pocket_File": str(atom_file),

                "Pocket_Score": safe_float(
                    info.get("Pocket_Score")
                ),

                "Druggability": safe_float(
                    info.get("Druggability")
                ),

                "Volume_A3": safe_float(
                    info.get("Volume_A3")
                ),

                "Alpha_Sphere_Count": safe_float(
                    info.get("Alpha_Sphere_Count")
                ),

                "Alpha_Sphere_Radius": safe_float(
                    info.get("Alpha_Sphere_Radius")
                ),

                "Hydrophobicity": safe_float(
                    info.get("Hydrophobicity")
                ),

                "Polarity": safe_float(
                    info.get("Polarity")
                ),

                "Charge": safe_float(
                    info.get("Charge")
                ),

                "B_Factor": safe_float(
                    info.get("B_Factor")
                ),

                "Local_Hydrophobic_Density": safe_float(
                    info.get(
                        "Local_Hydrophobic_Density"
                    )
                ),

                "Apolar_Sphere_Count": safe_float(
                    info.get(
                        "Apolar_Sphere_Count"
                    )
                ),

                "Apolar_Sphere_Proportion": safe_float(
                    info.get(
                        "Apolar_Sphere_Proportion"
                    )
                ),

                "Pocket_Center_X": center_x,
                "Pocket_Center_Y": center_y,
                "Pocket_Center_Z": center_z,

                "D166_Center_Distance_A":
                    center_distance,

                "D166_Min_Pocket_Atom_Distance_A":
                    atom_distance,

                "D166_Distance_Category":
                    mutation_proximity_category(
                        atom_distance
                    ),

                "D166_Directly_Lining_Pocket":
                    bool(direct_lining),

                "Lining_Residue_Count":
                    len(lining_map),

                "Lining_Residues":
                    ";".join(
                        sorted(lining_map.keys())
                    ),

                "_Lining_Position_Set":
                    lining_positions,

                "_Lining_Identity_Set":
                    set(lining_map.keys()),

                "Alpha_Sphere_Coordinate_Count":
                    len(pqr_coordinates)
            }
        )

    return pd.DataFrame(records)


# ============================================================
# POCKET MATCHING
# ============================================================

def calculate_center_distance(wt_row, mut_row):

    wt_center = np.array(
        [
            wt_row["Pocket_Center_X"],
            wt_row["Pocket_Center_Y"],
            wt_row["Pocket_Center_Z"]
        ],
        dtype=float
    )

    mut_center = np.array(
        [
            mut_row["Pocket_Center_X"],
            mut_row["Pocket_Center_Y"],
            mut_row["Pocket_Center_Z"]
        ],
        dtype=float
    )

    if (
        np.isnan(wt_center).any()
        or np.isnan(mut_center).any()
    ):
        return np.nan

    return float(
        np.linalg.norm(
            wt_center - mut_center
        )
    )


def match_pockets(wt_df, mut_df):

    comparison_records = []

    used_mutant_pockets = set()

    for _, wt_row in wt_df.iterrows():

        candidates = []

        wt_lining = wt_row["_Lining_Position_Set"]

        for _, mut_row in mut_df.iterrows():

            mut_id = int(mut_row["Pocket_ID"])

            if mut_id in used_mutant_pockets:
                continue

            center_distance = calculate_center_distance(
                wt_row,
                mut_row
            )

            lining_jaccard = jaccard_similarity(
                wt_lining,
                mut_row["_Lining_Position_Set"]
            )

            center_match = (
                not pd.isna(center_distance)
                and center_distance
                <= CENTER_MATCH_THRESHOLD_A
            )

            lining_match = (
                lining_jaccard
                >= LINING_JACCARD_THRESHOLD
            )

            if center_match or lining_match:

                composite = 0.0

                if not pd.isna(center_distance):

                    composite += max(
                        0.0,
                        1.0
                        - (
                            center_distance
                            / CENTER_MATCH_THRESHOLD_A
                        )
                    )

                composite += lining_jaccard

                candidates.append(
                    (
                        composite,
                        center_distance,
                        lining_jaccard,
                        mut_row
                    )
                )

        if candidates:

            candidates.sort(
                key=lambda x: x[0],
                reverse=True
            )

            (
                _,
                center_distance,
                lining_jaccard,
                best_mut
            ) = candidates[0]

            used_mutant_pockets.add(
                int(best_mut["Pocket_ID"])
            )

            if (
                not pd.isna(center_distance)
                and center_distance <= 4.0
                and lining_jaccard >= 0.50
            ):

                remodeling_category = "Minor remodeling"

            elif (
                not pd.isna(center_distance)
                and center_distance <= CENTER_MATCH_THRESHOLD_A
                and lining_jaccard >= LINING_JACCARD_THRESHOLD
            ):

                remodeling_category = "Moderate remodeling"

            else:

                remodeling_category = "Strong remodeling"

            comparison_records.append(
                {
                    "WT_Pocket_ID":
                        int(wt_row["Pocket_ID"]),

                    "D166V_Pocket_ID":
                        int(best_mut["Pocket_ID"]),

                    "Center_Distance_A":
                        center_distance,

                    "Lining_Jaccard":
                        lining_jaccard,

                    "Match_Status":
                        "Matched",

                    "Remodeling_Category":
                        remodeling_category,

                    "WT_Structure":
                        "WT",

                    "D166V_Structure":
                        "D166V",

                    "WT_Pocket_Score":
                        wt_row["Pocket_Score"],

                    "D166V_Pocket_Score":
                        best_mut["Pocket_Score"],

                    "Pocket_Score_Change":
                        best_mut["Pocket_Score"]
                        - wt_row["Pocket_Score"],

                    "WT_Druggability":
                        wt_row["Druggability"],

                    "D166V_Druggability":
                        best_mut["Druggability"],

                    "Druggability_Change":
                        best_mut["Druggability"]
                        - wt_row["Druggability"],

                    "WT_Volume_A3":
                        wt_row["Volume_A3"],

                    "D166V_Volume_A3":
                        best_mut["Volume_A3"],

                    "Volume_Change_A3":
                        best_mut["Volume_A3"]
                        - wt_row["Volume_A3"],

                    "WT_D166_Distance_A":
                        wt_row[
                            "D166_Min_Pocket_Atom_Distance_A"
                        ],

                    "D166V_D166_Distance_A":
                        best_mut[
                            "D166_Min_Pocket_Atom_Distance_A"
                        ],

                    "WT_D166_Category":
                        wt_row[
                            "D166_Distance_Category"
                        ],

                    "D166V_D166_Category":
                        best_mut[
                            "D166_Distance_Category"
                        ],

                    "WT_Direct_Lining":
                        wt_row[
                            "D166_Directly_Lining_Pocket"
                        ],

                    "D166V_Direct_Lining":
                        best_mut[
                            "D166_Directly_Lining_Pocket"
                        ],

                    "WT_Lining_Residues":
                        wt_row["Lining_Residues"],

                    "D166V_Lining_Residues":
                        best_mut["Lining_Residues"]
                }
            )

        else:

            comparison_records.append(
                {
                    "WT_Pocket_ID":
                        int(wt_row["Pocket_ID"]),

                    "D166V_Pocket_ID":
                        np.nan,

                    "Center_Distance_A":
                        np.nan,

                    "Lining_Jaccard":
                        np.nan,

                    "Match_Status":
                        "WT-specific",

                    "Remodeling_Category":
                        "WT-specific pocket",

                    "WT_Structure":
                        "WT",

                    "D166V_Structure":
                        "Absent",

                    "WT_Pocket_Score":
                        wt_row["Pocket_Score"],

                    "D166V_Pocket_Score":
                        np.nan,

                    "Pocket_Score_Change":
                        np.nan,

                    "WT_Druggability":
                        wt_row["Druggability"],

                    "D166V_Druggability":
                        np.nan,

                    "Druggability_Change":
                        np.nan,

                    "WT_Volume_A3":
                        wt_row["Volume_A3"],

                    "D166V_Volume_A3":
                        np.nan,

                    "Volume_Change_A3":
                        np.nan,

                    "WT_D166_Distance_A":
                        wt_row[
                            "D166_Min_Pocket_Atom_Distance_A"
                        ],

                    "D166V_D166_Distance_A":
                        np.nan,

                    "WT_D166_Category":
                        wt_row[
                            "D166_Distance_Category"
                        ],

                    "D166V_D166_Category":
                        "Absent",

                    "WT_Direct_Lining":
                        wt_row[
                            "D166_Directly_Lining_Pocket"
                        ],

                    "D166V_Direct_Lining":
                        np.nan,

                    "WT_Lining_Residues":
                        wt_row["Lining_Residues"],

                    "D166V_Lining_Residues":
                        np.nan
                }
            )

    # --------------------------------------------------------
    # Add unmatched D166V-specific pockets
    # --------------------------------------------------------

    for _, mut_row in mut_df.iterrows():

        mut_id = int(mut_row["Pocket_ID"])

        if mut_id not in used_mutant_pockets:

            comparison_records.append(
                {
                    "WT_Pocket_ID":
                        np.nan,

                    "D166V_Pocket_ID":
                        mut_id,

                    "Center_Distance_A":
                        np.nan,

                    "Lining_Jaccard":
                        np.nan,

                    "Match_Status":
                        "D166V-specific",

                    "Remodeling_Category":
                        "D166V-specific pocket",

                    "WT_Structure":
                        "Absent",

                    "D166V_Structure":
                        "D166V",

                    "WT_Pocket_Score":
                        np.nan,

                    "D166V_Pocket_Score":
                        mut_row["Pocket_Score"],

                    "Pocket_Score_Change":
                        np.nan,

                    "WT_Druggability":
                        np.nan,

                    "D166V_Druggability":
                        mut_row["Druggability"],

                    "Druggability_Change":
                        np.nan,

                    "WT_Volume_A3":
                        np.nan,

                    "D166V_Volume_A3":
                        mut_row["Volume_A3"],

                    "Volume_Change_A3":
                        np.nan,

                    "WT_D166_Distance_A":
                        np.nan,

                    "D166V_D166_Distance_A":
                        mut_row[
                            "D166_Min_Pocket_Atom_Distance_A"
                        ],

                    "WT_D166_Category":
                        "Absent",

                    "D166V_D166_Category":
                        mut_row[
                            "D166_Distance_Category"
                        ],

                    "WT_Direct_Lining":
                        np.nan,

                    "D166V_Direct_Lining":
                        mut_row[
                            "D166_Directly_Lining_Pocket"
                        ],

                    "WT_Lining_Residues":
                        np.nan,

                    "D166V_Lining_Residues":
                        mut_row["Lining_Residues"]
                }
            )

    return pd.DataFrame(
        comparison_records
    )


# ============================================================
# PRIORITIZATION
# ============================================================

def build_prioritization_table(
    comparison_df,
    wt_df,
    mut_df
):

    records = []

    # --------------------------------------------------------
    # Prioritize D166V pockets
    # --------------------------------------------------------

    mut_lookup = mut_df.set_index(
        "Pocket_ID"
    )

    for _, row in comparison_df.iterrows():

        if pd.isna(
            row["D166V_Pocket_ID"]
        ):
            continue

        pocket_id = int(
            row["D166V_Pocket_ID"]
        )

        mut_pocket = mut_lookup.loc[
            pocket_id
        ]

        druggability = safe_float(
            mut_pocket["Druggability"],
            0.0
        )

        pocket_score = safe_float(
            mut_pocket["Pocket_Score"],
            0.0
        )

        volume = safe_float(
            mut_pocket["Volume_A3"],
            np.nan
        )

        distance = safe_float(
            mut_pocket[
                "D166_Min_Pocket_Atom_Distance_A"
            ],
            np.nan
        )

        remodeling = mutation_remodeling_score(
            row["Remodeling_Category"],
            row["Lining_Jaccard"],
            row["Pocket_Score_Change"],
            row["Druggability_Change"],
            row["Volume_Change_A3"]
        )

        records.append(
            {
                "D166V_Pocket_ID":
                    pocket_id,

                "WT_Pocket_ID":
                    row["WT_Pocket_ID"],

                "Match_Status":
                    row["Match_Status"],

                "Remodeling_Category":
                    row["Remodeling_Category"],

                "Druggability":
                    druggability,

                "Pocket_Score":
                    pocket_score,

                "Volume_A3":
                    volume,

                "D166_Distance_A":
                    distance,

                "D166_Distance_Category":
                    mut_pocket[
                        "D166_Distance_Category"
                    ],

                "D166_Directly_Lining":
                    mut_pocket[
                        "D166_Directly_Lining_Pocket"
                    ],

                "Lining_Jaccard":
                    row["Lining_Jaccard"],

                "Mutation_Remodeling_Score":
                    remodeling
            }
        )

    priority_df = pd.DataFrame(records)

    if priority_df.empty:
        return priority_df

    priority_df[
        "Druggability_Normalized"
    ] = minmax_normalize(
        priority_df["Druggability"]
    )

    priority_df[
        "Pocket_Score_Normalized"
    ] = minmax_normalize(
        priority_df["Pocket_Score"]
    )

    priority_df[
        "Volume_Suitability"
    ] = priority_df[
        "Volume_A3"
    ].apply(
        volume_suitability
    )

    priority_df[
        "D166_Proximity_Score"
    ] = priority_df[
        "D166_Distance_A"
    ].apply(
        proximity_score
    )

    priority_df[
        "Prioritization_Score"
    ] = (
        W_DRUGGABILITY
        * priority_df[
            "Druggability_Normalized"
        ]
        +
        W_POCKET_SCORE
        * priority_df[
            "Pocket_Score_Normalized"
        ]
        +
        W_VOLUME
        * priority_df[
            "Volume_Suitability"
        ]
        +
        W_D166_PROXIMITY
        * priority_df[
            "D166_Proximity_Score"
        ]
        +
        W_MUTATION_REMODELING
        * priority_df[
            "Mutation_Remodeling_Score"
        ]
    )

    priority_df[
        "Prioritization_Percent"
    ] = (
        priority_df[
            "Prioritization_Score"
        ]
        * 100
    )

    priority_df = priority_df.sort_values(
        "Prioritization_Score",
        ascending=False
    ).reset_index(
        drop=True
    )

    priority_df.insert(
        0,
        "Priority_Rank",
        range(1, len(priority_df) + 1)
    )

    return priority_df


# ============================================================
# FIGURES
# ============================================================

def save_figure(fig, filename):

    output = FIGURE_DIR / filename

    fig.savefig(
        output,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print_pass(
        f"Figure saved: {output.name}"
    )


def create_d166_distance_figure(
    wt_df,
    mut_df
):

    fig, ax = plt.subplots(
        figsize=(12, 7)
    )

    wt_sorted = wt_df.sort_values(
        "D166_Min_Pocket_Atom_Distance_A"
    )

    mut_sorted = mut_df.sort_values(
        "D166_Min_Pocket_Atom_Distance_A"
    )

    ax.scatter(
        wt_sorted["Pocket_ID"],
        wt_sorted[
            "D166_Min_Pocket_Atom_Distance_A"
        ],
        label="WT",
        s=50
    )

    ax.scatter(
        mut_sorted["Pocket_ID"],
        mut_sorted[
            "D166_Min_Pocket_Atom_Distance_A"
        ],
        label="D166V",
        s=50
    )

    ax.axhline(
        VERY_CLOSE_A,
        linestyle="--",
        linewidth=1
    )

    ax.axhline(
        CLOSE_A,
        linestyle="--",
        linewidth=1
    )

    ax.axhline(
        MODERATE_A,
        linestyle="--",
        linewidth=1
    )

    ax.set_xlabel(
        "fpocket Pocket ID",
        fontsize=12
    )

    ax.set_ylabel(
        "Minimum Distance to Residue 166 (Å)",
        fontsize=12
    )

    ax.set_title(
        "PNPLA3 Pocket Proximity to Residue 166",
        fontsize=14
    )

    ax.legend()

    ax.grid(
        alpha=0.3
    )

    save_figure(
        fig,
        "Figure_16_01_D166_Pocket_Proximity.png"
    )


def create_pocket_score_comparison(
    comparison_df
):

    matched = comparison_df[
        comparison_df["Match_Status"] == "Matched"
    ].copy()

    if matched.empty:
        return

    matched = matched.dropna(
        subset=[
            "WT_Pocket_Score",
            "D166V_Pocket_Score"
        ]
    )

    if matched.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        matched["WT_Pocket_Score"],
        matched["D166V_Pocket_Score"],
        s=60
    )

    minimum = min(
        matched["WT_Pocket_Score"].min(),
        matched["D166V_Pocket_Score"].min()
    )

    maximum = max(
        matched["WT_Pocket_Score"].max(),
        matched["D166V_Pocket_Score"].max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel(
        "WT Pocket Score",
        fontsize=12
    )

    ax.set_ylabel(
        "D166V Pocket Score",
        fontsize=12
    )

    ax.set_title(
        "WT vs D166V Matched Pocket Scores",
        fontsize=14
    )

    ax.grid(alpha=0.3)

    save_figure(
        fig,
        "Figure_16_02_WT_vs_D166V_Pocket_Score.png"
    )


def create_druggability_comparison(
    comparison_df
):

    matched = comparison_df[
        comparison_df["Match_Status"] == "Matched"
    ].copy()

    matched = matched.dropna(
        subset=[
            "WT_Druggability",
            "D166V_Druggability"
        ]
    )

    if matched.empty:
        return

    fig, ax = plt.subplots(
        figsize=(8, 8)
    )

    ax.scatter(
        matched["WT_Druggability"],
        matched["D166V_Druggability"],
        s=60
    )

    minimum = min(
        matched["WT_Druggability"].min(),
        matched["D166V_Druggability"].min()
    )

    maximum = max(
        matched["WT_Druggability"].max(),
        matched["D166V_Druggability"].max()
    )

    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    ax.set_xlabel(
        "WT Druggability Score",
        fontsize=12
    )

    ax.set_ylabel(
        "D166V Druggability Score",
        fontsize=12
    )

    ax.set_title(
        "WT vs D166V Matched Pocket Druggability",
        fontsize=14
    )

    ax.grid(alpha=0.3)

    save_figure(
        fig,
        "Figure_16_03_WT_vs_D166V_Druggability.png"
    )


def create_prioritization_figure(
    priority_df
):

    if priority_df.empty:
        return

    top_df = priority_df.head(15).copy()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    labels = [
        f"D166V Pocket {int(x)}"
        for x in top_df[
            "D166V_Pocket_ID"
        ]
    ]

    ax.bar(
        labels,
        top_df[
            "Prioritization_Percent"
        ]
    )

    ax.set_ylabel(
        "Computational Prioritization Score (%)",
        fontsize=12
    )

    ax.set_xlabel(
        "D166V Pocket",
        fontsize=12
    )

    ax.set_title(
        "Top D166V Pocket Prioritization Scores",
        fontsize=14
    )

    plt.xticks(
        rotation=45,
        ha="right"
    )

    ax.grid(
        axis="y",
        alpha=0.3
    )

    save_figure(
        fig,
        "Figure_16_04_D166V_Pocket_Prioritization.png"
    )


# ============================================================
# START ANALYSIS
# ============================================================

print_header(
    "STEP 16 — WT vs D166V POCKET DISCOVERY"
)

print("PNPLA3 / Q9NST1")
print(f"Primary variant: {PRIMARY_VARIANT}")
print(f"Primary position: {PRIMARY_POSITION}")
print(f"Primary VariationID: {PRIMARY_VARIATION_ID}")


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for directory in [

    OUTPUT_DIR,
    POCKET_DIR,
    WT_POCKET_DIR,
    D166V_POCKET_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR

]:

    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# INPUT VALIDATION
# ============================================================

print_header(
    "CHECKING INPUT FILES"
)

required_inputs = {

    "Step 9F candidate file":
        STEP9F,

    "WT PDB":
        WT_PDB,

    "D166V PDB":
        D166V_PDB,

    "WT fpocket output":
        WT_FPOCKET,

    "D166V fpocket output":
        D166V_FPOCKET
}


input_failure = False

for name, path in required_inputs.items():

    if path.exists():

        print_pass(
            f"{name}: {path}"
        )

    else:

        print_fail(
            f"{name} missing: {path}"
        )

        input_failure = True


if input_failure:

    raise FileNotFoundError(
        "One or more required Step 16 inputs are missing."
    )


# ============================================================
# STEP 9F VALIDATION
# ============================================================

print_header(
    "VALIDATING STEP 9F PRIMARY CANDIDATE"
)

step9f_df = pd.read_csv(
    STEP9F
)

required_columns = [

    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",
    "UniProt_ID"
]

missing_columns = [

    column
    for column in required_columns
    if column not in step9f_df.columns
]

if missing_columns:

    raise ValueError(
        "Missing Step 9F columns: "
        + ", ".join(missing_columns)
    )


uniprot_values = (
    step9f_df["UniProt_ID"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

if uniprot_values != [UNIPROT_ID]:

    raise ValueError(
        "Unexpected UniProt IDs in Step 9F: "
        + str(uniprot_values)
    )


primary_rows = step9f_df[
    pd.to_numeric(
        step9f_df["VariationID"],
        errors="coerce"
    )
    == PRIMARY_VARIATION_ID
]

if len(primary_rows) != 1:

    raise ValueError(
        f"Expected exactly one VariationID "
        f"{PRIMARY_VARIATION_ID}, found "
        f"{len(primary_rows)}."
    )


primary_row = primary_rows.iloc[0]


if (
    str(
        primary_row["Protein_Change"]
    ).strip()
    != PRIMARY_VARIANT
):

    raise ValueError(
        f"Primary Protein_Change does not match "
        f"{PRIMARY_VARIANT}."
    )


if (
    normalize_position(
        primary_row["Protein_Position"]
    )
    != PRIMARY_POSITION
):

    raise ValueError(
        "Primary Protein_Position is not 166."
    )


print_pass(
    f"Primary candidate validated: "
    f"{PRIMARY_VARIANT}"
)


# ============================================================
# LOAD STRUCTURES
# ============================================================

print_header(
    "LOADING WT AND D166V STRUCTURES"
)

wt_structure = load_structure(
    WT_PDB,
    "PNPLA3_WT"
)

mut_structure = load_structure(
    D166V_PDB,
    "PNPLA3_D166V"
)


wt_residue_166 = residue_atom_coordinates(
    wt_structure,
    PRIMARY_POSITION
)

mut_residue_166 = residue_atom_coordinates(
    mut_structure,
    PRIMARY_POSITION
)


if len(wt_residue_166) == 0:

    raise ValueError(
        "Residue 166 not found in WT structure."
    )


if len(mut_residue_166) == 0:

    raise ValueError(
        "Residue 166 not found in D166V structure."
    )


print_pass(
    f"WT residue 166 heavy atoms: "
    f"{len(wt_residue_166)}"
)

print_pass(
    f"D166V residue 166 heavy atoms: "
    f"{len(mut_residue_166)}"
)


# ============================================================
# COPY EXISTING FPOCKET RESULTS
# ============================================================

print_header(
    "REUSING EXISTING FPOCKET OUTPUTS"
)

copy_existing_fpocket_results(
    WT_FPOCKET,
    WT_POCKET_DIR
)

copy_existing_fpocket_results(
    D166V_FPOCKET,
    D166V_POCKET_DIR
)


print_pass(
    "Existing WT fpocket results copied."
)

print_pass(
    "Existing D166V fpocket results copied."
)

print(
    "fpocket was NOT rerun."
)


# ============================================================
# LOCATE POCKET FILES
# ============================================================

print_header(
    "LOCATING FPOCKET POCKET FILES"
)

wt_atom_files, wt_pqr_files = get_pocket_files(
    WT_POCKET_DIR
)

mut_atom_files, mut_pqr_files = get_pocket_files(
    D166V_POCKET_DIR
)


print(
    f"WT pocket atom files: "
    f"{len(wt_atom_files)}"
)

print(
    f"D166V pocket atom files: "
    f"{len(mut_atom_files)}"
)


if len(wt_atom_files) != EXPECTED_POCKET_COUNT:

    raise ValueError(
        f"WT pocket count is "
        f"{len(wt_atom_files)}, expected "
        f"{EXPECTED_POCKET_COUNT}."
    )


if len(mut_atom_files) != EXPECTED_POCKET_COUNT:

    raise ValueError(
        f"D166V pocket count is "
        f"{len(mut_atom_files)}, expected "
        f"{EXPECTED_POCKET_COUNT}."
    )


print_pass(
    f"WT pocket count = "
    f"{EXPECTED_POCKET_COUNT}"
)

print_pass(
    f"D166V pocket count = "
    f"{EXPECTED_POCKET_COUNT}"
)


# ============================================================
# LOCATE FPOCKET INFO FILES
# ============================================================

print_header(
    "PARSING FPOCKET INFORMATION"
)

wt_info_candidates = list(
    WT_POCKET_DIR.rglob("*_info.txt")
)

mut_info_candidates = list(
    D166V_POCKET_DIR.rglob("*_info.txt")
)


if not wt_info_candidates:

    raise FileNotFoundError(
        "WT fpocket *_info.txt file not found."
    )


if not mut_info_candidates:

    raise FileNotFoundError(
        "D166V fpocket *_info.txt file not found."
    )


wt_info_file = wt_info_candidates[0]
mut_info_file = mut_info_candidates[0]


print_pass(
    f"WT info file: "
    f"{wt_info_file.name}"
)

print_pass(
    f"D166V info file: "
    f"{mut_info_file.name}"
)


wt_info = parse_fpocket_info(
    wt_info_file
)

mut_info = parse_fpocket_info(
    mut_info_file
)


print(
    f"WT fpocket info records parsed: "
    f"{len(wt_info)}"
)

print(
    f"D166V fpocket info records parsed: "
    f"{len(mut_info)}"
)


if len(wt_info) == EXPECTED_POCKET_COUNT:

    print_pass(
        "WT fpocket information contains "
        "35 pocket records."
    )

else:

    print_warn(
        "WT fpocket information did not parse "
        "35 records. Analysis will continue."
    )


if len(mut_info) == EXPECTED_POCKET_COUNT:

    print_pass(
        "D166V fpocket information contains "
        "35 pocket records."
    )

else:

    print_warn(
        "D166V fpocket information did not parse "
        "35 records. Analysis will continue."
    )


# ============================================================
# BUILD WT POCKET TABLE
# ============================================================

print_header(
    "BUILDING WT POCKET TABLE"
)

wt_df = build_pocket_table(
    wt_atom_files,
    wt_pqr_files,
    wt_info,
    wt_residue_166,
    "WT"
)

print_pass(
    f"WT pockets processed: "
    f"{len(wt_df)}"
)


# ============================================================
# BUILD D166V POCKET TABLE
# ============================================================

print_header(
    "BUILDING D166V POCKET TABLE"
)

mut_df = build_pocket_table(
    mut_atom_files,
    mut_pqr_files,
    mut_info,
    mut_residue_166,
    "D166V"
)

print_pass(
    f"D166V pockets processed: "
    f"{len(mut_df)}"
)


# ============================================================
# SAVE INDIVIDUAL POCKET TABLES
# ============================================================

print_header(
    "SAVING INDIVIDUAL POCKET TABLES"
)

# Remove internal Python set columns before CSV export

wt_export = wt_df.drop(
    columns=[
        "_Lining_Position_Set",
        "_Lining_Identity_Set"
    ],
    errors="ignore"
)

mut_export = mut_df.drop(
    columns=[
        "_Lining_Position_Set",
        "_Lining_Identity_Set"
    ],
    errors="ignore"
)


wt_export_file = (
    TABLE_DIR
    / "Table_16_01_WT_Pocket_Descriptors.csv"
)

mut_export_file = (
    TABLE_DIR
    / "Table_16_02_D166V_Pocket_Descriptors.csv"
)


wt_export.to_csv(
    wt_export_file,
    index=False
)

mut_export.to_csv(
    mut_export_file,
    index=False
)


print_pass(
    f"Saved: {wt_export_file.name}"
)

print_pass(
    f"Saved: {mut_export_file.name}"
)


# ============================================================
# MATCH WT AND D166V POCKETS
# ============================================================

print_header(
    "MATCHING WT AND D166V POCKETS"
)

comparison_df = match_pockets(
    wt_df,
    mut_df
)


comparison_file = (
    TABLE_DIR
    / "Table_16_03_WT_vs_D166V_Pocket_Comparison.csv"
)

comparison_df.to_csv(
    comparison_file,
    index=False
)


print_pass(
    f"Pocket comparison records: "
    f"{len(comparison_df)}"
)

print_pass(
    f"Saved: {comparison_file.name}"
)


# ============================================================
# MATCH SUMMARY
# ============================================================

print_header(
    "POCKET MATCH SUMMARY"
)

match_summary = (
    comparison_df[
        "Match_Status"
    ]
    .value_counts(
        dropna=False
    )
)

for category, count in match_summary.items():

    print(
        f"{category}: {count}"
    )


# ============================================================
# BUILD PRIORITIZATION TABLE
# ============================================================

print_header(
    "BUILDING D166V POCKET PRIORITIZATION"
)

priority_df = build_prioritization_table(
    comparison_df,
    wt_df,
    mut_df
)


priority_file = (
    TABLE_DIR
    / "Table_16_04_D166V_Pocket_Prioritization.csv"
)

priority_df.to_csv(
    priority_file,
    index=False
)


print_pass(
    f"Prioritized D166V pockets: "
    f"{len(priority_df)}"
)

print_pass(
    f"Saved: {priority_file.name}"
)


# ============================================================
# TOP POCKETS
# ============================================================

print_header(
    "TOP PRIORITIZED D166V POCKETS"
)

if not priority_df.empty:

    display_columns = [

        "Priority_Rank",
        "D166V_Pocket_ID",
        "WT_Pocket_ID",
        "Match_Status",
        "Remodeling_Category",
        "Druggability",
        "Pocket_Score",
        "Volume_A3",
        "D166_Distance_A",
        "Prioritization_Percent"
    ]

    print(
        priority_df[
            display_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

else:

    print_warn(
        "No D166V pockets available for prioritization."
    )


# ============================================================
# CREATE FIGURES
# ============================================================

print_header(
    "CREATING PUBLICATION-QUALITY FIGURES"
)

create_d166_distance_figure(
    wt_df,
    mut_df
)

create_pocket_score_comparison(
    comparison_df
)

create_druggability_comparison(
    comparison_df
)

create_prioritization_figure(
    priority_df
)


# ============================================================
# QC SUMMARY
# ============================================================

print_header(
    "GENERATING QC SUMMARY"
)

qc_summary = {

    "Step":
        "STEP16_WT_vs_D166V_POCKET_DISCOVERY",

    "UniProt_ID":
        UNIPROT_ID,

    "Expected_Protein_Length":
        EXPECTED_PROTEIN_LENGTH,

    "Primary_Variant":
        PRIMARY_VARIANT,

    "Primary_Position":
        PRIMARY_POSITION,

    "Primary_VariationID":
        PRIMARY_VARIATION_ID,

    "WT_Fpocket_Source":
        str(WT_FPOCKET),

    "D166V_Fpocket_Source":
        str(D166V_FPOCKET),

    "WT_Pocket_Count":
        int(len(wt_df)),

    "D166V_Pocket_Count":
        int(len(mut_df)),

    "Matched_Pockets":
        int(
            (
                comparison_df["Match_Status"]
                == "Matched"
            ).sum()
        ),

    "WT_Specific_Pockets":
        int(
            (
                comparison_df["Match_Status"]
                == "WT-specific"
            ).sum()
        ),

    "D166V_Specific_Pockets":
        int(
            (
                comparison_df["Match_Status"]
                == "D166V-specific"
            ).sum()
        ),

    "Prioritized_D166V_Pockets":
        int(len(priority_df)),

    "Center_Match_Threshold_A":
        CENTER_MATCH_THRESHOLD_A,

    "Lining_Jaccard_Threshold":
        LINING_JACCARD_THRESHOLD,

    "Prioritization_Weights": {

        "Druggability":
            W_DRUGGABILITY,

        "Pocket_Score":
            W_POCKET_SCORE,

        "Volume":
            W_VOLUME,

        "D166_Proximity":
            W_D166_PROXIMITY,

        "Mutation_Remodeling":
            W_MUTATION_REMODELING
    },

    "Methodological_Note":
        (
            "fpocket descriptors and computational "
            "prioritization scores are computational "
            "structural descriptors and do not constitute "
            "experimental evidence of therapeutic binding."
        )
}


qc_file = (
    QC_DIR
    / "Step16_QC_Summary.json"
)


with open(
    qc_file,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        make_json_safe(qc_summary),
        handle,
        indent=4
    )


print_pass(
    f"QC summary saved: "
    f"{qc_file.name}"
)


# ============================================================
# FINAL SUMMARY TEXT
# ============================================================

summary_file = (
    OUTPUT_DIR
    / "STEP16_ANALYSIS_SUMMARY.txt"
)


with open(
    summary_file,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "=" * 78 + "\n"
    )

    handle.write(
        "STEP 16 — WT vs D166V POCKET DISCOVERY\n"
    )

    handle.write(
        "=" * 78 + "\n\n"
    )

    handle.write(
        f"UniProt ID: {UNIPROT_ID}\n"
    )

    handle.write(
        f"Primary variant: {PRIMARY_VARIANT}\n"
    )

    handle.write(
        f"Primary position: {PRIMARY_POSITION}\n\n"
    )

    handle.write(
        f"WT pockets analyzed: {len(wt_df)}\n"
    )

    handle.write(
        f"D166V pockets analyzed: {len(mut_df)}\n"
    )

    handle.write(
        f"Matched pockets: "
        f"{int((comparison_df['Match_Status'] == 'Matched').sum())}\n"
    )

    handle.write(
        f"WT-specific pockets: "
        f"{int((comparison_df['Match_Status'] == 'WT-specific').sum())}\n"
    )

    handle.write(
        f"D166V-specific pockets: "
        f"{int((comparison_df['Match_Status'] == 'D166V-specific').sum())}\n\n"
    )

    handle.write(
        "Important interpretation note:\n"
    )

    handle.write(
        "fpocket scores, pocket descriptors, and the "
        "prioritization score are computational descriptors. "
        "They do not constitute experimental evidence of "
        "ligand binding, therapeutic activity, or mutation "
        "causality.\n"
    )


print_pass(
    f"Analysis summary saved: "
    f"{summary_file.name}"
)


# ============================================================
# FINAL COMPLETION
# ============================================================

print_header(
    "STEP 16 COMPLETED SUCCESSFULLY"
)

print(
    f"Output directory:\n{OUTPUT_DIR}"
)

print()
print("Generated directories:")

print(
    f"  Tables:  {TABLE_DIR}"
)

print(
    f"  Figures: {FIGURE_DIR}"
)

print(
    f"  QC:      {QC_DIR}"
)

print()
print(
    "fpocket was NOT rerun."
)

print(
    "No docking was performed."
)

print(
    "No molecular dynamics was performed."
)

print(
    "Step 16 analysis completed successfully."
)

print("=" * 78)