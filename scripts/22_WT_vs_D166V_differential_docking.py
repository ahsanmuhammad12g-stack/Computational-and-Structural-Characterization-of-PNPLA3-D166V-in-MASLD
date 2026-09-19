#!/usr/bin/env python3

"""
STEP 22 — WT vs D166V DIFFERENTIAL DOCKING

Purpose
-------
Dock the same Step 19 ligand set against:

    1. PNPLA3 WT
    2. PNPLA3 D166V

using the same:

    - AutoDock Vina 1.2.5
    - ligand PDBQT files
    - docking box
    - exhaustiveness
    - number of modes
    - energy range
    - random seed

Primary comparison
------------------
Delta_Vina = D166V_score - WT_score

Interpretation:
    Delta_Vina < 0  -> more favorable Vina score for D166V
    Delta_Vina > 0  -> more favorable Vina score for WT
    Delta_Vina ~ 0  -> little score difference

IMPORTANT:
    Delta_Vina is a computational docking difference.
    It is NOT experimental selectivity or binding affinity.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path.cwd()

STEP19_CSV = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP19_DRUG_LIKENESS_FILTERING"
    / "tables"
    / "STEP19_SCREENING_READY_LIBRARY.csv"
)

STEP16_CSV = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
    / "tables"
    / "Table_16_04_D166V_Pocket_Prioritization.csv"
)

STEP21_CSV = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
    / "tables"
    / "STEP21_BEST_DOCKING_RESULTS.csv"
)

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

STEP21_LIGAND_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
    / "ligands"
)

STEP21_POSE_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
    / "poses"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
)

DIR_RECEPTOR = OUT_ROOT / "receptor"
DIR_POSES = OUT_ROOT / "poses"
DIR_LOGS = OUT_ROOT / "logs"
DIR_TABLES = OUT_ROOT / "tables"
DIR_QC = OUT_ROOT / "qc"

for directory in (
    DIR_RECEPTOR,
    DIR_POSES,
    DIR_LOGS,
    DIR_TABLES,
    DIR_QC,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# PARAMETERS
# ============================================================

CHAIN = "A"
RESIDUE = 166
POCKET_ID = 1

BOX_SIZE = (
    20.0,
    20.0,
    20.0,
)

EXHAUSTIVENESS = 16
NUM_MODES = 10
ENERGY_RANGE = 4.0
SEED = 20260910


# ============================================================
# VINA
# ============================================================

VINA_BIN = shutil.which("vina")

if VINA_BIN is None:
    raise RuntimeError(
        "AutoDock Vina was not found.\n"
        "Run: which vina"
    )


# ============================================================
# HELPERS
# ============================================================

def find_ca(
    pdb_file: Path,
    chain_id: str,
    residue_number: int,
) -> dict:

    with pdb_file.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for line in handle:

            if not line.startswith("ATOM"):
                continue

            if line[21].strip() != chain_id:
                continue

            if line[12:16].strip() != "CA":
                continue

            try:
                resnum = int(
                    line[22:26].strip()
                )
            except ValueError:
                continue

            if resnum != residue_number:
                continue

            return {
                "x": float(line[30:38]),
                "y": float(line[38:46]),
                "z": float(line[46:54]),
                "resname": line[17:20].strip(),
                "chain": chain_id,
                "residue": residue_number,
            }

    raise RuntimeError(
        f"Could not find {chain_id}:{residue_number} "
        f"CA in {pdb_file}"
    )


def make_clean_receptor(
    source_pdb: Path,
    destination_pdb: Path,
    chain_id: str,
) -> int:

    count = 0

    with (
        source_pdb.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as source,
        destination_pdb.open(
            "w",
            encoding="utf-8",
        ) as destination,
    ):

        for line in source:

            if (
                line.startswith("ATOM")
                and line[21].strip() == chain_id
            ):
                destination.write(line)
                count += 1

        destination.write("END\n")

    if count == 0:
        raise RuntimeError(
            f"No ATOM records found for chain {chain_id} "
            f"in {source_pdb}"
        )

    return count


def prepare_receptor(
    receptor_pdb: Path,
    receptor_pdbqt: Path,
    center: tuple[float, float, float],
    log_file: Path,
) -> None:

    command = [
        "mk_prepare_receptor.py",

        "--read_pdb",
        str(receptor_pdb),

        "--write_pdbqt",
        str(receptor_pdbqt),

        "--box_center",
        f"{center[0]:.6f}",
        f"{center[1]:.6f}",
        f"{center[2]:.6f}",

        "--box_size",
        f"{BOX_SIZE[0]:.6f}",
        f"{BOX_SIZE[1]:.6f}",
        f"{BOX_SIZE[2]:.6f}",
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    print(result.stdout)

    log_file.write_text(
        result.stdout,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Meeko receptor preparation failed:\n"
            + result.stdout
        )

    if not receptor_pdbqt.exists():
        raise RuntimeError(
            f"Receptor PDBQT was not created:\n"
            f"{receptor_pdbqt}"
        )


def run_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_pose: Path,
    log_file: Path,
    center: tuple[float, float, float],
) -> None:

    # IMPORTANT:
    # Vina 1.2.5 used here does not support --log.
    # Stdout/stderr are captured by Python.

    command = [
        VINA_BIN,

        "--receptor",
        str(receptor_pdbqt),

        "--ligand",
        str(ligand_pdbqt),

        "--center_x",
        f"{center[0]:.6f}",

        "--center_y",
        f"{center[1]:.6f}",

        "--center_z",
        f"{center[2]:.6f}",

        "--size_x",
        f"{BOX_SIZE[0]:.6f}",

        "--size_y",
        f"{BOX_SIZE[1]:.6f}",

        "--size_z",
        f"{BOX_SIZE[2]:.6f}",

        "--exhaustiveness",
        str(EXHAUSTIVENESS),

        "--num_modes",
        str(NUM_MODES),

        "--energy_range",
        str(ENERGY_RANGE),

        "--seed",
        str(SEED),

        "--out",
        str(output_pose),
    ]

    print(
        "\n$",
        " ".join(command),
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    print(result.stdout)

    log_file.write_text(
        result.stdout,
        encoding="utf-8",
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Vina failed for {ligand_pdbqt.name}:\n"
            + result.stdout
        )

    if not output_pose.exists():
        raise RuntimeError(
            f"Vina completed without generating:\n"
            f"{output_pose}"
        )


def parse_vina_scores(
    pose_file: Path,
) -> list[dict]:

    pattern = re.compile(
        r"REMARK VINA RESULT:\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?)\s+"
        r"([-+]?\d+(?:\.\d+)?)"
    )

    records = []

    with pose_file.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for line in handle:

            match = pattern.search(line)

            if match:

                records.append(
                    {
                        "score":
                            float(match.group(1)),
                        "rmsd_lb":
                            float(match.group(2)),
                        "rmsd_ub":
                            float(match.group(3)),
                    }
                )

    if not records:
        raise RuntimeError(
            f"No Vina score records found in:\n"
            f"{pose_file}"
        )

    return records


# ============================================================
# CHECK INPUTS
# ============================================================

required_files = [
    STEP19_CSV,
    STEP16_CSV,
    STEP21_CSV,
    WT_PDB,
    D166V_PDB,
]

for file in required_files:

    if not file.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file}"
        )


# ============================================================
# VALIDATE STEP 16 POCKET 1
# ============================================================

pocket_df = pd.read_csv(
    STEP16_CSV
)

pocket_row = pocket_df[
    pocket_df["D166V_Pocket_ID"]
    == POCKET_ID
]

if pocket_row.empty:
    raise RuntimeError(
        "Step 16 Pocket 1 was not found."
    )

pocket_row = (
    pocket_row
    .sort_values("Priority_Rank")
    .iloc[0]
)

if str(
    pocket_row["Match_Status"]
) != "Matched":

    raise RuntimeError(
        "Step 16 Pocket 1 is not marked as Matched."
    )


# ============================================================
# DEFINE THE COMMON DOCKING CENTER
# ============================================================

d166v_ca = find_ca(
    D166V_PDB,
    CHAIN,
    RESIDUE,
)

wt_ca = find_ca(
    WT_PDB,
    CHAIN,
    RESIDUE,
)

# Use the D166V Pocket 1 center for BOTH structures.
#
# This keeps the WT vs D166V comparison under the same
# explicitly defined search-space coordinates.
#
# Step 17 established that the pocket-center displacement
# between WT and D166V is only ~0.0007 Å.

CENTER = (
    d166v_ca["x"],
    d166v_ca["y"],
    d166v_ca["z"],
)

center_displacement = float(
    np.sqrt(
        (
            wt_ca["x"]
            - d166v_ca["x"]
        ) ** 2
        +
        (
            wt_ca["y"]
            - d166v_ca["y"]
        ) ** 2
        +
        (
            wt_ca["z"]
            - d166v_ca["z"]
        ) ** 2
    )
)


print("\nTarget center comparison:")

print(
    "  D166V A:166 CA = "
    f"({d166v_ca['x']:.4f}, "
    f"{d166v_ca['y']:.4f}, "
    f"{d166v_ca['z']:.4f})"
)

print(
    "  WT A:166 CA    = "
    f"({wt_ca['x']:.4f}, "
    f"{wt_ca['y']:.4f}, "
    f"{wt_ca['z']:.4f})"
)

print(
    f"  WT-D166V CA displacement = "
    f"{center_displacement:.6f} Å"
)

print(
    "  COMMON DOCKING CENTER     = "
    f"({CENTER[0]:.4f}, "
    f"{CENTER[1]:.4f}, "
    f"{CENTER[2]:.4f})"
)


# ============================================================
# PREPARE WT RECEPTOR
# ============================================================

wt_clean_pdb = (
    DIR_RECEPTOR
    / "PNPLA3_WT_chainA_protein.pdb"
)

wt_atom_count = make_clean_receptor(
    WT_PDB,
    wt_clean_pdb,
    CHAIN,
)

print(
    f"\nWT receptor ATOM records: "
    f"{wt_atom_count}"
)

wt_pdbqt = (
    DIR_RECEPTOR
    / "PNPLA3_WT_chainA.pdbqt"
)

prepare_receptor(
    wt_clean_pdb,
    wt_pdbqt,
    CENTER,
    DIR_LOGS / "WT_receptor_preparation.log",
)


# ============================================================
# PREPARE D166V RECEPTOR
# ============================================================

d166v_clean_pdb = (
    DIR_RECEPTOR
    / "PNPLA3_D166V_chainA_protein.pdb"
)

d166v_atom_count = make_clean_receptor(
    D166V_PDB,
    d166v_clean_pdb,
    CHAIN,
)

print(
    f"D166V receptor ATOM records: "
    f"{d166v_atom_count}"
)

d166v_pdbqt = (
    DIR_RECEPTOR
    / "PNPLA3_D166V_chainA.pdbqt"
)

prepare_receptor(
    d166v_clean_pdb,
    d166v_pdbqt,
    CENTER,
    DIR_LOGS / "D166V_receptor_preparation.log",
)


# ============================================================
# READ STEP 21 LIBRARY
# ============================================================

step19_df = pd.read_csv(
    STEP19_CSV
)

step21_df = pd.read_csv(
    STEP21_CSV
)

required_step19 = [
    "CID",
    "IUPACName",
    "IsomericSMILES",
]

missing_step19 = [
    c for c in required_step19
    if c not in step19_df.columns
]

if missing_step19:
    raise RuntimeError(
        "Missing Step 19 columns:\n"
        + "\n".join(missing_step19)
    )


required_step21 = [
    "CID",
    "Best_Vina_Score_kcal_mol",
]

missing_step21 = [
    c for c in required_step21
    if c not in step21_df.columns
]

if missing_step21:
    raise RuntimeError(
        "Missing Step 21 columns:\n"
        + "\n".join(missing_step21)
    )


# The actual Step 21 successful ligand set is used.
input_ligands = (
    step21_df[
        step21_df["Docking_Status"]
        .astype(str)
        .str.upper()
        == "SUCCESS"
    ]
    .copy()
)

input_ligands["CID"] = (
    input_ligands["CID"]
    .astype(int)
)

print(
    f"\nStep 21 successful compounds available: "
    f"{len(input_ligands)}"
)


# ============================================================
# CHECK LIGAND FILES
# ============================================================

missing_ligands = []

for cid in input_ligands["CID"]:

    ligand_file = (
        STEP21_LIGAND_DIR
        / f"CID_{cid}.pdbqt"
    )

    if not ligand_file.exists():
        missing_ligands.append(
            cid
        )

if missing_ligands:

    raise RuntimeError(
        "Step 21 ligand PDBQT files missing:\n"
        + ", ".join(
            map(
                str,
                missing_ligands
            )
        )
    )


# ============================================================
# WT DOCKING
# ============================================================

wt_results = []
wt_failures = []

for cid in input_ligands["CID"]:

    ligand_pdbqt = (
        STEP21_LIGAND_DIR
        / f"CID_{cid}.pdbqt"
    )

    output_pose = (
        DIR_POSES
        / f"CID_{cid}_WT_docked.pdbqt"
    )

    log_file = (
        DIR_LOGS
        / f"CID_{cid}_WT_vina.log"
    )

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"WT docking CID {cid}"
    )

    try:

        run_vina(
            wt_pdbqt,
            ligand_pdbqt,
            output_pose,
            log_file,
            CENTER,
        )

        scores = parse_vina_scores(
            output_pose
        )

        best_pose_number, best_pose = min(
            enumerate(
                scores,
                start=1,
            ),
            key=lambda item:
            item[1]["score"]
        )

        wt_results.append(
            {
                "CID": cid,
                "WT_Number_of_Poses":
                    len(scores),
                "WT_Best_Pose":
                    best_pose_number,
                "WT_Best_Vina_Score_kcal_mol":
                    best_pose["score"],
                "WT_Best_RMSD_LB":
                    best_pose["rmsd_lb"],
                "WT_Best_RMSD_UB":
                    best_pose["rmsd_ub"],
                "WT_Status":
                    "SUCCESS",
            }
        )

        print(
            f"WT SUCCESS CID {cid}: "
            f"{best_pose['score']:.3f} kcal/mol"
        )

    except Exception as exc:

        wt_failures.append(
            {
                "CID": cid,
                "WT_Status": "FAILED",
                "Error": str(exc),
            }
        )

        print(
            f"WT FAILED CID {cid}: "
            f"{exc}"
        )


# ============================================================
# MERGE WT AND D166V RESULTS
# ============================================================

wt_df = pd.DataFrame(
    wt_results
)

if wt_df.empty:
    raise RuntimeError(
        "No WT compounds completed docking."
    )

comparison_df = (
    input_ligands
    .merge(
        wt_df,
        on="CID",
        how="inner",
    )
)


# ============================================================
# CALCULATE DIFFERENTIAL DOCKING
# ============================================================

comparison_df[
    "D166V_Best_Vina_Score_kcal_mol"
] = comparison_df[
    "Best_Vina_Score_kcal_mol"
].astype(float)

comparison_df[
    "Delta_Vina_D166V_minus_WT_kcal_mol"
] = (
    comparison_df[
        "D166V_Best_Vina_Score_kcal_mol"
    ]
    -
    comparison_df[
        "WT_Best_Vina_Score_kcal_mol"
    ]
)


# Negative delta = D166V more favorable
# Positive delta = WT more favorable

def classify_delta(delta: float) -> str:

    if delta <= -1.0:
        return "D166V_PREFERRED"

    if delta >= 1.0:
        return "WT_PREFERRED"

    return "SIMILAR_DOCKING_SCORE"


comparison_df[
    "Differential_Docking_Class"
] = comparison_df[
    "Delta_Vina_D166V_minus_WT_kcal_mol"
].apply(
    classify_delta
)


# ============================================================
# D166V SCORE + WT SCORE + DELTA RANKING
# ============================================================

comparison_df = comparison_df.sort_values(
    by="Delta_Vina_D166V_minus_WT_kcal_mol",
    ascending=True,
).reset_index(drop=True)

comparison_df.insert(
    0,
    "Differential_Rank",
    np.arange(
        1,
        len(comparison_df) + 1,
    ),
)


# ============================================================
# INCLUDE STEP 20/21 INFORMATION
# ============================================================

step19_small = step19_df[
    [
        c for c in [
            "CID",
            "Drug_Likeness_Class",
            "Chemical_Filter_Score",
            "Step19_Rank",
            "Step19_Screening_Rank",
        ]
        if c in step19_df.columns
    ]
].copy()

if "CID" in step19_small.columns:

    step19_small["CID"] = (
        step19_small["CID"]
        .astype(int)
    )

    comparison_df = comparison_df.merge(
        step19_small,
        on="CID",
        how="left",
        suffixes=("", "_Step19"),
    )


# ============================================================
# SAVE MAIN RESULTS
# ============================================================

comparison_df.to_csv(
    DIR_TABLES
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING.csv",
    index=False,
)


# ============================================================
# SAVE WT FAILURE TABLE
# ============================================================

if wt_failures:

    pd.DataFrame(
        wt_failures
    ).to_csv(
        DIR_TABLES
        / "STEP22_WT_DOCKING_FAILURES.csv",
        index=False,
    )


# ============================================================
# SAVE POSE-LEVEL WT RESULTS
# ============================================================

wt_pose_rows = []

for cid in comparison_df["CID"]:

    pose_file = (
        DIR_POSES
        / f"CID_{cid}_WT_docked.pdbqt"
    )

    scores = parse_vina_scores(
        pose_file
    )

    for pose_number, record in enumerate(
        scores,
        start=1,
    ):

        wt_pose_rows.append(
            {
                "CID": cid,
                "Pose_Number": pose_number,
                "WT_Vina_Score_kcal_mol":
                    record["score"],
                "WT_RMSD_LB":
                    record["rmsd_lb"],
                "WT_RMSD_UB":
                    record["rmsd_ub"],
            }
        )


pd.DataFrame(
    wt_pose_rows
).to_csv(
    DIR_TABLES
    / "STEP22_WT_POSE_LEVEL_RESULTS.csv",
    index=False,
)


# ============================================================
# ADD D166V POSE-LEVEL RESULTS FROM STEP 21
# ============================================================

d166v_pose_rows = []

for cid in comparison_df["CID"]:

    pose_file = (
        STEP21_POSE_DIR
        / f"CID_{cid}_docked.pdbqt"
    )

    if not pose_file.exists():
        continue

    scores = parse_vina_scores(
        pose_file
    )

    for pose_number, record in enumerate(
        scores,
        start=1,
    ):

        d166v_pose_rows.append(
            {
                "CID": cid,
                "Pose_Number": pose_number,
                "D166V_Vina_Score_kcal_mol":
                    record["score"],
                "D166V_RMSD_LB":
                    record["rmsd_lb"],
                "D166V_RMSD_UB":
                    record["rmsd_ub"],
            }
        )


pd.DataFrame(
    d166v_pose_rows
).to_csv(
    DIR_TABLES
    / "STEP22_D166V_POSE_LEVEL_RESULTS.csv",
    index=False,
)


# ============================================================
# QC
# ============================================================

qc = {
    "step": 22,
    "status": (
        "PASS"
        if not wt_failures
        else "PASS_WITH_FAILURES"
    ),
    "vina_executable":
        VINA_BIN,
    "vina_version":
        "1.2.5",
    "wt_receptor":
        str(WT_PDB),
    "d166v_receptor":
        str(D166V_PDB),
    "pocket_id":
        POCKET_ID,
    "chain":
        CHAIN,
    "target_residue":
        f"{CHAIN}:{RESIDUE}",
    "common_center":
        list(CENTER),
    "box_size_A":
        list(BOX_SIZE),
    "wt_ca":
        [
            wt_ca["x"],
            wt_ca["y"],
            wt_ca["z"],
        ],
    "d166v_ca":
        [
            d166v_ca["x"],
            d166v_ca["y"],
            d166v_ca["z"],
        ],
    "wt_d166v_ca_displacement_A":
        center_displacement,
    "exhaustiveness":
        EXHAUSTIVENESS,
    "num_modes":
        NUM_MODES,
    "energy_range":
        ENERGY_RANGE,
    "seed":
        SEED,
    "input_compounds":
        int(len(input_ligands)),
    "successful_wt":
        int(len(wt_df)),
    "failed_wt":
        int(len(wt_failures)),
    "differential_definition":
        "D166V score minus WT score",
    "interpretation":
        (
            "Negative delta indicates a more favorable "
            "D166V Vina score; positive delta indicates "
            "a more favorable WT Vina score."
        ),
    "scientific_limitation":
        (
            "Differential Vina score is a computational "
            "comparison and does not establish experimental "
            "selectivity or binding affinity."
        ),
}

(DIR_QC / "STEP22_QC.json").write_text(
    json.dumps(
        qc,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# INTERPRETATION REPORT
# ============================================================

d166v_preferred = comparison_df[
    comparison_df[
        "Differential_Docking_Class"
    ]
    == "D166V_PREFERRED"
]

wt_preferred = comparison_df[
    comparison_df[
        "Differential_Docking_Class"
    ]
    == "WT_PREFERRED"
]

similar = comparison_df[
    comparison_df[
        "Differential_Docking_Class"
    ]
    == "SIMILAR_DOCKING_SCORE"
]

best_d166v_preference = (
    comparison_df.iloc[0]
)

report = [
    "STEP 22 — WT vs D166V DIFFERENTIAL DOCKING",
    "",
    "Docking engine: AutoDock Vina 1.2.5",
    "Target pocket: Pocket 1",
    "Common docking box: 20 × 20 × 20 Å",
    f"Exhaustiveness: {EXHAUSTIVENESS}",
    f"Number of modes: {NUM_MODES}",
    f"Energy range: {ENERGY_RANGE}",
    f"Seed: {SEED}",
    "",
    f"Input compounds: {len(input_ligands)}",
    f"Successful WT dockings: {len(wt_df)}",
    f"Failed WT dockings: {len(wt_failures)}",
    "",
    "Differential definition:",
    "Delta_Vina = D166V score − WT score",
    "",
    f"D166V-preferred score differences: "
    f"{len(d166v_preferred)}",
    f"WT-preferred score differences: "
    f"{len(wt_preferred)}",
    f"Similar score differences: "
    f"{len(similar)}",
    "",
    "Largest D166V score preference:",
    f"CID: {int(best_d166v_preference['CID'])}",
    (
        "D166V score: "
        f"{best_d166v_preference['D166V_Best_Vina_Score_kcal_mol']:.3f} "
        "kcal/mol"
    ),
    (
        "WT score: "
        f"{best_d166v_preference['WT_Best_Vina_Score_kcal_mol']:.3f} "
        "kcal/mol"
    ),
    (
        "Delta Vina: "
        f"{best_d166v_preference['Delta_Vina_D166V_minus_WT_kcal_mol']:.3f} "
        "kcal/mol"
    ),
    "",
    "Scientific interpretation:",
    (
        "Compounds with negative Delta Vina values receive a "
        "more favorable Vina score in the D166V receptor under "
        "the same docking protocol."
    ),
    (
        "This comparison is intended to identify compounds whose "
        "computed docking behavior changes between WT and D166V."
    ),
    "",
    "Important limitation:",
    (
        "Differential docking does not by itself demonstrate "
        "mutation-specific experimental binding or therapeutic "
        "selectivity. Structural interaction analysis is required "
        "for mechanistic interpretation."
    ),
]

(DIR_QC / "STEP22_INTERPRETATION.txt").write_text(
    "\n".join(report),
    encoding="utf-8",
)


# ============================================================
# MANIFEST
# ============================================================

manifest = [
    f"Project root: {PROJECT_ROOT}",
    f"Step 19 input: {STEP19_CSV}",
    f"Step 16 input: {STEP16_CSV}",
    f"Step 21 input: {STEP21_CSV}",
    f"WT receptor: {WT_PDB}",
    f"D166V receptor: {D166V_PDB}",
    f"Vina executable: {VINA_BIN}",
    "Vina version: 1.2.5",
    f"Pocket ID: {POCKET_ID}",
    f"Target residue: {CHAIN}:{RESIDUE}",
    f"Common box center: {CENTER}",
    f"Box size: {BOX_SIZE}",
    f"Exhaustiveness: {EXHAUSTIVENESS}",
    f"Number of modes: {NUM_MODES}",
    f"Energy range: {ENERGY_RANGE}",
    f"Seed: {SEED}",
    "Delta_Vina = D166V - WT",
]

(DIR_QC / "STEP22_MANIFEST.txt").write_text(
    "\n".join(manifest),
    encoding="utf-8",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("STEP 22 COMPLETE")
print("=" * 78)

print(
    f"Input compounds       : "
    f"{len(input_ligands)}"
)

print(
    f"Successful WT dockings: "
    f"{len(wt_df)}"
)

print(
    f"Failed WT dockings    : "
    f"{len(wt_failures)}"
)

print(
    f"D166V-preferred       : "
    f"{len(d166v_preferred)}"
)

print(
    f"WT-preferred          : "
    f"{len(wt_preferred)}"
)

print(
    f"Similar               : "
    f"{len(similar)}"
)

print("\nDifferential docking ranking:")

print(
    comparison_df[
        [
            "Differential_Rank",
            "CID",
            "D166V_Best_Vina_Score_kcal_mol",
            "WT_Best_Vina_Score_kcal_mol",
            "Delta_Vina_D166V_minus_WT_kcal_mol",
            "Differential_Docking_Class",
        ]
    ].to_string(
        index=False
    )
)

print("\nResults:")
print(OUT_ROOT)

print("\nQC:")
print(DIR_QC)

print("=" * 78)