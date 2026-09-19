#!/usr/bin/env python3

"""
STEP 21 — VIRTUAL SCREENING / MOLECULAR DOCKING

Real AutoDock Vina 1.2.5 docking against:
    PNPLA3 D166V
    D166V Pocket 1

Inputs:
    Step 19 screening-ready library
    Step 16 Pocket 1
    Step 10 D166V structure

Pipeline:
    Step 19
       ↓
    RDKit 3D
       ↓
    Meeko ligand PDBQT
       ↓
    Meeko receptor PDBQT
       ↓
    AutoDock Vina 1.2.5
       ↓
    Docking poses + Vina scores

Important:
    Vina scores are computational scoring-function outputs.
    They are NOT experimental affinities or proof of inhibition.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


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

STEP20_CSV = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP20_PHARMACOPHORE_SHAPE_SCREENING"
    / "tables"
    / "STEP20_PHARMACOPHORE_SHAPE_SCREENING.csv"
)

D166V_PDB = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
    / "structures"
    / "D166V"
    / "PNPLA3_D166V_mutant.pdb"
)

OUT_ROOT = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
)

DIR_RECEPTOR = OUT_ROOT / "receptor"
DIR_LIGANDS = OUT_ROOT / "ligands"
DIR_POSES = OUT_ROOT / "poses"
DIR_LOGS = OUT_ROOT / "logs"
DIR_TABLES = OUT_ROOT / "tables"
DIR_FIGURES = OUT_ROOT / "figures"
DIR_QC = OUT_ROOT / "qc"

for directory in (
    DIR_RECEPTOR,
    DIR_LIGANDS,
    DIR_POSES,
    DIR_LOGS,
    DIR_TABLES,
    DIR_FIGURES,
    DIR_QC,
):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# FIXED SCIENTIFIC PARAMETERS
# ============================================================

EXPECTED_VARIANT = "D166V"
EXPECTED_POCKET_ID = 1
EXPECTED_CHAIN = "A"
EXPECTED_RESIDUE = 166

BOX_SIZE = (20.0, 20.0, 20.0)

EXHAUSTIVENESS = 16
NUM_MODES = 10
ENERGY_RANGE = 4.0
SEED = 20260910


# ============================================================
# LOCATE VINA
# ============================================================

VINA_BIN = shutil.which("vina")

if VINA_BIN is None:
    raise RuntimeError(
        "AutoDock Vina was not found in PATH.\n"
        "Run:\n"
        "    which vina\n"
        "inside the WSL docking environment."
    )

print("=" * 78)
print("STEP 21 — VIRTUAL SCREENING / MOLECULAR DOCKING")
print("=" * 78)
print(f"Project root : {PROJECT_ROOT}")
print(f"Vina         : {VINA_BIN}")
print(f"Step 19      : {STEP19_CSV}")
print(f"Step 16      : {STEP16_CSV}")
print(f"Step 20      : {STEP20_CSV}")
print(f"D166V PDB    : {D166V_PDB}")


# ============================================================
# GENERAL HELPERS
# ============================================================

def run_command(
    command: list[str],
    log_file: Path | None = None,
) -> subprocess.CompletedProcess:

    print("\n$", " ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    print(result.stdout)

    if log_file is not None:
        log_file.write_text(
            result.stdout,
            encoding="utf-8",
        )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code "
            f"{result.returncode}:\n"
            f"{result.stdout}"
        )

    return result


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
        f"Could not find "
        f"{chain_id}:{residue_number} CA in "
        f"{pdb_file}"
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
            "No receptor ATOM records retained."
        )

    return count


# ============================================================
# LIGAND GENERATION
# ============================================================

def build_3d_ligand(
    smiles: str,
    cid: int,
    output_sdf: Path,
) -> Chem.Mol:

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise RuntimeError(
            f"RDKit failed to parse CID {cid}."
        )

    mol.SetProp(
        "_Name",
        f"CID_{cid}",
    )

    mol.SetProp(
        "CID",
        str(cid),
    )

    # Keep explicit hydrogens.
    mol = Chem.AddHs(mol)

    parameters = AllChem.ETKDGv3()

    parameters.randomSeed = (
        SEED + (cid % 100000)
    )

    parameters.useRandomCoords = False

    status = AllChem.EmbedMolecule(
        mol,
        parameters,
    )

    if status != 0:

        parameters.useRandomCoords = True

        status = AllChem.EmbedMolecule(
            mol,
            parameters,
        )

    if status != 0:
        raise RuntimeError(
            f"3D embedding failed for CID {cid}."
        )

    try:
        AllChem.UFFOptimizeMolecule(
            mol,
            maxIters=500,
        )
    except Exception:
        pass

    # DO NOT RemoveHs().
    writer = Chem.SDWriter(
        str(output_sdf)
    )

    writer.write(mol)
    writer.close()

    return mol


# ============================================================
# RECEPTOR PREPARATION
# ============================================================

def prepare_receptor(
    receptor_pdb: Path,
    receptor_pdbqt: Path,
    center: tuple[float, float, float],
):

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

    run_command(
        command,
        DIR_LOGS / "receptor_preparation.log",
    )


# ============================================================
# LIGAND PREPARATION
# ============================================================

def prepare_ligand(
    sdf_file: Path,
    pdbqt_file: Path,
    cid: int,
):

    command = [
        "mk_prepare_ligand.py",
        "-i",
        str(sdf_file),
        "-o",
        str(pdbqt_file),
    ]

    run_command(
        command,
        DIR_LOGS / f"CID_{cid}_meeko.log",
    )


# ============================================================
# VINA EXECUTION
# ============================================================

def run_vina(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    output_pose: Path,
    log_file: Path,
    center: tuple[float, float, float],
):

    # IMPORTANT:
    # Vina 1.2.5 does NOT accept --log.
    # We capture stdout/stderr ourselves instead.

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

    print("\n$", " ".join(command))

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    print(result.stdout)

    # Save the complete Vina console output.
    log_file.write_text(
        result.stdout,
        encoding="utf-8",
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"Vina failed with exit code "
            f"{result.returncode}:\n"
            f"{result.stdout}"
        )

    if not output_pose.exists():

        raise RuntimeError(
            f"Vina completed but did not create:\n"
            f"{output_pose}"
        )


# ============================================================
# SCORE PARSER
# ============================================================

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
                        "score_kcal_mol":
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


def score_classification(
    score: float,
) -> str:

    if score <= -8.0:
        return "STRONGER_SCREENING_SCORE"

    if score <= -6.0:
        return "INTERMEDIATE_SCREENING_SCORE"

    return "WEAKER_SCREENING_SCORE"


# ============================================================
# INPUT CHECKS
# ============================================================

for file in (
    STEP19_CSV,
    STEP16_CSV,
    D166V_PDB,
):

    if not file.exists():

        raise FileNotFoundError(
            f"Missing required input:\n{file}"
        )


# ============================================================
# STEP 16 POCKET VALIDATION
# ============================================================

pocket_df = pd.read_csv(
    STEP16_CSV
)

required_pocket_columns = [
    "Priority_Rank",
    "D166V_Pocket_ID",
    "WT_Pocket_ID",
    "Match_Status",
    "Druggability",
    "Pocket_Score",
    "Volume_A3",
    "D166_Distance_A",
    "D166_Directly_Lining",
    "Prioritization_Score",
]

missing = [
    col
    for col in required_pocket_columns
    if col not in pocket_df.columns
]

if missing:

    raise RuntimeError(
        "Missing Step 16 columns:\n"
        + "\n".join(missing)
    )


pocket_match = pocket_df[
    pocket_df["D166V_Pocket_ID"]
    == EXPECTED_POCKET_ID
].copy()

if pocket_match.empty:

    raise RuntimeError(
        "D166V Pocket 1 not found."
    )

pocket_row = (
    pocket_match
    .sort_values("Priority_Rank")
    .iloc[0]
)

if str(pocket_row["Match_Status"]) != "Matched":

    raise RuntimeError(
        "Pocket 1 is not marked as Matched."
    )

print("\nPocket validation:")
print(
    f"  D166V Pocket ID    : "
    f"{pocket_row['D166V_Pocket_ID']}"
)
print(
    f"  WT Pocket ID       : "
    f"{pocket_row['WT_Pocket_ID']}"
)
print(
    f"  Match Status       : "
    f"{pocket_row['Match_Status']}"
)
print(
    f"  Druggability       : "
    f"{pocket_row['Druggability']}"
)
print(
    f"  Pocket score       : "
    f"{pocket_row['Pocket_Score']}"
)
print(
    f"  Volume (A^3)       : "
    f"{pocket_row['Volume_A3']}"
)
print(
    f"  D166 distance (A)  : "
    f"{pocket_row['D166_Distance_A']}"
)
print(
    f"  D166 directly lining: "
    f"{pocket_row['D166_Directly_Lining']}"
)
print(
    f"  Prioritization     : "
    f"{pocket_row['Prioritization_Score']}"
)


# ============================================================
# D166V COORDINATE / BOX
# ============================================================

ca = find_ca(
    D166V_PDB,
    EXPECTED_CHAIN,
    EXPECTED_RESIDUE,
)

BOX_CENTER = (
    ca["x"],
    ca["y"],
    ca["z"],
)

print("\nDocking center:")
print(
    f"  {ca['chain']}:{ca['residue']} "
    f"{ca['resname']} CA = "
    f"({ca['x']:.4f}, "
    f"{ca['y']:.4f}, "
    f"{ca['z']:.4f})"
)


# ============================================================
# CLEAN RECEPTOR
# ============================================================

clean_receptor = (
    DIR_RECEPTOR
    / "PNPLA3_D166V_chainA_protein.pdb"
)

atom_count = make_clean_receptor(
    D166V_PDB,
    clean_receptor,
    EXPECTED_CHAIN,
)

print(
    f"\nReceptor ATOM records retained: "
    f"{atom_count}"
)


# ============================================================
# RECEPTOR PDBQT
# ============================================================

receptor_pdbqt = (
    DIR_RECEPTOR
    / "PNPLA3_D166V_chainA.pdbqt"
)

prepare_receptor(
    clean_receptor,
    receptor_pdbqt,
    BOX_CENTER,
)


# ============================================================
# STEP 19 COMPOUND LIBRARY
# ============================================================

ligand_df = pd.read_csv(
    STEP19_CSV
)

required_ligand_columns = [
    "CID",
    "IsomericSMILES",
    "IUPACName",
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
]

missing = [
    col
    for col in required_ligand_columns
    if col not in ligand_df.columns
]

if missing:

    raise RuntimeError(
        "Missing Step 19 columns:\n"
        + "\n".join(missing)
    )


if "Retain_for_Step20" in ligand_df.columns:

    retained = ligand_df[
        ligand_df["Retain_for_Step20"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
    ].copy()

else:

    retained = ligand_df.copy()


retained["CID"] = (
    retained["CID"]
    .astype(int)
)


print(
    f"\nCompounds selected for docking: "
    f"{len(retained)}"
)


# ============================================================
# MAIN DOCKING
# ============================================================

successful = []
failed = []

for _, row in retained.iterrows():

    cid = int(row["CID"])

    smiles = str(
        row["IsomericSMILES"]
    )

    ligand_name = str(
        row["IUPACName"]
    )

    print(
        "\n"
        + "-" * 78
    )

    print(
        f"Docking CID {cid}"
    )

    print(
        ligand_name
    )

    sdf_file = (
        DIR_LIGANDS
        / f"CID_{cid}.sdf"
    )

    pdbqt_file = (
        DIR_LIGANDS
        / f"CID_{cid}.pdbqt"
    )

    pose_file = (
        DIR_POSES
        / f"CID_{cid}_docked.pdbqt"
    )

    vina_log = (
        DIR_LOGS
        / f"CID_{cid}_vina.log"
    )

    try:

        # ----------------------------------------------------
        # RDKit
        # ----------------------------------------------------

        build_3d_ligand(
            smiles,
            cid,
            sdf_file,
        )

        # ----------------------------------------------------
        # Meeko
        # ----------------------------------------------------

        prepare_ligand(
            sdf_file,
            pdbqt_file,
            cid,
        )

        # ----------------------------------------------------
        # Vina
        # ----------------------------------------------------

        run_vina(
            receptor_pdbqt,
            pdbqt_file,
            pose_file,
            vina_log,
            BOX_CENTER,
        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        scores = parse_vina_scores(
            pose_file
        )

        best_pose_number, best_pose = min(
            enumerate(
                scores,
                start=1,
            ),
            key=lambda x:
            x[1]["score_kcal_mol"],
        )

        best_score = (
            best_pose[
                "score_kcal_mol"
            ]
        )

        successful.append(
            {
                "CID": cid,
                "IUPACName": ligand_name,
                "MolecularWeight":
                    row["MolecularWeight"],
                "XLogP":
                    row["XLogP"],
                "TPSA":
                    row["TPSA"],
                "HBondDonorCount":
                    row["HBondDonorCount"],
                "HBondAcceptorCount":
                    row["HBondAcceptorCount"],
                "RotatableBondCount":
                    row["RotatableBondCount"],
                "Number_of_Poses":
                    len(scores),
                "Best_Pose_Number":
                    best_pose_number,
                "Best_Vina_Score_kcal_mol":
                    best_score,
                "Best_Pose_RMSD_LB":
                    best_pose["rmsd_lb"],
                "Best_Pose_RMSD_UB":
                    best_pose["rmsd_ub"],
                "Docking_Score_Class":
                    score_classification(
                        best_score
                    ),
                "Docking_Status":
                    "SUCCESS",
                "Receptor":
                    EXPECTED_VARIANT,
                "Pocket_ID":
                    EXPECTED_POCKET_ID,
                "Box_Center_X":
                    BOX_CENTER[0],
                "Box_Center_Y":
                    BOX_CENTER[1],
                "Box_Center_Z":
                    BOX_CENTER[2],
                "Box_Size_X":
                    BOX_SIZE[0],
                "Box_Size_Y":
                    BOX_SIZE[1],
                "Box_Size_Z":
                    BOX_SIZE[2],
                "Exhaustiveness":
                    EXHAUSTIVENESS,
                "Num_Modes":
                    NUM_MODES,
                "Energy_Range":
                    ENERGY_RANGE,
                "Seed":
                    SEED,
            }
        )

        print(
            f"SUCCESS CID {cid}"
        )

        print(
            f"Best Vina score: "
            f"{best_score:.3f} kcal/mol"
        )

        print(
            f"Best pose: "
            f"{best_pose_number}"
        )

    except Exception as exc:

        failed.append(
            {
                "CID": cid,
                "IUPACName": ligand_name,
                "Docking_Status": "FAILED",
                "Error": str(exc),
            }
        )

        print(
            f"FAILED CID {cid}: {exc}"
        )


# ============================================================
# FAILURE OUTPUT
# ============================================================

if failed:

    pd.DataFrame(
        failed
    ).to_csv(
        DIR_TABLES
        / "STEP21_DOCKING_FAILURES.csv",
        index=False,
    )


# ============================================================
# REQUIRE AT LEAST ONE SUCCESS
# ============================================================

if not successful:

    raise RuntimeError(
        "No compounds completed docking successfully.\n"
        "See STEP21_DOCKING_FAILURES.csv and the individual "
        "Vina/Meeko log files."
    )


# ============================================================
# RESULT DATAFRAME
# ============================================================

results_df = pd.DataFrame(
    successful
)

results_df = (
    results_df
    .sort_values(
        "Best_Vina_Score_kcal_mol",
        ascending=True,
    )
    .reset_index(drop=True)
)

results_df.insert(
    0,
    "Docking_Rank",
    np.arange(
        1,
        len(results_df) + 1,
    ),
)


# ============================================================
# MERGE STEP 20 SCREENING SCORE
# ============================================================

if STEP20_CSV.exists():

    step20_df = pd.read_csv(
        STEP20_CSV
    )

    if "CID" in step20_df.columns:

        possible_columns = [
            "Combined_Screening_Score",
            "Screening_Score",
            "Pharmacophore_Shape_Score",
            "Step20_Score",
        ]

        score_column = next(
            (
                col
                for col in possible_columns
                if col in step20_df.columns
            ),
            None,
        )

        if score_column:

            merge_df = (
                step20_df[
                    [
                        "CID",
                        score_column,
                    ]
                ]
                .drop_duplicates(
                    subset=["CID"]
                )
                .copy()
            )

            merge_df["CID"] = (
                merge_df["CID"]
                .astype(int)
            )

            merge_df = merge_df.rename(
                columns={
                    score_column:
                    "Step20_Screening_Score"
                }
            )

            results_df = results_df.merge(
                merge_df,
                on="CID",
                how="left",
            )


# ============================================================
# SAVE MAIN RESULTS
# ============================================================

results_df.to_csv(
    DIR_TABLES
    / "STEP21_BEST_DOCKING_RESULTS.csv",
    index=False,
)

results_df.to_csv(
    DIR_TABLES
    / "STEP21_COMPLETE_DOCKING_SUMMARY.csv",
    index=False,
)


# ============================================================
# POSE LEVEL RESULTS
# ============================================================

pose_records = []

for _, row in results_df.iterrows():

    cid = int(
        row["CID"]
    )

    pose_file = (
        DIR_POSES
        / f"CID_{cid}_docked.pdbqt"
    )

    scores = parse_vina_scores(
        pose_file
    )

    for pose_number, pose in enumerate(
        scores,
        start=1,
    ):

        pose_records.append(
            {
                "CID": cid,
                "Pose_Number": pose_number,
                "Vina_Score_kcal_mol":
                    pose["score_kcal_mol"],
                "RMSD_LB":
                    pose["rmsd_lb"],
                "RMSD_UB":
                    pose["rmsd_ub"],
                "Receptor":
                    EXPECTED_VARIANT,
                "Pocket_ID":
                    EXPECTED_POCKET_ID,
            }
        )


pose_df = pd.DataFrame(
    pose_records
)

pose_df.to_csv(
    DIR_TABLES
    / "STEP21_POSE_LEVEL_RESULTS.csv",
    index=False,
)


# ============================================================
# QC
# ============================================================

qc = {
    "step": 21,
    "status": (
        "PASS"
        if not failed
        else "PASS_WITH_FAILURES"
    ),
    "vina_executable":
        VINA_BIN,
    "vina_version":
        "1.2.5",
    "receptor":
        EXPECTED_VARIANT,
    "receptor_input":
        str(D166V_PDB),
    "chain":
        EXPECTED_CHAIN,
    "target_residue":
        f"{EXPECTED_CHAIN}:{EXPECTED_RESIDUE}",
    "target_residue_name":
        ca["resname"],
    "pocket_id":
        EXPECTED_POCKET_ID,
    "pocket_match_status":
        str(pocket_row["Match_Status"]),
    "pocket_druggability":
        float(
            pocket_row["Druggability"]
        ),
    "pocket_score":
        float(
            pocket_row["Pocket_Score"]
        ),
    "pocket_volume_A3":
        float(
            pocket_row["Volume_A3"]
        ),
    "pocket_prioritization_score":
        float(
            pocket_row[
                "Prioritization_Score"
            ]
        ),
    "box_center":
        list(BOX_CENTER),
    "box_size_A":
        list(BOX_SIZE),
    "exhaustiveness":
        EXHAUSTIVENESS,
    "num_modes":
        NUM_MODES,
    "energy_range":
        ENERGY_RANGE,
    "seed":
        SEED,
    "input_compounds":
        int(len(retained)),
    "successful_dockings":
        int(len(successful)),
    "failed_dockings":
        int(len(failed)),
}

(DIR_QC / "STEP21_QC.json").write_text(
    json.dumps(
        qc,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# FIGURE 21-01
# ============================================================

plot_df = results_df.sort_values(
    "Best_Vina_Score_kcal_mol",
    ascending=True,
)

plt.figure(
    figsize=(11, 6)
)

plt.bar(
    plot_df["CID"].astype(str),
    plot_df[
        "Best_Vina_Score_kcal_mol"
    ],
)

plt.axhline(
    -8.0,
    linestyle="--",
    linewidth=1,
)

plt.axhline(
    -6.0,
    linestyle="--",
    linewidth=1,
)

plt.xlabel(
    "PubChem CID"
)

plt.ylabel(
    "Best Vina score (kcal/mol)"
)

plt.title(
    "Step 21 — D166V Pocket 1 Docking Scores"
)

plt.xticks(
    rotation=45,
    ha="right",
)

plt.tight_layout()

plt.savefig(
    DIR_FIGURES
    / "Figure_21_01_Docking_Scores.png",
    dpi=300,
)

plt.close()


# ============================================================
# FIGURE 21-02
# ============================================================

if (
    "Step20_Screening_Score"
    in results_df.columns
):

    comparison_df = results_df.dropna(
        subset=[
            "Step20_Screening_Score"
        ]
    )

    if len(comparison_df) >= 2:

        plt.figure(
            figsize=(8, 6)
        )

        plt.scatter(
            comparison_df[
                "Step20_Screening_Score"
            ],
            comparison_df[
                "Best_Vina_Score_kcal_mol"
            ],
        )

        for _, row in comparison_df.iterrows():

            plt.annotate(
                str(int(row["CID"])),
                (
                    row[
                        "Step20_Screening_Score"
                    ],
                    row[
                        "Best_Vina_Score_kcal_mol"
                    ],
                ),
            )

        plt.xlabel(
            "Step 20 screening score"
        )

        plt.ylabel(
            "Best Vina score (kcal/mol)"
        )

        plt.title(
            "Step 20 Screening vs Step 21 Docking"
        )

        plt.tight_layout()

        plt.savefig(
            DIR_FIGURES
            / "Figure_21_02_Step20_vs_Docking.png",
            dpi=300,
        )

        plt.close()


# ============================================================
# FIGURE 21-03
# ============================================================

top_df = results_df.head(
    min(5, len(results_df))
)

plt.figure(
    figsize=(9, 6)
)

plt.barh(
    top_df["CID"].astype(str)[::-1],
    top_df[
        "Best_Vina_Score_kcal_mol"
    ][::-1],
)

plt.xlabel(
    "Best Vina score (kcal/mol)"
)

plt.ylabel(
    "PubChem CID"
)

plt.title(
    "Top Step 21 Docking Candidates"
)

plt.tight_layout()

plt.savefig(
    DIR_FIGURES
    / "Figure_21_03_Top_Docking_Candidates.png",
    dpi=300,
)

plt.close()


# ============================================================
# INTERPRETATION
# ============================================================

best = results_df.iloc[0]

report = [
    "STEP 21 — VIRTUAL SCREENING / MOLECULAR DOCKING",
    "",
    "Docking engine: AutoDock Vina 1.2.5",
    "Receptor: PNPLA3 D166V",
    "Target: D166V Pocket 1",
    "",
    (
        "Docking center: "
        f"{BOX_CENTER[0]:.4f}, "
        f"{BOX_CENTER[1]:.4f}, "
        f"{BOX_CENTER[2]:.4f} Å"
    ),
    (
        "Docking box: "
        f"{BOX_SIZE[0]:.1f} × "
        f"{BOX_SIZE[1]:.1f} × "
        f"{BOX_SIZE[2]:.1f} Å"
    ),
    f"Exhaustiveness: {EXHAUSTIVENESS}",
    f"Number of modes: {NUM_MODES}",
    f"Energy range: {ENERGY_RANGE}",
    f"Seed: {SEED}",
    "",
    f"Input compounds: {len(retained)}",
    f"Successful dockings: {len(successful)}",
    f"Failed dockings: {len(failed)}",
    "",
    "Top docking candidate:",
    f"CID: {int(best['CID'])}",
    f"IUPAC name: {best['IUPACName']}",
    (
        "Best Vina score: "
        f"{best['Best_Vina_Score_kcal_mol']:.3f} kcal/mol"
    ),
    (
        "Score category: "
        f"{best['Docking_Score_Class']}"
    ),
    "",
    "Scientific interpretation:",
    (
        "The docking results provide computational structural "
        "hypotheses for ligand placement within the predefined "
        "D166V Pocket 1."
    ),
    (
        "More negative Vina values indicate more favorable "
        "AutoDock Vina scoring-function results under the "
        "same computational protocol."
    ),
    "",
    "Important limitation:",
    (
        "Vina docking scores are not experimental binding "
        "affinities and do not establish biochemical inhibition, "
        "therapeutic efficacy, or D166V selectivity."
    ),
    (
        "WT-vs-D166V differential docking and subsequent "
        "interaction analysis are required for mutation-specific "
        "interpretation."
    ),
]

(DIR_QC / "STEP21_INTERPRETATION.txt").write_text(
    "\n".join(report),
    encoding="utf-8",
)


# ============================================================
# MANIFEST
# ============================================================

manifest = [
    f"Step 19 input: {STEP19_CSV}",
    f"Step 16 input: {STEP16_CSV}",
    f"Step 20 input: {STEP20_CSV}",
    f"Receptor input: {D166V_PDB}",
    f"Vina executable: {VINA_BIN}",
    "Vina version: 1.2.5",
    "Receptor: PNPLA3 D166V",
    "Chain: A",
    "Target residue: A:166",
    "Target: D166V Pocket 1",
    f"Box center: {BOX_CENTER}",
    f"Box size: {BOX_SIZE}",
    f"Exhaustiveness: {EXHAUSTIVENESS}",
    f"Number of modes: {NUM_MODES}",
    f"Energy range: {ENERGY_RANGE}",
    f"Seed: {SEED}",
    f"Input compounds: {len(retained)}",
    f"Successful: {len(successful)}",
    f"Failed: {len(failed)}",
]

(DIR_QC / "STEP21_MANIFEST.txt").write_text(
    "\n".join(manifest),
    encoding="utf-8",
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("STEP 21 COMPLETE")
print("=" * 78)

print(
    f"Input compounds     : "
    f"{len(retained)}"
)

print(
    f"Successful dockings : "
    f"{len(successful)}"
)

print(
    f"Failed dockings     : "
    f"{len(failed)}"
)

print(
    f"Best CID             : "
    f"{int(best['CID'])}"
)

print(
    f"Best Vina score      : "
    f"{best['Best_Vina_Score_kcal_mol']:.3f} kcal/mol"
)

print("\nTop docking candidates:")

print(
    results_df[
        [
            "Docking_Rank",
            "CID",
            "Best_Vina_Score_kcal_mol",
            "Docking_Score_Class",
        ]
    ].head(10).to_string(
        index=False
    )
)

print("\nResults:")
print(OUT_ROOT)

print("\nQC:")
print(DIR_QC)

print("\nFigures:")
print(DIR_FIGURES)

print("=" * 78)