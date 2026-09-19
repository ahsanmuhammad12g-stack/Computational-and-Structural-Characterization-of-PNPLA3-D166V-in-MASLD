# =============================================================================
# STEP 20 — PHARMACOPHORE / SHAPE SCREENING
# =============================================================================
# Project: MASLD–PNPLA3 Computational Variant Prioritization
#
# Purpose:
#   1. Read the chemically filtered Step 19 library.
#   2. Read the actual D166V structure from Step 10.
#   3. Define the prioritized Pocket 1 using the validated Step 16/17
#      D166-centered pocket definition.
#   4. Generate 3D ligand conformers locally using RDKit.
#   5. Identify pocket chemical interaction environments.
#   6. Compare ligand pharmacophore features with the pocket environment.
#   7. Evaluate coarse ligand size/shape compatibility with Pocket 1.
#   8. Produce a ranked library for subsequent structure-based screening.
#
# IMPORTANT:
#   This is NOT docking.
#   This does NOT calculate binding affinity.
#   This does NOT establish PNPLA3 inhibition.
#   This does NOT establish D166V selectivity.
#
#   Step 20 is a pre-docking pharmacophore/shape prioritization step.
# =============================================================================

from pathlib import Path
from collections import defaultdict
import math
import json
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# =============================================================================
# DEPENDENCY CHECK
# =============================================================================

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
except ImportError:
    raise ImportError(
        "\nRDKit is required for Step 20 but is not installed in the active "
        "environment.\n\n"
        "Install RDKit in the current .venv before running this script."
    )

try:
    from Bio.PDB import PDBParser
except ImportError:
    raise ImportError(
        "\nBiopython is required for Step 20 but is not installed in the "
        "active environment.\n\n"
        "Install Biopython in the current .venv before running this script."
    )


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

STEP19_LIBRARY = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP19_DRUG_LIKENESS_FILTERING"
    / "tables"
    / "STEP19_SCREENING_READY_LIBRARY.csv"
)

STEP16_POCKET_TABLE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
    / "tables"
    / "Table_16_04_D166V_Pocket_Prioritization.csv"
)

D166V_PDB = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
    / "structures"
    / "D166V"
    / "PNPLA3_D166V_mutant.pdb"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP20_PHARMACOPHORE_SHAPE_SCREENING"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
FIGURE_DIR = OUTPUT_ROOT / "figures"
QC_DIR = OUTPUT_ROOT / "QC"
REPORT_DIR = OUTPUT_ROOT / "reports"
CONFORMER_DIR = OUTPUT_ROOT / "ligand_conformers"

for directory in [
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
    REPORT_DIR,
    CONFORMER_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. PARAMETERS
# =============================================================================

POCKET_RADIUS = 10.0
D166_RESIDUE = 166

# Minimum distance used when selecting protein atoms around the ligand.
# This prevents excessive overlap between the ligand and protein feature point.
MIN_FEATURE_DISTANCE = 1.5

# Number of conformers generated per compound.
N_CONFORMERS = 20

# Maximum number of attempts for conformer generation.
MAX_CONFORMER_ATTEMPTS = 50

# Shape size tolerance.
#
# A ligand whose molecular volume is extremely small relative to the pocket
# is unlikely to occupy the complete pocket efficiently.
#
# This is deliberately a coarse pre-docking filter and is NOT a binding score.
MIN_VOLUME_RATIO = 0.05
MAX_VOLUME_RATIO = 0.90


# =============================================================================
# 3. OUTPUT HEADER
# =============================================================================

print("=" * 78)
print("STEP 20 — PHARMACOPHORE / SHAPE SCREENING")
print("=" * 78)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Step 19     : {STEP19_LIBRARY}")
print(f"Step 16     : {STEP16_POCKET_TABLE}")
print(f"D166V PDB   : {D166V_PDB}")


# =============================================================================
# 4. VALIDATE INPUTS
# =============================================================================

for required_file in [
    STEP19_LIBRARY,
    STEP16_POCKET_TABLE,
    D166V_PDB,
]:
    if not required_file.exists():
        raise FileNotFoundError(
            f"\nRequired file not found:\n{required_file}"
        )


# =============================================================================
# 5. LOAD STEP 19 LIBRARY
# =============================================================================

df = pd.read_csv(STEP19_LIBRARY)

print("\n" + "=" * 78)
print("STEP 19 LIBRARY")
print("=" * 78)

print(f"Compounds available : {len(df)}")

if len(df) == 0:
    raise ValueError("Step 19 screening-ready library is empty.")

# Identify structure column.
smiles_candidates = [
    "IsomericSMILES",
    "CanonicalSMILES",
    "SMILES",
]

smiles_column = None

for candidate in smiles_candidates:
    if candidate in df.columns:
        smiles_column = candidate
        break

if smiles_column is None:
    raise ValueError(
        "\nNo SMILES column found in Step 19 library.\n"
        f"Available columns:\n{list(df.columns)}"
    )

print(f"SMILES column       : {smiles_column}")


# =============================================================================
# 6. LOAD STEP 16 POCKET DEFINITION
# =============================================================================

pocket_df = pd.read_csv(STEP16_POCKET_TABLE)

required_pocket_columns = [
    "D166V_Pocket_ID",
    "WT_Pocket_ID",
    "Match_Status",
    "Druggability",
    "Volume_A3",
    "D166_Distance_A",
    "D166_Directly_Lining",
    "Prioritization_Score",
]

missing_pocket = [
    c for c in required_pocket_columns
    if c not in pocket_df.columns
]

if missing_pocket:
    raise ValueError(
        "\nStep 16 pocket table is missing required columns:\n"
        + "\n".join(missing_pocket)
    )

# Pocket 1 is the validated top-priority pocket.
pocket_row = pocket_df[
    pocket_df["D166V_Pocket_ID"].astype(str) == "1"
]

if pocket_row.empty:
    raise ValueError(
        "Validated Step 16 D166V Pocket 1 was not found."
    )

pocket_row = pocket_row.iloc[0]

pocket_id = int(pocket_row["D166V_Pocket_ID"])
pocket_volume = float(pocket_row["Volume_A3"])
pocket_druggability = float(pocket_row["Druggability"])
pocket_priority = float(pocket_row["Prioritization_Score"])

print("\nValidated Pocket 1:")
print(f"  Pocket ID          : {pocket_id}")
print(f"  Druggability       : {pocket_druggability}")
print(f"  Volume             : {pocket_volume:.3f} Å³")
print(f"  D166 distance      : {pocket_row['D166_Distance_A']}")
print(f"  D166 directly line : {pocket_row['D166_Directly_Lining']}")
print(f"  Prioritization     : {pocket_priority:.6f}")


# =============================================================================
# 7. PARSE D166V STRUCTURE
# =============================================================================

print("\n" + "=" * 78)
print("D166V POCKET ENVIRONMENT")
print("=" * 78)

parser = PDBParser(QUIET=True)
structure = parser.get_structure("PNPLA3_D166V", str(D166V_PDB))

# Prefer chain A, as established in Step 17.
model = next(structure.get_models())

if "A" not in model:
    available_chains = [chain.id for chain in model]
    raise ValueError(
        f"Chain A not found in D166V structure. "
        f"Available chains: {available_chains}"
    )

chain = model["A"]

if D166_RESIDUE not in [res.id[1] for res in chain]:
    raise ValueError(
        "Residue 166 was not found in D166V chain A."
    )

residue_166 = None

for residue in chain:
    if residue.id[1] == D166_RESIDUE:
        residue_166 = residue
        break

if residue_166 is None:
    raise ValueError("Unable to identify residue 166.")

print(
    f"Residue 166 in structure : "
    f"{residue_166.get_resname()}166"
)


# =============================================================================
# 8. IDENTIFY POCKET RESIDUES
# =============================================================================

# Use the same structural definition as Step 17:
# residues having at least one C-alpha atom within 10 Å of residue 166 CA.

ca166 = None

for atom in residue_166:
    if atom.get_name() == "CA":
        ca166 = atom
        break

if ca166 is None:
    raise ValueError("CA atom for residue 166 was not found.")

ca166_coord = np.asarray(ca166.coord, dtype=float)

pocket_residues = []

for residue in chain:

    if residue.id[0] != " ":
        continue

    ca = residue["CA"] if "CA" in residue else None

    if ca is None:
        continue

    distance = float(
        np.linalg.norm(
            np.asarray(ca.coord, dtype=float) - ca166_coord
        )
    )

    if distance <= POCKET_RADIUS:
        pocket_residues.append(
            (residue, distance)
        )

print(f"Pocket residues within {POCKET_RADIUS:.1f} Å : "
      f"{len(pocket_residues)}")


# =============================================================================
# 9. IDENTIFY PROTEIN CHEMICAL FEATURES
# =============================================================================
# Feature assignment is intentionally conservative.
#
# DONOR:
#   N-H capable atoms
#
# ACCEPTOR:
#   O/N/S atoms capable of accepting H bonds
#
# AROMATIC:
#   aromatic ring atoms
#
# HYDROPHOBIC:
#   carbon/sulfur-rich side-chain environment
#
# These are pocket-environment descriptors, not experimentally validated
# pharmacophore points.
# =============================================================================

AROMATIC_RESIDUES = {
    "PHE",
    "TYR",
    "TRP",
    "HIS",
}

HYDROPHOBIC_RESIDUES = {
    "ALA",
    "VAL",
    "LEU",
    "ILE",
    "MET",
    "PHE",
    "TRP",
    "PRO",
}

DONOR_RESIDUES = {
    "ARG",
    "LYS",
    "HIS",
    "SER",
    "THR",
    "TYR",
    "ASN",
    "GLN",
    "TRP",
    "CYS",
}

ACCEPTOR_RESIDUES = {
    "ASP",
    "GLU",
    "ASN",
    "GLN",
    "HIS",
    "SER",
    "THR",
    "TYR",
    "CYS",
}


def residue_sidechain_atoms(residue):
    """Return non-backbone atoms."""
    atoms = []

    for atom in residue:
        if atom.get_name() not in {"N", "CA", "C", "O"}:
            atoms.append(atom)

    return atoms


protein_features = []

for residue, residue_distance in pocket_residues:

    resname = residue.get_resname().strip()

    atoms = residue_sidechain_atoms(residue)

    if not atoms:
        continue

    # -------------------------------------------------------------------------
    # ACCEPTOR FEATURES
    # -------------------------------------------------------------------------
    if resname in ACCEPTOR_RESIDUES:

        for atom in atoms:

            element = (atom.element or "").upper()
            atom_name = atom.get_name().upper()

            if element in {"O", "N", "S"}:

                protein_features.append({
                    "Feature_Type": "Acceptor",
                    "Residue": f"{resname}{residue.id[1]}",
                    "Atom": atom_name,
                    "Coordinate": np.asarray(atom.coord, dtype=float),
                    "Residue_Distance_From_D166": residue_distance,
                })

    # -------------------------------------------------------------------------
    # DONOR FEATURES
    # -------------------------------------------------------------------------
    if resname in DONOR_RESIDUES:

        for atom in atoms:

            element = (atom.element or "").upper()

            if element in {"N", "O", "S"}:

                protein_features.append({
                    "Feature_Type": "Donor",
                    "Residue": f"{resname}{residue.id[1]}",
                    "Atom": atom.get_name(),
                    "Coordinate": np.asarray(atom.coord, dtype=float),
                    "Residue_Distance_From_D166": residue_distance,
                })

    # -------------------------------------------------------------------------
    # AROMATIC FEATURES
    # -------------------------------------------------------------------------
    if resname in AROMATIC_RESIDUES:

        aromatic_atoms = [
            atom for atom in atoms
            if atom.get_name().upper() in {
                "CG", "CD1", "CD2",
                "CE1", "CE2", "CZ",
                "CG", "ND1", "CD2",
                "CE1", "NE2",
            }
        ]

        if aromatic_atoms:

            center = np.mean(
                [np.asarray(atom.coord, dtype=float)
                 for atom in aromatic_atoms],
                axis=0
            )

            protein_features.append({
                "Feature_Type": "Aromatic",
                "Residue": f"{resname}{residue.id[1]}",
                "Atom": "RING_CENTER",
                "Coordinate": center,
                "Residue_Distance_From_D166": residue_distance,
            })

    # -------------------------------------------------------------------------
    # HYDROPHOBIC FEATURES
    # -------------------------------------------------------------------------
    if resname in HYDROPHOBIC_RESIDUES:

        hydrophobic_atoms = [
            atom for atom in atoms
            if (atom.element or "").upper() in {"C", "S"}
        ]

        if hydrophobic_atoms:

            center = np.mean(
                [np.asarray(atom.coord, dtype=float)
                 for atom in hydrophobic_atoms],
                axis=0
            )

            protein_features.append({
                "Feature_Type": "Hydrophobic",
                "Residue": f"{resname}{residue.id[1]}",
                "Atom": "HYDROPHOBIC_CENTER",
                "Coordinate": center,
                "Residue_Distance_From_D166": residue_distance,
            })


print("\nPocket chemical features:")

feature_counts = defaultdict(int)

for feature in protein_features:
    feature_counts[feature["Feature_Type"]] += 1

for feature_type in [
    "Donor",
    "Acceptor",
    "Aromatic",
    "Hydrophobic",
]:
    print(
        f"  {feature_type:<12}: "
        f"{feature_counts[feature_type]}"
    )

if len(protein_features) == 0:
    raise RuntimeError(
        "No pocket chemical features could be identified."
    )


# =============================================================================
# 10. SAVE POCKET FEATURE TABLE
# =============================================================================

feature_rows = []

for feature in protein_features:

    x, y, z = feature["Coordinate"]

    feature_rows.append({
        "Feature_Type": feature["Feature_Type"],
        "Residue": feature["Residue"],
        "Atom": feature["Atom"],
        "X": float(x),
        "Y": float(y),
        "Z": float(z),
        "Residue_Distance_From_D166_A":
            float(feature["Residue_Distance_From_D166"]),
    })

feature_df = pd.DataFrame(feature_rows)

feature_file = (
    TABLE_DIR / "STEP20_POCKET_PHARMACOPHORE_FEATURES.csv"
)

feature_df.to_csv(
    feature_file,
    index=False
)


# =============================================================================
# 11. RDKit LIGAND FEATURE DEFINITIONS
# =============================================================================

from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

feature_factory = ChemicalFeatures.BuildFeatureFactory(
    str(
        Path(RDConfig.RDDataDir)
        / "BaseFeatures.fdef"
    )
)


# =============================================================================
# 12. MOLECULAR VOLUME APPROXIMATION
# =============================================================================
# RDKit's LabuteASA is used as a molecular-size descriptor.
# It is not a true solvent-excluded molecular volume.
#
# A relative molecular-size compatibility measure is therefore used rather
# than claiming an exact ligand-vs-pocket volume fit.
# =============================================================================

def estimate_ligand_volume(mol):

    # Approximate volume from molecular weight and heavy atom count.
    # This is deliberately only a coarse size descriptor.
    mw = Descriptors.MolWt(mol)
    heavy = mol.GetNumHeavyAtoms()

    if heavy == 0:
        return np.nan

    # Empirical coarse estimate, used only for relative ranking.
    volume = (
        0.70 * mw
        + 5.0 * heavy
    )

    return float(volume)


# =============================================================================
# 13. GENERATE 3D CONFORMERS
# =============================================================================

print("\n" + "=" * 78)
print("GENERATING 3D LIGAND CONFORMERS")
print("=" * 78)

results = []
conformer_failures = []

for index, row in df.iterrows():

    cid = row["CID"] if "CID" in df.columns else f"compound_{index+1}"

    smiles = row[smiles_column]

    print(
        f"  [{index + 1}/{len(df)}] "
        f"CID {cid}"
    )

    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None:

        conformer_failures.append({
            "CID": cid,
            "Reason": "Invalid SMILES"
        })

        continue

    mol = Chem.AddHs(mol)

    # Generate conformers.
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    params.pruneRmsThresh = 0.5

    try:
        conf_ids = list(
            AllChem.EmbedMultipleConfs(
                mol,
                numConfs=N_CONFORMERS,
                params=params,
            )
        )
    except Exception as exc:

        conformer_failures.append({
            "CID": cid,
            "Reason": f"Embedding error: {exc}"
        })

        continue

    if not conf_ids:

        # Single-conformer fallback.
        try:

            status = AllChem.EmbedMolecule(
                mol,
                randomSeed=2026
            )

            if status == 0:
                conf_ids = [0]

        except Exception as exc:

            conformer_failures.append({
                "CID": cid,
                "Reason": f"Fallback embedding error: {exc}"
            })

            continue

    if not conf_ids:

        conformer_failures.append({
            "CID": cid,
            "Reason": "No 3D conformer generated"
        })

        continue

    # Optimize conformers.
    valid_confs = []

    for conf_id in conf_ids:

        try:
            result = AllChem.UFFOptimizeMolecule(
                mol,
                confId=int(conf_id),
                maxIters=500
            )

            valid_confs.append(int(conf_id))

        except Exception:
            pass

    if not valid_confs:
        valid_confs = [int(conf_ids[0])]

    # Save first valid conformer.
    mol_no_h = Chem.RemoveHs(mol)

    saved_mol = Chem.Mol(mol_no_h)

    # Keep the first conformer only in the saved SDF.
    first_conf = valid_confs[0]

    try:
        Chem.MolToMolFile(
            mol_no_h,
            str(
                CONFORMER_DIR
                / f"CID_{cid}.mol"
            ),
        )
    except Exception:
        pass

    results.append({
        "CID": cid,
        "Original_Index": int(index),
        "Mol": mol,
        "Conformer_IDs": valid_confs,
    })


print(
    f"\nSuccessful 3D conformer generation : "
    f"{len(results)}"
)

print(
    f"Conformer generation failures       : "
    f"{len(conformer_failures)}"
)

if len(results) == 0:
    raise RuntimeError(
        "No compounds received valid 3D conformers."
    )


# =============================================================================
# 14. LIGAND PHARMACOPHORE FEATURES
# =============================================================================

def get_ligand_features(mol, conf_id):

    features = []

    for feature in feature_factory.GetFeaturesForMol(mol):

        family = feature.GetFamily()

        if family not in {
            "Donor",
            "Acceptor",
            "Aromatic",
            "Hydrophobe",
        }:
            continue

        try:
            position = feature.GetPos(conf_id)
        except Exception:
            continue

        features.append({
            "Feature_Type": family,
            "Coordinate": np.array(
                [position.x, position.y, position.z],
                dtype=float
            ),
        })

    return features


# =============================================================================
# 15. PHARMACOPHORE COMPATIBILITY
# =============================================================================
# For every ligand conformer:
#
#   Donor ligand feature    ↔ pocket acceptor
#   Acceptor ligand feature ↔ pocket donor
#   Aromatic ligand feature ↔ pocket aromatic
#   Hydrophobe ligand feature ↔ pocket hydrophobic
#
# A feature is considered geometrically compatible when within 3.5 Å.
# This is a pharmacophore-environment compatibility measure, not docking.
# =============================================================================

FEATURE_MATCH_DISTANCE = {
    "Donor-Acceptor": 3.5,
    "Acceptor-Donor": 3.5,
    "Aromatic-Aromatic": 5.0,
    "Hydrophobe-Hydrophobic": 5.0,
}

FEATURE_TYPE_MAP = {
    "Donor": "Acceptor",
    "Acceptor": "Donor",
    "Aromatic": "Aromatic",
    "Hydrophobe": "Hydrophobic",
}


def pharmacophore_score(ligand_features):

    if not ligand_features:
        return {
            "Matched_Features": 0,
            "Ligand_Features": 0,
            "Feature_Match_Percent": 0.0,
            "Feature_Types_Matched": "",
        }

    matched = 0
    matched_types = set()

    for ligand_feature in ligand_features:

        ligand_type = ligand_feature["Feature_Type"]
        expected_protein_type = FEATURE_TYPE_MAP.get(
            ligand_type
        )

        if expected_protein_type is None:
            continue

        ligand_coord = ligand_feature["Coordinate"]

        best_distance = None

        for protein_feature in protein_features:

            if (
                protein_feature["Feature_Type"]
                != expected_protein_type
            ):
                continue

            protein_coord = protein_feature["Coordinate"]

            distance = float(
                np.linalg.norm(
                    ligand_coord - protein_coord
                )
            )

            threshold = FEATURE_MATCH_DISTANCE.get(
                f"{ligand_type}-{expected_protein_type}",
                3.5
            )

            if distance <= threshold:

                if (
                    best_distance is None
                    or distance < best_distance
                ):
                    best_distance = distance

        if best_distance is not None:

            matched += 1
            matched_types.add(
                f"{ligand_type}->{expected_protein_type}"
            )

    percent = (
        matched / len(ligand_features) * 100
        if ligand_features
        else 0.0
    )

    return {
        "Matched_Features": matched,
        "Ligand_Features": len(ligand_features),
        "Feature_Match_Percent": percent,
        "Feature_Types_Matched":
            ";".join(sorted(matched_types)),
    }


# =============================================================================
# 16. COARSE 3D SHAPE COMPATIBILITY
# =============================================================================
# Since docking has NOT yet been performed, there is no experimentally or
# computationally optimized ligand pose inside the pocket.
#
# Therefore we do NOT calculate "shape complementarity" from a nonexistent
# docked pose.
#
# Instead:
#   - estimate ligand molecular size
#   - compare it to Pocket 1 volume
#   - reject only extreme mismatches
#
# This is explicitly a coarse size-compatibility component.
# =============================================================================

def shape_compatibility(ligand_volume, pocket_volume):

    if not np.isfinite(ligand_volume):
        return {
            "Volume_Ratio": np.nan,
            "Shape_Compatibility": "INCOMPLETE",
            "Shape_Score": 0.0,
        }

    ratio = ligand_volume / pocket_volume

    if ratio < MIN_VOLUME_RATIO:
        classification = "Too small"
        score = ratio / MIN_VOLUME_RATIO

    elif ratio <= MAX_VOLUME_RATIO:
        classification = "Compatible"
        score = 1.0

    else:
        classification = "Large"
        score = max(
            0.0,
            MAX_VOLUME_RATIO / ratio
        )

    return {
        "Volume_Ratio": ratio,
        "Shape_Compatibility": classification,
        "Shape_Score": min(1.0, score),
    }


# =============================================================================
# 17. SCREEN ALL VALID CONFORMERS
# =============================================================================

print("\n" + "=" * 78)
print("PHARMACOPHORE / SHAPE SCREENING")
print("=" * 78)

screened_rows = []

for compound in results:

    cid = compound["CID"]
    mol = compound["Mol"]

    ligand_volume = estimate_ligand_volume(mol)

    best_result = None

    for conf_id in compound["Conformer_IDs"]:

        ligand_features = get_ligand_features(
            mol,
            conf_id
        )

        pharma = pharmacophore_score(
            ligand_features
        )

        shape = shape_compatibility(
            ligand_volume,
            pocket_volume
        )

        # Combined pre-docking score.
        #
        # Pharmacophore compatibility receives greater weight because
        # chemical interaction compatibility is more informative than
        # simple molecular size.
        combined_score = (
            0.70 * (pharma["Feature_Match_Percent"] / 100.0)
            + 0.30 * shape["Shape_Score"]
        )

        candidate_result = {
            "CID": cid,
            "Conformer_ID": conf_id,
            "Ligand_Volume_Estimate": ligand_volume,
            **pharma,
            **shape,
            "Combined_Screening_Score":
                combined_score * 100.0,
        }

        if (
            best_result is None
            or candidate_result["Combined_Screening_Score"]
            > best_result["Combined_Screening_Score"]
        ):
            best_result = candidate_result

    if best_result is not None:
        screened_rows.append(best_result)


screened_df = pd.DataFrame(screened_rows)

if screened_df.empty:
    raise RuntimeError(
        "No compounds could be evaluated."
    )


# =============================================================================
# 18. MERGE ORIGINAL COMPOUND INFORMATION
# =============================================================================

cid_column = "CID"

if cid_column not in df.columns:
    raise ValueError(
        "Step 19 library does not contain a CID column."
    )

original_df = df.copy()

original_df["CID"] = (
    original_df["CID"]
    .astype(str)
)

screened_df["CID"] = (
    screened_df["CID"]
    .astype(str)
)

final_df = screened_df.merge(
    original_df,
    on="CID",
    how="left",
    suffixes=("", "_Step19")
)


# =============================================================================
# 19. SCREENING CLASSIFICATION
# =============================================================================

def screening_class(row):

    score = float(
        row["Combined_Screening_Score"]
    )

    matched = int(
        row["Matched_Features"]
    )

    shape = row["Shape_Compatibility"]

    if matched >= 3 and shape == "Compatible" and score >= 50:
        return "HIGH_COMPATIBILITY"

    if matched >= 2 and score >= 30:
        return "MODERATE_COMPATIBILITY"

    if matched >= 1:
        return "LOW_COMPATIBILITY"

    return "NO_CLEAR_COMPATIBILITY"


final_df["Step20_Class"] = final_df.apply(
    screening_class,
    axis=1
)


# =============================================================================
# 20. RANK
# =============================================================================

final_df = final_df.sort_values(
    by=[
        "Combined_Screening_Score",
        "Matched_Features",
        "Feature_Match_Percent",
    ],
    ascending=False
).reset_index(drop=True)

final_df["Step20_Rank"] = (
    range(1, len(final_df) + 1)
)


# =============================================================================
# 21. SAVE FINAL SCREENING TABLE
# =============================================================================

output_columns = [
    "Step20_Rank",
    "CID",
    "Step20_Class",
    "Combined_Screening_Score",
    "Matched_Features",
    "Ligand_Features",
    "Feature_Match_Percent",
    "Feature_Types_Matched",
    "Shape_Compatibility",
    "Shape_Score",
    "Volume_Ratio",
    "Ligand_Volume_Estimate",
    "Conformer_ID",
]

# Add available chemical identifiers.
for column in [
    "IUPACName",
    "MolecularFormula",
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
    "IsomericSMILES",
    "InChIKey",
]:
    if column in final_df.columns:
        output_columns.append(column)

output_columns = [
    column
    for column in output_columns
    if column in final_df.columns
]

screening_output = final_df[
    output_columns
].copy()

screening_file = (
    TABLE_DIR
    / "STEP20_PHARMACOPHORE_SHAPE_SCREENING.csv"
)

screening_output.to_csv(
    screening_file,
    index=False
)


# =============================================================================
# 22. POCKET FEATURE SUMMARY
# =============================================================================

pocket_summary = pd.DataFrame([
    {
        "Pocket_ID": pocket_id,
        "Druggability": pocket_druggability,
        "Pocket_Volume_A3": pocket_volume,
        "D166_Centered_Radius_A": POCKET_RADIUS,
        "Pocket_Residue_Count": len(pocket_residues),
        "Donor_Features": feature_counts["Donor"],
        "Acceptor_Features": feature_counts["Acceptor"],
        "Aromatic_Features": feature_counts["Aromatic"],
        "Hydrophobic_Features": feature_counts["Hydrophobic"],
    }
])

pocket_summary_file = (
    TABLE_DIR
    / "STEP20_POCKET_SUMMARY.csv"
)

pocket_summary.to_csv(
    pocket_summary_file,
    index=False
)


# =============================================================================
# 23. CLASS SUMMARY
# =============================================================================

class_summary = (
    final_df["Step20_Class"]
    .value_counts()
    .rename_axis("Step20_Class")
    .reset_index(name="Compound_Count")
)

class_summary_file = (
    TABLE_DIR
    / "STEP20_CLASS_SUMMARY.csv"
)

class_summary.to_csv(
    class_summary_file,
    index=False
)


# =============================================================================
# 24. CONFORMER QC
# =============================================================================

conformer_qc = pd.DataFrame(conformer_failures)

conformer_qc_file = (
    QC_DIR
    / "STEP20_CONFORMER_FAILURES.csv"
)

conformer_qc.to_csv(
    conformer_qc_file,
    index=False
)


# =============================================================================
# 25. FIGURE 1 — SCREENING SCORE
# =============================================================================

import matplotlib.pyplot as plt

plot_df = final_df.sort_values(
    "Combined_Screening_Score",
    ascending=True
)

plt.figure(figsize=(9, 6))

plt.barh(
    plot_df["CID"].astype(str),
    plot_df["Combined_Screening_Score"]
)

plt.xlabel(
    "Combined Pharmacophore / Shape Compatibility Score"
)

plt.ylabel("PubChem CID")

plt.title(
    "Step 20 — Pharmacophore / Shape Screening"
)

plt.tight_layout()

figure1 = (
    FIGURE_DIR
    / "Figure_20_01_Compound_Screening_Scores.png"
)

plt.savefig(
    figure1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 26. FIGURE 2 — PHARMACOPHORE FEATURE MATCHING
# =============================================================================

plot_df = final_df.sort_values(
    "Feature_Match_Percent",
    ascending=True
)

plt.figure(figsize=(9, 6))

plt.barh(
    plot_df["CID"].astype(str),
    plot_df["Feature_Match_Percent"]
)

plt.xlabel(
    "Ligand Pharmacophore Feature Match (%)"
)

plt.ylabel("PubChem CID")

plt.title(
    "Step 20 — Pocket Pharmacophore Compatibility"
)

plt.xlim(0, 100)

plt.tight_layout()

figure2 = (
    FIGURE_DIR
    / "Figure_20_02_Pharmacophore_Matching.png"
)

plt.savefig(
    figure2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 27. FIGURE 3 — POCKET FEATURE MAP
# =============================================================================

feature_plot = feature_df.copy()

plt.figure(figsize=(8, 7))

for feature_type in sorted(
    feature_plot["Feature_Type"].unique()
):

    subset = feature_plot[
        feature_plot["Feature_Type"] == feature_type
    ]

    plt.scatter(
        subset["X"],
        subset["Y"],
        s=60,
        label=feature_type,
    )

plt.xlabel("X coordinate (Å)")
plt.ylabel("Y coordinate (Å)")

plt.title(
    "Step 20 — D166V Pocket 1 Chemical Feature Map"
)

plt.legend()

plt.tight_layout()

figure3 = (
    FIGURE_DIR
    / "Figure_20_03_Pocket_Chemical_Feature_Map.png"
)

plt.savefig(
    figure3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# 28. QC
# =============================================================================

high = int(
    (final_df["Step20_Class"] == "HIGH_COMPATIBILITY").sum()
)

moderate = int(
    (final_df["Step20_Class"] == "MODERATE_COMPATIBILITY").sum()
)

low = int(
    (final_df["Step20_Class"] == "LOW_COMPATIBILITY").sum()
)

none = int(
    (final_df["Step20_Class"] == "NO_CLEAR_COMPATIBILITY").sum()
)

qc = {
    "step": "Step 20 — Pharmacophore / Shape Screening",
    "timestamp": datetime.now().isoformat(),

    "input_library": str(STEP19_LIBRARY),

    "input_compounds": int(len(df)),

    "valid_3D_conformer_compounds": int(len(results)),

    "conformer_failures": int(len(conformer_failures)),

    "screened_compounds": int(len(final_df)),

    "high_compatibility": high,
    "moderate_compatibility": moderate,
    "low_compatibility": low,
    "no_clear_compatibility": none,

    "pocket_id": pocket_id,
    "pocket_volume_A3": pocket_volume,
    "pocket_druggability": pocket_druggability,

    "pocket_definition":
        "Residues within 10 Å of D166V residue 166 C-alpha, "
        "consistent with Step 17 structural analysis.",

    "n_conformers_requested": N_CONFORMERS,

    "scientific_limitations": [
        "No docking was performed.",
        "No binding affinity was calculated.",
        "No PNPLA3 inhibition was inferred.",
        "No D166V selectivity was inferred.",
        "Pocket pharmacophore features are structure-derived "
        "chemical environment descriptors.",
        "Shape compatibility is a coarse molecular-size "
        "compatibility measure before docking."
    ]
}

qc_file = (
    QC_DIR
    / "STEP20_QC.json"
)

with open(
    qc_file,
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        qc,
        handle,
        indent=2
    )


# =============================================================================
# 29. INTERPRETATION REPORT
# =============================================================================

report_file = (
    REPORT_DIR
    / "STEP20_INTERPRETATION.txt"
)

top_rows = final_df.head(10)

report_lines = [
    "STEP 20 — PHARMACOPHORE / SHAPE SCREENING",
    "=" * 70,
    "",
    f"Input compounds: {len(df)}",
    f"Compounds with valid 3D conformers: {len(results)}",
    f"Compounds screened: {len(final_df)}",
    "",
    "Validated structural target:",
    f"  D166V Pocket ID: {pocket_id}",
    f"  Pocket volume: {pocket_volume:.3f} Å³",
    f"  Pocket druggability: {pocket_druggability:.3f}",
    f"  D166-centered radius: {POCKET_RADIUS:.1f} Å",
    "",
    "STEP 20 CLASSIFICATION",
    "-" * 70,
    f"High compatibility: {high}",
    f"Moderate compatibility: {moderate}",
    f"Low compatibility: {low}",
    f"No clear compatibility: {none}",
    "",
    "TOP COMPOUNDS",
    "-" * 70,
]

for _, row in top_rows.iterrows():

    report_lines.append(
        f"Rank {int(row['Step20_Rank']):2d} | "
        f"CID {row['CID']} | "
        f"{row['Step20_Class']} | "
        f"Score {row['Combined_Screening_Score']:.2f} | "
        f"Features {int(row['Matched_Features'])}/"
        f"{int(row['Ligand_Features'])} | "
        f"Shape {row['Shape_Compatibility']}"
    )

report_lines.extend([
    "",
    "SCIENTIFIC INTERPRETATION",
    "-" * 70,
    "Step 20 provides a pre-docking prioritization of the Step 19",
    "chemically filtered compounds according to compatibility with",
    "the chemical environment of the validated D166V Pocket 1.",
    "",
    "The pharmacophore component evaluates whether ligand chemical",
    "features can geometrically correspond to donor, acceptor,",
    "aromatic, and hydrophobic environments identified from the",
    "actual D166V pocket structure.",
    "",
    "The shape component is deliberately coarse because no docked",
    "ligand pose exists at this stage. It therefore evaluates relative",
    "molecular-size compatibility with the experimentally modeled",
    "Pocket 1 volume rather than claiming true protein-ligand shape",
    "complementarity.",
    "",
    "The Step 20 score must NOT be interpreted as binding affinity,",
    "binding free energy, inhibitory potency, or D166V selectivity.",
    "",
    "Step 21 should proceed to structure-based virtual screening",
    "using the compounds prioritized by Step 20.",
])

with open(
    report_file,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "\n".join(report_lines)
    )


# =============================================================================
# 30. MANIFEST
# =============================================================================

manifest = (
    "STEP 20 — PHARMACOPHORE / SHAPE SCREENING\n"
    "==========================================\n\n"
    f"Completed: {datetime.now().isoformat()}\n\n"
    "INPUTS\n"
    f"- Step 19 library: {STEP19_LIBRARY}\n"
    f"- Step 16 pocket table: {STEP16_POCKET_TABLE}\n"
    f"- D166V structure: {D166V_PDB}\n\n"
    "OUTPUTS\n"
    f"- Screening table: {screening_file}\n"
    f"- Pocket features: {feature_file}\n"
    f"- Pocket summary: {pocket_summary_file}\n"
    f"- Class summary: {class_summary_file}\n"
    f"- Figure 1: {figure1}\n"
    f"- Figure 2: {figure2}\n"
    f"- Figure 3: {figure3}\n"
    f"- QC: {qc_file}\n"
    f"- Report: {report_file}\n\n"
    "SCIENTIFIC LIMITATION\n"
    "Step 20 is a pre-docking pharmacophore/shape prioritization.\n"
    "It does not establish binding, inhibition, affinity, or mutation\n"
    "selectivity.\n"
)

manifest_file = (
    QC_DIR
    / "STEP20_MANIFEST.txt"
)

with open(
    manifest_file,
    "w",
    encoding="utf-8"
) as handle:

    handle.write(manifest)


# =============================================================================
# 31. FINAL CONSOLE OUTPUT
# =============================================================================

print("\n" + "=" * 78)
print("STEP 20 COMPLETED")
print("=" * 78)

print(f"\nInput compounds                 : {len(df)}")
print(f"Valid 3D conformer compounds   : {len(results)}")
print(f"Screened compounds             : {len(final_df)}")

print("\nCompatibility classes:")
print(f"  HIGH_COMPATIBILITY            : {high}")
print(f"  MODERATE_COMPATIBILITY        : {moderate}")
print(f"  LOW_COMPATIBILITY             : {low}")
print(f"  NO_CLEAR_COMPATIBILITY        : {none}")

print("\nTop compounds:")

for _, row in final_df.head(10).iterrows():

    print(
        f"  Rank {int(row['Step20_Rank']):2d} | "
        f"CID {row['CID']} | "
        f"{row['Step20_Class']} | "
        f"Score {row['Combined_Screening_Score']:.2f}"
    )

print("\nOutputs:")

print(f"  Screening table:")
print(f"    {screening_file}")

print(f"  Pocket features:")
print(f"    {feature_file}")

print(f"  Pocket summary:")
print(f"    {pocket_summary_file}")

print(f"  Class summary:")
print(f"    {class_summary_file}")

print(f"  Figures:")
print(f"    {figure1}")
print(f"    {figure2}")
print(f"    {figure3}")

print(f"  QC:")
print(f"    {qc_file}")

print(f"  Interpretation:")
print(f"    {report_file}")

print("\nScientific note:")
print(
    "Step 20 is pre-docking pharmacophore/shape prioritization. "
    "It does not establish binding, inhibition, affinity, "
    "or D166V selectivity."
)

print("=" * 78)