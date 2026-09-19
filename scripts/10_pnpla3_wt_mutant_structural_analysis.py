"""
===============================================================================
STEP 10 — PNPLA3 WILD-TYPE AND MUTANT STRUCTURAL PREPARATION
              AND COMPARATIVE STRUCTURAL ANALYSIS
===============================================================================

Project:
    MASLD–PNPLA3 Variant Analysis

Primary structural candidate:
    p.Asp166Val (D166V)
    VariationID: 3308171
    UniProt: Q9NST1

IMPORTANT:
    PyMOL is executed through the working Conda Python environment:
        %USERPROFILE%\miniconda3\envs\pymol\python.exe

    This script does NOT require pymol.exe to be available on PATH.

WORKFLOW
--------
1. Locate and validate Step 9F final candidate-selection CSV
2. Validate 82-variant dataset
3. Validate 9 structural candidates
4. Confirm D166V as primary candidate
5. Retrieve canonical PNPLA3 sequence from UniProt
6. Search RCSB PDB for experimental structures
7. Fall back to current AlphaFold DB structure when required
8. Validate WT structure
9. Confirm residue 166 = Asp
10. Detect working PyMOL Python environment
11. Convert structure to PDB using PyMOL Python API
12. Generate D166V using PyMOL mutagenesis
13. Validate mutant residue 166 = Val
14. Compare WT vs D166V
15. Calculate structural metrics
16. Calculate local environment and contact changes
17. Generate quantitative figures
18. Generate PyMOL 3D figures
19. Perform final QC
20. Write final Step 10 report

===============================================================================
"""

from pathlib import Path
import sys
import os
import re
import json
import shutil
import subprocess
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_NAME = "MASLD_PNPLA3"

STEP9F_FILENAME = (
    "pnpla3_step9f_final_candidate_selection_FINAL.csv"
)

EXPECTED_VARIANTS = 82
EXPECTED_STRUCTURAL_CANDIDATES = 9

PRIMARY_VARIATION_ID = 3308171
PRIMARY_PROTEIN_CHANGE = "p.Asp166Val"
PRIMARY_PROTEIN_VARIANT = "D166V"

PRIMARY_POSITION = 166
PRIMARY_REFERENCE_AA = "Asp"
PRIMARY_ALTERNATE_AA = "Val"

EXPECTED_UNIPROT = "Q9NST1"
EXPECTED_TRANSCRIPT = "ENST00000216180.8"

LOCAL_RADIUS = 5.0
CONTACT_DISTANCE = 4.5
POLAR_CONTACT_DISTANCE = 3.5

REQUEST_TIMEOUT = 45


# =============================================================================
# PATHS
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

STEP10_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
)

STRUCTURE_DIR = STEP10_DIR / "structures"

WT_DIR = STRUCTURE_DIR / "WT"
MUTANT_DIR = STRUCTURE_DIR / "D166V"

TABLE_DIR = STEP10_DIR / "tables"
FIGURE_DIR = STEP10_DIR / "figures"
QC_DIR = STEP10_DIR / "QC"
LOG_DIR = STEP10_DIR / "logs"
TEMP_DIR = STEP10_DIR / "temporary"


for directory in [
    STEP10_DIR,
    STRUCTURE_DIR,
    WT_DIR,
    MUTANT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
    LOG_DIR,
    TEMP_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


LOG_FILE = LOG_DIR / "Step10_execution.log"


# =============================================================================
# LOGGING
# =============================================================================

def log(message=""):
    print(message)

    with open(LOG_FILE, "a", encoding="utf-8") as handle:
        handle.write(str(message) + "\n")


def section(title):
    log("")
    log("=" * 80)
    log(title)
    log("=" * 80)


def subsection(title):
    log("")
    log("-" * 80)
    log(title)
    log("-" * 80)


def status(label, passed):
    log(f"{label}: {'PASS' if passed else 'FAIL'}")


# =============================================================================
# JSON SAFE CONVERSION
# =============================================================================

def json_safe(value):

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(k): json_safe(v)
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [json_safe(v) for v in value]

    if isinstance(value, tuple):
        return [json_safe(v) for v in value]

    return value


def write_json(path, data):

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            json_safe(data),
            handle,
            indent=4
        )


# =============================================================================
# BIOPYTHON
# =============================================================================

try:

    from Bio.PDB import (
        PDBParser,
        MMCIFParser,
        PDBIO,
        Superimposer,
    )

    from Bio.PDB.Polypeptide import is_aa

    from Bio.SeqUtils import seq1

except ImportError as exc:

    log("ERROR: Biopython is required.")
    log(str(exc))
    sys.exit(1)


# =============================================================================
# MATPLOTLIB
# =============================================================================

try:

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

except ImportError:

    log("ERROR: matplotlib is required.")
    sys.exit(1)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return np.nan


def safe_seq1(resname):

    resname = str(resname).upper()

    if resname in AA3_TO_AA1:
        return AA3_TO_AA1[resname]

    try:
        return seq1(resname)
    except Exception:
        return "X"


def get_structure_parser(path):

    path = Path(path)

    if path.suffix.lower() == ".cif":

        return MMCIFParser(
            QUIET=True
        )

    return PDBParser(
        QUIET=True
    )


def load_structure(path, structure_id="STRUCTURE"):

    parser = get_structure_parser(path)

    return parser.get_structure(
        structure_id,
        str(path)
    )


def get_protein_residues(chain):

    residues = []

    for residue in chain:

        if not is_aa(residue, standard=False):
            continue

        residues.append(residue)

    return residues


def residue_number(residue):

    return int(residue.id[1])


def get_residue_by_number(chain, position):

    for residue in chain:

        if residue_number(residue) == position:
            return residue

    return None


def residue_centroid(residue):

    atoms = list(residue.get_atoms())

    if not atoms:
        return None

    coordinates = np.array(
        [atom.coord for atom in atoms],
        dtype=float
    )

    return coordinates.mean(axis=0)


def minimum_heavy_atom_distance(residue_a, residue_b):

    atoms_a = [
        atom
        for atom in residue_a.get_atoms()
        if atom.element.upper() != "H"
    ]

    atoms_b = [
        atom
        for atom in residue_b.get_atoms()
        if atom.element.upper() != "H"
    ]

    if not atoms_a or not atoms_b:
        return np.nan

    min_distance = np.inf

    for atom_a in atoms_a:

        for atom_b in atoms_b:

            distance = np.linalg.norm(
                atom_a.coord - atom_b.coord
            )

            if distance < min_distance:
                min_distance = distance

    return float(min_distance)


def atom_key(atom):

    residue = atom.get_parent()

    chain = residue.get_parent()

    return (
        chain.id,
        residue_number(residue),
        atom.name.strip()
    )


def collect_atoms_by_key(structure, chain_id=None):

    atoms = {}

    for model in structure:

        for chain in model:

            if chain_id is not None and chain.id != chain_id:
                continue

            for residue in chain:

                if not is_aa(residue, standard=False):
                    continue

                for atom in residue:

                    if atom.element.upper() == "H":
                        continue

                    atoms[atom_key(atom)] = atom

    return atoms


def get_chain(structure, preferred_chain="A"):

    model = next(structure.get_models())

    if preferred_chain in model.child_dict:

        return model[preferred_chain]

    chains = list(model.get_chains())

    if not chains:

        return None

    return chains[0]


def get_structure_residue_numbers(chain):

    return sorted(
        residue_number(residue)
        for residue in get_protein_residues(chain)
    )


def get_mean_plddt(chain):

    values = []

    for residue in get_protein_residues(chain):

        for atom in residue:

            try:

                value = float(atom.bfactor)

                if value > 0:
                    values.append(value)

            except Exception:
                pass

    if not values:
        return np.nan

    return float(np.mean(values))


def get_residue_plddt(residue):

    values = []

    for atom in residue:

        try:

            value = float(atom.bfactor)

            if value > 0:
                values.append(value)

        except Exception:
            pass

    if not values:
        return np.nan

    return float(np.mean(values))


# =============================================================================
# STEP 1
# =============================================================================

section(
    "STEP 10 — PNPLA3 STRUCTURAL ANALYSIS"
)

log(f"Project root: {PROJECT_ROOT}")
log(f"Step 10 directory: {STEP10_DIR}")


section(
    "STEP 1 — LOCATING STEP 9F FINAL CANDIDATE-SELECTION FILE"
)

candidate_files = list(
    PROJECT_ROOT.rglob(STEP9F_FILENAME)
)

if not candidate_files:

    log("ERROR: Step 9F final candidate-selection file was not found.")
    sys.exit(1)

if len(candidate_files) > 1:

    log("Multiple Step 9F files found:")

    for path in candidate_files:
        log(f"    {path}")

    step9f_file = candidate_files[0]

else:

    step9f_file = candidate_files[0]

log("Step 9F file found:")
log(f"    {step9f_file}")


# =============================================================================
# STEP 2
# =============================================================================

subsection(
    "STEP 2 — READING STEP 9F DATASET"
)

try:

    df = pd.read_csv(
        step9f_file,
        low_memory=False
    )

except Exception as exc:

    log(f"ERROR reading Step 9F file: {exc}")
    sys.exit(1)


log(f"Rows loaded: {len(df)}")
log(f"Columns loaded: {len(df.columns)}")


# =============================================================================
# STEP 3
# =============================================================================

subsection(
    "STEP 3 — STEP 9F DATASET VALIDATION"
)

variant_count_pass = (
    len(df) == EXPECTED_VARIANTS
)

status(
    f"Variant count ({EXPECTED_VARIANTS})",
    variant_count_pass
)

if not variant_count_pass:
    sys.exit(1)


required_columns = [
    "VariationID",
    "Protein_Change",
    "UniProt_ID",
    "Transcript_ID",
    "Protein_Variant",
    "Structural_Analysis_Candidate",
    "Priority_Rank",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",
    "Final_Mutant_Rank",
    "Final_Mutant_Selection_Score",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    log("ERROR: Missing required columns:")

    for column in missing_columns:
        log(f"    {column}")

    sys.exit(1)


# =============================================================================
# STEP 4
# =============================================================================

subsection(
    "STEP 4 — STRUCTURAL CANDIDATE VALIDATION"
)

structural_mask = (
    df["Structural_Analysis_Candidate"]
    .astype(str)
    .str.upper()
    .eq("YES")
)

structural_df = df.loc[
    structural_mask
].copy()

log(
    f"Structural candidates found: "
    f"{len(structural_df)}"
)

structural_count_pass = (
    len(structural_df)
    == EXPECTED_STRUCTURAL_CANDIDATES
)

status(
    "Structural candidate count",
    structural_count_pass
)

if not structural_count_pass:
    sys.exit(1)


structural_output = (
    TABLE_DIR
    / "Table_S10_1_structural_candidates.csv"
)

structural_df.to_csv(
    structural_output,
    index=False
)


# =============================================================================
# STEP 5
# =============================================================================

subsection(
    "STEP 5 — PRIMARY CANDIDATE VALIDATION"
)

primary_matches = df.loc[
    pd.to_numeric(
        df["VariationID"],
        errors="coerce"
    ) == PRIMARY_VARIATION_ID
].copy()

if len(primary_matches) != 1:

    log(
        "ERROR: Primary VariationID does not "
        "occur exactly once."
    )

    sys.exit(1)


primary = primary_matches.iloc[0]


checks = {}

checks["VariationID"] = (
    int(primary["VariationID"])
    == PRIMARY_VARIATION_ID
)

checks["Protein_Change"] = (
    str(primary["Protein_Change"]).strip()
    == PRIMARY_PROTEIN_CHANGE
)

checks["Protein_Variant"] = (
    str(primary["Protein_Variant"]).strip()
    == PRIMARY_PROTEIN_VARIANT
)

checks["Protein_Position"] = (
    int(float(primary["Protein_Position"]))
    == PRIMARY_POSITION
)

checks["Reference_AA"] = (
    str(primary["Reference_Amino_Acid"]).strip()
    == PRIMARY_REFERENCE_AA
)

checks["Alternate_AA"] = (
    str(primary["Alternate_Amino_Acid"]).strip()
    == PRIMARY_ALTERNATE_AA
)

checks["UniProt_ID"] = (
    str(primary["UniProt_ID"]).strip()
    == EXPECTED_UNIPROT
)

checks["Transcript_ID"] = (
    str(primary["Transcript_ID"]).strip()
    == EXPECTED_TRANSCRIPT
)

checks["Structural_Candidate"] = (
    str(
        primary["Structural_Analysis_Candidate"]
    ).upper()
    == "YES"
)

checks["Priority_Rank"] = (
    int(float(primary["Priority_Rank"]))
    == 1
)

checks["Final_Mutant_Rank"] = (
    int(float(primary["Final_Mutant_Rank"]))
    == 1
)


for label, passed in checks.items():

    status(
        label,
        passed
    )

if not all(checks.values()):

    log(
        "ERROR: Primary structural candidate "
        "validation failed."
    )

    sys.exit(1)


log("")
log("PRIMARY STRUCTURAL CANDIDATE CONFIRMED:")
log(
    f"    PNPLA3 {PRIMARY_PROTEIN_CHANGE} "
    f"({PRIMARY_PROTEIN_VARIANT})"
)
log(
    f"    VariationID: {PRIMARY_VARIATION_ID}"
)
log(
    f"    UniProt: {EXPECTED_UNIPROT}"
)
log(
    f"    Position: {PRIMARY_POSITION}"
)
log(
    f"    WT amino acid: {PRIMARY_REFERENCE_AA}"
)
log(
    f"    Mutant amino acid: {PRIMARY_ALTERNATE_AA}"
)


# =============================================================================
# STEP 6 — UNIPROT
# =============================================================================

subsection(
    "STEP 6 — RETRIEVING CANONICAL PNPLA3 SEQUENCE"
)

uniprot_url = (
    f"https://rest.uniprot.org/"
    f"uniprotkb/{EXPECTED_UNIPROT}.fasta"
)

wt_fasta = (
    WT_DIR
    / f"PNPLA3_{EXPECTED_UNIPROT}_UniProt.fasta"
)

try:

    response = requests.get(
        uniprot_url,
        timeout=REQUEST_TIMEOUT
    )

    response.raise_for_status()

    wt_fasta.write_text(
        response.text,
        encoding="utf-8"
    )

except Exception as exc:

    log(
        f"ERROR retrieving UniProt sequence: {exc}"
    )

    sys.exit(1)


sequence_lines = [
    line.strip()
    for line in response.text.splitlines()
    if not line.startswith(">")
]

sequence = "".join(sequence_lines).upper()

log(
    f"UniProt sequence length: {len(sequence)}"
)

if len(sequence) < PRIMARY_POSITION:

    log(
        "ERROR: UniProt sequence is shorter "
        "than residue 166."
    )

    sys.exit(1)


uniprot_residue = sequence[
    PRIMARY_POSITION - 1
]

log(
    f"UniProt residue {PRIMARY_POSITION}: "
    f"{uniprot_residue}"
)

uniprot_residue_pass = (
    uniprot_residue == "D"
)

status(
    "UniProt canonical residue 166 = Asp",
    uniprot_residue_pass
)

if not uniprot_residue_pass:
    sys.exit(1)


# =============================================================================
# STEP 7 — RCSB
# =============================================================================

subsection(
    "STEP 7 — SEARCHING RCSB PDB FOR EXPERIMENTAL STRUCTURES"
)

rcsb_url = (
    "https://search.rcsb.org/"
    "rcsbsearch/v2/query"
)

rcsb_query = {
    "query": {
        "type": "terminal",
        "service": "text",
        "parameters": {
            "attribute":
                "rcsb_polymer_entity_container_identifiers."
                "reference_sequence_identifiers.database_accession",
            "operator": "exact_match",
            "value": EXPECTED_UNIPROT
        }
    },
    "return_type": "entry",
    "request_options": {
        "paginate": {
            "start": 0,
            "rows": 100
        }
    }
}

rcsb_hits = []

try:

    rcsb_response = requests.post(
        rcsb_url,
        json=rcsb_query,
        timeout=REQUEST_TIMEOUT
    )

    log(
        f"RCSB search returned HTTP "
        f"{rcsb_response.status_code}"
    )

    if rcsb_response.status_code == 200:

        data = rcsb_response.json()

        for result in data.get(
            "result_set",
            []
        ):

            identifier = result.get(
                "identifier"
            )

            if identifier:
                rcsb_hits.append(
                    identifier
                )

except Exception as exc:

    log(
        f"RCSB search error: {exc}"
    )


log(
    f"RCSB candidate entries found: "
    f"{len(rcsb_hits)}"
)


# =============================================================================
# STEP 8 — STRUCTURE SELECTION
# =============================================================================

subsection(
    "STEP 8 — SELECTING WT STRUCTURE"
)

selected_structure = None
selected_source = None
selected_structure_id = None
selected_chain_id = None
selected_structure_url = None


def evaluate_structure_file(
    structure_file,
    source_name,
    structure_id
):

    try:

        structure = load_structure(
            structure_file,
            structure_id
        )

    except Exception:

        return None

    best = None

    for chain in structure[0]:

        residues = get_protein_residues(
            chain
        )

        numbers = [
            residue_number(residue)
            for residue in residues
        ]

        if PRIMARY_POSITION not in numbers:
            continue

        target = get_residue_by_number(
            chain,
            PRIMARY_POSITION
        )

        if target is None:
            continue

        aa = safe_seq1(
            target.resname
        )

        if aa != "D":
            continue

        coverage = len(residues)

        candidate = {
            "file": structure_file,
            "source": source_name,
            "structure_id": structure_id,
            "chain": chain.id,
            "coverage": coverage,
        }

        if (
            best is None
            or coverage > best["coverage"]
        ):
            best = candidate

    return best


# -------------------------------------------------------------------------
# RCSB candidate download
# -------------------------------------------------------------------------

for pdb_id in rcsb_hits:

    pdb_id = str(pdb_id).upper()

    candidate_url = (
        f"https://files.rcsb.org/download/"
        f"{pdb_id}.cif"
    )

    candidate_file = (
        TEMP_DIR
        / f"RCSB_{pdb_id}.cif"
    )

    try:

        response = requests.get(
            candidate_url,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:
            continue

        candidate_file.write_bytes(
            response.content
        )

        candidate = evaluate_structure_file(
            candidate_file,
            "RCSB_PDB",
            pdb_id
        )

        if candidate is None:
            continue

        if (
            selected_structure is None
            or candidate["coverage"]
            > selected_structure["coverage"]
        ):

            selected_structure = candidate

    except Exception:
        continue


# =============================================================================
# ALPHAFOLD FALLBACK
# =============================================================================

if selected_structure is None:

    log("")
    log(
        "No suitable RCSB structure covering "
        "residue 166 was found."
    )

    log(
        "Falling back to current AlphaFold DB record."
    )

    af_api = (
        f"https://alphafold.ebi.ac.uk/"
        f"api/prediction/{EXPECTED_UNIPROT}"
    )

    try:

        af_response = requests.get(
            af_api,
            timeout=REQUEST_TIMEOUT
        )

        af_response.raise_for_status()

        af_data = af_response.json()

    except Exception as exc:

        log(
            f"ERROR retrieving AlphaFold record: {exc}"
        )

        sys.exit(1)


    if isinstance(af_data, list):

        af_record = af_data[0]

    else:

        af_record = af_data


    cif_url = af_record.get(
        "cifUrl"
    )

    pdb_url = af_record.get(
        "pdbUrl"
    )

    if cif_url:

        structure_url = cif_url
        structure_suffix = ".cif"

    elif pdb_url:

        structure_url = pdb_url
        structure_suffix = ".pdb"

    else:

        log(
            "ERROR: AlphaFold record does not "
            "contain a structure URL."
        )

        sys.exit(1)


    af_file = (
        WT_DIR
        / (
            f"PNPLA3_{EXPECTED_UNIPROT}_AlphaFold"
            f"{structure_suffix}"
        )
    )


    try:

        af_structure_response = requests.get(
            structure_url,
            timeout=REQUEST_TIMEOUT
        )

        af_structure_response.raise_for_status()

        af_file.write_bytes(
            af_structure_response.content
        )

    except Exception as exc:

        log(
            f"ERROR downloading AlphaFold structure: "
            f"{exc}"
        )

        sys.exit(1)


    selected_structure = {
        "file": af_file,
        "source": "AlphaFold_DB",
        "structure_id": af_record.get(
            "entryId",
            f"{EXPECTED_UNIPROT}-F1"
        ),
        "chain": "A",
        "coverage": None,
        "url": structure_url,
    }


selected_structure_file = Path(
    selected_structure["file"]
)

selected_source = selected_structure[
    "source"
]

selected_structure_id = selected_structure[
    "structure_id"
]

selected_chain_id = selected_structure[
    "chain"
]

selected_structure_url = selected_structure.get(
    "url",
    None
)


# Copy/rename selected structure to stable WT filename

if selected_structure_file.suffix.lower() == ".cif":

    wt_structure_file = (
        WT_DIR
        / "PNPLA3_Q9NST1_AlphaFold.cif"
    )

else:

    wt_structure_file = (
        WT_DIR
        / "PNPLA3_Q9NST1_AlphaFold.pdb"
    )


if selected_structure_file.resolve() != wt_structure_file.resolve():

    shutil.copy2(
        selected_structure_file,
        wt_structure_file
    )

else:

    wt_structure_file = selected_structure_file


log("")
log("SELECTED WT STRUCTURE")
log(
    f"    Source: {selected_source}"
)
log(
    f"    Structure ID: {selected_structure_id}"
)
log(
    f"    File: {wt_structure_file}"
)
log(
    f"    Chain: {selected_chain_id}"
)


write_json(
    STEP10_DIR
    / "structure_source_metadata.json",
    {
        "source": selected_source,
        "structure_id": selected_structure_id,
        "chain": selected_chain_id,
        "file": wt_structure_file,
        "url": selected_structure_url,
    }
)


# =============================================================================
# STEP 9 — WT VALIDATION
# =============================================================================

subsection(
    "STEP 9 — WT STRUCTURE VALIDATION"
)

wt_structure = load_structure(
    wt_structure_file,
    "PNPLA3_WT"
)

wt_chain = get_chain(
    wt_structure,
    selected_chain_id
)

if wt_chain is None:

    log(
        "ERROR: No suitable protein chain found."
    )

    sys.exit(1)


wt_residues = get_protein_residues(
    wt_chain
)

wt_numbers = get_structure_residue_numbers(
    wt_chain
)

wt_target_residue = get_residue_by_number(
    wt_chain,
    PRIMARY_POSITION
)

if wt_target_residue is None:

    log(
        "ERROR: WT structure does not contain "
        "residue 166."
    )

    sys.exit(1)


wt_target_aa = safe_seq1(
    wt_target_residue.resname
)

log(
    f"WT protein residues in structure: "
    f"{len(wt_residues)}"
)

log(
    f"Structure residue {PRIMARY_POSITION}: "
    f"{wt_target_residue.resname} "
    f"({wt_target_aa})"
)

wt_residue_pass = (
    wt_target_aa == "D"
)

status(
    "WT residue 166 = Asp",
    wt_residue_pass
)

if not wt_residue_pass:
    sys.exit(1)


structure_start = min(wt_numbers)
structure_end = max(wt_numbers)

log(
    f"Structure numbering range: "
    f"{structure_start}-{structure_end}"
)


overlap_positions = [
    position
    for position in wt_numbers
    if 1 <= position <= len(sequence)
]

coverage_percent = (
    len(set(overlap_positions))
    / len(sequence)
    * 100
)

log(
    f"Structure sequence coverage: "
    f"{len(set(overlap_positions))}/"
    f"{len(sequence)} "
    f"({coverage_percent:.2f}%)"
)


mean_plddt = get_mean_plddt(
    wt_chain
)

target_plddt = get_residue_plddt(
    wt_target_residue
)

log(
    f"Mean structure pLDDT: "
    f"{mean_plddt:.2f}"
)

log(
    f"Residue {PRIMARY_POSITION} pLDDT: "
    f"{target_plddt:.2f}"
)


# =============================================================================
# STEP 10 — PYMOl PYTHON ENVIRONMENT
# =============================================================================

subsection(
    "STEP 10 — CHECKING PyMOL PYTHON ENVIRONMENT"
)

PYMOL_PYTHON_CANDIDATES = [

    Path.home()
    / "miniconda3"
    / "envs"
    / "pymol"
    / "python.exe",

    Path(
        os.environ.get(
            "USERPROFILE",
            ""
        )
    )
    / "miniconda3"
    / "envs"
    / "pymol"
    / "python.exe",
]


pymol_python = None

for candidate in PYMOL_PYTHON_CANDIDATES:

    if not candidate:
        continue

    if not candidate.exists():
        continue

    try:

        test = subprocess.run(
            [
                str(candidate),
                "-c",
                (
                    "import pymol; "
                    "print(pymol.__file__)"
                )
            ],
            capture_output=True,
            text=True,
            timeout=60
        )

        if test.returncode == 0:

            pymol_python = candidate
            break

    except Exception:
        continue


if pymol_python is None:

    log("")
    log(
        "ERROR: A working PyMOL Python environment "
        "was not found."
    )

    log("")
    log(
        "Expected environment:"
    )

    log(
        f"    {Path.home() / 'miniconda3' / 'envs' / 'pymol' / 'python.exe'}"
    )

    log("")
    log(
        "The script will NOT create a fake mutant."
    )

    sys.exit(1)


log(
    "PyMOL Python found:"
)

log(
    f"    {pymol_python}"
)


version_test = subprocess.run(
    [
        str(pymol_python),
        "-c",
        (
            "import pymol; "
            "pymol.finish_launching(['pymol','-cq']); "
            "from pymol import cmd; "
            "print(cmd.get_version()[0]); "
            "cmd.quit()"
        )
    ],
    capture_output=True,
    text=True,
    timeout=120
)


if version_test.returncode != 0:

    log(
        "ERROR: PyMOL Python failed initialization."
    )

    log(
        version_test.stderr
    )

    sys.exit(1)


version_lines = [
    line.strip()
    for line in version_test.stdout.splitlines()
    if line.strip()
]

pymol_version = (
    version_lines[-1]
    if version_lines
    else "unknown"
)

log(
    f"PyMOL version: {pymol_version}"
)

status(
    "PyMOL import",
    True
)


# =============================================================================
# PYMOl RUNNER
# =============================================================================

def run_pymol_python(
    script_text,
    script_name,
    timeout=300
):

    script_path = (
        TEMP_DIR
        / script_name
    )

    script_path.write_text(
        script_text,
        encoding="utf-8"
    )

    result = subprocess.run(
        [
            str(pymol_python),
            str(script_path)
        ],
        capture_output=True,
        text=True,
        timeout=timeout
    )

    output_file = (
        LOG_DIR
        / f"{script_name}.stdout.txt"
    )

    error_file = (
        LOG_DIR
        / f"{script_name}.stderr.txt"
    )

    output_file.write_text(
        result.stdout,
        encoding="utf-8"
    )

    error_file.write_text(
        result.stderr,
        encoding="utf-8"
    )

    if result.stdout.strip():

        log("")
        log(
            f"PyMOL output ({script_name}):"
        )

        log(
            result.stdout.strip()
        )

    if result.returncode != 0:

        log("")
        log(
            f"PyMOL script failed: "
            f"{script_name}"
        )

        if result.stderr.strip():
            log(
                result.stderr.strip()
            )

        return False

    return True


# =============================================================================
# STEP 11 — CONVERT WT CIF TO PDB
# =============================================================================

subsection(
    "STEP 11 — PREPARING WT PDB USING PyMOL PYTHON API"
)

wt_pdb_file = (
    WT_DIR
    / "PNPLA3_WT.pdb"
)


convert_script = f"""
import pymol

pymol.finish_launching(
    ["pymol", "-cq"]
)

from pymol import cmd

cmd.load(
    r"{wt_structure_file}",
    "PNPLA3_WT"
)

cmd.save(
    r"{wt_pdb_file}",
    "PNPLA3_WT"
)

cmd.quit()
"""


convert_ok = run_pymol_python(
    convert_script,
    "convert_wt_to_pdb.py"
)

if not convert_ok or not wt_pdb_file.exists():

    log(
        "ERROR: WT PDB preparation failed."
    )

    sys.exit(1)


log(
    f"WT PDB created:"
)

log(
    f"    {wt_pdb_file}"
)

status(
    "WT PDB preparation",
    True
)


# =============================================================================
# STEP 12 — GENERATE TRUE D166V MUTANT
# =============================================================================

subsection(
    "STEP 12 — GENERATING D166V MUTANT USING PyMOL MUTAGENESIS"
)

mutant_pdb_file = (
    MUTANT_DIR
    / "PNPLA3_D166V_mutant.pdb"
)

if mutant_pdb_file.exists():

    mutant_pdb_file.unlink()


mutation_script = f"""
import pymol

pymol.finish_launching(
    ["pymol", "-cq"]
)

from pymol import cmd

cmd.load(
    r"{wt_pdb_file}",
    "PNPLA3_WT"
)

# Select the exact residue.
cmd.select(
    "target_residue",
    "chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

# Start PyMOL mutagenesis wizard.
cmd.wizard("mutagenesis")

wiz = cmd.get_wizard()

# Set desired mutant amino acid.
wiz.set_mode("VAL")

# Apply mutagenesis to selected residue.
wiz.do_select("target_residue")

# Refresh wizard so that a rotamer is generated.
try:
    cmd.refresh_wizard()
except Exception:
    pass

# Apply the currently selected VAL rotamer.
wiz.apply()

# Exit wizard.
cmd.set_wizard()

# Save the actual remodeled mutant structure.
cmd.save(
    r"{mutant_pdb_file}",
    "PNPLA3_WT"
)

cmd.quit()
"""


mutation_ok = run_pymol_python(
    mutation_script,
    "generate_D166V_mutant.py",
    timeout=300
)

if (
    not mutation_ok
    or not mutant_pdb_file.exists()
):

    log(
        "ERROR: D166V mutant generation failed."
    )

    sys.exit(1)


log("")
log(
    "D166V mutant generated:"
)

log(
    f"    {mutant_pdb_file}"
)


# =============================================================================
# STEP 13 — MUTANT VALIDATION
# =============================================================================

subsection(
    "STEP 13 — VALIDATING D166V MUTANT"
)

mutant_structure = load_structure(
    mutant_pdb_file,
    "PNPLA3_D166V"
)

mutant_chain = get_chain(
    mutant_structure,
    selected_chain_id
)

if mutant_chain is None:

    log(
        "ERROR: Mutant chain could not be identified."
    )

    sys.exit(1)


mutant_target_residue = get_residue_by_number(
    mutant_chain,
    PRIMARY_POSITION
)

if mutant_target_residue is None:

    log(
        "ERROR: Mutant residue 166 not found."
    )

    sys.exit(1)


mutant_target_aa = safe_seq1(
    mutant_target_residue.resname
)

log(
    f"Mutant residue {PRIMARY_POSITION}: "
    f"{mutant_target_residue.resname} "
    f"({mutant_target_aa})"
)

mutant_residue_pass = (
    mutant_target_aa == "V"
)

status(
    "Mutant residue 166 = Val",
    mutant_residue_pass
)

if not mutant_residue_pass:
    sys.exit(1)


mutant_atom_names = {
    atom.name.strip()
    for atom in mutant_target_residue
}


required_val_atoms = {
    "CG1",
    "CG2"
}

val_sidechain_pass = (
    required_val_atoms
    .issubset(mutant_atom_names)
)

status(
    "VAL side-chain atoms CG1/CG2 present",
    val_sidechain_pass
)

if not val_sidechain_pass:
    log(
        "ERROR: Mutant VAL side chain does not "
        "contain expected CG1/CG2 atoms."
    )

    sys.exit(1)


invalid_asp_atoms = {
    "OD1",
    "OD2"
}.intersection(
    mutant_atom_names
)

asp_removed_pass = (
    len(invalid_asp_atoms) == 0
)

status(
    "ASP OD1/OD2 atoms removed",
    asp_removed_pass
)

if not asp_removed_pass:
    log(
        "ERROR: Residue 166 still contains "
        "ASP-specific side-chain atoms."
    )

    sys.exit(1)


# =============================================================================
# STEP 14 — GLOBAL STRUCTURAL COMPARISON
# =============================================================================

subsection(
    "STEP 14 — WT vs D166V STRUCTURAL COMPARISON"
)

wt_atoms = collect_atoms_by_key(
    wt_structure,
    selected_chain_id
)

mutant_atoms = collect_atoms_by_key(
    mutant_structure,
    selected_chain_id
)


# -------------------------------------------------------------------------
# C-alpha RMSD
# -------------------------------------------------------------------------

wt_ca = []
mutant_ca = []

common_ca_residues = []

for position in sorted(
    set(wt_numbers)
    & set(
        get_structure_residue_numbers(
            mutant_chain
        )
    )
):

    wt_residue = get_residue_by_number(
        wt_chain,
        position
    )

    mut_residue = get_residue_by_number(
        mutant_chain,
        position
    )

    if wt_residue is None or mut_residue is None:
        continue

    if "CA" not in wt_residue:
        continue

    if "CA" not in mut_residue:
        continue

    wt_ca.append(
        wt_residue["CA"]
    )

    mutant_ca.append(
        mut_residue["CA"]
    )

    common_ca_residues.append(
        position
    )


ca_superimposer = Superimposer()

ca_superimposer.set_atoms(
    wt_ca,
    mutant_ca
)

ca_rmsd = float(
    ca_superimposer.rms
)


# -------------------------------------------------------------------------
# Backbone RMSD
# -------------------------------------------------------------------------

wt_backbone = []
mutant_backbone = []

common_backbone_atoms = []

for position in common_ca_residues:

    wt_residue = get_residue_by_number(
        wt_chain,
        position
    )

    mut_residue = get_residue_by_number(
        mutant_chain,
        position
    )

    for atom_name in [
        "N",
        "CA",
        "C"
    ]:

        if (
            atom_name not in wt_residue
            or atom_name not in mut_residue
        ):
            continue

        wt_backbone.append(
            wt_residue[atom_name]
        )

        mutant_backbone.append(
            mut_residue[atom_name]
        )

        common_backbone_atoms.append(
            (
                position,
                atom_name
            )
        )


backbone_superimposer = Superimposer()

backbone_superimposer.set_atoms(
    wt_backbone,
    mutant_backbone
)

backbone_rmsd = float(
    backbone_superimposer.rms
)


# -------------------------------------------------------------------------
# Local C-alpha RMSD
# -------------------------------------------------------------------------

local_positions = [
    position
    for position in common_ca_residues
    if abs(
        position - PRIMARY_POSITION
    ) <= 10
]

local_wt_ca = []
local_mut_ca = []

for position in local_positions:

    wt_residue = get_residue_by_number(
        wt_chain,
        position
    )

    mut_residue = get_residue_by_number(
        mutant_chain,
        position
    )

    if (
        "CA" in wt_residue
        and "CA" in mut_residue
    ):

        local_wt_ca.append(
            wt_residue["CA"]
        )

        local_mut_ca.append(
            mut_residue["CA"]
        )


local_superimposer = Superimposer()

local_superimposer.set_atoms(
    local_wt_ca,
    local_mut_ca
)

local_ca_rmsd = float(
    local_superimposer.rms
)


log(
    f"C-alpha RMSD: "
    f"{ca_rmsd:.4f} Å"
)

log(
    f"Backbone RMSD: "
    f"{backbone_rmsd:.4f} Å"
)

log(
    f"Local C-alpha RMSD "
    f"(±10 residues): "
    f"{local_ca_rmsd:.4f} Å"
)


# =============================================================================
# STEP 15 — LOCAL RESIDUE ENVIRONMENT
# =============================================================================

subsection(
    "STEP 15 — LOCAL D166 ENVIRONMENT"
)


def build_local_environment(
    chain,
    target_position,
    radius
):

    target = get_residue_by_number(
        chain,
        target_position
    )

    records = []

    if target is None:
        return records

    for residue in get_protein_residues(chain):

        position = residue_number(
            residue
        )

        if position == target_position:
            continue

        distance = minimum_heavy_atom_distance(
            target,
            residue
        )

        if np.isnan(distance):
            continue

        if distance <= radius:

            records.append(
                {
                    "Residue_Position": position,
                    "Residue_Name": residue.resname,
                    "Residue_AA": safe_seq1(
                        residue.resname
                    ),
                    "Minimum_Heavy_Atom_Distance_A": distance,
                }
            )

    records.sort(
        key=lambda x:
        x["Minimum_Heavy_Atom_Distance_A"]
    )

    return records


wt_local_records = build_local_environment(
    wt_chain,
    PRIMARY_POSITION,
    LOCAL_RADIUS
)

mutant_local_records = build_local_environment(
    mutant_chain,
    PRIMARY_POSITION,
    LOCAL_RADIUS
)


wt_local_df = pd.DataFrame(
    wt_local_records
)

mutant_local_df = pd.DataFrame(
    mutant_local_records
)


wt_local_file = (
    TABLE_DIR
    / "Table_S10_2_WT_D166_local_environment.csv"
)

mutant_local_file = (
    TABLE_DIR
    / "Table_S10_3_D166V_local_environment.csv"
)


wt_local_df.to_csv(
    wt_local_file,
    index=False
)

mutant_local_df.to_csv(
    mutant_local_file,
    index=False
)


log(
    f"WT neighboring residues within "
    f"{LOCAL_RADIUS:.1f} Å: "
    f"{len(wt_local_df)}"
)

log(
    f"D166V neighboring residues within "
    f"{LOCAL_RADIUS:.1f} Å: "
    f"{len(mutant_local_df)}"
)


# =============================================================================
# STEP 16 — CONTACT ANALYSIS
# =============================================================================

subsection(
    "STEP 16 — CONTACT ANALYSIS"
)


def build_contact_map(
    chain,
    target_position,
    cutoff
):

    target = get_residue_by_number(
        chain,
        target_position
    )

    records = []

    if target is None:
        return records

    for residue in get_protein_residues(chain):

        position = residue_number(
            residue
        )

        if position == target_position:
            continue

        distance = minimum_heavy_atom_distance(
            target,
            residue
        )

        if np.isnan(distance):
            continue

        if distance <= cutoff:

            records.append(
                {
                    "Residue_Position": position,
                    "Residue_Name": residue.resname,
                    "Residue_AA": safe_seq1(
                        residue.resname
                    ),
                    "Minimum_Heavy_Atom_Distance_A": distance,
                }
            )

    records.sort(
        key=lambda x:
        x["Minimum_Heavy_Atom_Distance_A"]
    )

    return records


wt_contacts = build_contact_map(
    wt_chain,
    PRIMARY_POSITION,
    CONTACT_DISTANCE
)

mutant_contacts = build_contact_map(
    mutant_chain,
    PRIMARY_POSITION,
    CONTACT_DISTANCE
)


wt_contact_dict = {
    item["Residue_Position"]:
    item["Minimum_Heavy_Atom_Distance_A"]
    for item in wt_contacts
}

mutant_contact_dict = {
    item["Residue_Position"]:
    item["Minimum_Heavy_Atom_Distance_A"]
    for item in mutant_contacts
}


all_contact_positions = sorted(
    set(wt_contact_dict)
    | set(mutant_contact_dict)
)


contact_records = []

for position in all_contact_positions:

    wt_distance = wt_contact_dict.get(
        position,
        np.nan
    )

    mutant_distance = mutant_contact_dict.get(
        position,
        np.nan
    )

    if position in wt_contact_dict:
        wt_present = True
    else:
        wt_present = False

    if position in mutant_contact_dict:
        mutant_present = True
    else:
        mutant_present = False

    if wt_present and mutant_present:

        change = (
            mutant_distance
            - wt_distance
        )

        category = "Retained"

    elif wt_present and not mutant_present:

        change = np.nan
        category = "Lost"

    else:

        change = np.nan
        category = "Gained"

    residue = (
        get_residue_by_number(
            wt_chain,
            position
        )
        or
        get_residue_by_number(
            mutant_chain,
            position
        )
    )

    contact_records.append(
        {
            "Residue_Position": position,
            "Residue_Name": residue.resname,
            "Residue_AA": safe_seq1(
                residue.resname
            ),
            "WT_Contact": wt_present,
            "D166V_Contact": mutant_present,
            "WT_Min_Distance_A": wt_distance,
            "D166V_Min_Distance_A": mutant_distance,
            "Distance_Change_A": change,
            "Contact_Status": category,
        }
    )


contact_df = pd.DataFrame(
    contact_records
)

contact_file = (
    TABLE_DIR
    / "Table_S10_4_D166V_contact_comparison.csv"
)

contact_df.to_csv(
    contact_file,
    index=False
)


lost_contacts = contact_df.loc[
    (
        contact_df["Contact_Status"]
        == "Lost"
    )
]

gained_contacts = contact_df.loc[
    (
        contact_df["Contact_Status"]
        == "Gained"
    )
]

retained_contacts = contact_df.loc[
    (
        contact_df["Contact_Status"]
        == "Retained"
    )
]


log(
    f"WT contacts within "
    f"{CONTACT_DISTANCE:.1f} Å: "
    f"{len(wt_contacts)}"
)

log(
    f"D166V contacts within "
    f"{CONTACT_DISTANCE:.1f} Å: "
    f"{len(mutant_contacts)}"
)

log(
    f"Contacts retained: "
    f"{len(retained_contacts)}"
)

log(
    f"Contacts lost: "
    f"{len(lost_contacts)}"
)

log(
    f"Contacts gained: "
    f"{len(gained_contacts)}"
)


# =============================================================================
# STEP 17 — POLAR CONTACT ANALYSIS
# =============================================================================

subsection(
    "STEP 17 — POLAR CONTACT ANALYSIS"
)

log(
    "NOTE: Hydrogen atoms are not explicitly "
    "modeled in the AlphaFold structure."
)

log(
    "Therefore these are reported as "
    "candidate polar heavy-atom contacts, "
    "not definitive hydrogen bonds."
)


POLAR_ELEMENTS = {
    "N",
    "O",
    "S"
}


def build_polar_contacts(
    chain,
    target_position,
    cutoff
):

    target = get_residue_by_number(
        chain,
        target_position
    )

    records = []

    if target is None:
        return records

    target_atoms = [
        atom
        for atom in target
        if atom.element.upper()
        in POLAR_ELEMENTS
    ]

    for residue in get_protein_residues(chain):

        position = residue_number(
            residue
        )

        if position == target_position:
            continue

        for target_atom in target_atoms:

            for partner_atom in residue:

                if (
                    partner_atom.element.upper()
                    not in POLAR_ELEMENTS
                ):
                    continue

                distance = np.linalg.norm(
                    target_atom.coord
                    - partner_atom.coord
                )

                if distance <= cutoff:

                    records.append(
                        {
                            "Target_Residue":
                                target_position,
                            "Target_Atom":
                                target_atom.name.strip(),
                            "Partner_Residue":
                                position,
                            "Partner_Residue_Name":
                                residue.resname,
                            "Partner_AA":
                                safe_seq1(
                                    residue.resname
                                ),
                            "Partner_Atom":
                                partner_atom.name.strip(),
                            "Distance_A":
                                float(distance),
                        }
                    )

    records.sort(
        key=lambda x:
        x["Distance_A"]
    )

    return records


wt_polar = build_polar_contacts(
    wt_chain,
    PRIMARY_POSITION,
    POLAR_CONTACT_DISTANCE
)

mutant_polar = build_polar_contacts(
    mutant_chain,
    PRIMARY_POSITION,
    POLAR_CONTACT_DISTANCE
)


wt_polar_df = pd.DataFrame(
    wt_polar
)

mutant_polar_df = pd.DataFrame(
    mutant_polar
)


wt_polar_file = (
    TABLE_DIR
    / "Table_S10_5_WT_D166_candidate_polar_contacts.csv"
)

mutant_polar_file = (
    TABLE_DIR
    / "Table_S10_6_D166V_candidate_polar_contacts.csv"
)


wt_polar_df.to_csv(
    wt_polar_file,
    index=False
)

mutant_polar_df.to_csv(
    mutant_polar_file,
    index=False
)


log(
    f"WT candidate polar contacts: "
    f"{len(wt_polar_df)}"
)

log(
    f"D166V candidate polar contacts: "
    f"{len(mutant_polar_df)}"
)


# =============================================================================
# STEP 18 — PHYSICOCHEMICAL CONTEXT
# =============================================================================

subsection(
    "STEP 18 — LOCAL PHYSICOCHEMICAL CONTEXT"
)

physicochemical = {

    "Wild_Type_Amino_Acid":
        "Aspartate (D)",

    "Mutant_Amino_Acid":
        "Valine (V)",

    "Wild_Type_Charge":
        "Negative",

    "Mutant_Charge":
        "Neutral",

    "Wild_Type_Polarity":
        "Polar",

    "Mutant_Polarity":
        "Non-polar",

    "Wild_Type_Functional_Group":
        "Carboxylate",

    "Mutant_Functional_Group":
        "Hydrocarbon side chain",

    "Wild_Type_Residue_Position":
        PRIMARY_POSITION,

    "Mutation":
        PRIMARY_PROTEIN_VARIANT,

    "Interpretation":
        (
            "D166V replaces an acidic, polar, "
            "negatively charged side chain with "
            "a neutral, non-polar valine side chain."
        ),
}


physchem_df = pd.DataFrame(
    [
        {
            "Feature": key,
            "Value": value
        }
        for key, value in physicochemical.items()
    ]
)


physchem_file = (
    TABLE_DIR
    / "Table_S10_7_D166V_physicochemical_context.csv"
)

physchem_df.to_csv(
    physchem_file,
    index=False
)


# =============================================================================
# STEP 19 — STRUCTURAL METRICS TABLE
# =============================================================================

subsection(
    "STEP 19 — STRUCTURAL METRICS SUMMARY"
)

metrics = {

    "Protein": "PNPLA3",

    "UniProt": EXPECTED_UNIPROT,

    "Mutation": PRIMARY_PROTEIN_VARIANT,

    "Protein_Change": PRIMARY_PROTEIN_CHANGE,

    "VariationID": PRIMARY_VARIATION_ID,

    "Residue_Position": PRIMARY_POSITION,

    "WT_Amino_Acid": PRIMARY_REFERENCE_AA,

    "Mutant_Amino_Acid": PRIMARY_ALTERNATE_AA,

    "Structure_Source": selected_source,

    "Structure_ID": selected_structure_id,

    "Chain": selected_chain_id,

    "Structure_Residues":
        len(wt_residues),

    "Canonical_Sequence_Length":
        len(sequence),

    "Coverage_Percent":
        coverage_percent,

    "Mean_pLDDT":
        mean_plddt,

    "Residue_166_pLDDT":
        target_plddt,

    "C_alpha_RMSD_A":
        ca_rmsd,

    "Backbone_RMSD_A":
        backbone_rmsd,

    "Local_C_alpha_RMSD_A":
        local_ca_rmsd,

    "WT_Local_Environment_Count":
        len(wt_local_df),

    "D166V_Local_Environment_Count":
        len(mutant_local_df),

    "WT_Contact_Count":
        len(wt_contacts),

    "D166V_Contact_Count":
        len(mutant_contacts),

    "Contacts_Retained":
        len(retained_contacts),

    "Contacts_Lost":
        len(lost_contacts),

    "Contacts_Gained":
        len(gained_contacts),

    "WT_Candidate_Polar_Contacts":
        len(wt_polar_df),

    "D166V_Candidate_Polar_Contacts":
        len(mutant_polar_df),
}


metrics_df = pd.DataFrame(
    [
        {
            "Metric": key,
            "Value": value
        }
        for key, value in metrics.items()
    ]
)


metrics_file = (
    TABLE_DIR
    / "Table_S10_8_D166V_structural_metrics.csv"
)

metrics_df.to_csv(
    metrics_file,
    index=False
)


# =============================================================================
# STEP 20 — LOCAL DISTANCE COMPARISON
# =============================================================================

subsection(
    "STEP 20 — LOCAL DISTANCE COMPARISON"
)

local_distance_records = []

for position in sorted(
    set(
        wt_local_df["Residue_Position"]
        if not wt_local_df.empty
        else []
    )
    |
    set(
        mutant_local_df["Residue_Position"]
        if not mutant_local_df.empty
        else []
    )
):

    wt_row = wt_local_df.loc[
        wt_local_df["Residue_Position"]
        == position
    ]

    mutant_row = mutant_local_df.loc[
        mutant_local_df["Residue_Position"]
        == position
    ]

    wt_distance = (
        float(
            wt_row.iloc[0][
                "Minimum_Heavy_Atom_Distance_A"
            ]
        )
        if not wt_row.empty
        else np.nan
    )

    mutant_distance = (
        float(
            mutant_row.iloc[0][
                "Minimum_Heavy_Atom_Distance_A"
            ]
        )
        if not mutant_row.empty
        else np.nan
    )

    residue = (
        get_residue_by_number(
            wt_chain,
            position
        )
        or
        get_residue_by_number(
            mutant_chain,
            position
        )
    )

    local_distance_records.append(
        {
            "Residue_Position": position,
            "Residue_Name": residue.resname,
            "Residue_AA": safe_seq1(
                residue.resname
            ),
            "WT_Distance_A": wt_distance,
            "D166V_Distance_A": mutant_distance,
            "Distance_Change_A":
                (
                    mutant_distance
                    - wt_distance
                    if not np.isnan(wt_distance)
                    and not np.isnan(mutant_distance)
                    else np.nan
                ),
        }
    )


local_distance_df = pd.DataFrame(
    local_distance_records
)

local_distance_file = (
    TABLE_DIR
    / "Table_S10_9_local_distance_comparison.csv"
)

local_distance_df.to_csv(
    local_distance_file,
    index=False
)


# =============================================================================
# STEP 21 — FIGURE 1: RMSD
# =============================================================================

subsection(
    "STEP 21 — GENERATING QUANTITATIVE STRUCTURAL FIGURES"
)

rmsd_figure = (
    FIGURE_DIR
    / "Figure_S10_1_WT_D166V_RMSD_comparison.png"
)

labels = [
    "Cα RMSD",
    "Backbone RMSD",
    "Local Cα RMSD"
]

values = [
    ca_rmsd,
    backbone_rmsd,
    local_ca_rmsd
]

plt.figure(
    figsize=(8, 6),
    dpi=300
)

plt.bar(
    labels,
    values
)

plt.ylabel(
    "RMSD (Å)"
)

plt.title(
    "WT PNPLA3 vs D166V Structural RMSD"
)

plt.tight_layout()

plt.savefig(
    rmsd_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# STEP 22 — FIGURE 2: LOCAL DISTANCES
# =============================================================================

local_distance_figure = (
    FIGURE_DIR
    / "Figure_S10_2_D166V_local_distance_comparison.png"
)

if not local_distance_df.empty:

    plt.figure(
        figsize=(10, 6),
        dpi=300
    )

    x = local_distance_df[
        "Residue_Position"
    ].values

    plt.plot(
        x,
        local_distance_df[
            "WT_Distance_A"
        ].values,
        marker="o",
        label="WT D166"
    )

    plt.plot(
        x,
        local_distance_df[
            "D166V_Distance_A"
        ].values,
        marker="s",
        label="D166V"
    )

    plt.axvline(
        PRIMARY_POSITION,
        linestyle="--",
        linewidth=1.5
    )

    plt.xlabel(
        "Residue position"
    )

    plt.ylabel(
        "Minimum heavy-atom distance (Å)"
    )

    plt.title(
        "Local Structural Environment of PNPLA3 Residue 166"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        local_distance_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# STEP 23 — FIGURE 3: CONTACT CHANGES
# =============================================================================

contact_figure = (
    FIGURE_DIR
    / "Figure_S10_3_D166V_contact_comparison.png"
)

contact_counts = [
    len(wt_contacts),
    len(mutant_contacts),
]

plt.figure(
    figsize=(7, 6),
    dpi=300
)

plt.bar(
    [
        "WT D166",
        "D166V"
    ],
    contact_counts
)

plt.ylabel(
    f"Residues within {CONTACT_DISTANCE:.1f} Å"
)

plt.title(
    "Local Contact Comparison"
)

plt.tight_layout()

plt.savefig(
    contact_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# STEP 24 — FIGURE 4: POLAR CONTACTS
# =============================================================================

polar_figure = (
    FIGURE_DIR
    / "Figure_S10_4_D166V_candidate_polar_contact_comparison.png"
)

plt.figure(
    figsize=(7, 6),
    dpi=300
)

plt.bar(
    [
        "WT D166",
        "D166V"
    ],
    [
        len(wt_polar_df),
        len(mutant_polar_df)
    ]
)

plt.ylabel(
    "Candidate polar heavy-atom contacts"
)

plt.title(
    "Candidate Polar Contact Comparison"
)

plt.tight_layout()

plt.savefig(
    polar_figure,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# =============================================================================
# STEP 25 — FIGURE 5: LOCAL pLDDT
# =============================================================================

plddt_figure = (
    FIGURE_DIR
    / "Figure_S10_5_local_pLDDT_around_D166.png"
)

plddt_records = []

for position in range(
    max(1, PRIMARY_POSITION - 20),
    min(
        len(sequence),
        PRIMARY_POSITION + 20
    ) + 1
):

    residue = get_residue_by_number(
        wt_chain,
        position
    )

    if residue is None:
        continue

    plddt = get_residue_plddt(
        residue
    )

    plddt_records.append(
        {
            "Position": position,
            "pLDDT": plddt
        }
    )


plddt_df = pd.DataFrame(
    plddt_records
)

plddt_file = (
    TABLE_DIR
    / "Table_S10_10_local_pLDDT.csv"
)

plddt_df.to_csv(
    plddt_file,
    index=False
)


if not plddt_df.empty:

    plt.figure(
        figsize=(10, 6),
        dpi=300
    )

    plt.plot(
        plddt_df["Position"],
        plddt_df["pLDDT"],
        marker="o"
    )

    plt.axvline(
        PRIMARY_POSITION,
        linestyle="--",
        linewidth=1.5
    )

    plt.xlabel(
        "Residue position"
    )

    plt.ylabel(
        "pLDDT"
    )

    plt.ylim(
        0,
        100
    )

    plt.title(
        "AlphaFold Confidence Around PNPLA3 D166"
    )

    plt.tight_layout()

    plt.savefig(
        plddt_figure,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# STEP 26 — PYMOl 3D FIGURES
# =============================================================================

subsection(
    "STEP 26 — GENERATING PyMOL 3D STRUCTURAL FIGURES"
)


wt_3d_figure = (
    FIGURE_DIR
    / "Figure_S10_6_PNPLA3_WT_D166_3D.png"
)

mutant_3d_figure = (
    FIGURE_DIR
    / "Figure_S10_7_PNPLA3_D166V_3D.png"
)

comparison_3d_figure = (
    FIGURE_DIR
    / "Figure_S10_8_PNPLA3_WT_vs_D166V_3D.png"
)


render_script = f"""
import pymol

pymol.finish_launching(
    ["pymol", "-cq"]
)

from pymol import cmd

cmd.bg_color("white")

# -------------------------------------------------------------------------
# WT
# -------------------------------------------------------------------------

cmd.load(
    r"{wt_pdb_file}",
    "WT"
)

cmd.hide(
    "everything",
    "WT"
)

cmd.show(
    "cartoon",
    "WT"
)

cmd.color(
    "cyan",
    "WT"
)

cmd.show(
    "sticks",
    f"WT and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.color(
    "red",
    f"WT and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.zoom(
    f"WT and chain {selected_chain_id} and resi {PRIMARY_POSITION}",
    10
)

cmd.png(
    r"{wt_3d_figure}",
    width=1800,
    height=1400,
    dpi=300,
    ray=1
)

cmd.delete("all")

# -------------------------------------------------------------------------
# MUTANT
# -------------------------------------------------------------------------

cmd.load(
    r"{mutant_pdb_file}",
    "MUTANT"
)

cmd.hide(
    "everything",
    "MUTANT"
)

cmd.show(
    "cartoon",
    "MUTANT"
)

cmd.color(
    "cyan",
    "MUTANT"
)

cmd.show(
    "sticks",
    f"MUTANT and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.color(
    "yellow",
    f"MUTANT and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.zoom(
    f"MUTANT and chain {selected_chain_id} and resi {PRIMARY_POSITION}",
    10
)

cmd.png(
    r"{mutant_3d_figure}",
    width=1800,
    height=1400,
    dpi=300,
    ray=1
)

cmd.delete("all")

# -------------------------------------------------------------------------
# COMPARISON
# -------------------------------------------------------------------------

cmd.load(
    r"{wt_pdb_file}",
    "WT"
)

cmd.load(
    r"{mutant_pdb_file}",
    "D166V"
)

cmd.hide(
    "everything",
    "WT D166V"
)

cmd.show(
    "cartoon",
    "WT"
)

cmd.show(
    "cartoon",
    "D166V"
)

cmd.color(
    "cyan",
    "WT"
)

cmd.color(
    "green",
    "D166V"
)

cmd.show(
    "sticks",
    f"WT and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.show(
    "sticks",
    f"D166V and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.color(
    "red",
    f"WT and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.color(
    "yellow",
    f"D166V and chain {selected_chain_id} and resi {PRIMARY_POSITION}"
)

cmd.zoom(
    f"(WT or D166V) and chain {selected_chain_id} and resi {PRIMARY_POSITION}",
    10
)

cmd.png(
    r"{comparison_3d_figure}",
    width=2000,
    height=1500,
    dpi=300,
    ray=1
)

cmd.quit()
"""


render_ok = run_pymol_python(
    render_script,
    "render_step10_3d_figures.py",
    timeout=600
)


# =============================================================================
# STEP 27 — FINAL QC
# =============================================================================

subsection(
    "STEP 27 — FINAL STEP 10 QUALITY CONTROL"
)


qc = {

    "Step9F_File_Exists":
        step9f_file.exists(),

    "Step9F_Variant_Count_82":
        len(df) == 82,

    "Structural_Candidate_Count_9":
        len(structural_df) == 9,

    "Primary_VariationID":
        int(primary["VariationID"])
        == PRIMARY_VARIATION_ID,

    "Primary_D166V":
        str(primary["Protein_Variant"]).strip()
        == "D166V",

    "Primary_UniProt_Q9NST1":
        str(primary["UniProt_ID"]).strip()
        == EXPECTED_UNIPROT,

    "Primary_Position_166":
        int(float(primary["Protein_Position"]))
        == 166,

    "UniProt_Residue_166_D":
        uniprot_residue == "D",

    "WT_Structure_Exists":
        wt_structure_file.exists(),

    "WT_Residue_166_D":
        wt_target_aa == "D",

    "WT_Coverage_Valid":
        coverage_percent > 0,

    "PyMOL_Python_Found":
        pymol_python is not None,

    "WT_PDB_Exists":
        wt_pdb_file.exists(),

    "D166V_Mutant_Exists":
        mutant_pdb_file.exists(),

    "Mutant_Residue_166_V":
        mutant_target_aa == "V",

    "Mutant_VAL_Sidechain_Valid":
        val_sidechain_pass,

    "Mutant_ASP_Sidechain_Removed":
        asp_removed_pass,

    "Structural_Metrics_Exists":
        metrics_file.exists(),

    "Local_Environment_Exists":
        wt_local_file.exists()
        and mutant_local_file.exists(),

    "Contact_Comparison_Exists":
        contact_file.exists(),

    "RMSD_Figure_Exists":
        rmsd_figure.exists(),

    "3D_Figures_Generated":
        render_ok
        and wt_3d_figure.exists()
        and mutant_3d_figure.exists()
        and comparison_3d_figure.exists(),
}


overall_qc_pass = all(
    qc.values()
)


for label, passed in qc.items():

    status(
        label,
        passed
    )


# =============================================================================
# FINAL QC JSON
# =============================================================================

qc_report = {

    "project":
        PROJECT_NAME,

    "step":
        10,

    "primary_candidate":
        PRIMARY_PROTEIN_VARIANT,

    "variation_id":
        PRIMARY_VARIATION_ID,

    "uniprot":
        EXPECTED_UNIPROT,

    "structure_source":
        selected_source,

    "structure_id":
        selected_structure_id,

    "chain":
        selected_chain_id,

    "structure_file":
        wt_structure_file,

    "mutant_file":
        mutant_pdb_file,

    "pymol_python":
        pymol_python,

    "pymol_version":
        pymol_version,

    "metrics": {
        "coverage_percent":
            coverage_percent,

        "mean_pLDDT":
            mean_plddt,

        "residue_166_pLDDT":
            target_plddt,

        "ca_rmsd_A":
            ca_rmsd,

        "backbone_rmsd_A":
            backbone_rmsd,

        "local_ca_rmsd_A":
            local_ca_rmsd,

        "wt_contacts":
            len(wt_contacts),

        "mutant_contacts":
            len(mutant_contacts),

        "contacts_lost":
            len(lost_contacts),

        "contacts_gained":
            len(gained_contacts),

        "contacts_retained":
            len(retained_contacts),

        "wt_candidate_polar_contacts":
            len(wt_polar_df),

        "mutant_candidate_polar_contacts":
            len(mutant_polar_df),
    },

    "qc":
        qc,

    "overall_qc":
        "PASS"
        if overall_qc_pass
        else "FAIL",

    "note":
        (
            "D166V was generated using the PyMOL "
            "mutagenesis wizard. The AlphaFold "
            "structure is a predicted model and "
            "the mutant is a side-chain-remodeled "
            "model, not an energy-minimized or "
            "molecular-dynamics-equilibrated "
            "structure. Polar contacts are "
            "reported as candidate heavy-atom "
            "polar contacts because explicit "
            "hydrogens and hydrogen-bond geometry "
            "were not modeled."
        ),
}


qc_json_file = (
    QC_DIR
    / "Step10_final_QC_report.json"
)

write_json(
    qc_json_file,
    qc_report
)


# =============================================================================
# FINAL SUMMARY
# =============================================================================

summary_file = (
    STEP10_DIR
    / "Step10_final_summary.txt"
)


summary_lines = [

    "STEP 10 — PNPLA3 STRUCTURAL ANALYSIS",
    "=" * 70,
    "",
    f"Primary candidate: {PRIMARY_PROTEIN_CHANGE} ({PRIMARY_PROTEIN_VARIANT})",
    f"VariationID: {PRIMARY_VARIATION_ID}",
    f"UniProt: {EXPECTED_UNIPROT}",
    f"Residue: {PRIMARY_POSITION}",
    "",
    "STRUCTURE",
    "-" * 70,
    f"Source: {selected_source}",
    f"Structure ID: {selected_structure_id}",
    f"Chain: {selected_chain_id}",
    f"Coverage: {coverage_percent:.2f}%",
    f"Mean pLDDT: {mean_plddt:.2f}",
    f"Residue 166 pLDDT: {target_plddt:.2f}",
    "",
    "MUTATION",
    "-" * 70,
    "WT residue 166: ASP (D)",
    "Mutant residue 166: VAL (V)",
    "Mutation method: PyMOL mutagenesis wizard",
    "",
    "STRUCTURAL COMPARISON",
    "-" * 70,
    f"C-alpha RMSD: {ca_rmsd:.4f} Å",
    f"Backbone RMSD: {backbone_rmsd:.4f} Å",
    f"Local C-alpha RMSD: {local_ca_rmsd:.4f} Å",
    "",
    "LOCAL ENVIRONMENT",
    "-" * 70,
    f"WT local residues: {len(wt_local_df)}",
    f"D166V local residues: {len(mutant_local_df)}",
    "",
    "CONTACTS",
    "-" * 70,
    f"WT contacts: {len(wt_contacts)}",
    f"D166V contacts: {len(mutant_contacts)}",
    f"Retained: {len(retained_contacts)}",
    f"Lost: {len(lost_contacts)}",
    f"Gained: {len(gained_contacts)}",
    "",
    "POLAR CONTACTS",
    "-" * 70,
    f"WT candidate polar contacts: {len(wt_polar_df)}",
    f"D166V candidate polar contacts: {len(mutant_polar_df)}",
    "",
    "PHYSICOCHEMICAL CHANGE",
    "-" * 70,
    "Aspartate → Valine",
    "Negative/polar → Neutral/non-polar",
    "Carboxylate side chain → Hydrocarbon side chain",
    "",
    "QUALITY CONTROL",
    "-" * 70,
    f"Overall Step 10 QC: {'PASS' if overall_qc_pass else 'FAIL'}",
    "",
    "IMPORTANT INTERPRETATION NOTE",
    "-" * 70,
    "The D166V model is a PyMOL side-chain-remodeled structure.",
    "It is not an energy-minimized structure or molecular-dynamics result.",
    "The AlphaFold structure is a predicted structural model.",
    "Polar contacts are candidate heavy-atom polar contacts rather",
    "than definitive hydrogen bonds because explicit hydrogens were",
    "not modeled.",
    "",
]


summary_file.write_text(
    "\n".join(summary_lines),
    encoding="utf-8"
)


# =============================================================================
# FINAL OUTPUT INVENTORY
# =============================================================================

subsection(
    "STEP 28 — FINAL OUTPUT INVENTORY"
)

output_files = {

    "WT_structure":
        wt_structure_file,

    "WT_PDB":
        wt_pdb_file,

    "D166V_mutant":
        mutant_pdb_file,

    "Structural_candidates":
        structural_output,

    "Structural_metrics":
        metrics_file,

    "WT_local_environment":
        wt_local_file,

    "D166V_local_environment":
        mutant_local_file,

    "Contact_comparison":
        contact_file,

    "WT_polar_contacts":
        wt_polar_file,

    "D166V_polar_contacts":
        mutant_polar_file,

    "Physicochemical_context":
        physchem_file,

    "Local_distance_comparison":
        local_distance_file,

    "Local_pLDDT":
        plddt_file,

    "RMSD_figure":
        rmsd_figure,

    "Local_distance_figure":
        local_distance_figure,

    "Contact_figure":
        contact_figure,

    "Polar_contact_figure":
        polar_figure,

    "pLDDT_figure":
        plddt_figure,

    "WT_3D_figure":
        wt_3d_figure,

    "D166V_3D_figure":
        mutant_3d_figure,

    "WT_vs_D166V_3D_figure":
        comparison_3d_figure,

    "QC_report":
        qc_json_file,

    "Final_summary":
        summary_file,
}


for label, path in output_files.items():

    exists = Path(path).exists()

    log(
        f"{label}: "
        f"{'CREATED' if exists else 'MISSING'}"
    )

    if exists:
        log(
            f"    {path}"
        )


# =============================================================================
# FINAL STATUS
# =============================================================================

section(
    "STEP 10 COMPLETE"
)

if overall_qc_pass:

    log("")
    log(
        "FINAL STEP 10 QC: PASS"
    )

    log("")
    log(
        "Primary structural model:"
    )

    log(
        "    PNPLA3 p.Asp166Val (D166V)"
    )

    log("")
    log(
        "WT structure:"
    )

    log(
        f"    {wt_structure_file}"
    )

    log("")
    log(
        "D166V mutant:"
    )

    log(
        f"    {mutant_pdb_file}"
    )

    log("")
    log(
        "Structural metrics:"
    )

    log(
        f"    C-alpha RMSD = "
        f"{ca_rmsd:.4f} Å"
    )

    log(
        f"    Backbone RMSD = "
        f"{backbone_rmsd:.4f} Å"
    )

    log(
        f"    Local C-alpha RMSD = "
        f"{local_ca_rmsd:.4f} Å"
    )

    log("")
    log(
        "Step 10 outputs are ready for downstream "
        "structural interpretation."
    )

else:

    log("")
    log(
        "FINAL STEP 10 QC: FAIL"
    )

    log(
        "Review:"
    )

    log(
        f"    {qc_json_file}"
    )

    sys.exit(1)