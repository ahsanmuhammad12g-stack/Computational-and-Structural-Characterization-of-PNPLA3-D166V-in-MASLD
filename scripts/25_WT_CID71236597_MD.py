from pathlib import Path
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import openmm
from openmm import app, unit
from openmm import XmlSerializer

from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from pdbfixer import PDBFixer

from rdkit import Chem
from rdkit.Geometry import Point3D


# =============================================================================
# STEP 25 — WT PNPLA3 + CID 71236597 MOLECULAR DYNAMICS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

WT_RECEPTOR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP10_STRUCTURAL_ANALYSIS"
    / "structures"
    / "WT"
    / "PNPLA3_WT.pdb"
)

PREVIOUS_PREPARED_PROTEIN = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP25_WT_CID71236597_MD"
    / "preparation"
    / "PNPLA3_WT_prepared.pdb"
)

LIGAND_PDBQT = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
    / "ligands"
    / "CID_71236597.pdbqt"
)

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP25_WT_CID71236597_MD"
)

PREP_DIR = OUT_DIR / "preparation"
RUN_DIR = OUT_DIR / "md"
TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
QC_DIR = OUT_DIR / "qc"

for folder in [
    OUT_DIR,
    PREP_DIR,
    RUN_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# CONSTANTS
# =============================================================================

CID = 71236597

EXPECTED_PROTEIN_RESIDUES = 481

EXPECTED_LIGAND_HEAVY_ATOMS = 33
EXPECTED_LIGAND_HYDROGENS = 17
EXPECTED_LIGAND_TOTAL_ATOMS = 50

EXPECTED_D166 = 166

TEMPERATURE = 300.0 * unit.kelvin
PRESSURE = 1.0 * unit.atmosphere

TIMESTEP = 0.002 * unit.picoseconds

NVT_PS = 100.0
NPT_PS = 500.0
PRODUCTION_NS = 10.0

REPORT_PS = 10.0
DCD_REPORT_PS = 20.0
CHECKPOINT_PS = 100.0

SOLVENT_PADDING_ANGSTROM = 10.0

IONIC_STRENGTH = 0.15 * unit.molar

NONBONDED_CUTOFF = 1.0 * unit.nanometer
EWALD_ERROR_TOLERANCE = 0.0005

WATER_LIGAND_CUTOFF = 1.5 * unit.angstrom

SEED = 20260910

NVT_STEPS = int(
    round(
        NVT_PS
        / TIMESTEP.value_in_unit(
            unit.picoseconds
        )
    )
)

NPT_STEPS = int(
    round(
        NPT_PS
        / TIMESTEP.value_in_unit(
            unit.picoseconds
        )
    )
)

PRODUCTION_STEPS = int(
    round(
        PRODUCTION_NS
        * 1000.0
        / TIMESTEP.value_in_unit(
            unit.picoseconds
        )
    )
)

REPORT_INTERVAL_STEPS = max(
    1,
    int(
        round(
            REPORT_PS
            / TIMESTEP.value_in_unit(
                unit.picoseconds
            )
        )
    )
)

DCD_INTERVAL_STEPS = max(
    1,
    int(
        round(
            DCD_REPORT_PS
            / TIMESTEP.value_in_unit(
                unit.picoseconds
            )
        )
    )
)

CHECKPOINT_INTERVAL_STEPS = max(
    1,
    int(
        round(
            CHECKPOINT_PS
            / TIMESTEP.value_in_unit(
                unit.picoseconds
            )
        )
    )
)


# =============================================================================
# EXACT CID 71236597 SMILES
# =============================================================================

LIGAND_SMILES = (
    "CN1CCCn2c1cc("
    "OCc1cc(F)c("
    "Oc3ccc(C(F)(F)F)nc3"
    ")c(F)c1"
    ")nc2=O"
)


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def fail(message):
    print()
    print("=" * 78)
    print("ERROR")
    print("=" * 78)
    print(message)
    print()
    sys.exit(1)


def count_atoms(topology):
    return sum(
        1 for _ in topology.atoms()
    )


def count_residues(topology):
    return sum(
        1 for _ in topology.residues()
    )


def positions_to_numpy(positions):
    return np.asarray(
        positions.value_in_unit(
            unit.angstrom
        ),
        dtype=float,
    )


def find_ca_coordinate(
    topology,
    positions,
    residue_number,
):
    xyz = positions_to_numpy(
        positions
    )

    for residue in topology.residues():

        try:
            rid = int(residue.id)
        except Exception:
            continue

        if rid != residue_number:
            continue

        for atom in residue.atoms():

            if atom.name == "CA":
                return xyz[atom.index]

    fail(
        f"CA atom for residue {residue_number} "
        f"was not found."
    )


# =============================================================================
# PDBQT PARSING
# =============================================================================

def parse_pdbqt_atoms(path):

    atoms = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for line in handle:

            if not (
                line.startswith("ATOM")
                or line.startswith("HETATM")
            ):
                continue

            try:

                atom_id = int(
                    line[6:11].strip()
                )

                x = float(
                    line[30:38]
                )

                y = float(
                    line[38:46]
                )

                z = float(
                    line[46:54]
                )

                atoms[atom_id] = {
                    "x": x,
                    "y": y,
                    "z": z,
                }

            except Exception:
                continue

    if not atoms:
        fail(
            f"No ligand atoms found in:\n{path}"
        )

    return atoms


def parse_smiles_idx(path):

    mapping = {}

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for line in handle:

            if not line.startswith(
                "REMARK SMILES IDX"
            ):
                continue

            parts = line.strip().split()

            values = parts[3:]

            if len(values) % 2 != 0:
                continue

            for i in range(
                0,
                len(values),
                2,
            ):

                try:

                    pdb_atom_id = int(
                        values[i]
                    )

                    smiles_index = int(
                        values[i + 1]
                    )

                    mapping[
                        smiles_index
                    ] = pdb_atom_id

                except ValueError:
                    continue

    if not mapping:
        fail(
            "No REMARK SMILES IDX mapping was found "
            "in the ligand PDBQT."
        )

    return mapping


# =============================================================================
# BUILD EXACT DOCKED LIGAND
# =============================================================================

def build_exact_docked_ligand():

    print(
        "Building exact CID 71236597 from docking pose..."
    )
    print()

    pdbqt_atoms = parse_pdbqt_atoms(
        LIGAND_PDBQT
    )

    mapping = parse_smiles_idx(
        LIGAND_PDBQT
    )

    rd_mol = Chem.MolFromSmiles(
        LIGAND_SMILES
    )

    if rd_mol is None:
        fail(
            "RDKit could not parse the CID 71236597 SMILES."
        )

    heavy_count = (
        rd_mol.GetNumAtoms()
    )

    if heavy_count != EXPECTED_LIGAND_HEAVY_ATOMS:
        fail(
            f"Expected {EXPECTED_LIGAND_HEAVY_ATOMS} "
            f"heavy atoms but found {heavy_count}."
        )

    conformer = Chem.Conformer(
        heavy_count
    )

    for rd_index in range(
        heavy_count
    ):

        smiles_index = (
            rd_index + 1
        )

        if smiles_index not in mapping:
            fail(
                f"Missing SMILES atom index "
                f"{smiles_index}."
            )

        pdb_atom_id = mapping[
            smiles_index
        ]

        if pdb_atom_id not in pdbqt_atoms:
            fail(
                f"Mapped PDBQT atom {pdb_atom_id} "
                f"was not found."
            )

        atom = pdbqt_atoms[
            pdb_atom_id
        ]

        conformer.SetAtomPosition(
            rd_index,
            Point3D(
                atom["x"],
                atom["y"],
                atom["z"],
            ),
        )

    rd_mol.AddConformer(
        conformer,
        assignId=True,
    )

    rd_h = Chem.AddHs(
        rd_mol,
        addCoords=True,
    )

    total_atoms = rd_h.GetNumAtoms()

    hydrogen_count = (
        total_atoms - heavy_count
    )

    if hydrogen_count != EXPECTED_LIGAND_HYDROGENS:
        fail(
            f"Expected {EXPECTED_LIGAND_HYDROGENS} "
            f"hydrogens but RDKit generated "
            f"{hydrogen_count}."
        )

    if total_atoms != EXPECTED_LIGAND_TOTAL_ATOMS:
        fail(
            f"Expected {EXPECTED_LIGAND_TOTAL_ATOMS} "
            f"total ligand atoms but found "
            f"{total_atoms}."
        )

    ligand = Molecule.from_rdkit(
        rd_h,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )

    if len(ligand.atoms) != EXPECTED_LIGAND_TOTAL_ATOMS:
        fail(
            "OpenFF ligand atom count does not match "
            "the expected 50 atoms."
        )

    rd_positions = np.asarray(
        rd_h.GetConformer().GetPositions(),
        dtype=float,
    )

    ligand_topology = (
        ligand
        .to_topology()
        .to_openmm()
    )

    ligand_positions = (
        rd_positions
        * unit.angstrom
    )

    ligand_residues = list(
        ligand_topology.residues()
    )

    if len(ligand_residues) != 1:
        fail(
            "Expected one ligand residue."
        )

    ligand_residues[0].name = "LIG"

    # Exact coordinate validation against PDBQT
    for rd_index in range(
        heavy_count
    ):

        smiles_index = (
            rd_index + 1
        )

        pdb_atom_id = mapping[
            smiles_index
        ]

        expected = np.array(
            [
                pdbqt_atoms[
                    pdb_atom_id
                ]["x"],
                pdbqt_atoms[
                    pdb_atom_id
                ]["y"],
                pdbqt_atoms[
                    pdb_atom_id
                ]["z"],
            ],
            dtype=float,
        )

        observed = (
            rd_positions[
                rd_index
            ]
        )

        error = np.linalg.norm(
            observed - expected
        )

        if error > 1e-6:
            fail(
                f"Ligand coordinate mismatch at "
                f"SMILES atom {smiles_index}: "
                f"{error:.6f} Å."
            )

    ligand_pdb = (
        PREP_DIR
        / "CID_71236597_exact_docked_with_H.pdb"
    )

    with open(
        ligand_pdb,
        "w",
        encoding="utf-8",
    ) as handle:

        app.PDBFile.writeFile(
            ligand_topology,
            ligand_positions,
            handle,
            keepIds=True,
        )

    print(
        f"Ligand heavy atoms : {heavy_count}"
    )

    print(
        f"Ligand hydrogens   : {hydrogen_count}"
    )

    print(
        f"Ligand total atoms : {total_atoms}"
    )

    print(
        f"Ligand SMILES      : {LIGAND_SMILES}"
    )

    print()

    return (
        ligand,
        ligand_topology,
        ligand_positions,
    )


# =============================================================================
# WT PROTEIN PREPARATION
# =============================================================================

def prepare_wt_protein():

    if PREVIOUS_PREPARED_PROTEIN.exists():

        print(
            "Using previously validated prepared WT protein..."
        )

        pdb = app.PDBFile(
            str(PREVIOUS_PREPARED_PROTEIN)
        )

        residues = count_residues(
            pdb.topology
        )

        atoms = count_atoms(
            pdb.topology
        )

        print(
            f"Prepared protein residues : {residues}"
        )

        print(
            f"Prepared protein atoms    : {atoms}"
        )

        print()

        if residues != EXPECTED_PROTEIN_RESIDUES:
            fail(
                f"Expected {EXPECTED_PROTEIN_RESIDUES} "
                f"protein residues but found {residues}."
            )

        return (
            pdb.topology,
            pdb.positions,
        )

    print(
        "Preparing WT PNPLA3..."
    )
    print()

    fixer = PDBFixer(
        filename=str(
            WT_RECEPTOR
        )
    )

    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(
        7.4
    )

    topology = fixer.topology
    positions = fixer.positions

    residues = count_residues(
        topology
    )

    atoms = count_atoms(
        topology
    )

    if residues != EXPECTED_PROTEIN_RESIDUES:
        fail(
            f"Expected {EXPECTED_PROTEIN_RESIDUES} "
            f"protein residues but found {residues}."
        )

    with open(
        PREVIOUS_PREPARED_PROTEIN,
        "w",
        encoding="utf-8",
    ) as handle:

        app.PDBFile.writeFile(
            topology,
            positions,
            handle,
            keepIds=True,
        )

    print(
        f"Prepared protein residues : {residues}"
    )

    print(
        f"Prepared protein atoms    : {atoms}"
    )

    print()

    return (
        topology,
        positions,
    )


# =============================================================================
# RECTANGULAR BOX
# =============================================================================

def calculate_box_lengths(
    positions,
):

    xyz = positions_to_numpy(
        positions
    )

    mins = xyz.min(
        axis=0
    )

    maxs = xyz.max(
        axis=0
    )

    spans = (
        maxs - mins
    )

    box_lengths = (
        spans
        + 2.0
        * SOLVENT_PADDING_ANGSTROM
    )

    print(
        "WT coordinate dimensions:"
    )

    print(
        f"X span : {spans[0]:.3f} Å"
    )

    print(
        f"Y span : {spans[1]:.3f} Å"
    )

    print(
        f"Z span : {spans[2]:.3f} Å"
    )

    print()

    print(
        "Rectangular periodic box:"
    )

    print(
        f"X box : {box_lengths[0]:.3f} Å"
    )

    print(
        f"Y box : {box_lengths[1]:.3f} Å"
    )

    print(
        f"Z box : {box_lengths[2]:.3f} Å"
    )

    print()

    return box_lengths


# =============================================================================
# SOLVATION
# =============================================================================

def solvate_protein_only(
    protein_topology,
    protein_positions,
    box_lengths,
):

    print(
        "Solvating WT protein only with rectangular box..."
    )

    forcefield = app.ForceField(
        "amber14/protein.ff14SB.xml",
        "amber14/tip3p.xml",
    )

    modeller = app.Modeller(
        protein_topology,
        protein_positions,
    )

    ca166_before = find_ca_coordinate(
        modeller.topology,
        modeller.positions,
        EXPECTED_D166,
    )

    box_size = (
        openmm.Vec3(
            float(box_lengths[0]),
            float(box_lengths[1]),
            float(box_lengths[2]),
        )
        * unit.angstrom
    )

    modeller.addSolvent(
        forcefield,
        model="tip3p",
        boxSize=box_size,
        neutralize=True,
        ionicStrength=IONIC_STRENGTH,
    )

    ca166_after = find_ca_coordinate(
        modeller.topology,
        modeller.positions,
        EXPECTED_D166,
    )

    translation = (
        ca166_after
        - ca166_before
    )

    print(
        f"Solvated protein atoms : "
        f"{count_atoms(modeller.topology)}"
    )

    print(
        f"CA166 translation       : "
        f"({translation[0]:.3f}, "
        f"{translation[1]:.3f}, "
        f"{translation[2]:.3f}) Å"
    )

    print()

    return (
        modeller,
        translation,
    )


# =============================================================================
# ADD LIGAND
# =============================================================================

def add_ligand(
    modeller,
    translation,
    ligand_topology,
    ligand_positions,
):

    print(
        "Adding exact docked ligand to solvated system..."
    )

    ligand_xyz = positions_to_numpy(
        ligand_positions
    )

    ligand_xyz = (
        ligand_xyz
        + translation
    )

    shifted_positions = (
        ligand_xyz
        * unit.angstrom
    )

    modeller.add(
        ligand_topology,
        shifted_positions,
    )

    residues = list(
        modeller.topology.residues()
    )

    ligand_residue = residues[-1]

    ligand_residue.name = "LIG"

    ligand_atoms = [
        atom
        for atom in ligand_residue.atoms()
    ]

    if len(ligand_atoms) != EXPECTED_LIGAND_TOTAL_ATOMS:
        fail(
            f"LIG contains {len(ligand_atoms)} atoms; "
            f"expected {EXPECTED_LIGAND_TOTAL_ATOMS}."
        )

    print(
        f"Ligand residue : {ligand_residue.name}"
    )

    print(
        f"Ligand atoms   : {len(ligand_atoms)}"
    )

    print()

    return modeller


# =============================================================================
# REMOVE OVERLAPPING WATERS
# =============================================================================

def remove_overlapping_waters(
    modeller,
):

    print(
        "Checking for solvent molecules overlapping the ligand..."
    )

    xyz = positions_to_numpy(
        modeller.positions
    )

    ligand_residue = None

    for residue in modeller.topology.residues():

        if residue.name == "LIG":
            ligand_residue = residue
            break

    if ligand_residue is None:
        fail(
            "LIG residue was not found."
        )

    ligand_indices = [
        atom.index
        for atom in ligand_residue.atoms()
    ]

    ligand_xyz = xyz[
        ligand_indices,
        :,
    ]

    water_to_delete = []

    cutoff = (
        WATER_LIGAND_CUTOFF.value_in_unit(
            unit.angstrom
        )
    )

    for residue in modeller.topology.residues():

        if residue.name not in {
            "HOH",
            "WAT",
            "TIP3",
        }:
            continue

        water_indices = [
            atom.index
            for atom in residue.atoms()
        ]

        if not water_indices:
            continue

        water_xyz = xyz[
            water_indices,
            :,
        ]

        delta = (
            water_xyz[:, None, :]
            - ligand_xyz[None, :, :]
        )

        distances = np.sqrt(
            np.sum(
                delta * delta,
                axis=2,
            )
        )

        if np.any(
            distances < cutoff
        ):
            water_to_delete.append(
                residue
            )

    if water_to_delete:
        modeller.delete(
            water_to_delete
        )

    print(
        f"Overlapping waters removed : "
        f"{len(water_to_delete)}"
    )

    print()

    return modeller


# =============================================================================
# WRITE PREPARED COMPLEX
# =============================================================================

def write_prepared_complex(
    modeller,
):

    path = (
        PREP_DIR
        / "PNPLA3_WT_CID71236597_rectangular_solvated_complex.pdb"
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as handle:

        app.PDBFile.writeFile(
            modeller.topology,
            modeller.positions,
            handle,
            keepIds=True,
        )

    print(
        f"Prepared solvated complex : {path}"
    )

    print(
        f"Complex atoms              : "
        f"{count_atoms(modeller.topology)}"
    )

    print()

    return path


# =============================================================================
# INITIAL D166-LIGAND DISTANCE
# =============================================================================

def initial_d166_ligand_distance(
    topology,
    positions,
):

    xyz = positions_to_numpy(
        positions
    )

    d166_indices = []
    ligand_indices = []

    for residue in topology.residues():

        if residue.name == "LIG":

            ligand_indices.extend(
                atom.index
                for atom in residue.atoms()
            )

        try:
            rid = int(
                residue.id
            )
        except Exception:
            continue

        if rid == EXPECTED_D166:

            d166_indices.extend(
                atom.index
                for atom in residue.atoms()
            )

    if not ligand_indices:
        fail(
            "Ligand atoms not found "
            "for initial distance QC."
        )

    if not d166_indices:
        fail(
            "Residue 166 atoms not found "
            "for initial distance QC."
        )

    d166_xyz = xyz[
        d166_indices,
        :,
    ]

    ligand_xyz = xyz[
        ligand_indices,
        :,
    ]

    delta = (
        d166_xyz[:, None, :]
        - ligand_xyz[None, :, :]
    )

    distances = np.sqrt(
        np.sum(
            delta * delta,
            axis=2,
        )
    )

    return float(
        distances.min()
    )


# =============================================================================
# SYSTEM GENERATION
# =============================================================================

def build_system(
    topology,
    ligand,
):

    print(
        "Generating AMBER14 ff14SB + TIP3P + OpenFF 2.2.1 system..."
    )

    periodic_forcefield_kwargs = {
        "nonbondedMethod": app.PME,
        "nonbondedCutoff": NONBONDED_CUTOFF,
        "constraints": app.HBonds,
        "rigidWater": True,
        "ewaldErrorTolerance": EWALD_ERROR_TOLERANCE,
    }

    nonperiodic_forcefield_kwargs = {
        "constraints": app.HBonds,
        "rigidWater": True,
    }

    system_generator = SystemGenerator(
        forcefields=[
            "amber14/protein.ff14SB.xml",
            "amber14/tip3p.xml",
        ],
        small_molecule_forcefield="openff-2.2.1",
        molecules=[ligand],
        cache=None,
        periodic_forcefield_kwargs=periodic_forcefield_kwargs,
        nonperiodic_forcefield_kwargs=nonperiodic_forcefield_kwargs,
    )

    system = system_generator.create_system(
        topology
    )

    print(
        f"System particles : {system.getNumParticles()}"
    )
    print()

    return system


# =============================================================================
# CPU PLATFORM
# =============================================================================

def get_cpu_platform():

    platform = openmm.Platform.getPlatformByName(
        "CPU"
    )

    cpu_count = (
        os.cpu_count()
        or 1
    )

    threads = min(
        8,
        cpu_count,
    )

    properties = {
        "Threads": str(
            threads
        )
    }

    print(
        f"OpenMM CPU threads : {threads}"
    )

    print()

    return (
        platform,
        properties,
    )


# =============================================================================
# MINIMIZATION
# =============================================================================

def minimize_system(
    system,
    topology,
    positions,
    platform,
    properties,
):

    print("=" * 78)
    print("ENERGY MINIMIZATION")
    print("=" * 78)

    integrator = (
        openmm.LangevinMiddleIntegrator(
            TEMPERATURE,
            1.0 / unit.picosecond,
            TIMESTEP,
        )
    )

    integrator.setRandomNumberSeed(
        SEED
    )

    simulation = app.Simulation(
        topology,
        system,
        integrator,
        platform,
        properties,
    )

    simulation.context.setPositions(
        positions
    )

    print(
        "Minimizing energy..."
    )

    simulation.minimizeEnergy(
        maxIterations=5000
    )

    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
    )

    energy = (
        state.getPotentialEnergy()
    )

    print(
        f"Minimized potential energy : "
        f"{energy}"
    )

    print()

    with open(
        RUN_DIR
        / "minimized_state.xml",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            XmlSerializer.serialize(
                state
            )
        )

    return state


# =============================================================================
# NVT EQUILIBRATION
# =============================================================================

def run_nvt(
    system,
    topology,
    minimized_state,
    platform,
    properties,
):

    print("=" * 78)
    print("NVT EQUILIBRATION — 100 ps")
    print("=" * 78)

    integrator = (
        openmm.LangevinMiddleIntegrator(
            TEMPERATURE,
            1.0 / unit.picosecond,
            TIMESTEP,
        )
    )

    integrator.setRandomNumberSeed(
        SEED + 1
    )

    simulation = app.Simulation(
        topology,
        system,
        integrator,
        platform,
        properties,
    )

    simulation.context.setState(
        minimized_state
    )

    simulation.context.setVelocitiesToTemperature(
        TEMPERATURE,
        SEED + 1,
    )

    csv_path = (
        RUN_DIR
        / "NVT_equilibration.csv"
    )

    chk_path = (
        RUN_DIR
        / "NVT_equilibration.chk"
    )

    simulation.reporters.append(
        app.StateDataReporter(
            str(csv_path),
            REPORT_INTERVAL_STEPS,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            separator=",",
        )
    )

    simulation.reporters.append(
        app.CheckpointReporter(
            str(chk_path),
            CHECKPOINT_INTERVAL_STEPS,
        )
    )

    print(
        "Running NVT..."
    )

    simulation.step(
        NVT_STEPS
    )

    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
    )

    with open(
        RUN_DIR
        / "nvt_final_state.xml",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            XmlSerializer.serialize(
                state
            )
        )

    print(
        "NVT completed."
    )

    print()

    return state


# =============================================================================
# NPT EQUILIBRATION
# =============================================================================

def run_npt(
    base_system,
    topology,
    nvt_state,
    platform,
    properties,
):

    print("=" * 78)
    print("NPT EQUILIBRATION — 500 ps")
    print("=" * 78)

    npt_system = (
        XmlSerializer.deserialize(
            XmlSerializer.serialize(
                base_system
            )
        )
    )

    barostat = (
        openmm.MonteCarloBarostat(
            PRESSURE,
            TEMPERATURE,
            100,
        )
    )

    barostat.setRandomNumberSeed(
        SEED + 2
    )

    npt_system.addForce(
        barostat
    )

    integrator = (
        openmm.LangevinMiddleIntegrator(
            TEMPERATURE,
            1.0 / unit.picosecond,
            TIMESTEP,
        )
    )

    integrator.setRandomNumberSeed(
        SEED + 2
    )

    simulation = app.Simulation(
        topology,
        npt_system,
        integrator,
        platform,
        properties,
    )

    simulation.context.setState(
        nvt_state
    )

    csv_path = (
        RUN_DIR
        / "NPT_equilibration.csv"
    )

    chk_path = (
        RUN_DIR
        / "NPT_equilibration.chk"
    )

    simulation.reporters.append(
        app.StateDataReporter(
            str(csv_path),
            REPORT_INTERVAL_STEPS,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            separator=",",
        )
    )

    simulation.reporters.append(
        app.CheckpointReporter(
            str(chk_path),
            CHECKPOINT_INTERVAL_STEPS,
        )
    )

    print(
        "Running NPT..."
    )

    simulation.step(
        NPT_STEPS
    )

    state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
    )

    with open(
        RUN_DIR
        / "npt_final_state.xml",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            XmlSerializer.serialize(
                state
            )
        )

    print(
        "NPT completed."
    )

    print()

    return (
        npt_system,
        state,
    )


# =============================================================================
# PRODUCTION MD
# =============================================================================

def run_production(
    system,
    topology,
    npt_state,
    platform,
    properties,
):

    print("=" * 78)
    print(
        f"PRODUCTION MD — {PRODUCTION_NS:.1f} ns"
    )
    print("=" * 78)

    integrator = (
        openmm.LangevinMiddleIntegrator(
            TEMPERATURE,
            1.0 / unit.picosecond,
            TIMESTEP,
        )
    )

    integrator.setRandomNumberSeed(
        SEED + 3
    )

    simulation = app.Simulation(
        topology,
        system,
        integrator,
        platform,
        properties,
    )

    simulation.context.setState(
        npt_state
    )

    production_csv = (
        RUN_DIR
        / "production.csv"
    )

    production_chk = (
        RUN_DIR
        / "production.chk"
    )

    production_dcd = (
        RUN_DIR
        / "production.dcd"
    )

    simulation.reporters.append(
        app.StateDataReporter(
            str(production_csv),
            REPORT_INTERVAL_STEPS,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            separator=",",
        )
    )

    simulation.reporters.append(
        app.CheckpointReporter(
            str(production_chk),
            CHECKPOINT_INTERVAL_STEPS,
        )
    )

    simulation.reporters.append(
        app.DCDReporter(
            str(production_dcd),
            DCD_INTERVAL_STEPS,
            enforcePeriodicBox=True,
        )
    )

    print(
        "Running production MD..."
    )

    simulation.step(
        PRODUCTION_STEPS
    )

    final_state = simulation.context.getState(
        getPositions=True,
        getVelocities=True,
        getEnergy=True,
    )

    final_pdb = (
        RUN_DIR
        / "production_final_frame.pdb"
    )

    with open(
        final_pdb,
        "w",
        encoding="utf-8",
    ) as handle:

        app.PDBFile.writeFile(
            topology,
            final_state.getPositions(),
            handle,
            keepIds=True,
        )

    with open(
        RUN_DIR
        / "production_final_state.xml",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            XmlSerializer.serialize(
                final_state
            )
        )

    print()
    print(
        "Production MD completed."
    )

    print(
        f"Trajectory : {production_dcd}"
    )

    print(
        f"Final PDB   : {final_pdb}"
    )

    print(
        f"Production log : {production_csv}"
    )

    print()

    return (
        production_csv,
        production_dcd,
        final_pdb,
    )


# =============================================================================
# BASIC PRODUCTION QC FIGURES
# =============================================================================

def make_qc_figures(
    production_csv,
):

    if not production_csv.exists():
        return

    try:

        df = pd.read_csv(
            production_csv
        )

        # Temperature
        if {
            "Time (ps)",
            "Temperature (K)",
        }.issubset(df.columns):

            plt.figure(
                figsize=(9, 5)
            )

            plt.plot(
                df["Time (ps)"],
                df["Temperature (K)"],
            )

            plt.axhline(
                300.0,
                linestyle="--",
            )

            plt.xlabel(
                "Time (ps)"
            )

            plt.ylabel(
                "Temperature (K)"
            )

            plt.title(
                "Step 25 — WT PNPLA3 + CID 71236597 Temperature"
            )

            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR
                / "Figure_25_01_Production_Temperature.png",
                dpi=300,
            )

            plt.close()

        # Potential energy
        if {
            "Time (ps)",
            "Potential Energy (kJ/mole)",
        }.issubset(df.columns):

            plt.figure(
                figsize=(9, 5)
            )

            plt.plot(
                df["Time (ps)"],
                df["Potential Energy (kJ/mole)"],
            )

            plt.xlabel(
                "Time (ps)"
            )

            plt.ylabel(
                "Potential Energy (kJ/mol)"
            )

            plt.title(
                "Step 25 — WT PNPLA3 + CID 71236597 Potential Energy"
            )

            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR
                / "Figure_25_02_Production_Potential_Energy.png",
                dpi=300,
            )

            plt.close()

        # Box volume
        if {
            "Time (ps)",
            "Box Volume (nm^3)",
        }.issubset(df.columns):

            plt.figure(
                figsize=(9, 5)
            )

            plt.plot(
                df["Time (ps)"],
                df["Box Volume (nm^3)"],
            )

            plt.xlabel(
                "Time (ps)"
            )

            plt.ylabel(
                "Box Volume (nm³)"
            )

            plt.title(
                "Step 25 — WT PNPLA3 + CID 71236597 Box Volume"
            )

            plt.tight_layout()

            plt.savefig(
                FIGURE_DIR
                / "Figure_25_03_Production_Box_Volume.png",
                dpi=300,
            )

            plt.close()

    except Exception as exc:

        print(
            f"QC figure warning: {exc}"
        )


# =============================================================================
# QC / MANIFEST
# =============================================================================

def write_qc(
    box_lengths,
    protein_atoms,
    complex_atoms,
    system_particles,
    initial_distance,
):

    qc = {
        "step": 25,
        "system": "WT PNPLA3 + CID 71236597",
        "cid": CID,
        "protein_residues": EXPECTED_PROTEIN_RESIDUES,
        "protein_atoms_prepared": int(
            protein_atoms
        ),
        "ligand_heavy_atoms": EXPECTED_LIGAND_HEAVY_ATOMS,
        "ligand_hydrogens": EXPECTED_LIGAND_HYDROGENS,
        "ligand_total_atoms": EXPECTED_LIGAND_TOTAL_ATOMS,
        "complex_atoms": int(
            complex_atoms
        ),
        "system_particles": int(
            system_particles
        ),
        "box_x_A": float(
            box_lengths[0]
        ),
        "box_y_A": float(
            box_lengths[1]
        ),
        "box_z_A": float(
            box_lengths[2]
        ),
        "solvent_padding_A":
            SOLVENT_PADDING_ANGSTROM,
        "ionic_strength_M": 0.15,
        "temperature_K": 300.0,
        "pressure_atm": 1.0,
        "timestep_fs": 2.0,
        "nvt_ps": NVT_PS,
        "npt_ps": NPT_PS,
        "production_ns": PRODUCTION_NS,
        "nonbonded_method": "PME",
        "nonbonded_cutoff_nm": 1.0,
        "ewald_error_tolerance":
            EWALD_ERROR_TOLERANCE,
        "initial_D166_LIG_min_distance_A":
            float(initial_distance),
        "seed": SEED,
        "status": "COMPLETED",
    }

    with open(
        QC_DIR / "STEP25_QC.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    interpretation = (
        "Step 25 simulated WT PNPLA3 bound to the exact "
        "CID 71236597 docking pose. The system used "
        "AMBER14 ff14SB for protein, OpenFF 2.2.1 for "
        "the ligand, TIP3P explicit water, 0.15 M ionic "
        "strength, PME electrostatics, and a 1.0 nm "
        "nonbonded cutoff. NVT equilibration preceded "
        "NPT equilibration; the Monte Carlo barostat was "
        "introduced only for NPT. Production MD was "
        "performed for 10 ns at 300 K and 1 atm. "
        "Trajectory-derived observations must be based "
        "on subsequent analysis and should not be treated "
        "as experimental binding or efficacy evidence."
    )

    with open(
        QC_DIR / "STEP25_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            interpretation
        )

    manifest = {
        "WT_receptor": str(
            WT_RECEPTOR
        ),
        "prepared_WT": str(
            PREVIOUS_PREPARED_PROTEIN
        ),
        "ligand_PDBQT": str(
            LIGAND_PDBQT
        ),
        "ligand_SMILES": LIGAND_SMILES,
        "output_directory": str(
            OUT_DIR
        ),
    }

    with open(
        QC_DIR / "STEP25_MANIFEST.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            manifest,
            handle,
            indent=2,
        )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 25 — WT PNPLA3 + CID 71236597 MOLECULAR DYNAMICS"
    )
    print("=" * 78)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"WT receptor  : {WT_RECEPTOR}"
    )

    print(
        f"Ligand PDBQT : {LIGAND_PDBQT}"
    )

    print()

    # -------------------------------------------------------------------------
    # FILE CHECKS
    # -------------------------------------------------------------------------

    if not WT_RECEPTOR.exists():
        fail(
            f"WT receptor not found:\n{WT_RECEPTOR}"
        )

    if not LIGAND_PDBQT.exists():
        fail(
            f"Ligand PDBQT not found:\n{LIGAND_PDBQT}"
        )

    # -------------------------------------------------------------------------
    # EXACT LIGAND
    # -------------------------------------------------------------------------

    (
        ligand,
        ligand_topology,
        ligand_positions,
    ) = build_exact_docked_ligand()

    # -------------------------------------------------------------------------
    # WT PROTEIN
    # -------------------------------------------------------------------------

    (
        protein_topology,
        protein_positions,
    ) = prepare_wt_protein()

    protein_atoms = count_atoms(
        protein_topology
    )

    # -------------------------------------------------------------------------
    # BOX
    # -------------------------------------------------------------------------

    box_lengths = (
        calculate_box_lengths(
            protein_positions
        )
    )

    # -------------------------------------------------------------------------
    # SOLVATION
    # -------------------------------------------------------------------------

    (
        modeller,
        translation,
    ) = solvate_protein_only(
        protein_topology,
        protein_positions,
        box_lengths,
    )

    # -------------------------------------------------------------------------
    # ADD LIGAND
    # -------------------------------------------------------------------------

    modeller = add_ligand(
        modeller,
        translation,
        ligand_topology,
        ligand_positions,
    )

    # -------------------------------------------------------------------------
    # REMOVE WATER OVERLAP
    # -------------------------------------------------------------------------

    modeller = (
        remove_overlapping_waters(
            modeller
        )
    )

    # -------------------------------------------------------------------------
    # WRITE PREPARED COMPLEX
    # -------------------------------------------------------------------------

    write_prepared_complex(
        modeller
    )

    complex_atoms = count_atoms(
        modeller.topology
    )

    # -------------------------------------------------------------------------
    # INITIAL D166-LIGAND DISTANCE
    # -------------------------------------------------------------------------

    initial_distance = (
        initial_d166_ligand_distance(
            modeller.topology,
            modeller.positions,
        )
    )

    print(
        f"Initial D166–ligand minimum distance : "
        f"{initial_distance:.3f} Å"
    )

    print()

    # -------------------------------------------------------------------------
    # FORCE FIELD / SYSTEM
    # -------------------------------------------------------------------------

    system = build_system(
        modeller.topology,
        ligand,
    )

    # -------------------------------------------------------------------------
    # CPU
    # -------------------------------------------------------------------------

    (
        platform,
        properties,
    ) = get_cpu_platform()

    # -------------------------------------------------------------------------
    # MINIMIZATION
    # -------------------------------------------------------------------------

    minimized_state = (
        minimize_system(
            system,
            modeller.topology,
            modeller.positions,
            platform,
            properties,
        )
    )

    # -------------------------------------------------------------------------
    # NVT
    # -------------------------------------------------------------------------

    nvt_state = (
        run_nvt(
            system,
            modeller.topology,
            minimized_state,
            platform,
            properties,
        )
    )

    # -------------------------------------------------------------------------
    # NPT
    # -------------------------------------------------------------------------

    (
        npt_system,
        npt_state,
    ) = run_npt(
        system,
        modeller.topology,
        nvt_state,
        platform,
        properties,
    )

    # -------------------------------------------------------------------------
    # PRODUCTION
    # -------------------------------------------------------------------------

    (
        production_csv,
        production_dcd,
        final_pdb,
    ) = run_production(
        npt_system,
        modeller.topology,
        npt_state,
        platform,
        properties,
    )

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------

    make_qc_figures(
        production_csv
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    write_qc(
        box_lengths,
        protein_atoms,
        complex_atoms,
        npt_system.getNumParticles(),
        initial_distance,
    )

    # -------------------------------------------------------------------------
    # FINAL
    # -------------------------------------------------------------------------

    print("=" * 78)
    print("STEP 25 COMPLETED")
    print("=" * 78)

    print(
        f"Complex atoms   : {complex_atoms}"
    )

    print(
        f"System particles: "
        f"{npt_system.getNumParticles()}"
    )

    print(
        f"Production DCD  : {production_dcd}"
    )

    print(
        f"Production log  : {production_csv}"
    )

    print(
        f"Final PDB       : {final_pdb}"
    )

    print()


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Step 25 interrupted by user."
        )

        sys.exit(130)

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
