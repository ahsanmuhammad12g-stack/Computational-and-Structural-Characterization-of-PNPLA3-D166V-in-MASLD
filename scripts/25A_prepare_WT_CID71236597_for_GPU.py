from pathlib import Path
import json
import os
import sys
import traceback

import numpy as np

import openmm
from openmm import app, unit, XmlSerializer

from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator


# =============================================================================
# STEP 25A — PREPARE WT PNPLA3 + CID 71236597 FOR GPU MD
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

PREPARED_PROTEIN = (
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
GPU_DIR = OUT_DIR / "GPU_READY"
QC_DIR = OUT_DIR / "qc"

for folder in [
    PREP_DIR,
    GPU_DIR,
    QC_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# EXPECTED VALUES
# =============================================================================

EXPECTED_RESIDUES = 481
EXPECTED_LIGAND_HEAVY = 33
EXPECTED_LIGAND_H = 17
EXPECTED_LIGAND_TOTAL = 50
EXPECTED_D166 = 166

SOLVENT_PADDING_A = 10.0

TEMPERATURE = 300.0 * unit.kelvin
PRESSURE = 1.0 * unit.atmosphere

IONIC_STRENGTH = 0.15 * unit.molar

NONBONDED_CUTOFF = 1.0 * unit.nanometer
EWALD_ERROR_TOLERANCE = 0.0005

TIMESTEP = 0.002 * unit.picoseconds

SEED = 20260910

LIGAND_SMILES = (
    "CN1CCCn2c1cc("
    "OCc1cc(F)c("
    "Oc3ccc(C(F)(F)F)nc3"
    ")c(F)c1"
    ")nc2=O"
)


# =============================================================================
# UTILITIES
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


def to_numpy(positions):
    return np.asarray(
        positions.value_in_unit(
            unit.angstrom
        ),
        dtype=float,
    )


# =============================================================================
# READ PDBQT
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

                atoms[atom_id] = (
                    x,
                    y,
                    z,
                )

            except Exception:
                continue

    if not atoms:
        fail(
            f"No atoms found in ligand PDBQT:\n{path}"
        )

    return atoms


# =============================================================================
# READ MEEKO SMILES INDEX
# =============================================================================

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

            values = (
                line.strip()
                .split()[3:]
            )

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
            "REMARK SMILES IDX mapping was not found."
        )

    return mapping


# =============================================================================
# BUILD EXACT DOCKED LIGAND
# =============================================================================

def build_ligand():

    print(
        "Building exact CID 71236597 from Step 21 docking coordinates..."
    )
    print()

    pdbqt_atoms = parse_pdbqt_atoms(
        LIGAND_PDBQT
    )

    mapping = parse_smiles_idx(
        LIGAND_PDBQT
    )

    rd_mol = (
        __import__("rdkit")
    )

    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.MolFromSmiles(
        LIGAND_SMILES
    )

    if mol is None:
        fail(
            "RDKit could not parse CID 71236597."
        )

    heavy = mol.GetNumAtoms()

    if heavy != EXPECTED_LIGAND_HEAVY:
        fail(
            f"Expected {EXPECTED_LIGAND_HEAVY} heavy atoms; "
            f"found {heavy}."
        )

    conformer = Chem.Conformer(
        heavy
    )

    for i in range(heavy):

        smiles_index = i + 1

        if smiles_index not in mapping:
            fail(
                f"SMILES atom {smiles_index} missing from PDBQT mapping."
            )

        pdb_atom_id = mapping[
            smiles_index
        ]

        if pdb_atom_id not in pdbqt_atoms:
            fail(
                f"PDBQT atom {pdb_atom_id} is missing."
            )

        x, y, z = pdbqt_atoms[
            pdb_atom_id
        ]

        conformer.SetAtomPosition(
            i,
            Point3D(
                x,
                y,
                z,
            ),
        )

    mol.AddConformer(
        conformer,
        assignId=True,
    )

    mol_h = Chem.AddHs(
        mol,
        addCoords=True,
    )

    total = mol_h.GetNumAtoms()
    hydrogens = total - heavy

    if hydrogens != EXPECTED_LIGAND_H:
        fail(
            f"Expected {EXPECTED_LIGAND_H} H atoms; "
            f"found {hydrogens}."
        )

    if total != EXPECTED_LIGAND_TOTAL:
        fail(
            f"Expected {EXPECTED_LIGAND_TOTAL} total atoms; "
            f"found {total}."
        )

    ligand = Molecule.from_rdkit(
        mol_h,
        allow_undefined_stereo=True,
        hydrogens_are_explicit=True,
    )

    positions = (
        np.asarray(
            mol_h.GetConformer()
            .GetPositions(),
            dtype=float,
        )
        * unit.angstrom
    )

    ligand_topology = (
        ligand
        .to_topology()
        .to_openmm()
    )

    residues = list(
        ligand_topology.residues()
    )

    if len(residues) != 1:
        fail(
            "Expected exactly one ligand residue."
        )

    residues[0].name = "LIG"

    # Exact heavy-atom coordinate QC
    coords = positions.value_in_unit(
        unit.angstrom
    )

    for i in range(heavy):

        smiles_index = i + 1

        pdb_atom_id = mapping[
            smiles_index
        ]

        expected = np.array(
            pdbqt_atoms[
                pdb_atom_id
            ],
            dtype=float,
        )

        observed = coords[i]

        error = np.linalg.norm(
            observed - expected
        )

        if error > 1e-6:
            fail(
                f"Docking coordinate mismatch at ligand atom "
                f"{smiles_index}: {error:.6f} Å."
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
            positions,
            handle,
            keepIds=True,
        )

    print(
        f"Ligand heavy atoms : {heavy}"
    )

    print(
        f"Ligand hydrogens   : {hydrogens}"
    )

    print(
        f"Ligand total atoms : {total}"
    )

    print()

    return (
        ligand,
        ligand_topology,
        positions,
    )


# =============================================================================
# PREPARED WT PROTEIN
# =============================================================================

def load_prepared_protein():

    if not PREPARED_PROTEIN.exists():
        fail(
            f"Previously validated prepared WT protein not found:\n"
            f"{PREPARED_PROTEIN}"
        )

    pdb = app.PDBFile(
        str(PREPARED_PROTEIN)
    )

    residues = count_residues(
        pdb.topology
    )

    atoms = count_atoms(
        pdb.topology
    )

    print(
        "Using previously validated prepared WT protein..."
    )

    print(
        f"Prepared protein residues : {residues}"
    )

    print(
        f"Prepared protein atoms    : {atoms}"
    )

    print()

    if residues != EXPECTED_RESIDUES:
        fail(
            f"Expected {EXPECTED_RESIDUES} residues; "
            f"found {residues}."
        )

    return (
        pdb.topology,
        pdb.positions,
    )


# =============================================================================
# RECTANGULAR BOX
# =============================================================================

def calculate_box(
    positions
):

    xyz = to_numpy(
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

    box = (
        spans
        + 2.0
        * SOLVENT_PADDING_A
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
        f"X box : {box[0]:.3f} Å"
    )

    print(
        f"Y box : {box[1]:.3f} Å"
    )

    print(
        f"Z box : {box[2]:.3f} Å"
    )

    print()

    return box


# =============================================================================
# SOLVATE
# =============================================================================

def solvate(
    protein_topology,
    protein_positions,
    box_lengths,
):

    print(
        "Solvating WT protein..."
    )

    forcefield = app.ForceField(
        "amber14/protein.ff14SB.xml",
        "amber14/tip3p.xml",
    )

    modeller = app.Modeller(
        protein_topology,
        protein_positions,
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

    print(
        f"Solvated protein atoms : "
        f"{count_atoms(modeller.topology)}"
    )

    print()

    return modeller


# =============================================================================
# ADD LIGAND
# =============================================================================

def add_ligand(
    modeller,
    ligand_topology,
    ligand_positions,
):

    print(
        "Adding exact docked ligand..."
    )

    modeller.add(
        ligand_topology,
        ligand_positions,
    )

    residues = list(
        modeller.topology.residues()
    )

    ligand_residue = residues[-1]
    ligand_residue.name = "LIG"

    ligand_atom_count = sum(
        1 for _ in ligand_residue.atoms()
    )

    if ligand_atom_count != EXPECTED_LIGAND_TOTAL:
        fail(
            f"LIG contains {ligand_atom_count} atoms; "
            f"expected {EXPECTED_LIGAND_TOTAL}."
        )

    print(
        f"Ligand residue : {ligand_residue.name}"
    )

    print(
        f"Ligand atoms   : {ligand_atom_count}"
    )

    print()

    return modeller


# =============================================================================
# INITIAL DISTANCE
# =============================================================================

def calculate_d166_distance(
    topology,
    positions,
):

    xyz = to_numpy(
        positions
    )

    d166_atoms = []
    ligand_atoms = []

    for residue in topology.residues():

        if residue.name == "LIG":

            ligand_atoms.extend(
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

            d166_atoms.extend(
                atom.index
                for atom in residue.atoms()
            )

    if not ligand_atoms:
        fail(
            "LIG not found for distance QC."
        )

    if not d166_atoms:
        fail(
            "Residue 166 not found for distance QC."
        )

    a = xyz[
        d166_atoms,
        :
    ]

    b = xyz[
        ligand_atoms,
        :
    ]

    delta = (
        a[:, None, :]
        - b[None, :, :]
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
# BUILD OPENMM SYSTEM
# =============================================================================

def build_system(
    topology,
    ligand,
):

    print(
        "Generating OpenMM system..."
    )

    periodic_kwargs = {
        "nonbondedMethod": app.PME,
        "nonbondedCutoff": NONBONDED_CUTOFF,
        "constraints": app.HBonds,
        "rigidWater": True,
        "ewaldErrorTolerance":
            EWALD_ERROR_TOLERANCE,
    }

    nonperiodic_kwargs = {
        "nonbondedMethod": app.NoCutoff,
        "constraints": app.HBonds,
        "rigidWater": True,
    }

    generator = SystemGenerator(
        forcefields=[
            "amber14/protein.ff14SB.xml",
            "amber14/tip3p.xml",
        ],
        small_molecule_forcefield="openff-2.2.1",
        molecules=[
            ligand
        ],
        cache=None,
        periodic_forcefield_kwargs=
            periodic_kwargs,
        nonperiodic_forcefield_kwargs=
            nonperiodic_kwargs,
    )

    system = generator.create_system(
        topology
    )

    print(
        f"System particles : "
        f"{system.getNumParticles()}"
    )

    print()

    return system


# =============================================================================
# MINIMIZATION
# =============================================================================

def minimize_system(
    system,
    topology,
    positions,
):

    print("=" * 78)
    print("ENERGY MINIMIZATION")
    print("=" * 78)

    platform = (
        openmm.Platform.getPlatformByName(
            "CPU"
        )
    )

    threads = min(
        8,
        os.cpu_count() or 1,
    )

    properties = {
        "Threads": str(
            threads
        )
    }

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

    return state


# =============================================================================
# WRITE GPU-READY FILES
# =============================================================================

def write_gpu_ready_files(
    topology,
    positions,
    system,
    minimized_state,
    box_lengths,
    initial_distance,
):

    topology_pdb = (
        GPU_DIR
        / "WT_CID71236597_topology.pdb"
    )

    system_xml = (
        GPU_DIR
        / "WT_CID71236597_system.xml"
    )

    minimized_xml = (
        GPU_DIR
        / "WT_CID71236597_minimized_state.xml"
    )

    initial_pdb = (
        GPU_DIR
        / "WT_CID71236597_minimized.pdb"
    )

    with open(
        topology_pdb,
        "w",
        encoding="utf-8",
    ) as handle:

        app.PDBFile.writeFile(
            topology,
            positions,
            handle,
            keepIds=True,
        )

    with open(
        system_xml,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            XmlSerializer.serialize(
                system
            )
        )

    with open(
        minimized_xml,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            XmlSerializer.serialize(
                minimized_state
            )
        )

    with open(
        initial_pdb,
        "w",
        encoding="utf-8",
    ) as handle:

        app.PDBFile.writeFile(
            topology,
            minimized_state.getPositions(),
            handle,
            keepIds=True,
        )

    qc = {
        "step": "25A",
        "system": "WT PNPLA3 + CID 71236597",
        "cid": 71236597,
        "protein_residues": EXPECTED_RESIDUES,
        "ligand_heavy_atoms":
            EXPECTED_LIGAND_HEAVY,
        "ligand_hydrogens":
            EXPECTED_LIGAND_H,
        "ligand_total_atoms":
            EXPECTED_LIGAND_TOTAL,
        "system_particles":
            system.getNumParticles(),
        "box_x_A":
            float(box_lengths[0]),
        "box_y_A":
            float(box_lengths[1]),
        "box_z_A":
            float(box_lengths[2]),
        "solvent_padding_A":
            SOLVENT_PADDING_A,
        "ionic_strength_M": 0.15,
        "nonbonded_method": "PME",
        "nonbonded_cutoff_nm": 1.0,
        "ewald_error_tolerance":
            EWALD_ERROR_TOLERANCE,
        "temperature_K": 300.0,
        "pressure_atm": 1.0,
        "timestep_fs": 2.0,
        "initial_D166_LIG_min_distance_A":
            initial_distance,
        "minimization_completed": True,
        "ready_for_gpu_md": True,
    }

    with open(
        GPU_DIR / "WT_CID71236597_GPU_MANIFEST.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    with open(
        QC_DIR / "STEP25A_GPU_PREPARATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "WT PNPLA3 + CID 71236597 was prepared for "
            "subsequent GPU molecular dynamics. The system "
            "uses AMBER14 ff14SB for protein, OpenFF 2.2.1 "
            "for the ligand, explicit TIP3P water, 0.15 M "
            "ionic strength, PME electrostatics, and a 1.0 nm "
            "nonbonded cutoff. NVT/NPT/production MD was NOT "
            "performed in this preparation-only step. The "
            "saved OpenMM system and minimized state are "
            "intended for GPU execution."
        )

    print(
        "GPU-ready files written:"
    )

    print(
        f"  {topology_pdb}"
    )

    print(
        f"  {system_xml}"
    )

    print(
        f"  {minimized_xml}"
    )

    print(
        f"  {initial_pdb}"
    )

    print()


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 25A — WT PNPLA3 + CID 71236597 "
        "GPU PREPARATION"
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

    if not WT_RECEPTOR.exists():
        fail(
            f"WT receptor not found:\n{WT_RECEPTOR}"
        )

    if not PREPARED_PROTEIN.exists():
        fail(
            f"Prepared protein not found:\n{PREPARED_PROTEIN}"
        )

    if not LIGAND_PDBQT.exists():
        fail(
            f"Ligand PDBQT not found:\n{LIGAND_PDBQT}"
        )

    (
        ligand,
        ligand_topology,
        ligand_positions,
    ) = build_ligand()

    (
        protein_topology,
        protein_positions,
    ) = load_prepared_protein()

    box_lengths = calculate_box(
        protein_positions
    )

    modeller = solvate(
        protein_topology,
        protein_positions,
        box_lengths,
    )

    modeller = add_ligand(
        modeller,
        ligand_topology,
        ligand_positions,
    )

    complex_atoms = count_atoms(
        modeller.topology
    )

    initial_distance = (
        calculate_d166_distance(
            modeller.topology,
            modeller.positions,
        )
    )

    print(
        f"Prepared complex atoms : "
        f"{complex_atoms}"
    )

    print(
        f"Initial D166–ligand minimum distance : "
        f"{initial_distance:.3f} Å"
    )

    print()

    complex_pdb = (
        PREP_DIR
        / "PNPLA3_WT_CID71236597_rectangular_solvated_complex.pdb"
    )

    with open(
        complex_pdb,
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
        f"Prepared complex saved : {complex_pdb}"
    )

    print()

    system = build_system(
        modeller.topology,
        ligand,
    )

    minimized_state = minimize_system(
        system,
        modeller.topology,
        modeller.positions,
    )

    write_gpu_ready_files(
        modeller.topology,
        modeller.positions,
        system,
        minimized_state,
        box_lengths,
        initial_distance,
    )

    print("=" * 78)
    print("STEP 25A COMPLETED")
    print("=" * 78)

    print(
        "WT + CID 71236597 is prepared for GPU MD."
    )

    print(
        f"System particles : "
        f"{system.getNumParticles()}"
    )

    print(
        "No NVT, NPT, or production MD was run."
    )

    print()


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print()
        print(
            "Step 25A interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 25A FAILED")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        traceback.print_exc()

        sys.exit(1)