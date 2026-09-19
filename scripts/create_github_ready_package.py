from pathlib import Path
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime


# =============================================================================
# FINAL GITHUB-READY PACKAGE BUILDER
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GITHUB_READY = PROJECT_ROOT / "GITHUB_READY"

MAX_FILE_MB = 9.0

SCRIPT_DIR = GITHUB_READY / "scripts"
RESULT_DIR = GITHUB_READY / "results"
FIGURE_DIR = GITHUB_READY / "figures"
STRUCTURE_DIR = GITHUB_READY / "structures"
DOC_DIR = GITHUB_READY / "docs"
META_DIR = GITHUB_READY / "metadata"

PACKAGE_DIRS = [
    SCRIPT_DIR,
    RESULT_DIR,
    FIGURE_DIR,
    STRUCTURE_DIR,
    DOC_DIR,
    META_DIR,
]

manifest = []
exclusions = []


# =============================================================================
# HELPERS
# =============================================================================

def sha256_file(path):

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            chunk = handle.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def size_mb(path):

    return (
        path.stat().st_size
        /
        (1024.0 * 1024.0)
    )


def rel(path):

    try:
        return str(
            path.relative_to(PROJECT_ROOT)
        ).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def exclude(path, reason):

    if path.exists():

        exclusions.append(
            {
                "source": rel(path),
                "reason": reason,
                "size_mb": round(
                    size_mb(path),
                    3,
                ),
            }
        )

    else:

        exclusions.append(
            {
                "source": rel(path),
                "reason": (
                    reason
                    + " File not present."
                ),
                "size_mb": 0.0,
            }
        )


def copy_and_record(
    source,
    destination,
    category,
):

    if not source.exists():

        exclude(
            source,
            "Optional source file not found.",
        )

        return False

    if not source.is_file():

        exclude(
            source,
            "Source is not a regular file.",
        )

        return False

    if size_mb(source) > MAX_FILE_MB:

        exclude(
            source,
            (
                f"File exceeds "
                f"{MAX_FILE_MB:.1f} MB GitHub package limit."
            ),
        )

        return False

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )

    manifest.append(
        {
            "category": category,
            "source": rel(source),
            "github_path": str(
                destination.relative_to(
                    GITHUB_READY
                )
            ).replace("\\", "/"),
            "size_bytes": source.stat().st_size,
            "sha256": sha256_file(source),
        }
    )

    return True


def print_header(text):

    print()
    print("=" * 78)
    print(text)
    print("=" * 78)


# =============================================================================
# INITIALIZE
# =============================================================================

def initialize():

    if GITHUB_READY.exists():

        print(
            "Removing previous GITHUB_READY folder..."
        )

        shutil.rmtree(
            GITHUB_READY
        )

    for directory in PACKAGE_DIRS:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# =============================================================================
# SCRIPTS
# =============================================================================

def collect_scripts():

    print_header(
        "COLLECTING ANALYSIS SCRIPTS"
    )

    source_dir = PROJECT_ROOT / "SCRIPTS"

    if not source_dir.exists():

        print(
            "WARNING: SCRIPTS folder not found."
        )

        return

    for source in sorted(
        source_dir.glob("*.py")
    ):

        if source.name.startswith(
            ("test_", "TEST_")
        ):

            exclude(
                source,
                "Development/test script.",
            )

            continue

        destination = (
            SCRIPT_DIR
            / source.name
        )

        if copy_and_record(
            source,
            destination,
            "script",
        ):

            print(
                f"Included: {source.name}"
            )


# =============================================================================
# RESULTS
# =============================================================================

def collect_results():

    print_header(
        "COLLECTING RESULT TABLES AND QC"
    )

    source_dir = PROJECT_ROOT / "RESULTS"

    if not source_dir.exists():

        print(
            "WARNING: RESULTS folder not found."
        )

        return

    allowed = {
        ".csv",
        ".json",
        ".txt",
        ".md",
        ".cff",
    }

    excluded_exact_names = {
        "sqm.in",
        "sqm.out",
        "leap.log",
    }

    for source in sorted(
        source_dir.rglob("*")
    ):

        if not source.is_file():
            continue

        if GITHUB_READY in source.parents:
            continue

        if (
            "__pycache__"
            in source.parts
        ):

            continue

        if (
            source.name
            in excluded_exact_names
        ):

            exclude(
                source,
                "AmberTools temporary file.",
            )

            continue

        if source.suffix.lower() not in allowed:
            continue

        if source.name.startswith(
            "ANTECHAMBER"
        ):

            exclude(
                source,
                "AmberTools temporary file.",
            )

            continue

        if source.name == "ATOMTYPE.INF":

            exclude(
                source,
                "AmberTools temporary file.",
            )

            continue

        relative = source.relative_to(
            source_dir
        )

        destination = (
            RESULT_DIR
            / relative
        )

        copy_and_record(
            source,
            destination,
            "result",
        )


# =============================================================================
# FIGURES
# =============================================================================

def collect_figures():

    print_header(
        "COLLECTING FIGURES"
    )

    source_dir = PROJECT_ROOT / "RESULTS"

    if not source_dir.exists():
        return

    allowed = {
        ".png",
        ".jpg",
        ".jpeg",
        ".svg",
    }

    for source in sorted(
        source_dir.rglob("*")
    ):

        if not source.is_file():
            continue

        if GITHUB_READY in source.parents:
            continue

        if source.suffix.lower() not in allowed:
            continue

        relative = source.relative_to(
            source_dir
        )

        destination = (
            FIGURE_DIR
            / relative
        )

        copy_and_record(
            source,
            destination,
            "figure",
        )


# =============================================================================
# KEY STRUCTURES
# =============================================================================

def collect_structures():

    print_header(
        "COLLECTING KEY STRUCTURES"
    )

    candidates = [

        (
            PROJECT_ROOT
            / "RESULTS"
            / "STEP10_STRUCTURAL_ANALYSIS"
            / "structures"
            / "WT"
            / "PNPLA3_WT.pdb",
            "PNPLA3_WT.pdb",
        ),

        (
            PROJECT_ROOT
            / "RESULTS"
            / "STEP10_STRUCTURAL_ANALYSIS"
            / "structures"
            / "D166V"
            / "PNPLA3_D166V_mutant.pdb",
            "PNPLA3_D166V_mutant.pdb",
        ),

        (
            PROJECT_ROOT
            / "RESULTS"
            / "STEP21_VIRTUAL_SCREENING_DOCKING"
            / "ligands"
            / "CID_71236597.pdbqt",
            "CID_71236597.pdbqt",
        ),

        # Correct Step 25 location.
        (
            PROJECT_ROOT
            / "RESULTS"
            / "STEP25_WT_CID71236597_MD"
            / "preparation"
            / "CID_71236597_exact_docked_with_H.pdb",
            "CID_71236597_exact_docked_with_H.pdb",
        ),
    ]

    for source, filename in candidates:

        destination = (
            STRUCTURE_DIR
            / filename
        )

        if copy_and_record(
            source,
            destination,
            "key_structure",
        ):

            print(
                f"Included: {filename}"
            )


# =============================================================================
# STEP 33 DOCUMENTATION
# =============================================================================

def collect_step33():

    print_header(
        "COLLECTING STEP 33 FINAL DOCUMENTATION"
    )

    source_dir = (
        PROJECT_ROOT
        / "RESULTS"
        / "STEP33_FINAL_PUBLICATION_PACKAGE"
    )

    if not source_dir.exists():

        print(
            "WARNING: Step 33 final package not found."
        )

        return

    allowed = {
        ".csv",
        ".json",
        ".txt",
        ".md",
        ".cff",
    }

    for source in sorted(
        source_dir.rglob("*")
    ):

        if not source.is_file():
            continue

        if source.suffix.lower() not in allowed:
            continue

        relative = source.relative_to(
            source_dir
        )

        destination = (
            DOC_DIR
            / "step33"
            / relative
        )

        copy_and_record(
            source,
            destination,
            "step33_documentation",
        )


# =============================================================================
# STATIC DOCUMENTS
# =============================================================================

def write_static_documents():

    readme = """
# PNPLA3 D166V Computational Mechanistic and Therapeutic Prioritization

## Target

PNPLA3 p.Asp166Val (D166V)

rsID: rs1381635405

ClinVar VariationID: 3308171

## Objective

This project computationally investigates an unresolved PNPLA3 variant
through variant prioritization, structural modeling, mechanistic analysis,
therapeutic-pocket analysis, virtual screening, differential docking,
lead prioritization, physicochemical development assessment and
independent biological/literature validation.

## Main finding

The completed analysis supports D166V as a computationally plausible
function-altering PNPLA3 variant.

The Asp166→Val substitution produces a substantial local chemical and
interaction-environment change while the global protein backbone remains
highly similar.

## Therapeutic candidate

CID 71236597 was prioritized as the principal computational therapeutic
candidate.

WT docking:
-9.296 kcal/mol

D166V docking:
-10.262 kcal/mol

D166V − WT:
-0.966 kcal/mol

These are docking scores and are not experimentally measured affinities.

## Scientific interpretation

The predicted D166V structural effect is consistent with established
PNPLA3 biology and the known relationship between PNPLA3 variation and
steatotic liver disease.

## Important limitation

The study does not experimentally establish:

- D166V enzymatic activity;
- D166V cellular phenotype;
- D166V clinical/disease causality;
- experimental compound binding;
- therapeutic rescue;
- experimental ADMET;
- clinical efficacy.

The principal remaining gap is direct variant-specific experimental
validation.

## Repository contents

scripts/
results/
figures/
structures/
docs/
metadata/

## Final status

COMPUTATIONALLY SUPPORTED WITH AN EXPERIMENTAL VALIDATION GAP
""".strip()

    requirements = """
numpy
pandas
matplotlib
biopython
openmm
openff-toolkit
openmmforcefields
pdbfixer
rdkit
""".strip()

    gitignore = """
__pycache__/
*.py[cod]
*.pyo

.venv/
venv/
env/

.vscode/
.idea/

RAW DATA/
PROCESSED DATA/

*.gz
*.zip
*.tar
*.tar.gz

*.dcd
*.xtc
*.trr
*.nc
*.prmtop
*.inpcrd
*.rst7
*.mdcrd

ANTECHAMBER*
ATOMTYPE.INF
sqm.in
sqm.out
leap.log
*.mol2
*.frcmod

*.tmp
*.bak
*.log

*solvated*
*equilibrated*
*production*
GPU_READY/

.env
*.key
*.pem
""".strip()

    citation = """
cff-version: 1.2.0

title: "Computational mechanistic and therapeutic prioritization of PNPLA3 D166V in MASLD"

message: "Please cite the associated publication or repository when reusing the workflow or derived results."

type: software

version: "1.0.0"

keywords:
  - PNPLA3
  - D166V
  - MASLD
  - computational biology
  - molecular docking
  - virtual screening
  - molecular modeling

authors:
  - family-names: "REPLACE"
    given-names: "WITH-AUTHOR-NAME"

repository-code: "https://github.com/REPLACE-WITH-USERNAME/REPLACE-WITH-REPOSITORY"
""".strip()

    license_status = """
# License Status

No open-source license has been automatically assigned.

Before public release, the author should select an appropriate license
if reuse of the repository is intended.
""".strip()

    scientific_status = """
# Final Scientific Status

PNPLA3 p.Asp166Val (D166V) is supported by convergent computational
structural evidence as a plausible function-altering variant.

The mutation changes the local physicochemical and interaction environment
of residue 166 while causing negligible global backbone displacement in
the generated models.

The predicted mechanism is consistent with established PNPLA3/MASLD
biology.

CID 71236597 is the principal computational therapeutic candidate and
shows a D166V-preferred docking score of -10.262 kcal/mol compared with
-9.296 kcal/mol for WT.

This does not establish experimental binding, biochemical inhibition,
functional rescue, pathogenicity, ADMET or clinical efficacy.

Direct D166V experimental validation remains the principal evidence gap.
""".strip()

    documents = {

        GITHUB_READY / "README.md":
            readme,

        GITHUB_READY / "requirements.txt":
            requirements,

        GITHUB_READY / ".gitignore":
            gitignore,

        GITHUB_READY / "CITATION.cff":
            citation,

        GITHUB_READY / "LICENSE_STATUS.txt":
            license_status,

        DOC_DIR / "FINAL_SCIENTIFIC_STATUS.md":
            scientific_status,
    }

    for path, content in documents.items():

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        path.write_text(
            content + "\n",
            encoding="utf-8",
        )


# =============================================================================
# ENVIRONMENT
# =============================================================================

def capture_environment():

    lines = [

        "GITHUB-READY ENVIRONMENT",
        "========================",
        "",
        f"Capture date: {datetime.now().isoformat()}",
        f"Python executable: {sys.executable}",
        f"Python version: {sys.version.replace(chr(10), ' ')}",
        f"Platform: {platform.platform()}",
        f"System: {platform.system()}",
        f"Machine: {platform.machine()}",
        "",
    ]

    try:

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "freeze",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )

        lines.append(
            "pip freeze:"
        )

        lines.append(
            result.stdout.strip()
        )

    except Exception as exc:

        lines.append(
            f"pip freeze unavailable: {exc}"
        )

    (
        META_DIR
        / "environment.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# MANIFESTS
# =============================================================================

def write_manifests():

    import csv

    manifest_file = (
        META_DIR
        / "GITHUB_READY_FILE_MANIFEST.csv"
    )

    with manifest_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "category",
                "source",
                "github_path",
                "size_bytes",
                "sha256",
            ],
        )

        writer.writeheader()

        for row in manifest:
            writer.writerow(row)

    exclusions_file = (
        META_DIR
        / "EXCLUDED_FILES.csv"
    )

    with exclusions_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source",
                "reason",
                "size_mb",
            ],
        )

        writer.writeheader()

        for row in exclusions:
            writer.writerow(row)


# =============================================================================
# DIRECTORY TREE
# =============================================================================

def tree(
    directory,
    prefix="",
):

    lines = []

    items = sorted(
        directory.iterdir(),
        key=lambda p: (
            not p.is_dir(),
            p.name.lower(),
        ),
    )

    for index, item in enumerate(items):

        last = (
            index
            ==
            len(items) - 1
        )

        connector = (
            "└── "
            if last
            else "├── "
        )

        lines.append(
            prefix
            + connector
            + item.name
        )

        if item.is_dir():

            extension = (
                "    "
                if last
                else "│   "
            )

            lines.extend(
                tree(
                    item,
                    prefix + extension,
                )
            )

    return lines


def write_tree():

    lines = tree(
        GITHUB_READY
    )

    (
        META_DIR
        / "DIRECTORY_TREE.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# SUMMARY
# =============================================================================

def write_summary():

    counts = {}

    for row in manifest:

        category = row[
            "category"
        ]

        counts[category] = (
            counts.get(
                category,
                0,
            )
            + 1
        )

    lines = [

        "GITHUB-READY PACKAGE SUMMARY",
        "============================",
        "",
        f"Project root: {PROJECT_ROOT}",
        f"Package: {GITHUB_READY}",
        "",
        f"Files included: {len(manifest)}",
        f"Files excluded: {len(exclusions)}",
        "",
        "Included categories:",
    ]

    for category in sorted(counts):

        lines.append(
            f"  {category}: {counts[category]}"
        )

    lines.extend(
        [
            "",
            "Original project files were not modified.",
            "Large/raw/local working files were excluded.",
            "",
            "Scientific status:",
            "COMPUTATIONALLY SUPPORTED WITH AN "
            "EXPERIMENTAL VALIDATION GAP",
        ]
    )

    (
        META_DIR
        / "PACKAGE_SUMMARY.txt"
    ).write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


# =============================================================================
# FINAL QC
# =============================================================================

def run_qc():

    required = [

        GITHUB_READY / "README.md",
        GITHUB_READY / "requirements.txt",
        GITHUB_READY / ".gitignore",
        GITHUB_READY / "CITATION.cff",
        GITHUB_READY / "LICENSE_STATUS.txt",

        DOC_DIR / "FINAL_SCIENTIFIC_STATUS.md",

        META_DIR / "environment.txt",
        META_DIR / "GITHUB_READY_FILE_MANIFEST.csv",
        META_DIR / "EXCLUDED_FILES.csv",
        META_DIR / "DIRECTORY_TREE.txt",
        META_DIR / "PACKAGE_SUMMARY.txt",
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    oversized = []

    for path in GITHUB_READY.rglob("*"):

        if not path.is_file():
            continue

        if size_mb(path) > MAX_FILE_MB:

            oversized.append(
                rel(path)
            )

    qc = {

        "package_exists":
            GITHUB_READY.exists(),

        "files_included":
            len(manifest),

        "files_excluded":
            len(exclusions),

        "required_files":
            len(required),

        "missing_required_files":
            missing,

        "oversized_files":
            oversized,

        "status":
            (
                "PASS"
                if not missing
                and not oversized
                else "CHECK_REQUIRED"
            ),
    }

    (
        META_DIR
        / "GITHUB_READY_QC.json"
    ).write_text(
        json.dumps(
            qc,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return qc


# =============================================================================
# MAIN
# =============================================================================

def main():

    print_header(
        "FINAL GITHUB-READY PROJECT PACKAGE"
    )

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    print(
        f"Output       : {GITHUB_READY}"
    )

    initialize()

    collect_scripts()
    collect_results()
    collect_figures()
    collect_structures()
    collect_step33()

    write_static_documents()
    capture_environment()
    write_manifests()
    write_tree()
    write_summary()

    qc = run_qc()

    print_header(
        "FINAL GITHUB PACKAGE RESULTS"
    )

    print(
        f"Files included : {qc['files_included']}"
    )

    print(
        f"Files excluded : {qc['files_excluded']}"
    )

    print(
        f"QC status      : {qc['status']}"
    )

    print()

    print(
        "Package:"
    )

    print(
        GITHUB_READY
    )

    print()

    print(
        "Manifest:"
    )

    print(
        META_DIR
        / "GITHUB_READY_FILE_MANIFEST.csv"
    )

    print()

    print(
        "QC:"
    )

    print(
        META_DIR
        / "GITHUB_READY_QC.json"
    )

    print()

    print(
        "Completed successfully."
    )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nPackage creation interrupted."
        )

        sys.exit(130)

    except Exception as exc:

        print_header(
            "GITHUB PACKAGE CREATION FAILED"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise