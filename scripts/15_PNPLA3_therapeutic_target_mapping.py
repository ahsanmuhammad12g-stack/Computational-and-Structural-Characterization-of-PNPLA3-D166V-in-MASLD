# ============================================================
# STEP 15
# PNPLA3 THERAPEUTIC TARGET AND MECHANISTIC SITE MAPPING
# CORRECTED VERSION — DUPLICATE PROTEIN-POSITION SAFE
# ============================================================

import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from Bio.PDB import PDBParser, ShrakeRupley


# ============================================================
# 0. PROJECT PATHS
# ============================================================

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

STEP13_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP13_COMPARATIVE_STRUCTURAL_ANALYSIS"
)

STEP14_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP14_FINAL_VARIANT_EVIDENCE_MATRIX"
)

WT_PDB = (
    STEP10_DIR
    / "structures"
    / "WT"
    / "PNPLA3_WT.pdb"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP15_THERAPEUTIC_SITE_MAPPING"
)

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
QC_DIR = OUTPUT_DIR / "QC"

for directory in [
    OUTPUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. CONSTANTS
# ============================================================

UNIPROT_ID = "Q9NST1"
EXPECTED_LENGTH = 481

PRIMARY_VARIATION_ID = 3308171
PRIMARY_PROTEIN_CHANGE = "p.Asp166Val"
PRIMARY_POSITION = 166

D166_NEIGHBOR_RADIUS = 10.0
CATALYTIC_NEIGHBOR_RADIUS = 10.0
VARIANT_CLUSTER_RADIUS = 10.0

CATALYTIC_RESIDUES = {
    47: {
        "Residue": "Ser47",
        "Amino_Acid": "Ser",
        "Role": "Catalytic residue",
        "Evidence_Source":
            "Step 11 validated catalytic-context analysis",
    },
    166: {
        "Residue": "Asp166",
        "Amino_Acid": "Asp",
        "Role": "Catalytic residue",
        "Evidence_Source":
            "Step 11 validated catalytic-context analysis",
    },
}

WEIGHTS = {
    "Biological_Relevance": 0.25,
    "Structural_Relevance": 0.20,
    "Mutation_Relevance": 0.20,
    "Accessibility": 0.15,
    "Mechanistic_Relevance": 0.20,
}


# ============================================================
# 2. QC
# ============================================================

QC = {
    "Step": "Step 15",
    "Title":
        "PNPLA3 Therapeutic Target and Mechanistic Site Mapping",
    "Start_Time": datetime.now().isoformat(),
    "Checks": [],
    "Warnings": [],
    "Errors": [],
}


def qc_check(name, status, details=""):

    QC["Checks"].append({
        "Check": name,
        "Status": status,
        "Details": details,
    })

    symbol = "PASS" if status == "PASS" else "FAIL"

    print(f"[{symbol}] {name}")

    if details:
        print(f"       {details}")


def qc_warning(message):

    QC["Warnings"].append(message)

    print(f"[WARNING] {message}")


def qc_error(message):

    QC["Errors"].append(message)

    print(f"[ERROR] {message}")


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 78)
print(
    "STEP 15 — PNPLA3 THERAPEUTIC TARGET AND "
    "MECHANISTIC SITE MAPPING"
)
print("=" * 78)

print()
print("Project:", PROJECT_ROOT)
print("UniProt:", UNIPROT_ID)
print("Primary candidate:", PRIMARY_PROTEIN_CHANGE)
print("Primary position:", PRIMARY_POSITION)
print()


# ============================================================
# 4. INPUT VALIDATION
# ============================================================

print("-" * 78)
print("CHECKING INPUT FILES")
print("-" * 78)

required_inputs = {
    "Step9F_final_candidate_selection": STEP9F_FILE,
    "Step10_WT_PDB": WT_PDB,
    "Step10_directory": STEP10_DIR,
    "Step11_directory": STEP11_DIR,
    "Step13_directory": STEP13_DIR,
    "Step14_directory": STEP14_DIR,
}

for name, path in required_inputs.items():

    exists = path.exists()

    qc_check(
        name,
        "PASS" if exists else "FAIL",
        str(path),
    )

    if not exists:
        qc_error(
            f"Required input missing: {path}"
        )

if QC["Errors"]:

    QC["End_Time"] = datetime.now().isoformat()

    with open(
        QC_DIR / "Step15_final_QC_report.json",
        "w",
    ) as handle:

        json.dump(
            QC,
            handle,
            indent=4,
        )

    sys.exit(
        "STEP 15 ABORTED: required input missing."
    )


# ============================================================
# 5. LOAD STEP 9F
# ============================================================

print()
print("-" * 78)
print("READING AUTHORITATIVE STEP 9F DATASET")
print("-" * 78)

df9 = pd.read_csv(STEP9F_FILE)

print("Rows:", len(df9))
print("Columns:", len(df9.columns))

required_columns = [
    "VariationID",
    "Protein_Change",
    "Protein_Position",
    "Reference_Amino_Acid",
    "Alternate_Amino_Acid",
    "Candidate_Group",
    "UniProt_ID",
    "Structural_Analysis_Candidate",
    "Structural_Relevance_Score",
    "Integrated_Priority_Score",
    "Integrated_Priority_Category",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df9.columns
]

qc_check(
    "Step9F required columns",
    "PASS" if not missing_columns else "FAIL",
    (
        f"{len(required_columns)} required columns present"
        if not missing_columns
        else str(missing_columns)
    ),
)

if missing_columns:
    sys.exit(
        "STEP 15 ABORTED: required Step9F columns missing."
    )


# ============================================================
# 6. VALIDATE UNIPROT
# ============================================================

uniprot_values = (
    df9["UniProt_ID"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

qc_check(
    "Step9F UniProt identity",
    "PASS"
    if UNIPROT_ID in uniprot_values
    else "FAIL",
    f"Observed UniProt IDs: {uniprot_values}",
)


# ============================================================
# 7. EXTRACT STRUCTURAL CANDIDATES
# ============================================================

structural_candidates = df9[
    df9["Structural_Analysis_Candidate"]
    .astype(str)
    .str.upper()
    .isin(["YES", "TRUE", "1"])
].copy()

structural_candidates["Protein_Position"] = pd.to_numeric(
    structural_candidates["Protein_Position"],
    errors="coerce",
)

structural_candidates = structural_candidates.dropna(
    subset=["Protein_Position"]
)

structural_candidates["Protein_Position"] = (
    structural_candidates["Protein_Position"]
    .astype(int)
)

print()
print("Structural candidates:", len(structural_candidates))

qc_check(
    "Nine structural candidates",
    "PASS"
    if len(structural_candidates) == 9
    else "FAIL",
    f"Observed: {len(structural_candidates)}",
)


# ============================================================
# 8. EXPLICIT DUPLICATE-POSITION CHECK
# ============================================================

position_counts = (
    structural_candidates["Protein_Position"]
    .value_counts()
)

duplicate_positions = (
    position_counts[position_counts > 1]
)

if len(duplicate_positions) > 0:

    print()
    print(
        "Multiple structural variants share protein positions:"
    )

    for position, count in duplicate_positions.items():

        variants = structural_candidates[
            structural_candidates["Protein_Position"] == position
        ]["Protein_Change"].tolist()

        print(
            f"  Position {position}: "
            f"{count} variants -> {variants}"
        )

    qc_warning(
        "Multiple variants share a protein position. "
        "Spatial clustering will therefore be calculated "
        "at unique-position level and merged safely."
    )

qc_check(
    "Structural candidate VariationID uniqueness",
    "PASS"
    if structural_candidates["VariationID"].is_unique
    else "FAIL",
    "Each structural candidate VariationID is unique.",
)


# ============================================================
# 9. VALIDATE D166V
# ============================================================

primary_rows = df9[
    (df9["VariationID"] == PRIMARY_VARIATION_ID)
    &
    (df9["Protein_Position"] == PRIMARY_POSITION)
]

qc_check(
    "Primary D166V identification",
    "PASS"
    if len(primary_rows) == 1
    else "FAIL",
    f"Matching rows: {len(primary_rows)}",
)

if len(primary_rows) != 1:
    sys.exit(
        "STEP 15 ABORTED: D166V not uniquely identified."
    )


# ============================================================
# 10. LOAD WT STRUCTURE
# ============================================================

print()
print("-" * 78)
print("LOADING VALIDATED PNPLA3 WT STRUCTURE")
print("-" * 78)

parser = PDBParser(QUIET=True)

structure = parser.get_structure(
    "PNPLA3_WT",
    str(WT_PDB),
)

model = structure[0]

chains = list(model.get_chains())

print(
    "Chains:",
    [chain.id for chain in chains],
)

if "A" in model:
    chain = model["A"]
else:
    chain = chains[0]
    qc_warning(
        f"Chain A unavailable; using chain {chain.id}."
    )

residues = [
    residue
    for residue in chain.get_residues()
    if residue.id[0] == " "
]

print(
    "Protein residues:",
    len(residues),
)

qc_check(
    "WT structure residue count",
    "PASS"
    if len(residues) == EXPECTED_LENGTH
    else "FAIL",
    f"Expected {EXPECTED_LENGTH}; observed {len(residues)}",
)

residue_dict = {
    residue.id[1]: residue
    for residue in residues
}


# ============================================================
# 11. VALIDATE D166
# ============================================================

if PRIMARY_POSITION not in residue_dict:
    sys.exit(
        "STEP 15 ABORTED: residue 166 missing."
    )

d166_residue = residue_dict[PRIMARY_POSITION]

qc_check(
    "WT residue 166 identity",
    "PASS"
    if d166_residue.resname == "ASP"
    else "FAIL",
    f"Observed residue 166: {d166_residue.resname}",
)


# ============================================================
# 12. SOLVENT ACCESSIBILITY
# ============================================================

print()
print("-" * 78)
print("CALCULATING RESIDUE SOLVENT ACCESSIBILITY")
print("-" * 78)

sr = ShrakeRupley(
    probe_radius=1.4,
    n_points=960,
)

sr.compute(
    structure,
    level="R",
)

accessibility_records = []

for residue in residues:

    position = residue.id[1]

    try:
        sasa = float(residue.sasa)
    except Exception:
        sasa = np.nan

    accessibility_records.append({
        "Protein_Position": position,
        "Residue": residue.resname,
        "Residue_SASA_A2": sasa,
    })

accessibility_df = pd.DataFrame(
    accessibility_records
)

max_sasa = accessibility_df[
    "Residue_SASA_A2"
].max()

if pd.notna(max_sasa) and max_sasa > 0:

    accessibility_df[
        "Relative_SASA"
    ] = (
        accessibility_df["Residue_SASA_A2"]
        / max_sasa
    )

else:

    accessibility_df[
        "Relative_SASA"
    ] = np.nan

accessibility_df[
    "Accessibility_Category"
] = pd.cut(
    accessibility_df["Relative_SASA"],
    bins=[
        -np.inf,
        0.10,
        0.30,
        np.inf,
    ],
    labels=[
        "Low",
        "Intermediate",
        "High",
    ],
)

qc_check(
    "Residue SASA calculation",
    "PASS"
    if accessibility_df[
        "Residue_SASA_A2"
    ].notna().any()
    else "FAIL",
    (
        "Residues with SASA: "
        f"{accessibility_df['Residue_SASA_A2'].notna().sum()}"
    ),
)


# ============================================================
# 13. THREE-LETTER TO ONE-LETTER
# ============================================================

three_to_one = {
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


# ============================================================
# 14. CATALYTIC MAP
# ============================================================

print()
print("-" * 78)
print("BUILDING CATALYTIC / MECHANISTIC RESIDUE MAP")
print("-" * 78)

catalytic_records = []

for position, info in CATALYTIC_RESIDUES.items():

    if position not in residue_dict:
        qc_warning(
            f"Catalytic residue {position} absent."
        )
        continue

    observed = residue_dict[position].resname

    catalytic_records.append({
        "Protein_Position": position,
        "Residue": info["Residue"],
        "Observed_Three_Letter": observed,
        "Observed_One_Letter":
            three_to_one.get(observed, "?"),
        "Role": info["Role"],
        "Evidence_Source":
            info["Evidence_Source"],
    })

catalytic_df = pd.DataFrame(
    catalytic_records
)

qc_check(
    "Catalytic residues mapped",
    "PASS"
    if len(catalytic_df) == 2
    else "FAIL",
    f"Mapped residues: {len(catalytic_df)}",
)


# ============================================================
# 15. DISTANCE FUNCTION
# ============================================================

def minimum_heavy_atom_distance(
    residue_a,
    residue_b,
):

    atoms_a = [
        atom
        for atom in residue_a
        if atom.element.upper() != "H"
    ]

    atoms_b = [
        atom
        for atom in residue_b
        if atom.element.upper() != "H"
    ]

    if not atoms_a or not atoms_b:
        return np.nan

    minimum = np.inf

    for atom_a in atoms_a:

        for atom_b in atoms_b:

            distance = atom_a - atom_b

            if distance < minimum:
                minimum = distance

    return float(minimum)


# ============================================================
# 16. D166 NEIGHBORHOOD
# ============================================================

print()
print("-" * 78)
print("MAPPING D166V MECHANISTIC NEIGHBORHOOD")
print("-" * 78)

d166_records = []

for position, residue in residue_dict.items():

    distance = minimum_heavy_atom_distance(
        d166_residue,
        residue,
    )

    if (
        pd.notna(distance)
        and distance <= D166_NEIGHBOR_RADIUS
    ):

        d166_records.append({
            "Protein_Position": position,
            "Residue": residue.resname,
            "Distance_to_D166_A": distance,
            "D166_Neighborhood": True,
        })

d166_neighborhood_df = pd.DataFrame(
    d166_records
).sort_values(
    "Distance_to_D166_A"
)

d166_neighborhood_df = d166_neighborhood_df.merge(
    accessibility_df,
    on=[
        "Protein_Position",
        "Residue",
    ],
    how="left",
)

print(
    "Residues within",
    D166_NEIGHBOR_RADIUS,
    "Å of D166:",
    len(d166_neighborhood_df),
)


# ============================================================
# 17. CATALYTIC NEIGHBORHOOD
# ============================================================

catalytic_neighborhood_records = []

for catalytic_position in CATALYTIC_RESIDUES:

    catalytic_residue = residue_dict[
        catalytic_position
    ]

    for position, residue in residue_dict.items():

        distance = minimum_heavy_atom_distance(
            catalytic_residue,
            residue,
        )

        if (
            pd.notna(distance)
            and distance <= CATALYTIC_NEIGHBOR_RADIUS
        ):

            catalytic_neighborhood_records.append({
                "Catalytic_Position":
                    catalytic_position,

                "Protein_Position":
                    position,

                "Residue":
                    residue.resname,

                "Distance_to_Catalytic_Residue_A":
                    distance,
            })

catalytic_neighborhood_df = pd.DataFrame(
    catalytic_neighborhood_records
)


# ============================================================
# 18. MAP THE NINE STRUCTURAL CANDIDATES
# ============================================================

print()
print("-" * 78)
print("MAPPING ALL NINE STRUCTURAL CANDIDATES")
print("-" * 78)

candidate_records = []

for _, row in structural_candidates.iterrows():

    position = int(
        row["Protein_Position"]
    )

    residue = residue_dict[position]

    distance_to_d166 = (
        minimum_heavy_atom_distance(
            residue,
            d166_residue,
        )
    )

    distance_to_ser47 = (
        minimum_heavy_atom_distance(
            residue,
            residue_dict[47],
        )
    )

    distance_to_asp166 = (
        minimum_heavy_atom_distance(
            residue,
            residue_dict[166],
        )
    )

    sasa_row = accessibility_df[
        accessibility_df["Protein_Position"]
        == position
    ]

    if len(sasa_row):

        sasa = float(
            sasa_row["Residue_SASA_A2"].iloc[0]
        )

        relative_sasa = float(
            sasa_row["Relative_SASA"].iloc[0]
        )

        accessibility_category = str(
            sasa_row["Accessibility_Category"].iloc[0]
        )

    else:

        sasa = np.nan
        relative_sasa = np.nan
        accessibility_category = "Unknown"

    candidate_records.append({

        "VariationID":
            int(row["VariationID"]),

        "Protein_Change":
            row["Protein_Change"],

        "Protein_Position":
            position,

        "WT_Amino_Acid":
            row["Reference_Amino_Acid"],

        "Mutant_Amino_Acid":
            row["Alternate_Amino_Acid"],

        "Candidate_Group":
            row["Candidate_Group"],

        "Integrated_Priority_Score":
            row["Integrated_Priority_Score"],

        "Structural_Relevance_Score":
            row["Structural_Relevance_Score"],

        "Distance_to_D166_A":
            distance_to_d166,

        "Distance_to_Ser47_A":
            distance_to_ser47,

        "Distance_to_Asp166_A":
            distance_to_asp166,

        "Residue_SASA_A2":
            sasa,

        "Relative_SASA":
            relative_sasa,

        "Accessibility_Category":
            accessibility_category,

        "Within_D166_10A":
            bool(
                pd.notna(distance_to_d166)
                and distance_to_d166
                <= D166_NEIGHBOR_RADIUS
            ),

        "Within_Catalytic_10A":
            bool(
                (
                    pd.notna(distance_to_ser47)
                    and distance_to_ser47
                    <= CATALYTIC_NEIGHBOR_RADIUS
                )
                or
                (
                    pd.notna(distance_to_asp166)
                    and distance_to_asp166
                    <= CATALYTIC_NEIGHBOR_RADIUS
                )
            ),
    })


candidate_mapping_df = pd.DataFrame(
    candidate_records
)

qc_check(
    "All nine structural candidates structurally mapped",
    "PASS"
    if len(candidate_mapping_df) == 9
    else "FAIL",
    f"Mapped: {len(candidate_mapping_df)} / 9",
)


# ============================================================
# 19. CRITICAL DUPLICATE-SAFE SPATIAL CLUSTERING
# ============================================================

print()
print("-" * 78)
print("CALCULATING STRUCTURAL-VARIANT SPATIAL CLUSTERS")
print("-" * 78)

# IMPORTANT:
# Multiple variants can occupy the same protein position.
# Therefore spatial clustering is calculated from UNIQUE
# protein positions, not from variant rows.

unique_candidate_positions = sorted(
    candidate_mapping_df[
        "Protein_Position"
    ]
    .drop_duplicates()
    .tolist()
)

cluster_records = []

for position in unique_candidate_positions:

    residue = residue_dict[position]

    nearby_positions = []

    for other_position in unique_candidate_positions:

        other_residue = residue_dict[
            other_position
        ]

        distance = minimum_heavy_atom_distance(
            residue,
            other_residue,
        )

        if (
            pd.notna(distance)
            and distance <= VARIANT_CLUSTER_RADIUS
        ):

            nearby_positions.append(
                other_position
            )

    cluster_records.append({

        "Protein_Position":
            position,

        "Nearby_Structural_Candidate_Count":
            len(set(nearby_positions)),

        "Nearby_Structural_Candidate_Positions":
            ",".join(
                str(x)
                for x in sorted(
                    set(nearby_positions)
                )
            ),
    })

variant_cluster_df = pd.DataFrame(
    cluster_records
)

# HARD QC: one row per unique protein position.
qc_check(
    "Unique-position spatial cluster table",
    "PASS"
    if (
        len(variant_cluster_df)
        == len(unique_candidate_positions)
    )
    else "FAIL",
    (
        f"{len(variant_cluster_df)} unique positions "
        f"for {len(unique_candidate_positions)} "
        f"unique candidate positions"
    ),
)

# SAFE MERGE:
# one unique-position cluster row is merged onto each
# variant row.
candidate_mapping_df = candidate_mapping_df.merge(
    variant_cluster_df,
    on="Protein_Position",
    how="left",
    validate="many_to_one",
)

# HARD QC:
# the merge MUST preserve the nine candidate variants.
qc_check(
    "Duplicate-safe candidate mapping merge",
    "PASS"
    if len(candidate_mapping_df) == 9
    else "FAIL",
    f"Rows after merge: {len(candidate_mapping_df)} / 9",
)


# ============================================================
# 20. FUNCTIONAL ARCHITECTURE
# ============================================================

print()
print("-" * 78)
print("BUILDING COMPLETE PNPLA3 FUNCTIONAL ARCHITECTURE")
print("-" * 78)

architecture_df = accessibility_df.copy()

architecture_df["Is_Catalytic_Residue"] = (
    architecture_df["Protein_Position"]
    .isin(CATALYTIC_RESIDUES.keys())
)

architecture_df["Catalytic_Roles"] = ""

for position, info in CATALYTIC_RESIDUES.items():

    architecture_df.loc[
        architecture_df["Protein_Position"]
        == position,
        "Catalytic_Roles",
    ] = info["Role"]

architecture_df["Is_D166_Neighborhood"] = (
    architecture_df["Protein_Position"]
    .isin(
        d166_neighborhood_df[
            "Protein_Position"
        ].tolist()
    )
)

architecture_df["Is_Structural_Candidate"] = (
    architecture_df["Protein_Position"]
    .isin(
        candidate_mapping_df[
            "Protein_Position"
        ].tolist()
    )
)

architecture_df[
    "Structural_Candidate_Variants"
] = ""

for _, row in structural_candidates.iterrows():

    position = row["Protein_Position"]

    current = architecture_df.loc[
        architecture_df["Protein_Position"]
        == position,
        "Structural_Candidate_Variants",
    ]

    if len(current):

        existing = str(
            current.iloc[0]
        )

        variant = str(
            row["Protein_Change"]
        )

        if existing in ["", "nan"]:

            new_value = variant

        elif variant not in existing.split(";"):

            new_value = existing + ";" + variant

        else:

            new_value = existing

        architecture_df.loc[
            architecture_df["Protein_Position"]
            == position,
            "Structural_Candidate_Variants",
        ] = new_value


def classify_residue(row):

    position = int(
        row["Protein_Position"]
    )

    if position in CATALYTIC_RESIDUES:
        return "Catalytic residue"

    if (
        bool(row["Is_D166_Neighborhood"])
        and bool(row["Is_Structural_Candidate"])
    ):
        return "D166V-proximal structural candidate"

    if bool(row["Is_D166_Neighborhood"]):
        return "D166V mechanistic neighborhood"

    if bool(row["Is_Structural_Candidate"]):
        return "Structural candidate residue"

    return "General structural region"


architecture_df[
    "Functional_Region_Class"
] = architecture_df.apply(
    classify_residue,
    axis=1,
)


# ============================================================
# 21. CANDIDATE THERAPEUTIC REGIONS
# ============================================================

print()
print("-" * 78)
print("GENERATING CANDIDATE THERAPEUTIC REGIONS")
print("-" * 78)

therapeutic_regions = []

catalytic_positions = sorted(
    CATALYTIC_RESIDUES.keys()
)

catalytic_region_positions = sorted(
    set(
        catalytic_neighborhood_df[
            "Protein_Position"
        ].tolist()
    )
)

therapeutic_regions.append({

    "Site_ID":
        "SITE_01",

    "Site_Name":
        "PNPLA3 catalytic-associated region",

    "Site_Type":
        "Catalytic-associated region",

    "Defining_Residues":
        ",".join(
            f"{three_to_one.get(
                residue_dict[p].resname, '?'
            )}{p}"
            for p in catalytic_positions
        ),

    "Region_Residue_Count":
        len(catalytic_region_positions),

    "D166_Included":
        PRIMARY_POSITION
        in catalytic_region_positions,

    "Biological_Rationale":
        (
            "Contains the validated PNPLA3 catalytic-context "
            "residues Ser47 and Asp166."
        ),

    "Structural_Rationale":
        (
            "Defined from the validated PNPLA3 WT structure "
            "using a 10 Å heavy-atom neighborhood around "
            "catalytic residues."
        ),

    "Mutation_Rationale":
        (
            "Includes the primary D166V catalytic-context "
            "candidate."
        ),

    "Evidence_Level":
        "High",

    "Therapeutic_Status":
        "Candidate therapeutic region; pocket not yet established",
})


d166_region_positions = sorted(
    d166_neighborhood_df[
        "Protein_Position"
    ].tolist()
)

therapeutic_regions.append({

    "Site_ID":
        "SITE_02",

    "Site_Name":
        "D166V-centered mechanistic region",

    "Site_Type":
        "Mutation-proximal mechanistic region",

    "Defining_Residues":
        f"D{PRIMARY_POSITION}",

    "Region_Residue_Count":
        len(d166_region_positions),

    "D166_Included":
        True,

    "Biological_Rationale":
        (
            "Centered on the primary high-priority PNPLA3 "
            "D166V candidate and its catalytic context."
        ),

    "Structural_Rationale":
        (
            "Defined from actual WT 3D coordinates using a "
            "10 Å heavy-atom neighborhood around residue 166."
        ),

    "Mutation_Rationale":
        (
            "Directly represents the structural neighborhood "
            "of the selected primary mutation."
        ),

    "Evidence_Level":
        "High",

    "Therapeutic_Status":
        "Candidate mechanistic region; pocket not yet established",
})


hotspot_candidates = candidate_mapping_df[
    candidate_mapping_df[
        "Nearby_Structural_Candidate_Count"
    ] > 1
]

hotspot_positions = sorted(
    hotspot_candidates[
        "Protein_Position"
    ]
    .drop_duplicates()
    .tolist()
)

if hotspot_positions:

    therapeutic_regions.append({

        "Site_ID":
            "SITE_03",

        "Site_Name":
            "Structural-variant spatial hotspot",

        "Site_Type":
            "Comparative structural hotspot",

        "Defining_Residues":
            ",".join(
                f"{three_to_one.get(
                    residue_dict[p].resname, '?'
                )}{p}"
                for p in hotspot_positions
            ),

        "Region_Residue_Count":
            len(hotspot_positions),

        "D166_Included":
            PRIMARY_POSITION in hotspot_positions,

        "Biological_Rationale":
            (
                "Contains multiple structurally prioritized "
                "PNPLA3 variant positions in spatial proximity."
            ),

        "Structural_Rationale":
            (
                "Identified by 3D spatial proximity among the "
                "nine Step 13 structural candidates."
            ),

        "Mutation_Rationale":
            (
                "Represents a mutation-enriched structural region."
            ),

        "Evidence_Level":
            "Moderate",

        "Therapeutic_Status":
            "Exploratory therapeutic region; pocket not yet established",
    })


accessible_catalytic = d166_neighborhood_df[
    d166_neighborhood_df["Relative_SASA"] >= 0.30
]

accessible_positions = sorted(
    accessible_catalytic[
        "Protein_Position"
    ].tolist()
)

if accessible_positions:

    therapeutic_regions.append({

        "Site_ID":
            "SITE_04",

        "Site_Name":
            "Accessible catalytic-context surface",

        "Site_Type":
            "Surface-accessible mechanistic region",

        "Defining_Residues":
            ",".join(
                f"{three_to_one.get(
                    residue_dict[p].resname, '?'
                )}{p}"
                for p in accessible_positions
            ),

        "Region_Residue_Count":
            len(accessible_positions),

        "D166_Included":
            PRIMARY_POSITION
            in accessible_positions,

        "Biological_Rationale":
            (
                "Contains surface-accessible residues within "
                "the D166V mechanistic neighborhood."
            ),

        "Structural_Rationale":
            (
                "Combines actual 3D proximity with residue "
                "solvent accessibility."
            ),

        "Mutation_Rationale":
            (
                "May provide a testable region for "
                "mutation-sensitive ligandability analysis."
            ),

        "Evidence_Level":
            "Moderate",

        "Therapeutic_Status":
            "Exploratory ligandability region; pocket not yet established",
    })


therapeutic_regions_df = pd.DataFrame(
    therapeutic_regions
)


# ============================================================
# 22. REGION METRICS
# ============================================================

print()
print("-" * 78)
print("CALCULATING THERAPEUTIC REGION METRICS")
print("-" * 78)


def region_positions_from_row(row):

    text = str(
        row["Defining_Residues"]
    )

    positions = []

    for token in text.split(","):

        digits = "".join(
            character
            for character in token
            if character.isdigit()
        )

        if digits:
            positions.append(
                int(digits)
            )

    return sorted(
        set(positions)
    )


region_metric_records = []

for _, row in therapeutic_regions_df.iterrows():

    positions = region_positions_from_row(
        row
    )

    region_data = architecture_df[
        architecture_df["Protein_Position"]
        .isin(positions)
    ].copy()

    if len(region_data) == 0:
        continue

    mean_relative_sasa = (
        region_data["Relative_SASA"]
        .mean()
    )

    high_accessibility_fraction = (
        (
            region_data[
                "Accessibility_Category"
            ].astype(str)
            == "High"
        ).mean()
    )

    catalytic_fraction = (
        region_data[
            "Is_Catalytic_Residue"
        ].mean()
    )

    d166_fraction = (
        region_data[
            "Is_D166_Neighborhood"
        ].mean()
    )

    structural_candidate_fraction = (
        region_data[
            "Is_Structural_Candidate"
        ].mean()
    )

    if (
        row["Site_Type"]
        == "Catalytic-associated region"
    ):

        biological_score = 1.0

    elif (
        "mechanistic"
        in row["Site_Type"].lower()
    ):

        biological_score = 0.85

    elif (
        "structural hotspot"
        in row["Site_Type"].lower()
    ):

        biological_score = 0.65

    else:

        biological_score = 0.50

    structural_score = min(
        1.0,
        0.5 * structural_candidate_fraction
        + 0.5,
    )

    mutation_score = min(
        1.0,
        0.5 * d166_fraction
        + 0.5 * structural_candidate_fraction,
    )

    accessibility_score = min(
        1.0,
        max(
            0.0,
            float(mean_relative_sasa),
        ),
    )

    if (
        row["Site_Type"]
        == "Catalytic-associated region"
    ):

        mechanistic_score = 1.0

    elif (
        "mechanistic"
        in row["Site_Type"].lower()
    ):

        mechanistic_score = 0.95

    elif (
        "catalytic"
        in row["Site_Type"].lower()
    ):

        mechanistic_score = 0.90

    else:

        mechanistic_score = 0.60

    final_score = (
        WEIGHTS["Biological_Relevance"]
        * biological_score
        +
        WEIGHTS["Structural_Relevance"]
        * structural_score
        +
        WEIGHTS["Mutation_Relevance"]
        * mutation_score
        +
        WEIGHTS["Accessibility"]
        * accessibility_score
        +
        WEIGHTS["Mechanistic_Relevance"]
        * mechanistic_score
    )

    region_metric_records.append({

        "Site_ID":
            row["Site_ID"],

        "Site_Name":
            row["Site_Name"],

        "Site_Type":
            row["Site_Type"],

        "Defining_Residues":
            row["Defining_Residues"],

        "Region_Residue_Count":
            len(region_data),

        "Mean_Relative_SASA":
            mean_relative_sasa,

        "High_Accessibility_Fraction":
            high_accessibility_fraction,

        "Biological_Relevance_Score":
            biological_score,

        "Structural_Relevance_Score":
            structural_score,

        "Mutation_Relevance_Score":
            mutation_score,

        "Accessibility_Score":
            accessibility_score,

        "Mechanistic_Relevance_Score":
            mechanistic_score,

        "Therapeutic_Region_Score":
            final_score,

        "Evidence_Level":
            row["Evidence_Level"],

        "Therapeutic_Status":
            row["Therapeutic_Status"],

        "Biological_Rationale":
            row["Biological_Rationale"],

        "Structural_Rationale":
            row["Structural_Rationale"],

        "Mutation_Rationale":
            row["Mutation_Rationale"],
    })


region_metrics_df = pd.DataFrame(
    region_metric_records
)

region_metrics_df = (
    region_metrics_df
    .sort_values(
        "Therapeutic_Region_Score",
        ascending=False,
    )
    .reset_index(drop=True)
)

region_metrics_df[
    "Therapeutic_Region_Rank"
] = np.arange(
    1,
    len(region_metrics_df) + 1,
)


def region_priority_category(score):

    if score >= 0.75:
        return "High-priority candidate region"

    if score >= 0.55:
        return "Moderate-priority candidate region"

    return "Exploratory candidate region"


region_metrics_df[
    "Therapeutic_Region_Category"
] = region_metrics_df[
    "Therapeutic_Region_Score"
].apply(
    region_priority_category
)


# ============================================================
# 23. D166V SUMMARY
# ============================================================

print()
print("-" * 78)
print("BUILDING D166V MECHANISTIC SUMMARY")
print("-" * 78)

d166_sasa = accessibility_df[
    accessibility_df["Protein_Position"]
    == PRIMARY_POSITION
]

if len(d166_sasa):

    d166_sasa_value = float(
        d166_sasa[
            "Residue_SASA_A2"
        ].iloc[0]
    )

    d166_relative_sasa = float(
        d166_sasa[
            "Relative_SASA"
        ].iloc[0]
    )

    d166_accessibility = str(
        d166_sasa[
            "Accessibility_Category"
        ].iloc[0]
    )

else:

    d166_sasa_value = np.nan
    d166_relative_sasa = np.nan
    d166_accessibility = "Unknown"


d166_mechanistic_summary = pd.DataFrame([{

    "VariationID":
        PRIMARY_VARIATION_ID,

    "Protein_Change":
        PRIMARY_PROTEIN_CHANGE,

    "Protein_Position":
        PRIMARY_POSITION,

    "WT_Amino_Acid":
        "Asp",

    "Mutant_Amino_Acid":
        "Val",

    "Catalytic_Context":
        "Catalytic residue",

    "Nearest_Catalytic_Residue":
        "Asp166",

    "D166_SASA_A2":
        d166_sasa_value,

    "D166_Relative_SASA":
        d166_relative_sasa,

    "D166_Accessibility_Category":
        d166_accessibility,

    "D166_Neighborhood_Radius_A":
        D166_NEIGHBOR_RADIUS,

    "D166_Neighborhood_Residue_Count":
        len(d166_neighborhood_df),

    "D166_Neighborhood_Residues":
        ",".join(
            f"{three_to_one.get(
                residue_dict[p].resname, '?'
            )}{p}"
            for p in d166_neighborhood_df[
                "Protein_Position"
            ]
        ),

    "Interpretation":
        (
            "D166V is located within the validated PNPLA3 "
            "catalytic context. The therapeutic relevance "
            "of this region is mechanistically motivated, "
            "but a drug-binding pocket has not yet been "
            "established."
        ),
}])


# ============================================================
# 24. WRITE TABLES
# ============================================================

print()
print("-" * 78)
print("WRITING STEP 15 TABLES")
print("-" * 78)

tables = {

    "Table_S15_1_PNPLA3_functional_architecture.csv":
        architecture_df,

    "Table_S15_2_catalytic_mechanistic_residues.csv":
        catalytic_df,

    "Table_S15_3_D166V_mechanistic_neighborhood.csv":
        d166_neighborhood_df,

    "Table_S15_4_nine_structural_candidates_site_mapping.csv":
        candidate_mapping_df,

    "Table_S15_5_candidate_therapeutic_regions.csv":
        therapeutic_regions_df,

    "Table_S15_6_therapeutic_region_prioritization.csv":
        region_metrics_df,

    "Table_S15_7_D166V_mechanistic_summary.csv":
        d166_mechanistic_summary,

    "Table_S15_8_residue_accessibility.csv":
        accessibility_df,

    "Table_S15_9_catalytic_neighborhood.csv":
        catalytic_neighborhood_df,
}

for filename, dataframe in tables.items():

    dataframe.to_csv(
        TABLE_DIR / filename,
        index=False,
    )

    print(
        "Created:",
        filename,
    )


# ============================================================
# 25. FIGURE 1
# ============================================================

print()
print("-" * 78)
print("GENERATING FIGURES")
print("-" * 78)

fig, ax = plt.subplots(
    figsize=(14, 5)
)

ax.plot(
    architecture_df[
        "Protein_Position"
    ],
    architecture_df[
        "Relative_SASA"
    ],
    linewidth=1.2,
    label="Relative solvent accessibility",
)

for position in CATALYTIC_RESIDUES:

    ax.axvline(
        position,
        linestyle="--",
        linewidth=1.2,
    )

    ax.text(
        position,
        1.02,
        (
            f"{three_to_one.get(
                residue_dict[position].resname,
                '?'
            )}{position}"
        ),
        rotation=90,
        ha="center",
        va="bottom",
        transform=ax.get_xaxis_transform(),
    )

ax.axvline(
    PRIMARY_POSITION,
    linewidth=2.0,
)

ax.set_xlabel(
    "PNPLA3 protein position"
)

ax.set_ylabel(
    "Relative solvent accessibility"
)

ax.set_title(
    "PNPLA3 Functional Architecture and Catalytic Context"
)

ax.set_xlim(
    1,
    EXPECTED_LENGTH,
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "Figure_S15_1_PNPLA3_functional_architecture.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 26. FIGURE 2
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)

plot_df = (
    d166_neighborhood_df
    .head(20)
    .sort_values(
        "Distance_to_D166_A",
        ascending=True,
    )
)

labels = [
    (
        f"{three_to_one.get(
            residue_dict[p].resname,
            '?'
        )}{p}"
    )
    for p in plot_df[
        "Protein_Position"
    ]
]

ax.barh(
    labels,
    plot_df[
        "Distance_to_D166_A"
    ],
)

ax.axvline(
    D166_NEIGHBOR_RADIUS,
    linestyle="--",
    linewidth=1.2,
    label="10 Å neighborhood threshold",
)

ax.set_xlabel(
    "Minimum heavy-atom distance to D166 (Å)"
)

ax.set_ylabel(
    "Residue"
)

ax.set_title(
    "D166V-Centered PNPLA3 Structural Neighborhood"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "Figure_S15_2_D166V_mechanistic_neighborhood.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 27. FIGURE 3
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.barh(
    region_metrics_df[
        "Site_Name"
    ],
    region_metrics_df[
        "Therapeutic_Region_Score"
    ],
)

ax.set_xlabel(
    "Therapeutic-region prioritization score"
)

ax.set_ylabel(
    "Candidate region"
)

ax.set_title(
    "PNPLA3 Candidate Therapeutic Region Prioritization"
)

ax.set_xlim(
    0,
    1,
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "Figure_S15_3_therapeutic_region_prioritization.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 28. FIGURE 4
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

plot_candidates = (
    candidate_mapping_df
    .sort_values(
        "Distance_to_D166_A"
    )
)

ax.barh(
    plot_candidates[
        "Protein_Change"
    ],
    plot_candidates[
        "Distance_to_D166_A"
    ],
)

ax.axvline(
    D166_NEIGHBOR_RADIUS,
    linestyle="--",
    linewidth=1.2,
    label="10 Å D166 neighborhood",
)

ax.set_xlabel(
    "Distance to D166 (Å)"
)

ax.set_ylabel(
    "Structural candidate"
)

ax.set_title(
    "Spatial Relationship of Prioritized PNPLA3 Variants to D166"
)

ax.legend()

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "Figure_S15_4_variant_D166_spatial_relationship.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 29. FIGURE 5
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)

plot_accessibility = (
    candidate_mapping_df
    .sort_values(
        "Relative_SASA"
    )
)

ax.barh(
    plot_accessibility[
        "Protein_Change"
    ],
    plot_accessibility[
        "Relative_SASA"
    ],
)

ax.set_xlabel(
    "Relative solvent accessibility"
)

ax.set_ylabel(
    "Structural candidate"
)

ax.set_title(
    "Surface Accessibility of the Nine PNPLA3 Structural Candidates"
)

ax.set_xlim(
    0,
    max(
        1.0,
        float(
            candidate_mapping_df[
                "Relative_SASA"
            ].max()
        ) + 0.05,
    ),
)

fig.tight_layout()

fig.savefig(
    FIGURE_DIR
    / "Figure_S15_5_candidate_residue_accessibility.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 30. FINAL RECOMMENDATION
# ============================================================

top_region = (
    region_metrics_df.iloc[0]
)

final_recommendation = {

    "Primary_Therapeutic_Region":
        top_region["Site_Name"],

    "Primary_Site_ID":
        top_region["Site_ID"],

    "Primary_Site_Type":
        top_region["Site_Type"],

    "Primary_Site_Score":
        float(
            top_region[
                "Therapeutic_Region_Score"
            ]
        ),

    "Primary_Site_Category":
        top_region[
            "Therapeutic_Region_Category"
        ],

    "D166V_Remains_Primary_Mechanistic_Variant":
        True,

    "Pocket_Discovery_Required":
        True,

    "Docking_Performed":
        False,

    "Molecular_Dynamics_Performed":
        False,

    "Energy_Minimization_Performed":
        False,

    "Scientific_Interpretation":
        (
            "Step 15 identifies candidate PNPLA3 therapeutic "
            "regions using biological relevance, validated "
            "structural geometry, mutation proximity, residue "
            "accessibility and mechanistic context. These are "
            "candidate regions rather than experimentally "
            "validated ligand-binding pockets. Step 16 must "
            "perform explicit WT-versus-D166V pocket discovery "
            "and characterization."
        ),
}

with open(
    OUTPUT_DIR
    / "Step15_final_site_recommendation.json",
    "w",
) as handle:

    json.dump(
        final_recommendation,
        handle,
        indent=4,
    )


# ============================================================
# 31. SUMMARY
# ============================================================

summary_lines = [

    "=" * 78,

    "STEP 15 — PNPLA3 THERAPEUTIC TARGET AND MECHANISTIC SITE MAPPING",

    "=" * 78,

    "",

    f"UniProt: {UNIPROT_ID}",

    f"Protein length: {EXPECTED_LENGTH}",

    f"Primary variant: {PRIMARY_PROTEIN_CHANGE}",

    f"Primary VariationID: {PRIMARY_VARIATION_ID}",

    f"Primary residue: {PRIMARY_POSITION}",

    "",

    "INPUT VALIDATION",

    "----------------",

    f"Step9F rows: {len(df9)}",

    f"Structural candidates: {len(structural_candidates)}",

    f"Unique structural candidate positions: "
    f"{len(unique_candidate_positions)}",

    f"WT structure residues: {len(residues)}",

    "",

    "CATALYTIC CONTEXT",

    "-----------------",

    "Ser47: validated catalytic-context residue",

    "Asp166: validated catalytic-context residue",

    f"D166 neighborhood radius: "
    f"{D166_NEIGHBOR_RADIUS} Å",

    f"D166 neighborhood residues: "
    f"{len(d166_neighborhood_df)}",

    "",

    "DUPLICATE POSITION HANDLING",

    "----------------------------",

    (
        "Spatial clustering was calculated at unique "
        "protein-position level."
    ),

    (
        "Multiple variants sharing one residue position "
        "were retained as separate variants."
    ),

    "",

    "STRUCTURAL CANDIDATE COUNT",

    "---------------------------",

    (
        f"Variant-level structural candidates: "
        f"{len(candidate_mapping_df)}"
    ),

    (
        f"Unique structural candidate positions: "
        f"{len(unique_candidate_positions)}"
    ),

    "",

    "PRIMARY D166V INTERPRETATION",

    "---------------------------",

    (
        "D166V remains the primary mechanistic candidate "
        "because it directly occupies the validated "
        "catalytic context."
    ),

    (
        "Step 15 does not claim that D166V itself is a "
        "drug-binding pocket."
    ),

    (
        "The D166-centered region is carried forward as "
        "a mechanistically justified candidate region "
        "for explicit pocket discovery."
    ),

    "",

    "THERAPEUTIC REGION PRIORITIZATION",

    "--------------------------------",
]

for _, row in region_metrics_df.iterrows():

    summary_lines.append(
        f"{int(row['Therapeutic_Region_Rank'])}. "
        f"{row['Site_Name']} | "
        f"Score={row['Therapeutic_Region_Score']:.4f} | "
        f"{row['Therapeutic_Region_Category']}"
    )

summary_lines.extend([

    "",

    "IMPORTANT LIMITATIONS",

    "---------------------",

    "1. No docking was performed in Step 15.",

    "2. No molecular dynamics was performed in Step 15.",

    "3. No energy minimization was performed in Step 15.",

    (
        "4. Candidate therapeutic regions are hypotheses, "
        "not proven drug-binding sites."
    ),

    (
        "5. Step 16 must explicitly identify and compare "
        "WT/D166V pockets."
    ),

    "",

    "NEXT STEP",

    "---------",

    (
        "STEP 16 — WT vs D166V Pocket Discovery: identify, "
        "characterize and compare candidate binding pockets, "
        "including mutation-sensitive and potentially "
        "cryptic pockets."
    ),

    "",

    "=" * 78,
])

with open(
    OUTPUT_DIR / "Step15_final_summary.txt",
    "w",
    encoding="utf-8",
) as handle:

    handle.write(
        "\n".join(summary_lines)
    )


# ============================================================
# 32. FINAL QC
# ============================================================

print()
print("-" * 78)
print("FINAL STEP 15 QC")
print("-" * 78)

qc_check(
    "Final functional architecture table",
    "PASS"
    if (
        TABLE_DIR
        / "Table_S15_1_PNPLA3_functional_architecture.csv"
    ).exists()
    else "FAIL",
)

qc_check(
    "Catalytic/mechanistic table",
    "PASS"
    if (
        TABLE_DIR
        / "Table_S15_2_catalytic_mechanistic_residues.csv"
    ).exists()
    else "FAIL",
)

qc_check(
    "D166V neighborhood table",
    "PASS"
    if (
        TABLE_DIR
        / "Table_S15_3_D166V_mechanistic_neighborhood.csv"
    ).exists()
    else "FAIL",
)

qc_check(
    "Nine-candidate mapping table",
    "PASS"
    if len(candidate_mapping_df) == 9
    else "FAIL",
    f"{len(candidate_mapping_df)} candidates mapped",
)

qc_check(
    "Candidate VariationID uniqueness after mapping",
    "PASS"
    if candidate_mapping_df[
        "VariationID"
    ].is_unique
    else "FAIL",
)

qc_check(
    "Unique candidate-position clustering",
    "PASS"
    if len(variant_cluster_df)
    == len(unique_candidate_positions)
    else "FAIL",
)

qc_check(
    "Therapeutic-region prioritization",
    "PASS"
    if len(region_metrics_df) >= 2
    else "FAIL",
    f"{len(region_metrics_df)} candidate regions",
)

qc_check(
    "D166V present in therapeutic mapping",
    "PASS"
    if PRIMARY_POSITION
    in candidate_mapping_df[
        "Protein_Position"
    ].values
    else "FAIL",
)


expected_figures = [

    "Figure_S15_1_PNPLA3_functional_architecture.png",

    "Figure_S15_2_D166V_mechanistic_neighborhood.png",

    "Figure_S15_3_therapeutic_region_prioritization.png",

    "Figure_S15_4_variant_D166_spatial_relationship.png",

    "Figure_S15_5_candidate_residue_accessibility.png",
]

for figure_name in expected_figures:

    qc_check(
        figure_name,
        "PASS"
        if (
            FIGURE_DIR / figure_name
        ).exists()
        else "FAIL",
    )


# ============================================================
# 33. FINAL STATUS
# ============================================================

failed_checks = [
    check
    for check in QC["Checks"]
    if check["Status"] == "FAIL"
]

QC["End_Time"] = datetime.now().isoformat()

QC["Summary"] = {

    "Total_QC_Checks":
        len(QC["Checks"]),

    "Passed":
        len(QC["Checks"]) - len(failed_checks),

    "Failed":
        len(failed_checks),

    "Warnings":
        len(QC["Warnings"]),

    "Errors":
        len(QC["Errors"]),

    "Final_Status":
        "PASS"
        if not failed_checks
        and not QC["Errors"]
        else "FAIL",
}

with open(
    QC_DIR / "Step15_final_QC_report.json",
    "w",
) as handle:

    json.dump(
        QC,
        handle,
        indent=4,
    )


# ============================================================
# 34. FINAL REPORT
# ============================================================

print()
print("=" * 78)
print("STEP 15 FINAL REPORT")
print("=" * 78)

print(
    "Final QC:",
    QC["Summary"]["Final_Status"],
)

print(
    "Structural candidates mapped:",
    len(candidate_mapping_df),
    "/ 9",
)

print(
    "Unique candidate positions:",
    len(unique_candidate_positions),
)

print(
    "Candidate therapeutic regions:",
    len(region_metrics_df),
)

print(
    "Primary mechanistic candidate:",
    PRIMARY_PROTEIN_CHANGE,
)

print(
    "Top candidate therapeutic region:",
    top_region["Site_Name"],
)

print(
    "Top region score:",
    f"{top_region['Therapeutic_Region_Score']:.4f}",
)

print()
print("Output directory:")
print(OUTPUT_DIR)

print()
print(
    "Step 15 completed with duplicate-position-safe clustering."
)

print(
    "Next step: STEP 16 — WT vs D166V Pocket Discovery."
)

print("=" * 78)