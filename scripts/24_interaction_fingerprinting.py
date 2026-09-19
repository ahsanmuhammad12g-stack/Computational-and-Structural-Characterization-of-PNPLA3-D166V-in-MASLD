from pathlib import Path
import math
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# =============================================================================
# STEP 24 — INTERACTION FINGERPRINTING
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP23_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP23_LEAD_PRIORITIZATION"
    / "tables"
    / "STEP23_LEAD_PRIORITIZATION.csv"
)

WT_RECEPTOR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
    / "receptor"
    / "PNPLA3_WT_chainA.pdbqt"
)

D166V_RECEPTOR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
    / "receptor"
    / "PNPLA3_D166V_chainA.pdbqt"
)

D166V_POSE_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP21_VIRTUAL_SCREENING_DOCKING"
    / "poses"
)

WT_POSE_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
    / "poses"
)

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP24_INTERACTION_FINGERPRINTING"
)

TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
QC_DIR = OUT_DIR / "qc"

for folder in [
    OUT_DIR,
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True
    )


# =============================================================================
# SETTINGS
# =============================================================================

POCKET_CENTER = np.array(
    [2.1950, 8.9330, -5.5020],
    dtype=float
)

RESIDUE_166 = 166

# Contact cutoffs
CONTACT_CUTOFF = 4.0
HBOND_CUTOFF = 3.5
IONIC_CUTOFF = 4.0
AROMATIC_CUTOFF = 4.5

# Top leads to visualize
N_TOP_LEADS = 4


# =============================================================================
# BASIC HELPERS
# =============================================================================

def require_file(path, label):
    if not path.exists():
        raise FileNotFoundError(
            f"{label} not found:\n{path}"
        )


def parse_atom_line(line):
    """
    Parse PDB/PDBQT ATOM/HETATM line.
    """

    record = line[0:6].strip()

    if record not in {"ATOM", "HETATM"}:
        return None

    try:
        atom_name = line[12:16].strip()
        res_name = line[17:20].strip()
        chain = line[21].strip() or "A"
        res_seq = int(line[22:26].strip())

        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        element = ""

        # PDBQT element is not always reliably aligned, so infer from atom.
        if len(line) >= 78:
            element = line[76:78].strip()

        if not element:
            atom_clean = re.sub(
                r"[^A-Za-z]",
                "",
                atom_name
            )

            if atom_clean:
                if len(atom_clean) >= 2 and atom_clean[:2].upper() in {
                    "CL",
                    "BR",
                    "NA",
                    "MG",
                    "CA",
                    "FE",
                    "ZN",
                }:
                    element = atom_clean[:2].upper()
                else:
                    element = atom_clean[0].upper()

        return {
            "record": record,
            "atom_name": atom_name,
            "res_name": res_name,
            "chain": chain,
            "res_seq": res_seq,
            "x": x,
            "y": y,
            "z": z,
            "element": element.upper(),
        }

    except Exception:
        return None


def read_first_model_pdbqt(path):
    """
    Read ATOM/HETATM coordinates from first MODEL only.
    If no MODEL records exist, read all coordinates.
    """

    atoms = []

    in_first_model = True
    model_seen = False
    model_number = None

    with open(
        path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as handle:

        for line in handle:

            stripped = line.strip()

            if stripped.startswith("MODEL"):

                model_seen = True

                try:
                    model_number = int(
                        stripped.split()[1]
                    )
                except Exception:
                    model_number = 1

                in_first_model = (
                    model_number == 1
                )

                continue

            if model_seen and stripped.startswith("ENDMDL"):

                if model_number == 1:
                    break

                continue

            if not in_first_model:
                continue

            parsed = parse_atom_line(line)

            if parsed is not None:
                atoms.append(parsed)

    if not atoms:
        raise RuntimeError(
            f"No ATOM/HETATM coordinates found in:\n{path}"
        )

    return pd.DataFrame(atoms)


def atom_distance(a, b):
    return math.sqrt(
        (a["x"] - b["x"]) ** 2
        + (a["y"] - b["y"]) ** 2
        + (a["z"] - b["z"]) ** 2
    )


def is_protein_atom(atom):
    return (
        atom["record"] == "ATOM"
        and atom["res_seq"] > 0
    )


def residue_key(atom):
    return (
        f"{atom['chain']}:"
        f"{atom['res_name']}"
        f"{atom['res_seq']}"
    )


# =============================================================================
# CONTACT CLASSIFICATION
# =============================================================================

HETERO = {
    "N",
    "O",
    "S",
    "F",
    "CL",
    "BR",
    "I",
}

HYDROPHOBIC = {
    "C",
    "S",
    "F",
    "CL",
    "BR",
    "I",
}

POSITIVE_RESIDUES = {
    "LYS",
    "ARG",
    "HIS",
}

NEGATIVE_RESIDUES = {
    "ASP",
    "GLU",
}


def classify_contact(
    protein_atom,
    ligand_atom,
    distance
):
    """
    Conservative distance-based interaction candidate assignment.

    These are computational contact categories, not experimentally
    validated interactions.
    """

    p_elem = protein_atom["element"]
    l_elem = ligand_atom["element"]

    p_res = protein_atom["res_name"].upper()

    # Hydrogen-bond candidate
    if (
        distance <= HBOND_CUTOFF
        and p_elem in {"N", "O", "S"}
        and l_elem in {"N", "O", "S"}
    ):
        return "H_BOND_CANDIDATE"

    # Ionic candidate
    if (
        distance <= IONIC_CUTOFF
        and (
            (
                p_res in POSITIVE_RESIDUES
                and l_elem in {"O", "S"}
            )
            or
            (
                p_res in NEGATIVE_RESIDUES
                and l_elem in {"N"}
            )
        )
    ):
        return "IONIC_CANDIDATE"

    # Aromatic proximity
    if (
        distance <= AROMATIC_CUTOFF
        and p_res in {
            "PHE",
            "TYR",
            "TRP",
            "HIS",
        }
        and l_elem == "C"
    ):
        return "AROMATIC_PROXIMITY"

    # Hydrophobic contact
    if (
        distance <= CONTACT_CUTOFF
        and p_elem in HYDROPHOBIC
        and l_elem in HYDROPHOBIC
    ):
        return "HYDROPHOBIC_CONTACT"

    return "VAN_DER_WAALS_CONTACT"


# =============================================================================
# BUILD INTERACTION FINGERPRINT
# =============================================================================

def interaction_fingerprint(
    receptor_df,
    ligand_df,
    variant_name,
    cid
):

    receptor_atoms = [
        row
        for _, row in receptor_df.iterrows()
        if is_protein_atom(row)
    ]

    ligand_atoms = [
        row
        for _, row in ligand_df.iterrows()
    ]

    contacts = []

    for p_atom in receptor_atoms:

        for l_atom in ligand_atoms:

            distance = atom_distance(
                p_atom,
                l_atom
            )

            if distance <= CONTACT_CUTOFF:

                interaction = classify_contact(
                    p_atom,
                    l_atom,
                    distance
                )

                contacts.append(
                    {
                        "CID": int(cid),
                        "Variant": variant_name,

                        "Protein_Chain":
                            p_atom["chain"],

                        "Protein_Residue":
                            p_atom["res_name"],

                        "Protein_Residue_Number":
                            int(p_atom["res_seq"]),

                        "Protein_Residue_Key":
                            residue_key(p_atom),

                        "Protein_Atom":
                            p_atom["atom_name"],

                        "Protein_Element":
                            p_atom["element"],

                        "Ligand_Atom":
                            l_atom["atom_name"],

                        "Ligand_Element":
                            l_atom["element"],

                        "Distance_A":
                            distance,

                        "Interaction_Type":
                            interaction,
                    }
                )

    return pd.DataFrame(contacts)


# =============================================================================
# RESIDUE SUMMARY
# =============================================================================

def summarize_residues(contact_df):

    if contact_df.empty:
        return pd.DataFrame(
            columns=[
                "CID",
                "Variant",
                "Protein_Residue_Key",
                "Protein_Residue",
                "Protein_Residue_Number",
                "Minimum_Distance_A",
                "Contact_Count",
                "Interaction_Types",
            ]
        )

    grouped = []

    for key, group in contact_df.groupby(
        [
            "CID",
            "Variant",
            "Protein_Residue_Key",
            "Protein_Residue",
            "Protein_Residue_Number",
        ]
    ):

        grouped.append(
            {
                "CID": key[0],
                "Variant": key[1],
                "Protein_Residue_Key": key[2],
                "Protein_Residue": key[3],
                "Protein_Residue_Number": key[4],
                "Minimum_Distance_A":
                    group["Distance_A"].min(),
                "Contact_Count":
                    len(group),
                "Interaction_Types":
                    "; ".join(
                        sorted(
                            group[
                                "Interaction_Type"
                            ]
                            .unique()
                        )
                    ),
            }
        )

    return pd.DataFrame(grouped)


# =============================================================================
# MOLECULAR 3D FIGURE
# =============================================================================

def make_3d_figure(
    receptor_df,
    ligand_df,
    contact_df,
    cid,
    variant,
    output_path,
):

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    # ---------------------------------------------------------
    # Protein atoms
    # ---------------------------------------------------------

    protein = receptor_df[
        receptor_df["record"] == "ATOM"
    ].copy()

    # Show only Pocket 1 neighborhood
    distances = np.sqrt(
        (
            protein[
                ["x", "y", "z"]
            ].values
            - POCKET_CENTER
        ) ** 2
    ).sum(axis=1)

    # Correct Euclidean calculation
    coordinates = protein[
        ["x", "y", "z"]
    ].values

    distances = np.sqrt(
        (
            coordinates
            - POCKET_CENTER
        ) ** 2
    ).sum(axis=1)

    local = protein[
        distances <= 12.0
    ]

    if local.empty:
        local = protein

    ax.scatter(
        local["x"],
        local["y"],
        local["z"],
        s=5,
        alpha=0.20,
    )

    # ---------------------------------------------------------
    # Ligand
    # ---------------------------------------------------------

    ax.scatter(
        ligand_df["x"],
        ligand_df["y"],
        ligand_df["z"],
        s=70,
        depthshade=True,
    )

    # ---------------------------------------------------------
    # Residue 166
    # ---------------------------------------------------------

    residue166 = protein[
        protein["res_seq"] == RESIDUE_166
    ]

    if not residue166.empty:

        ax.scatter(
            residue166["x"],
            residue166["y"],
            residue166["z"],
            s=80,
            depthshade=True,
        )

    # ---------------------------------------------------------
    # Contact residue centroids
    # ---------------------------------------------------------

    if not contact_df.empty:

        residues = []

        for residue_key_value, group in contact_df.groupby(
            "Protein_Residue_Key"
        ):

            matching = receptor_df[
                receptor_df.apply(
                    lambda row:
                        residue_key(row)
                        == residue_key_value,
                    axis=1
                )
            ]

            if matching.empty:
                continue

            residues.append(
                matching[
                    ["x", "y", "z"]
                ].mean()
            )

        if residues:

            residues = pd.DataFrame(
                residues
            )

            ax.scatter(
                residues["x"],
                residues["y"],
                residues["z"],
                s=35,
                alpha=0.8,
            )

    # ---------------------------------------------------------
    # View around Pocket 1
    # ---------------------------------------------------------

    ax.set_xlim(
        POCKET_CENTER[0] - 12,
        POCKET_CENTER[0] + 12
    )

    ax.set_ylim(
        POCKET_CENTER[1] - 12,
        POCKET_CENTER[1] + 12
    )

    ax.set_zlim(
        POCKET_CENTER[2] - 12,
        POCKET_CENTER[2] + 12
    )

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")

    ax.set_title(
        f"CID {cid} — {variant} docking pose\n"
        "Pocket 1 / residue 166 region"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# START
# =============================================================================

print()
print("=" * 78)
print("STEP 24 — INTERACTION FINGERPRINTING")
print("=" * 78)

print(
    f"Project root : {PROJECT_ROOT}"
)


# =============================================================================
# VERIFY INPUTS
# =============================================================================

require_file(
    STEP23_FILE,
    "Step 23 table"
)

require_file(
    WT_RECEPTOR,
    "WT receptor"
)

require_file(
    D166V_RECEPTOR,
    "D166V receptor"
)

if not D166V_POSE_DIR.exists():
    raise FileNotFoundError(
        f"D166V pose directory not found:\n{D166V_POSE_DIR}"
    )

if not WT_POSE_DIR.exists():
    raise FileNotFoundError(
        f"WT pose directory not found:\n{WT_POSE_DIR}"
    )


# =============================================================================
# READ STEP 23
# =============================================================================

step23 = pd.read_csv(
    STEP23_FILE
)

required_step23 = [
    "Lead_Rank",
    "CID",
    "Lead_Class",
]

missing = [
    c
    for c in required_step23
    if c not in step23.columns
]

if missing:
    raise KeyError(
        f"Step 23 missing columns:\n{missing}"
    )


step23["CID"] = pd.to_numeric(
    step23["CID"],
    errors="coerce"
)

step23.dropna(
    subset=["CID"],
    inplace=True
)

step23["CID"] = (
    step23["CID"]
    .astype(int)
)

step23.sort_values(
    "Lead_Rank",
    inplace=True
)

top_leads = (
    step23
    .head(N_TOP_LEADS)
    .copy()
)


print()
print(
    f"Step 23 compounds available : {len(step23)}"
)

print(
    f"Top leads selected           : {len(top_leads)}"
)

print()
print(
    top_leads[
        [
            "Lead_Rank",
            "CID",
            "Lead_Class",
        ]
    ].to_string(
        index=False
    )
)


# =============================================================================
# READ RECEPTORS
# =============================================================================

print()
print("Reading receptors...")

wt_receptor = read_first_model_pdbqt(
    WT_RECEPTOR
)

d166v_receptor = read_first_model_pdbqt(
    D166V_RECEPTOR
)

print(
    f"WT receptor atoms     : {len(wt_receptor)}"
)

print(
    f"D166V receptor atoms  : {len(d166v_receptor)}"
)


# =============================================================================
# PROCESS TOP LEADS
# =============================================================================

all_contacts = []

all_molecular = []


for _, lead in top_leads.iterrows():

    cid = int(
        lead["CID"]
    )

    # ---------------------------------------------------------
    # Pose paths
    # ---------------------------------------------------------

    d166v_pose = (
        D166V_POSE_DIR
        / f"CID_{cid}_docked.pdbqt"
    )

    wt_pose = (
        WT_POSE_DIR
        / f"CID_{cid}_WT_docked.pdbqt"
    )

    if not d166v_pose.exists():
        print(
            f"\nWARNING: missing D166V pose for CID {cid}"
        )
        continue

    if not wt_pose.exists():
        print(
            f"\nWARNING: missing WT pose for CID {cid}"
        )
        continue

    print()
    print("-" * 78)
    print(
        f"CID {cid}"
    )

    # ---------------------------------------------------------
    # Read poses
    # ---------------------------------------------------------

    dlig = read_first_model_pdbqt(
        d166v_pose
    )

    wlig = read_first_model_pdbqt(
        wt_pose
    )

    print(
        f"D166V ligand atoms : {len(dlig)}"
    )

    print(
        f"WT ligand atoms     : {len(wlig)}"
    )

    # ---------------------------------------------------------
    # D166V
    # ---------------------------------------------------------

    d_contact = interaction_fingerprint(
        d166v_receptor,
        dlig,
        "D166V",
        cid
    )

    # ---------------------------------------------------------
    # WT
    # ---------------------------------------------------------

    w_contact = interaction_fingerprint(
        wt_receptor,
        wlig,
        "WT",
        cid
    )

    if not d_contact.empty:
        all_contacts.append(
            d_contact
        )

    if not w_contact.empty:
        all_contacts.append(
            w_contact
        )

    # ---------------------------------------------------------
    # Molecular figures
    # ---------------------------------------------------------

    make_3d_figure(
        d166v_receptor,
        dlig,
        d_contact,
        cid,
        "D166V",
        FIGURE_DIR
        / f"Figure_24_CID_{cid}_D166V_3D.png"
    )

    make_3d_figure(
        wt_receptor,
        wlig,
        w_contact,
        cid,
        "WT",
        FIGURE_DIR
        / f"Figure_24_CID_{cid}_WT_3D.png"
    )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    d_res = summarize_residues(
        d_contact
    )

    w_res = summarize_residues(
        w_contact
    )

    all_molecular.append(
        {
            "CID": cid,
            "D166V_Contact_Count":
                len(d_contact),
            "WT_Contact_Count":
                len(w_contact),
            "D166V_Residue_Count":
                d_res[
                    "Protein_Residue_Key"
                ].nunique()
                if not d_res.empty
                else 0,
            "WT_Residue_Count":
                w_res[
                    "Protein_Residue_Key"
                ].nunique()
                if not w_res.empty
                else 0,
            "D166V_Min_Contact_Distance_A":
                d_contact["Distance_A"].min()
                if not d_contact.empty
                else np.nan,
            "WT_Min_Contact_Distance_A":
                w_contact["Distance_A"].min()
                if not w_contact.empty
                else np.nan,
            "D166V_D166_Contact":
                (
                    d_contact[
                        "Protein_Residue_Number"
                    ]
                    == 166
                ).any()
                if not d_contact.empty
                else False,
            "WT_D166_Contact":
                (
                    w_contact[
                        "Protein_Residue_Number"
                    ]
                    == 166
                ).any()
                if not w_contact.empty
                else False,
        }
    )


# =============================================================================
# COMBINE CONTACTS
# =============================================================================

if all_contacts:

    contact_df = pd.concat(
        all_contacts,
        ignore_index=True
    )

else:

    contact_df = pd.DataFrame()


if not contact_df.empty:

    contact_df.sort_values(
        [
            "CID",
            "Variant",
            "Distance_A",
        ],
        inplace=True
    )

    contact_df.to_csv(
        TABLE_DIR
        / "STEP24_ATOM_LEVEL_CONTACTS.csv",
        index=False
    )

    residue_summary = summarize_residues(
        contact_df
    )

else:

    residue_summary = pd.DataFrame()


# =============================================================================
# SAVE RESIDUE SUMMARY
# =============================================================================

residue_summary.to_csv(
    TABLE_DIR
    / "STEP24_RESIDUE_INTERACTION_FINGERPRINT.csv",
    index=False
)


# =============================================================================
# SAVE LEAD SUMMARY
# =============================================================================

lead_summary = pd.DataFrame(
    all_molecular
)

lead_summary.to_csv(
    TABLE_DIR
    / "STEP24_LEAD_INTERACTION_SUMMARY.csv",
    index=False
)


# =============================================================================
# WT vs D166V RESIDUE COMPARISON
# =============================================================================

comparison_rows = []

if not residue_summary.empty:

    for cid in top_leads["CID"]:

        cid = int(cid)

        wt = residue_summary[
            (
                residue_summary["CID"]
                == cid
            )
            &
            (
                residue_summary["Variant"]
                == "WT"
            )
        ]

        mutant = residue_summary[
            (
                residue_summary["CID"]
                == cid
            )
            &
            (
                residue_summary["Variant"]
                == "D166V"
            )
        ]

        wt_residues = set(
            wt["Protein_Residue_Key"]
        )

        mutant_residues = set(
            mutant["Protein_Residue_Key"]
        )

        shared = (
            wt_residues
            & mutant_residues
        )

        wt_only = (
            wt_residues
            - mutant_residues
        )

        mutant_only = (
            mutant_residues
            - wt_residues
        )

        comparison_rows.append(
            {
                "CID": cid,
                "WT_Residue_Count":
                    len(wt_residues),
                "D166V_Residue_Count":
                    len(mutant_residues),
                "Shared_Residue_Count":
                    len(shared),
                "WT_Only_Residue_Count":
                    len(wt_only),
                "D166V_Only_Residue_Count":
                    len(mutant_only),
                "Residue_Linking_Jaccard":
                    (
                        len(shared)
                        /
                        len(
                            wt_residues
                            | mutant_residues
                        )
                        if (
                            wt_residues
                            | mutant_residues
                        )
                        else np.nan
                    ),
                "WT_Only_Residues":
                    ";".join(
                        sorted(
                            wt_only
                        )
                    ),
                "D166V_Only_Residues":
                    ";".join(
                        sorted(
                            mutant_only
                        )
                    ),
            }
        )

comparison_df = pd.DataFrame(
    comparison_rows
)

comparison_df.to_csv(
    TABLE_DIR
    / "STEP24_WT_vs_D166V_INTERACTION_COMPARISON.csv",
    index=False
)


# =============================================================================
# INTERACTION-TYPE SUMMARY
# =============================================================================

if not contact_df.empty:

    interaction_summary = (
        contact_df
        .groupby(
            [
                "CID",
                "Variant",
                "Interaction_Type",
            ]
        )
        .size()
        .reset_index(
            name="Contact_Count"
        )
    )

else:

    interaction_summary = pd.DataFrame(
        columns=[
            "CID",
            "Variant",
            "Interaction_Type",
            "Contact_Count",
        ]
    )

interaction_summary.to_csv(
    TABLE_DIR
    / "STEP24_INTERACTION_TYPE_SUMMARY.csv",
    index=False
)


# =============================================================================
# FIGURE 24.01 — CONTACT COUNTS
# =============================================================================

if not lead_summary.empty:

    plot_df = lead_summary.copy()

    x = np.arange(
        len(plot_df)
    )

    width = 0.38

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x - width / 2,
        plot_df[
            "WT_Contact_Count"
        ],
        width,
        label="WT"
    )

    plt.bar(
        x + width / 2,
        plot_df[
            "D166V_Contact_Count"
        ],
        width,
        label="D166V"
    )

    plt.xticks(
        x,
        plot_df[
            "CID"
        ].astype(str)
    )

    plt.xlabel(
        "PubChem CID"
    )

    plt.ylabel(
        "Protein–ligand contact count"
    )

    plt.title(
        "Step 24 — WT vs D166V Interaction Contacts"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "Figure_24_01_Contact_Counts.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# FIGURE 24.02 — RESIDUE OVERLAP
# =============================================================================

if not comparison_df.empty:

    plot_df = comparison_df.copy()

    x = np.arange(
        len(plot_df)
    )

    width = 0.25

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        x - width,
        plot_df[
            "Shared_Residue_Count"
        ],
        width,
        label="Shared"
    )

    plt.bar(
        x,
        plot_df[
            "WT_Only_Residue_Count"
        ],
        width,
        label="WT-only"
    )

    plt.bar(
        x + width,
        plot_df[
            "D166V_Only_Residue_Count"
        ],
        width,
        label="D166V-only"
    )

    plt.xticks(
        x,
        plot_df[
            "CID"
        ].astype(str)
    )

    plt.xlabel(
        "PubChem CID"
    )

    plt.ylabel(
        "Interacting residue count"
    )

    plt.title(
        "Step 24 — WT vs D166V Interaction-Residue Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "Figure_24_02_Residue_Comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# FIGURE 24.03 — INTERACTION TYPES
# =============================================================================

if not interaction_summary.empty:

    pivot = interaction_summary.pivot_table(
        index="CID",
        columns=[
            "Variant",
            "Interaction_Type"
        ],
        values="Contact_Count",
        fill_value=0
    )

    pivot.columns = [
        f"{a}_{b}"
        for a, b
        in pivot.columns
    ]

    pivot.to_csv(
        TABLE_DIR
        / "STEP24_INTERACTION_TYPE_MATRIX.csv"
    )

    plt.figure(
        figsize=(11, 6)
    )

    pivot.plot(
        kind="bar",
        figsize=(11, 6)
    )

    plt.xlabel(
        "PubChem CID"
    )

    plt.ylabel(
        "Contact count"
    )

    plt.title(
        "Step 24 — Interaction-Type Fingerprint"
    )

    plt.xticks(
        rotation=0
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR
        / "Figure_24_03_Interaction_Types.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# =============================================================================
# TOP INTERACTING RESIDUES
# =============================================================================

if not residue_summary.empty:

    top_residue_rows = []

    for cid in top_leads["CID"]:

        cid = int(cid)

        subset = residue_summary[
            residue_summary["CID"]
            == cid
        ].copy()

        subset.sort_values(
            [
                "Minimum_Distance_A",
                "Contact_Count",
            ],
            ascending=[
                True,
                False,
            ],
            inplace=True
        )

        for _, row in subset.head(15).iterrows():

            top_residue_rows.append(
                row.to_dict()
            )

    top_residues = pd.DataFrame(
        top_residue_rows
    )

else:

    top_residues = pd.DataFrame()

top_residues.to_csv(
    TABLE_DIR
    / "STEP24_TOP_INTERACTING_RESIDUES.csv",
    index=False
)


# =============================================================================
# QC
# =============================================================================

qc = {

    "step": 24,

    "top_leads_requested":
        int(N_TOP_LEADS),

    "top_leads_processed":
        int(len(all_molecular)),

    "contact_rows":
        int(len(contact_df)),

    "receptors": {
        "WT":
            str(WT_RECEPTOR),
        "D166V":
            str(D166V_RECEPTOR),
    },

    "contact_cutoffs_A": {
        "general":
            CONTACT_CUTOFF,
        "hbond_candidate":
            HBOND_CUTOFF,
        "ionic_candidate":
            IONIC_CUTOFF,
        "aromatic_proximity":
            AROMATIC_CUTOFF,
    },

    "scientific_note": (
        "Interaction categories are distance-based computational "
        "contact assignments from rigid docking poses. They should "
        "not be interpreted as experimentally confirmed hydrogen bonds, "
        "ionic interactions, binding affinity, or biological activity."
    ),
}

with open(
    QC_DIR
    / "STEP24_QC.json",
    "w",
    encoding="utf-8"
) as handle:

    json.dump(
        qc,
        handle,
        indent=2
    )


# =============================================================================
# INTERPRETATION
# =============================================================================

interpretation_lines = [
    "STEP 24 — INTERACTION FINGERPRINTING",
    "",
    f"Top leads requested: {N_TOP_LEADS}",
    f"Top leads processed: {len(all_molecular)}",
    "",
    "Purpose:",
    "Extract distance-based protein-ligand contacts from the",
    "actual WT and D166V docking poses and compare the interacting",
    "residue fingerprints.",
    "",
    "Important limitation:",
    "These interaction categories come from rigid docking poses.",
    "They are computational contact candidates and are not proof",
    "of experimental hydrogen bonding, binding affinity, inhibition,",
    "selectivity, or therapeutic activity.",
]

with open(
    QC_DIR
    / "STEP24_INTERPRETATION.txt",
    "w",
    encoding="utf-8"
) as handle:

    handle.write(
        "\n".join(
            interpretation_lines
        )
    )


# =============================================================================
# FINAL OUTPUT
# =============================================================================

print()
print("=" * 78)
print("STEP 24 COMPLETE")
print("=" * 78)

print(
    f"Top leads processed : {len(all_molecular)}"
)

print(
    f"Contact records     : {len(contact_df)}"
)

print()
print(
    "Tables:"
)

for path in sorted(
    TABLE_DIR.glob("*.csv")
):

    print(
        f"  {path}"
    )

print()
print(
    "Figures:"
)

for path in sorted(
    FIGURE_DIR.glob("*.png")
):

    print(
        f"  {path}"
    )

print()
print(
    "QC:"
)

print(
    QC_DIR
)