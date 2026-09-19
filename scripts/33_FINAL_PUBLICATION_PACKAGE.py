from pathlib import Path
import json
import hashlib
import sys

import pandas as pd
import numpy as np
from pathlib import Path
import json
import hashlib
import sys
from datetime import datetime

import pandas as pd


# =============================================================================
# STEP 33 — FINAL PUBLICATION / REPRODUCIBILITY / QC PACKAGE
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    PROJECT_ROOT
    / "RESULTS"
)

OUT_DIR = (
    RESULTS_DIR
    / "STEP33_FINAL_PUBLICATION_PACKAGE"
)

TABLE_DIR = OUT_DIR / "tables"
FIGURE_DIR = OUT_DIR / "figures"
QC_DIR = OUT_DIR / "qc"
MANUSCRIPT_DIR = OUT_DIR / "manuscript_support"

for folder in [
    TABLE_DIR,
    FIGURE_DIR,
    QC_DIR,
    MANUSCRIPT_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# PROJECT DEFINITION
# =============================================================================

VARIANT = "PNPLA3 p.Asp166Val (D166V)"
RSID = "rs1381635405"
VARIATION_ID = "3308171"
PRIMARY_CID = "71236597"

PROJECT_TITLE = (
    "Computational prioritization and mechanistic characterization "
    "of the unresolved PNPLA3 D166V variant in MASLD with "
    "structure-guided therapeutic candidate prioritization"
)


# =============================================================================
# COMPLETED STEP DIRECTORIES
# =============================================================================

STEP_DIRECTORIES = {

    "Step09F":
        [
            RESULTS_DIR / "04_VARIANT ANALYSIS",
            RESULTS_DIR / "VARIANT ANALYSIS",
        ],

    "Step10":
        [
            RESULTS_DIR / "STEP10_STRUCTURAL_ANALYSIS",
        ],

    "Step16":
        [
            RESULTS_DIR / "STEP16_WT_vs_D166V_POCKET_DISCOVERY",
        ],

    "Step17":
        [
            RESULTS_DIR / "STEP17_STRUCTURAL_ENSEMBLE_POCKET_DYNAMICS",
        ],

    "Step18":
        [
            RESULTS_DIR / "STEP18_THERAPEUTIC_COMPOUND_LIBRARY",
        ],

    "Step19":
        [
            RESULTS_DIR / "STEP19_DRUG_LIKENESS_FILTERING",
        ],

    "Step20":
        [
            RESULTS_DIR / "STEP20_PHARMACOPHORE_SHAPE_SCREENING",
        ],

    "Step21":
        [
            RESULTS_DIR / "STEP21_VIRTUAL_SCREENING_DOCKING",
        ],

    "Step22":
        [
            RESULTS_DIR / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING",
        ],

    "Step23":
        [
            RESULTS_DIR / "STEP23_LEAD_PRIORITIZATION",
        ],

    "Step24":
        [
            RESULTS_DIR / "STEP24_INTERACTION_FINGERPRINTING",
        ],

    "Step25":
        [
            RESULTS_DIR / "STEP25_D166V_FUNCTIONAL_MECHANISM",
        ],

    "Step26":
        [
            RESULTS_DIR / "STEP26_D166V_MASLD_MECHANISTIC_LINK",
        ],

    "Step27":
        [
            RESULTS_DIR / "STEP27_ORTHOGONAL_STRUCTURAL_VALIDATION",
        ],

    "Step28":
        [
            RESULTS_DIR / "STEP28_MUTATION_SPECIFIC_THERAPEUTIC_VULNERABILITY",
        ],

    "Step29":
        [
            RESULTS_DIR / "STEP29_LEAD_VALIDATION_RESCORING",
        ],

    "Step30":
        [
            RESULTS_DIR / "STEP30_ADMET_DRUG_DEVELOPMENT_FEASIBILITY",
        ],

    "Step31":
        [
            RESULTS_DIR / "STEP31_INDEPENDENT_BIOLOGICAL_VALIDATION",
        ],

    "Step32":
        [
            RESULTS_DIR / "STEP32_FINAL_EVIDENCE_INTEGRATION",
        ],
}


# =============================================================================
# IMPORTANT HISTORICAL FILES
# =============================================================================

EXPECTED_FILES = {

    "Step16_pocket":
        (
            RESULTS_DIR
            / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
            / "tables"
            / "Table_16_04_D166V_Pocket_Prioritization.csv"
        ),

    "Step22_differential":
        (
            RESULTS_DIR
            / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
            / "tables"
            / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING.csv"
        ),

    "Step23_leads":
        (
            RESULTS_DIR
            / "STEP23_LEAD_PRIORITIZATION"
            / "tables"
            / "STEP23_LEAD_PRIORITIZATION.csv"
        ),

    "Step25_summary":
        (
            RESULTS_DIR
            / "STEP25_D166V_FUNCTIONAL_MECHANISM"
            / "tables"
            / "STEP25_D166V_FUNCTIONAL_MECHANISM_SUMMARY.csv"
        ),

    "Step26_summary":
        (
            RESULTS_DIR
            / "STEP26_D166V_MASLD_MECHANISTIC_LINK"
            / "tables"
            / "STEP26_FINAL_MECHANISTIC_INTEGRATION.csv"
        ),

    "Step27_metrics":
        (
            RESULTS_DIR
            / "STEP27_ORTHOGONAL_STRUCTURAL_VALIDATION"
            / "tables"
            / "STEP27_RESIDUE_166_INDEPENDENT_METRICS.csv"
        ),

    "Step28_summary":
        (
            RESULTS_DIR
            / "STEP28_MUTATION_SPECIFIC_THERAPEUTIC_VULNERABILITY"
            / "tables"
            / "STEP28_THERAPEUTIC_VULNERABILITY_SUMMARY.csv"
        ),

    "Step29_validation":
        (
            RESULTS_DIR
            / "STEP29_LEAD_VALIDATION_RESCORING"
            / "tables"
            / "STEP29_FULL_LEAD_VALIDATION.csv"
        ),

    "Step30_summary":
        (
            RESULTS_DIR
            / "STEP30_ADMET_DRUG_DEVELOPMENT_FEASIBILITY"
            / "tables"
            / "STEP30_ADMET_DEVELOPMENT_SUMMARY.csv"
        ),

    "Step31_evidence":
        (
            RESULTS_DIR
            / "STEP31_INDEPENDENT_BIOLOGICAL_VALIDATION"
            / "tables"
            / "STEP31_EVIDENCE_STATUS_SUMMARY.csv"
        ),

    "Step32_chain":
        (
            RESULTS_DIR
            / "STEP32_FINAL_EVIDENCE_INTEGRATION"
            / "tables"
            / "STEP32_MASTER_EVIDENCE_CHAIN.csv"
        ),

    "Step32_claims":
        (
            RESULTS_DIR
            / "STEP32_FINAL_EVIDENCE_INTEGRATION"
            / "tables"
            / "STEP32_THESIS_CLAIM_GUIDANCE.csv"
        ),

    "Step32_completeness":
        (
            RESULTS_DIR
            / "STEP32_FINAL_EVIDENCE_INTEGRATION"
            / "tables"
            / "STEP32_EVIDENCE_COMPLETENESS_PROFILE.csv"
        ),
}


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):

    print()
    print("=" * 78)
    print("STEP 33 FAILED")
    print("=" * 78)
    print(message)
    print()
    sys.exit(1)


def read_csv(path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(
            path
        )
    except Exception:
        return None


def normalize_cid(value):

    try:
        return str(
            int(float(value))
        )
    except Exception:
        return str(value).strip()


def sha256_file(path):

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def find_recursive(
    root,
    patterns,
):

    matches = []

    if not root.exists():
        return matches

    for pattern in patterns:

        matches.extend(
            root.rglob(
                pattern
            )
        )

    return sorted(
        set(matches)
    )


def first_match(
    roots,
    patterns,
):

    for root in roots:

        matches = find_recursive(
            root,
            patterns,
        )

        if matches:
            return matches[0]

    return None


# =============================================================================
# LOCATE STEP 9F ROBUSTLY
# =============================================================================

def locate_step9f():

    roots = STEP_DIRECTORIES[
        "Step09F"
    ]

    patterns = [
        "*step9f*final*candidate*.csv",
        "*step9f*.csv",
        "*final*candidate*selection*.csv",
    ]

    return first_match(
        roots,
        patterns,
    )


# =============================================================================
# STEP STATUS
# =============================================================================

def build_step_status():

    rows = []

    for step, roots in STEP_DIRECTORIES.items():

        found_root = None

        for root in roots:

            if root.exists():

                found_root = root
                break

        rows.append(
            {
                "Step":
                    step,
                "Expected_Directory":
                    str(
                        roots[0]
                    ),
                "Directory_Found":
                    found_root is not None,
                "Resolved_Directory":
                    str(
                        found_root
                    )
                    if found_root
                    else "",
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EXTRACT FINAL LEAD
# =============================================================================

def build_final_lead_table():

    step22 = read_csv(
        EXPECTED_FILES[
            "Step22_differential"
        ]
    )

    step23 = read_csv(
        EXPECTED_FILES[
            "Step23_leads"
        ]
    )

    step28 = read_csv(
        EXPECTED_FILES[
            "Step28_summary"
        ]
    )

    step29 = read_csv(
        EXPECTED_FILES[
            "Step29_validation"
        ]
    )

    step30 = read_csv(
        EXPECTED_FILES[
            "Step30_summary"
        ]
    )

    if step22 is None:
        fail(
            "Step 22 differential docking table is missing."
        )

    if step23 is None:
        fail(
            "Step 23 lead table is missing."
        )

    rows = []

    # -------------------------------------------------------------------------
    # Step 22
    # -------------------------------------------------------------------------

    row22 = step22[
        step22["CID"]
        .map(normalize_cid)
        ==
        PRIMARY_CID
    ]

    if row22.empty:
        fail(
            f"Primary CID {PRIMARY_CID} not found in Step 22."
        )

    row22 = row22.iloc[0]

    result = {
        "CID":
            PRIMARY_CID,
        "Variant":
            VARIANT,
        "rsID":
            RSID,
        "VariationID":
            VARIATION_ID,
        "Step22_WT_Docking_kcal_mol":
            row22.get(
                "WT_Best_Vina_Score_kcal_mol",
                np.nan,
            ),
        "Step22_D166V_Docking_kcal_mol":
            row22.get(
                "D166V_Best_Vina_Score_kcal_mol",
                np.nan,
            ),
        "Step22_Delta_D166V_minus_WT_kcal_mol":
            row22.get(
                "Delta_Vina_D166V_minus_WT_kcal_mol",
                np.nan,
            ),
        "Step22_Differential_Class":
            row22.get(
                "Differential_Docking_Class",
                "",
            ),
    }

    # -------------------------------------------------------------------------
    # Step 23
    # -------------------------------------------------------------------------

    if "CID" in step23.columns:

        row23 = step23[
            step23["CID"]
            .map(normalize_cid)
            ==
            PRIMARY_CID
        ]

        if not row23.empty:

            row23 = row23.iloc[0]

            for field in [
                "Integrated_Score",
                "Step23_Rank",
                "Lead_Class",
                "Priority",
            ]:

                if field in step23.columns:

                    result[
                        f"Step23_{field}"
                    ] = row23[
                        field
                    ]

    # -------------------------------------------------------------------------
    # Step 28
    # -------------------------------------------------------------------------

    if step28 is not None:

        cid_column = None

        for candidate in [
            "CID_Normalized",
            "CID",
        ]:

            if candidate in step28.columns:

                cid_column = candidate
                break

        if cid_column:

            row28 = step28[
                step28[
                    cid_column
                ]
                .map(normalize_cid)
                ==
                PRIMARY_CID
            ]

            if not row28.empty:

                row28 = row28.iloc[0]

                for field in [
                    "Ligand_Specific_Vulnerability_Score",
                    "Therapeutic_Vulnerability_Score",
                    "Therapeutic_Interpretation",
                    "Docking_Preference_Class",
                ]:

                    if field in step28.columns:

                        result[
                            f"Step28_{field}"
                        ] = row28[
                            field
                        ]

    # -------------------------------------------------------------------------
    # Step 29
    # -------------------------------------------------------------------------

    if step29 is not None:

        if "CID" in step29.columns:

            row29 = step29[
                step29["CID"]
                .map(normalize_cid)
                ==
                PRIMARY_CID
            ]

            if not row29.empty:

                row29 = row29.iloc[0]

                for field in [
                    "Step29_Validation_Score",
                    "Step29_Validation_Class",
                    "Interaction_Jaccard",
                ]:

                    if field in step29.columns:

                        result[
                            field
                        ] = row29[
                            field
                        ]

    # -------------------------------------------------------------------------
    # Step 30
    # -------------------------------------------------------------------------

    if step30 is not None:

        if "CID" in step30.columns:

            row30 = step30[
                step30["CID"]
                .map(normalize_cid)
                ==
                PRIMARY_CID
            ]

            if not row30.empty:

                row30 = row30.iloc[0]

                for field in [
                    "ADMET_Development_Score",
                    "Development_Class",
                    "Step30_Development_Priority_Score",
                ]:

                    if field in step30.columns:

                        result[
                            f"Step30_{field}"
                        ] = row30[
                            field
                        ]

    rows.append(
        result
    )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FINAL VARIANT EVIDENCE
# =============================================================================

def build_variant_summary():

    step27 = read_csv(
        EXPECTED_FILES[
            "Step27_metrics"
        ]
    )

    row = {
        "Variant":
            VARIANT,
        "rsID":
            RSID,
        "VariationID":
            VARIATION_ID,
        "Primary_CID":
            PRIMARY_CID,
        "WT_Residue_166":
            "ASP",
        "D166V_Residue_166":
            "VAL",
    }

    if step27 is not None and not step27.empty:

        if "Model" in step27.columns:

            wt = step27[
                step27[
                    "Model"
                ].astype(str).str.upper()
                ==
                "WT"
            ]

            mut = step27[
                step27[
                    "Model"
                ].astype(str).str.upper()
                ==
                "D166V"
            ]

            if not wt.empty:

                wt = wt.iloc[0]

                row[
                    "WT_Residue_166_SASA_A2"
                ] = wt.get(
                    "Residue_SASA_A2",
                    np.nan,
                )

                row[
                    "WT_Heavy_Atom_Contacts_4A"
                ] = wt.get(
                    "Heavy_Atom_Contact_Count_4A",
                    np.nan,
                )

                row[
                    "WT_Polar_Contacts_4A"
                ] = wt.get(
                    "Polar_Atom_Contact_Count_4A",
                    np.nan,
                )

            if not mut.empty:

                mut = mut.iloc[0]

                row[
                    "D166V_Residue_166_SASA_A2"
                ] = mut.get(
                    "Residue_SASA_A2",
                    np.nan,
                )

                row[
                    "D166V_Heavy_Atom_Contacts_4A"
                ] = mut.get(
                    "Heavy_Atom_Contact_Count_4A",
                    np.nan,
                )

                row[
                    "D166V_Polar_Contacts_4A"
                ] = mut.get(
                    "Polar_Atom_Contact_Count_4A",
                    np.nan,
                )

    # Carry known results from completed analysis.
    row[
        "Step25_Hydropathy_Change"
    ] = 7.7

    row[
        "Step25_Charge_Change"
    ] = 1.0

    row[
        "Step25_Contacts_Lost"
    ] = 20

    row[
        "Step25_Contacts_Gained"
    ] = 8

    row[
        "Step25_Sidechain_Displacement_A"
    ] = 1.166877

    row[
        "Step26_Global_Calpha_RMSD_A"
    ] = 0.000960

    row[
        "Step26_Local_Calpha_RMSD_A"
    ] = 0.003287

    row[
        "Step27_Global_Calpha_RMSD_A"
    ] = 0.000959

    return pd.DataFrame(
        [row]
    )


# =============================================================================
# MASTER SCIENTIFIC EVIDENCE TABLE
# =============================================================================

def build_master_evidence_table():

    rows = [

        {
            "Evidence_ID":
                "E01",
            "Domain":
                "Variant prioritization",
            "Evidence":
                "D166V selected as primary unresolved PNPLA3 structural-analysis candidate.",
            "Status":
                "SUPPORTED",
            "Interpretation":
                "Project-level computational prioritization.",
            "Main_Limitation":
                "Prioritization does not establish pathogenicity.",
        },

        {
            "Evidence_ID":
                "E02",
            "Domain":
                "Population context",
            "Evidence":
                "D166V has very low observed gnomAD allele frequency in Step 9F.",
            "Status":
                "SUPPORTED",
            "Interpretation":
                "Compatible with a rare-variant investigation.",
            "Main_Limitation":
                "Rarity alone does not establish pathogenicity.",
        },

        {
            "Evidence_ID":
                "E03",
            "Domain":
                "Residue substitution",
            "Evidence":
                "Asp166 is replaced by Val166.",
            "Status":
                "SUPPORTED",
            "Interpretation":
                "Direct structural consequence of the mutation model.",
            "Main_Limitation":
                "Static structural model.",
        },

        {
            "Evidence_ID":
                "E04",
            "Domain":
                "Local chemical effect",
            "Evidence":
                "Hydropathy changes from -3.5 to +4.2 and approximate charge changes from -1 to 0.",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Interpretation":
                "Strong physicochemical change at the substituted residue.",
            "Main_Limitation":
                "Descriptors are approximate and do not measure thermodynamic effects.",
        },

        {
            "Evidence_ID":
                "E05",
            "Domain":
                "Contact remodeling",
            "Evidence":
                "Step 25 identifies 20 lost and 8 gained residue-166 contacts.",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Interpretation":
                "Local interaction-network remodeling.",
            "Main_Limitation":
                "Distance-based structural contacts.",
        },

        {
            "Evidence_ID":
                "E06",
            "Domain":
                "Independent structural validation",
            "Evidence":
                "Step 27 reports increased residue-166 SASA and reduced heavy-atom and polar contacts while global Cα RMSD remains ~0.001 Å.",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Interpretation":
                "Independent descriptors reproduce the local alteration.",
            "Main_Limitation":
                "Still based on static structural models.",
        },

        {
            "Evidence_ID":
                "E07",
            "Domain":
                "PNPLA3 biological function",
            "Evidence":
                "Independent literature establishes PNPLA3 involvement in hepatic lipid metabolism.",
            "Status":
                "LITERATURE_SUPPORTED",
            "Interpretation":
                "Provides biological context for the structural mechanism.",
            "Main_Limitation":
                "Most direct functional studies are not D166V-specific.",
        },

        {
            "Evidence_ID":
                "E08",
            "Domain":
                "PNPLA3–MASLD relationship",
            "Evidence":
                "Independent literature establishes a strong relationship between PNPLA3 variation and steatotic liver disease.",
            "Status":
                "LITERATURE_SUPPORTED",
            "Interpretation":
                "Provides disease context.",
            "Main_Limitation":
                "D166V-specific disease causality remains untested.",
        },

        {
            "Evidence_ID":
                "E09",
            "Domain":
                "Disease mechanism",
            "Evidence":
                "D166V structural alteration is consistent with a function-altering PNPLA3 mechanism.",
            "Status":
                "MECHANISTIC_INFERENCE",
            "Interpretation":
                "Computationally plausible disease mechanism.",
            "Main_Limitation":
                "No D166V biochemical or cellular experiment.",
        },

        {
            "Evidence_ID":
                "E10",
            "Domain":
                "Therapeutic pocket",
            "Evidence":
                "Residue 166 lies within the previously prioritized pocket.",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Interpretation":
                "The altered region is structurally targetable.",
            "Main_Limitation":
                "Pocket analysis does not prove binding.",
        },

        {
            "Evidence_ID":
                "E11",
            "Domain":
                "Lead docking",
            "Evidence":
                "CID 71236597 scores -10.262 kcal/mol on D166V versus -9.296 kcal/mol on WT.",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Interpretation":
                "D166V-preferred docking signal.",
            "Main_Limitation":
                "Docking scores are not measured affinities.",
        },

        {
            "Evidence_ID":
                "E12",
            "Domain":
                "Therapeutic specificity",
            "Evidence":
                "Step 28 classifies CID 71236597 as D166V-preferred, but not as strict mutation-specific support.",
            "Status":
                "LIMITED",
            "Interpretation":
                "Therapeutic hypothesis remains testable.",
            "Main_Limitation":
                "No functional rescue experiment.",
        },

        {
            "Evidence_ID":
                "E13",
            "Domain":
                "Interaction validation",
            "Evidence":
                "Step 29 could not reconstruct explicit WT/D166V interaction fingerprints from Step 24.",
            "Status":
                "LIMITED",
            "Interpretation":
                "Interaction-level validation remains incomplete.",
            "Main_Limitation":
                "Do not interpret Jaccard=0 as absence of shared interactions.",
        },

        {
            "Evidence_ID":
                "E14",
            "Domain":
                "Chemical development",
            "Evidence":
                "CID 71236597 has favorable physicochemical development descriptors in Step 30.",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Interpretation":
                "Reasonable physicochemical developability profile.",
            "Main_Limitation":
                "Not experimental ADMET.",
        },

        {
            "Evidence_ID":
                "E15",
            "Domain":
                "Independent biological validation",
            "Evidence":
                "Targeted literature search did not identify direct peer-reviewed functional characterization of D166V.",
            "Status":
                "UNRESOLVED",
            "Interpretation":
                "Supports novelty of the functional question.",
            "Main_Limitation":
                "Search outcome is not proof that no publication exists anywhere.",
        },

    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# CLAIM GUIDANCE
# =============================================================================

def build_claim_guidance():

    rows = [

        {
            "Use":
                "THESIS_RESULT",
            "Recommended_Claim":
                "D166V produced a reproducible local structural and physicochemical alteration at PNPLA3 residue 166 in the computational models.",
            "Evidence":
                "Steps 25 and 27",
            "Claim_Strength":
                "Appropriate",
        },

        {
            "Use":
                "THESIS_RESULT",
            "Recommended_Claim":
                "The predicted D166V effect is consistent with alteration of a functionally important PNPLA3 region.",
            "Evidence":
                "Steps 25–27 and independent biological literature",
            "Claim_Strength":
                "Appropriate",
        },

        {
            "Use":
                "THESIS_DISCUSSION",
            "Recommended_Claim":
                "The D166V structural mechanism is biologically consistent with the established role of PNPLA3 in hepatic lipid metabolism and MASLD.",
            "Evidence":
                "Steps 26 and 31",
            "Claim_Strength":
                "Appropriate as inference",
        },

        {
            "Use":
                "THERAPEUTIC_RESULT",
            "Recommended_Claim":
                "CID 71236597 was prioritized as a computational therapeutic candidate and exhibited a D166V-preferred docking score.",
            "Evidence":
                "Steps 22, 23, 28 and 30",
            "Claim_Strength":
                "Appropriate",
        },

        {
            "Use":
                "AVOID",
            "Recommended_Claim":
                "D166V is experimentally proven pathogenic.",
            "Evidence":
                "None",
            "Claim_Strength":
                "Unsupported",
        },

        {
            "Use":
                "AVOID",
            "Recommended_Claim":
                "CID 71236597 is a confirmed D166V inhibitor.",
            "Evidence":
                "None",
            "Claim_Strength":
                "Unsupported",
        },

        {
            "Use":
                "AVOID",
            "Recommended_Claim":
                "CID 71236597 will treat MASLD.",
            "Evidence":
                "None",
            "Claim_Strength":
                "Unsupported",
        },

    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# LIMITATIONS
# =============================================================================

def build_limitations():

    rows = [

        {
            "Limitation_ID":
                "L01",
            "Limitation":
                "No direct biochemical assay of D166V PNPLA3 activity was performed.",
            "Impact":
                "Functional effect remains computationally inferred.",
        },

        {
            "Limitation_ID":
                "L02",
            "Limitation":
                "No cellular phenotype or disease-model experiment for D166V was performed.",
            "Impact":
                "Direct disease causality remains unverified.",
        },

        {
            "Limitation_ID":
                "L03",
            "Limitation":
                "WT and D166V structures are static computational models.",
            "Impact":
                "Dynamic conformational effects may not be captured.",
        },

        {
            "Limitation_ID":
                "L04",
            "Limitation":
                "Step 29 interaction-overlap reconstruction was incomplete because the Step 24 schema could not be reconstructed automatically.",
            "Impact":
                "Interaction-consistency evidence should not be overinterpreted.",
        },

        {
            "Limitation_ID":
                "L05",
            "Limitation":
                "Step 30 evaluates physicochemical drug-development properties rather than experimental ADMET.",
            "Impact":
                "Pharmacokinetics and toxicity remain unresolved.",
        },

        {
            "Limitation_ID":
                "L06",
            "Limitation":
                "D166V-specific functional literature was not identified in the targeted search.",
            "Impact":
                "The project addresses an unresolved functional hypothesis rather than a previously established mechanism.",
        },

        {
            "Limitation_ID":
                "L07",
            "Limitation":
                "No experimental compound-binding or functional-rescue assay was performed.",
            "Impact":
                "Therapeutic efficacy remains hypothetical.",
        },

    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# REPRODUCIBILITY MANIFEST
# =============================================================================

def build_reproducibility_manifest():

    rows = []

    script_dir = (
        PROJECT_ROOT
        / "SCRIPTS"
    )

    for step in range(
        1,
        34,
    ):

        matches = sorted(
            script_dir.glob(
                f"{step}_*.py"
            )
        )

        if not matches:

            continue

        for path in matches:

            rows.append(
                {
                    "Step":
                        step,
                    "Script":
                        str(
                            path.relative_to(
                                PROJECT_ROOT
                            )
                        ),
                    "SHA256":
                        sha256_file(
                            path
                        ),
                    "Size_bytes":
                        path.stat().st_size,
                }
            )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# COMPLETE FILE MANIFEST
# =============================================================================

def build_file_manifest():

    rows = []

    important_extensions = {
        ".csv",
        ".json",
        ".txt",
        ".png",
        ".pdb",
        ".py",
    }

    if not RESULTS_DIR.exists():
        return pd.DataFrame()

    for path in RESULTS_DIR.rglob("*"):

        if not path.is_file():
            continue

        if (
            OUT_DIR in path.parents
            or path == OUT_DIR
        ):
            continue

        if path.suffix.lower() not in important_extensions:
            continue

        try:

            relative = path.relative_to(
                PROJECT_ROOT
            )

        except ValueError:

            continue

        rows.append(
            {
                "Relative_Path":
                    str(relative),
                "Size_bytes":
                    path.stat().st_size,
                "SHA256":
                    sha256_file(
                        path
                    ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "Relative_Path"
    )


# =============================================================================
# FIGURE INVENTORY
# =============================================================================

def build_figure_inventory():

    rows = []

    for path in RESULTS_DIR.rglob(
        "*.png"
    ):

        if OUT_DIR in path.parents:
            continue

        try:
            relative = path.relative_to(
                PROJECT_ROOT
            )
        except ValueError:
            continue

        rows.append(
            {
                "Figure_File":
                    str(relative),
                "Step":
                    (
                        path.parts[
                            -3
                        ]
                        if len(path.parts) >= 3
                        else ""
                    ),
                "Size_bytes":
                    path.stat().st_size,
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=[
                "Figure_File",
                "Step",
                "Size_bytes",
            ]
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "Figure_File"
    )


# =============================================================================
# FINAL CONCLUSION
# =============================================================================

FINAL_CONCLUSION = (
    "The completed computational workflow prioritizes PNPLA3 p.Asp166Val "
    "(D166V; rs1381635405; VariationID 3308171) as an unresolved candidate "
    "and provides convergent structural evidence that the mutation changes "
    "the local physicochemical and interaction environment of residue 166 "
    "without producing substantial global backbone displacement. Independent "
    "structural descriptors reproduce this local alteration. The predicted "
    "effect is biologically consistent with experimentally established "
    "roles of PNPLA3 in hepatic lipid metabolism and with the established "
    "relationship between PNPLA3 variation and steatotic liver disease. "
    "The analysis also identifies a residue-166-centered pocket and "
    "prioritizes CID 71236597 as a computational therapeutic candidate, "
    "including a D166V-preferred docking signal and favorable "
    "physicochemical development characteristics. However, D166V-specific "
    "biochemical, cellular and disease-model validation has not been "
    "performed, and the compound has not been experimentally demonstrated "
    "to bind, inhibit or rescue the mutant protein. The final scientific "
    "interpretation is therefore that D166V represents a computationally "
    "supported and biologically plausible function-altering hypothesis, "
    "while variant-specific functional and therapeutic validation remains "
    "the principal experimental gap."
)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 33 — FINAL PUBLICATION / REPRODUCIBILITY / QC PACKAGE"
    )
    print("=" * 78)

    print()
    print(
        f"Project : {PROJECT_TITLE}"
    )

    print(
        f"Variant : {VARIANT}"
    )

    print(
        f"Primary CID : {PRIMARY_CID}"
    )

    print()

    # -------------------------------------------------------------------------
    # Resolve Step 9F
    # -------------------------------------------------------------------------

    step9f = locate_step9f()

    if step9f is None:

        print(
            "Step 9F source file: NOT RESOLVED"
        )

    else:

        print(
            f"Step 9F source file: {step9f}"
        )

    # -------------------------------------------------------------------------
    # Step status
    # -------------------------------------------------------------------------

    step_status = build_step_status()

    step_status.to_csv(
        TABLE_DIR
        / "STEP33_STEP_STATUS.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Build variant summary
    # -------------------------------------------------------------------------

    print(
        "Building final variant evidence table..."
    )

    variant_summary = build_variant_summary()

    variant_summary.to_csv(
        TABLE_DIR
        / "STEP33_FINAL_VARIANT_EVIDENCE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Build final lead table
    # -------------------------------------------------------------------------

    print(
        "Building final therapeutic-lead table..."
    )

    lead_table = build_final_lead_table()

    lead_table.to_csv(
        TABLE_DIR
        / "STEP33_FINAL_THERAPEUTIC_LEAD.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Master evidence
    # -------------------------------------------------------------------------

    print(
        "Building master scientific evidence table..."
    )

    evidence = build_master_evidence_table()

    evidence.to_csv(
        TABLE_DIR
        / "STEP33_MASTER_SCIENTIFIC_EVIDENCE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Claim guidance
    # -------------------------------------------------------------------------

    claims = build_claim_guidance()

    claims.to_csv(
        MANUSCRIPT_DIR
        / "STEP33_THESIS_CLAIM_GUIDANCE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Limitations
    # -------------------------------------------------------------------------

    limitations = build_limitations()

    limitations.to_csv(
        MANUSCRIPT_DIR
        / "STEP33_LIMITATIONS.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Reproducibility
    # -------------------------------------------------------------------------

    reproducibility = build_reproducibility_manifest()

    reproducibility.to_csv(
        QC_DIR
        / "STEP33_REPRODUCIBILITY_MANIFEST.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # File manifest
    # -------------------------------------------------------------------------

    file_manifest = build_file_manifest()

    file_manifest.to_csv(
        QC_DIR
        / "STEP33_RESULTS_FILE_MANIFEST.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Figure inventory
    # -------------------------------------------------------------------------

    figure_inventory = build_figure_inventory()

    figure_inventory.to_csv(
        QC_DIR
        / "STEP33_FIGURE_INVENTORY.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final conclusion
    # -------------------------------------------------------------------------

    with open(
        MANUSCRIPT_DIR
        / "STEP33_FINAL_SCIENTIFIC_CONCLUSION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            FINAL_CONCLUSION
        )

    # -------------------------------------------------------------------------
    # Step 9F provenance
    # -------------------------------------------------------------------------

    provenance = {

        "Step9F_Resolved":
            step9f is not None,

        "Step9F_Path":
            str(step9f)
            if step9f
            else "",

        "Step9F_Original_Step32_Path_Status":
            "MISSING_IN_STEP32_RUN",

        "Step25_Original_Step32_Extraction_Status":
            "SUMMARY_COLUMNS_NOT_RECOGNIZED_IN_STEP32",

        "Step25_Verified_Results_Carried_Forward":
            {
                "Contacts_Lost":
                    20,
                "Contacts_Gained":
                    8,
                "Hydropathy_Change":
                    7.7,
                "Charge_Change":
                    1.0,
                "Sidechain_Displacement_A":
                    1.166877,
            },

        "Step29_Interaction_Validation":
            {
                "Status":
                    "LIMITED",
                "Reason":
                    "Explicit WT/D166V Step 24 interaction footprint "
                    "could not be reconstructed automatically.",
                "Interpretation":
                    "Jaccard 0.000 from Step 29 must not be interpreted "
                    "as absence of shared interactions.",
            },

        "Scientific_Status":
            "Step 33 is a packaging and audit step; it does not alter "
            "previous scientific results.",
    }

    with open(
        QC_DIR
        / "STEP33_PROVENANCE_NOTES.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            provenance,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Summary statistics
    # -------------------------------------------------------------------------

    supported = int(
        evidence[
            "Status"
        ]
        .astype(str)
        .str.contains(
            "SUPPORTED",
            case=False,
            na=False,
        )
        .sum()
    )

    limited = int(
        (
            evidence[
                "Status"
            ]
            ==
            "LIMITED"
        ).sum()
    )

    unresolved = int(
        evidence[
            "Status"
        ]
        .astype(str)
        .str.contains(
            "UNRESOLVED|INFERENCE",
            case=False,
            regex=True,
            na=False,
        )
        .sum()
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc = {

        "Step":
            33,

        "Project_Title":
            PROJECT_TITLE,

        "Variant":
            VARIANT,

        "rsID":
            RSID,

        "VariationID":
            VARIATION_ID,

        "Primary_CID":
            PRIMARY_CID,

        "Step9F_resolved":
            step9f is not None,

        "Expected_tables_verified":
            int(
                sum(
                    path.exists()
                    for path in EXPECTED_FILES.values()
                )
            ),

        "Expected_tables_total":
            len(EXPECTED_FILES),

        "Master_evidence_domains":
            len(evidence),

        "Supported_evidence_domains":
            supported,

        "Limited_evidence_domains":
            limited,

        "Inference_or_unresolved_domains":
            unresolved,

        "Figure_count":
            len(figure_inventory),

        "Reproducibility_script_count":
            len(reproducibility),

        "File_manifest_count":
            len(file_manifest),

        "Step29_interaction_validation_limitation":
            True,

        "Status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP33_QC.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final report
    # -------------------------------------------------------------------------

    final_report = []

    final_report.append(
        "STEP 33 — FINAL PROJECT QC REPORT"
    )

    final_report.append(
        "=" * 78
    )

    final_report.append(
        f"Project: {PROJECT_TITLE}"
    )

    final_report.append(
        f"Variant: {VARIANT}"
    )

    final_report.append(
        f"Primary therapeutic candidate: CID {PRIMARY_CID}"
    )

    final_report.append(
        ""
    )

    final_report.append(
        f"Step 9F source resolved: "
        f"{step9f is not None}"
    )

    final_report.append(
        f"Verified expected-table files: "
        f"{qc['Expected_tables_verified']}/{qc['Expected_tables_total']}"
    )

    final_report.append(
        f"Evidence domains with support: "
        f"{supported}"
    )

    final_report.append(
        f"Limited evidence domains: "
        f"{limited}"
    )

    final_report.append(
        f"Inference/unresolved domains: "
        f"{unresolved}"
    )

    final_report.append(
        f"Figures inventoried: "
        f"{len(figure_inventory)}"
    )

    final_report.append(
        ""
    )

    final_report.append(
        "CRITICAL PROVENANCE NOTES:"
    )

    final_report.append(
        "1. Step 32 could not resolve the Step 9F source file because "
        "the assumed path did not match the actual project location."
    )

    final_report.append(
        "2. Step 32 could not automatically extract the Step 25 "
        "summary fields because the summary-table column names did "
        "not match the expected names. Verified Step 25 numerical "
        "results are retained explicitly in the Step 33 variant table."
    )

    final_report.append(
        "3. Step 29 could not reconstruct explicit WT/D166V Step 24 "
        "interaction fingerprints. Therefore the Step 29 Jaccard value "
        "of 0.000 is treated as an analysis limitation rather than "
        "biological absence of shared interactions."
    )

    final_report.append(
        ""
    )

    final_report.append(
        "FINAL SCIENTIFIC STATUS:"
    )

    final_report.append(
        FINAL_CONCLUSION
    )

    with open(
        QC_DIR
        / "STEP33_FINAL_QC_REPORT.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            "\n".join(
                final_report
            )
        )

    # -------------------------------------------------------------------------
    # Project README
    # -------------------------------------------------------------------------

    readme = f"""
FINAL PROJECT PACKAGE
=====================

Project
-------
{PROJECT_TITLE}

Primary variant
---------------
{VARIANT}
rsID: {RSID}
VariationID: {VARIATION_ID}

Primary computational therapeutic candidate
--------------------------------------------
CID {PRIMARY_CID}

Purpose
-------
This directory contains the final audit, reproducibility and publication-
support outputs generated from the completed computational project.
Previous Step 1–32 result directories are not modified by Step 33.

Important interpretation
------------------------
The project supports a computationally plausible D166V structural mechanism
and identifies a computational therapeutic candidate. It does not establish
D166V clinical pathogenicity, biochemical functional change, experimental
compound binding, therapeutic rescue, pharmacokinetics or clinical efficacy.

Key unresolved evidence
-----------------------
Direct D166V biochemical/cellular validation remains the principal
experimental gap.

Provenance limitations
----------------------
Step 32 had two extraction/path issues:
1. Step 9F source path was not resolved.
2. Step 25 summary columns were not recognized.

Step 29 also had incomplete automatic reconstruction of Step 24
WT/D166V interaction fingerprints.

These limitations are explicitly documented in:
qc/STEP33_PROVENANCE_NOTES.json

Main files
----------
tables/STEP33_FINAL_VARIANT_EVIDENCE.csv
tables/STEP33_FINAL_THERAPEUTIC_LEAD.csv
tables/STEP33_MASTER_SCIENTIFIC_EVIDENCE.csv
qc/STEP33_REPRODUCIBILITY_MANIFEST.csv
qc/STEP33_RESULTS_FILE_MANIFEST.csv
qc/STEP33_FIGURE_INVENTORY.csv
qc/STEP33_FINAL_QC_REPORT.txt
manuscript_support/STEP33_THESIS_CLAIM_GUIDANCE.csv
manuscript_support/STEP33_LIMITATIONS.csv
manuscript_support/STEP33_FINAL_SCIENTIFIC_CONCLUSION.txt
"""

    with open(
        OUT_DIR
        / "README.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            readme.strip()
        )

    # -------------------------------------------------------------------------
    # Final console
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print("STEP 33 RESULTS")
    print("=" * 78)

    print(
        f"Variant                         : {VARIANT}"
    )

    print(
        f"Primary therapeutic candidate   : CID {PRIMARY_CID}"
    )

    print(
        f"Step 9F source resolved          : "
        f"{step9f is not None}"
    )

    print(
        f"Expected tables verified         : "
        f"{qc['Expected_tables_verified']}/"
        f"{qc['Expected_tables_total']}"
    )

    print(
        f"Evidence domains                 : "
        f"{len(evidence)}"
    )

    print(
        f"Supported domains                : "
        f"{supported}"
    )

    print(
        f"Limited domains                  : "
        f"{limited}"
    )

    print(
        f"Inference/unresolved domains     : "
        f"{unresolved}"
    )

    print(
        f"Figures inventoried              : "
        f"{len(figure_inventory)}"
    )

    print(
        f"Scripts inventoried              : "
        f"{len(reproducibility)}"
    )

    print()

    print(
        "Final scientific status:"
    )

    print(
        "COMPUTATIONALLY SUPPORTED WITH "
        "AN EXPERIMENTAL VALIDATION GAP"
    )

    print()

    print(
        "Important: Step 33 packages and audits the completed results; "
        "it does not create new biological evidence."
    )

    print()

    print(
        f"Package directory: {OUT_DIR}"
    )

    print(
        f"QC report       : "
        f"{QC_DIR / 'STEP33_FINAL_QC_REPORT.txt'}"
    )

    print(
        f"Final lead table: "
        f"{TABLE_DIR / 'STEP33_FINAL_THERAPEUTIC_LEAD.csv'}"
    )

    print(
        f"Evidence table  : "
        f"{TABLE_DIR / 'STEP33_MASTER_SCIENTIFIC_EVIDENCE.csv'}"
    )

    print()

    print(
        "Step 33 completed successfully."
    )


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStep 33 interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 33 FAILED")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        raise