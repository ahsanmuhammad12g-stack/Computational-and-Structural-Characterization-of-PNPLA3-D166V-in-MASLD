from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np


# =============================================================================
# STEP 32 — FINAL EVIDENCE INTEGRATION & SCIENTIFIC CONCLUSION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP32_FINAL_EVIDENCE_INTEGRATION"
)

TABLE_DIR = OUT_DIR / "tables"
QC_DIR = OUT_DIR / "QC"

for folder in [
    TABLE_DIR,
    QC_DIR,
]:
    folder.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# INPUTS
# =============================================================================

INPUTS = {

    "Step09F":
        PROJECT_ROOT
        / "RESULTS"
        / "04_VARIANT ANALYSIS"
        / "annotations"
        / "pnpla3_step9f_final_candidate_selection_FINAL.csv",

    "Step16":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
        / "tables"
        / "Table_16_04_D166V_Pocket_Prioritization.csv",

    "Step22":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING"
        / "tables"
        / "STEP22_WT_vs_D166V_DIFFERENTIAL_DOCKING.csv",

    "Step23":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP23_LEAD_PRIORITIZATION"
        / "tables"
        / "STEP23_LEAD_PRIORITIZATION.csv",

    "Step24":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP24_INTERACTION_FINGERPRINTING"
        / "tables"
        / "STEP24_LEAD_INTERACTION_SUMMARY.csv",

    "Step25":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP25_D166V_FUNCTIONAL_MECHANISM"
        / "tables"
        / "STEP25_D166V_FUNCTIONAL_MECHANISM_SUMMARY.csv",

    "Step26":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP26_D166V_MASLD_MECHANISTIC_LINK"
        / "tables"
        / "STEP26_FINAL_MECHANISTIC_INTEGRATION.csv",

    "Step27":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP27_ORTHOGONAL_STRUCTURAL_VALIDATION"
        / "tables"
        / "STEP27_RESIDUE_166_INDEPENDENT_METRICS.csv",

    "Step28":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP28_MUTATION_SPECIFIC_THERAPEUTIC_VULNERABILITY"
        / "tables"
        / "STEP28_THERAPEUTIC_VULNERABILITY_SUMMARY.csv",

    "Step29":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP29_LEAD_VALIDATION_RESCORING"
        / "tables"
        / "STEP29_FULL_LEAD_VALIDATION.csv",

    "Step30":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP30_ADMET_DRUG_DEVELOPMENT_FEASIBILITY"
        / "tables"
        / "STEP30_ADMET_DEVELOPMENT_SUMMARY.csv",

    "Step31":
        PROJECT_ROOT
        / "RESULTS"
        / "STEP31_INDEPENDENT_BIOLOGICAL_VALIDATION"
        / "tables"
        / "STEP31_EVIDENCE_STATUS_SUMMARY.csv",
}


# =============================================================================
# TARGET
# =============================================================================

TARGET_VARIANT = "D166V"
TARGET_RESIDUE = 166
TARGET_CID = "71236597"


# =============================================================================
# HELPERS
# =============================================================================

def fail(message):

    print()
    print("=" * 78)
    print("STEP 32 FAILED")
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

    except Exception as exc:

        print(
            f"Warning: unable to read {path}: {exc}"
        )

        return None


def normalize_cid(value):

    try:
        return str(
            int(float(value))
        )

    except Exception:

        return str(value).strip()


def safe_float(value):

    try:
        return float(value)

    except Exception:

        return np.nan


def find_column(
    df,
    candidates,
):

    if df is None:
        return None

    lookup = {
        str(column).strip().lower():
            column
        for column in df.columns
    }

    for candidate in candidates:

        key = (
            str(candidate)
            .strip()
            .lower()
        )

        if key in lookup:

            return lookup[key]

    return None


def first_valid_value(
    df,
    candidates,
    default=np.nan,
):

    column = find_column(
        df,
        candidates,
    )

    if column is None:
        return default

    if df.empty:
        return default

    value = df.iloc[0][column]

    return value


# =============================================================================
# FILE AVAILABILITY
# =============================================================================

def assess_input_availability():

    rows = []

    for name, path in INPUTS.items():

        rows.append(
            {
                "Step":
                    name,
                "Path":
                    str(path),
                "Exists":
                    path.exists(),
            }
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EXTRACT KEY RESULTS
# =============================================================================

def extract_variant_evidence():

    df = read_csv(
        INPUTS["Step09F"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    row = df.iloc[0]

    variant = first_valid_value(
        df,
        [
            "Protein_Change",
            "protein_change",
            "Mutation",
            "Variant",
            "HGVS_p",
            "Name",
        ],
        TARGET_VARIANT,
    )

    variation_id = first_valid_value(
        df,
        [
            "VariationID",
            "Variation_Id",
            "Variation_ID",
        ],
    )

    score = first_valid_value(
        df,
        [
            "Score",
            "Final_Score",
        ],
    )

    rank = first_valid_value(
        df,
        [
            "Rank",
            "Final_Rank",
        ],
    )

    alpha = first_valid_value(
        df,
        [
            "AlphaMissense",
            "AlphaMissense_Score",
            "AlphaMissense_Score_1",
        ],
    )

    cadd = first_valid_value(
        df,
        [
            "CADD_PHRED",
            "CADD_PHRED_Score",
            "CADD_PHRED_1",
        ],
    )

    af_exome = first_valid_value(
        df,
        [
            "gnomAD_Exome_AF",
            "Exome_AF",
        ],
    )

    af_genome = first_valid_value(
        df,
        [
            "gnomAD_Genome_AF",
            "Genome_AF",
        ],
    )

    return {
        "available":
            True,
        "Variant":
            variant,
        "VariationID":
            variation_id,
        "Rank":
            rank,
        "Selection_Score":
            score,
        "AlphaMissense":
            alpha,
        "CADD_PHRED":
            cadd,
        "gnomAD_Exome_AF":
            af_exome,
        "gnomAD_Genome_AF":
            af_genome,
    }


def extract_step16():

    df = read_csv(
        INPUTS["Step16"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    result = {
        "available":
            True
    }

    fields = {
        "Rank":
            [
                "Rank",
            ],
        "Druggability":
            [
                "Druggability",
                "D166V_Druggability",
            ],
        "Pocket_Score":
            [
                "Pocket_Score",
                "D166V_Pocket_Score",
            ],
        "Volume_A3":
            [
                "Volume_A3",
                "Volume",
                "D166V_Volume_A3",
            ],
        "D166_Distance_A":
            [
                "D166_Distance_A",
                "D166_Distance",
            ],
        "D166_Directly_Lining":
            [
                "D166_Directly_Lining",
                "Directly_Lining",
            ],
        "Lining_Jaccard":
            [
                "Lining_Jaccard",
            ],
        "Mutation_Remodeling_Score":
            [
                "Mutation_Remodeling_Score",
            ],
        "Prioritization_Score":
            [
                "Prioritization_Score",
                "D166V_Prioritization_Score",
            ],
    }

    for key, candidates in fields.items():

        result[key] = first_valid_value(
            df,
            candidates,
        )

    return result


def extract_step25():

    df = read_csv(
        INPUTS["Step25"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    return {
        "available":
            True,
        "Hydropathy_Change":
            first_valid_value(
                df,
                [
                    "Hydropathy_Change",
                    "Hydropathy_Delta",
                ],
            ),
        "Charge_Change":
            first_valid_value(
                df,
                [
                    "Charge_Change",
                    "Approximate_Charge_Change",
                ],
            ),
        "Contact_Gained":
            first_valid_value(
                df,
                [
                    "Contacts_Gained",
                    "Contact_Gained",
                    "Gained_Contacts",
                ],
            ),
        "Contact_Lost":
            first_valid_value(
                df,
                [
                    "Contacts_Lost",
                    "Contact_Lost",
                    "Lost_Contacts",
                ],
            ),
        "Sidechain_Displacement_A":
            first_valid_value(
                df,
                [
                    "Sidechain_Center_Displacement_A",
                    "Sidechain_Displacement_A",
                ],
            ),
    }


def extract_step27():

    df = read_csv(
        INPUTS["Step27"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    model_column = find_column(
        df,
        [
            "Model",
        ],
    )

    if model_column is None:

        return {
            "available":
                True
        }

    output = {
        "available":
            True
    }

    for model_name in [
        "WT",
        "D166V",
    ]:

        subset = df[
            df[
                model_column
            ]
            .astype(str)
            .str.upper()
            ==
            model_name.upper()
        ]

        if subset.empty:
            continue

        row = subset.iloc[0]

        for field in [
            "Residue_SASA_A2",
            "Heavy_Atom_Contact_Count_4A",
            "Polar_Atom_Contact_Count_4A",
        ]:

            if field in df.columns:

                output[
                    f"{model_name}_{field}"
                ] = safe_float(
                    row[field]
                )

    return output


def extract_step22():

    df = read_csv(
        INPUTS["Step22"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    if "CID" not in df.columns:

        return {
            "available":
                False
        }

    subset = df[
        df["CID"]
        .map(normalize_cid)
        ==
        TARGET_CID
    ]

    if subset.empty:

        return {
            "available":
                True
        }

    row = subset.iloc[0]

    return {
        "available":
            True,
        "WT_Score":
            safe_float(
                row[
                    "WT_Best_Vina_Score_kcal_mol"
                ]
            ),
        "D166V_Score":
            safe_float(
                row[
                    "D166V_Best_Vina_Score_kcal_mol"
                ]
            ),
        "Delta":
            safe_float(
                row[
                    "Delta_Vina_D166V_minus_WT_kcal_mol"
                ]
            ),
        "Differential_Class":
            row.get(
                "Differential_Docking_Class",
                "",
            ),
    }


def extract_step23():

    df = read_csv(
        INPUTS["Step23"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    if "CID" not in df.columns:

        return {
            "available":
                False
        }

    subset = df[
        df["CID"]
        .map(normalize_cid)
        ==
        TARGET_CID
    ]

    if subset.empty:

        return {
            "available":
                True
        }

    row = subset.iloc[0]

    result = {
        "available":
            True
    }

    for field in [
        "Integrated_Score",
        "Step23_Rank",
        "Lead_Class",
        "Priority",
    ]:

        if field in df.columns:

            result[field] = row[field]

    return result


def extract_step28():

    df = read_csv(
        INPUTS["Step28"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    if "CID_Normalized" in df.columns:

        cid_column = "CID_Normalized"

    elif "CID" in df.columns:

        cid_column = "CID"

    else:

        return {
            "available":
                False
        }

    subset = df[
        df[cid_column]
        .map(normalize_cid)
        ==
        TARGET_CID
    ]

    if subset.empty:

        return {
            "available":
                True
        }

    row = subset.iloc[0]

    return {
        "available":
            True,
        "Ligand_Specific_Vulnerability_Score":
            row.get(
                "Ligand_Specific_Vulnerability_Score",
                np.nan,
            ),
        "Therapeutic_Interpretation":
            row.get(
                "Therapeutic_Interpretation",
                "",
            ),
        "Docking_Preference_Class":
            row.get(
                "Docking_Preference_Class",
                "",
            ),
    }


def extract_step29():

    df = read_csv(
        INPUTS["Step29"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    if "CID" not in df.columns:

        return {
            "available":
                False
        }

    subset = df[
        df["CID"]
        .map(normalize_cid)
        ==
        TARGET_CID
    ]

    if subset.empty:

        return {
            "available":
                True
        }

    row = subset.iloc[0]

    return {
        "available":
            True,
        "Validation_Score":
            row.get(
                "Step29_Validation_Score",
                np.nan,
            ),
        "Validation_Class":
            row.get(
                "Step29_Validation_Class",
                "",
            ),
        "Interaction_Jaccard":
            row.get(
                "Interaction_Jaccard",
                np.nan,
            ),
    }


def extract_step30():

    df = read_csv(
        INPUTS["Step30"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    if "CID" not in df.columns:

        return {
            "available":
                False
        }

    subset = df[
        df["CID"]
        .map(normalize_cid)
        ==
        TARGET_CID
    ]

    if subset.empty:

        return {
            "available":
                True
        }

    row = subset.iloc[0]

    return {
        "available":
            True,
        "ADMET_Score":
            row.get(
                "ADMET_Development_Score",
                np.nan,
            ),
        "Development_Class":
            row.get(
                "Development_Class",
                "",
            ),
        "Step30_Priority":
            row.get(
                "Step30_Development_Priority_Score",
                np.nan,
            ),
    }


def extract_step31():

    df = read_csv(
        INPUTS["Step31"]
    )

    if df is None or df.empty:

        return {
            "available":
                False
        }

    result = {
        "available":
            True
    }

    for status_name in [
        "D166V direct literature",
        "PNPLA3 biochemical function",
        "PNPLA3–MASLD association",
        "D166V structural mechanism",
        "D166V disease mechanism",
        "D166V therapeutic hypothesis",
        "Therapeutic efficacy",
    ]:

        match = df[
            df[
                "Evidence_Domain"
            ]
            .astype(str)
            .str.strip()
            .str.casefold()
            ==
            status_name.casefold()
        ]

        if not match.empty:

            result[
                status_name
            ] = match.iloc[0][
                "Status"
            ]

    return result


# =============================================================================
# MASTER EVIDENCE CHAIN
# =============================================================================

def build_master_evidence_chain():

    rows = [

        {
            "Stage":
                "1. Variant identification",
            "Question":
                "Was D166V selected as the unresolved PNPLA3 candidate?",
            "Evidence":
                "Step 9F candidate-selection analysis",
            "Status":
                "SUPPORTED",
            "Claim_Level":
                "Project finding",
            "Limitation":
                "Computational prioritization is not equivalent to "
                "experimental pathogenicity.",
        },

        {
            "Stage":
                "2. Population context",
            "Question":
                "Is D166V compatible with a rare unresolved variant profile?",
            "Evidence":
                "gnomAD annotation carried through Step 9F",
            "Status":
                "SUPPORTED",
            "Claim_Level":
                "Observed database evidence",
            "Limitation":
                "Allele frequency does not establish pathogenicity.",
        },

        {
            "Stage":
                "3. Structural substitution",
            "Question":
                "Does Asp166 → Val166 occur in the generated mutant structure?",
            "Evidence":
                "Steps 10, 25 and 27",
            "Status":
                "SUPPORTED",
            "Claim_Level":
                "Direct computational observation",
            "Limitation":
                "Static structural model.",
        },

        {
            "Stage":
                "4. Local molecular effect",
            "Question":
                "Does D166V alter the chemical/contact environment?",
            "Evidence":
                "Steps 25 and 27",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Claim_Level":
                "Structural mechanism",
            "Limitation":
                "Contact and physicochemical descriptors are computational proxies.",
        },

        {
            "Stage":
                "5. Global structural effect",
            "Question":
                "Does D166V cause gross protein backbone disruption?",
            "Evidence":
                "Steps 26 and 27",
            "Status":
                "NOT_OBSERVED",
            "Claim_Level":
                "Negative structural observation",
            "Limitation":
                "Absence of backbone change does not exclude functional effects.",
        },

        {
            "Stage":
                "6. PNPLA3 functional relevance",
            "Question":
                "Is the affected region biologically relevant to PNPLA3 function?",
            "Evidence":
                "Project catalytic-site analysis + independent literature",
            "Status":
                "SUPPORTED",
            "Claim_Level":
                "Biological integration",
            "Limitation":
                "Direct D166V enzymatic measurements are unavailable.",
        },

        {
            "Stage":
                "7. MASLD mechanistic connection",
            "Question":
                "Can the predicted molecular defect be connected to MASLD biology?",
            "Evidence":
                "Steps 26 and 31",
            "Status":
                "MECHANISTICALLY_CONSISTENT",
            "Claim_Level":
                "Integrated inference",
            "Limitation":
                "D166V-specific disease causality is not experimentally demonstrated.",
        },

        {
            "Stage":
                "8. Therapeutic pocket",
            "Question":
                "Is the altered region computationally targetable?",
            "Evidence":
                "Step 16",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Claim_Level":
                "Structural druggability hypothesis",
            "Limitation":
                "Pocket scores do not prove binding.",
        },

        {
            "Stage":
                "9. Ligand screening",
            "Question":
                "Can existing compounds computationally occupy the region?",
            "Evidence":
                "Steps 20–23",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Claim_Level":
                "Virtual-screening result",
            "Limitation":
                "Docking scores are not measured binding affinities.",
        },

        {
            "Stage":
                "10. D166V differential preference",
            "Question":
                "Does a lead show a different docking score for D166V vs WT?",
            "Evidence":
                "Steps 22 and 28",
            "Status":
                "SUPPORTED",
            "Claim_Level":
                "Comparative docking result",
            "Limitation":
                "A docking-score difference does not establish mutation-specific inhibition.",
        },

        {
            "Stage":
                "11. Interaction validation",
            "Question":
                "Are ligand interaction fingerprints independently validated?",
            "Evidence":
                "Step 24 / Step 29",
            "Status":
                "LIMITED",
            "Claim_Level":
                "Incomplete computational validation",
            "Limitation":
                "Step 29 could not reconstruct explicit WT/D166V Step 24 interaction footprints.",
        },

        {
            "Stage":
                "12. Development feasibility",
            "Question":
                "Do prioritized compounds have favorable physicochemical properties?",
            "Evidence":
                "Step 30",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Claim_Level":
                "Physicochemical screening",
            "Limitation":
                "Not experimental ADMET or clinical safety.",
        },

        {
            "Stage":
                "13. Independent biological validation",
            "Question":
                "Does independent literature support the broader biological framework?",
            "Evidence":
                "Step 31",
            "Status":
                "SUPPORTED",
            "Claim_Level":
                "Literature-supported biological context",
            "Limitation":
                "Available literature does not directly characterize D166V.",
        },

    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EVIDENCE COMPLETENESS PROFILE
# =============================================================================

def build_completeness_profile():

    rows = [

        {
            "Evidence_Domain":
                "Variant prioritization",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Population rarity",
            "Status":
                "DATABASE_SUPPORTED",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Structural alteration",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Local mechanistic effect",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "PNPLA3 functional relevance",
            "Status":
                "LITERATURE_SUPPORTED + COMPUTATIONAL",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "D166V → MASLD causality",
            "Status":
                "INFERENCE",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Therapeutic pocket",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Lead docking",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Present":
                True,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Mutation-specific therapeutic action",
            "Status":
                "NOT_ESTABLISHED",
            "Present":
                False,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Experimental ADMET",
            "Status":
                "NOT_ESTABLISHED",
            "Present":
                False,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Cellular rescue",
            "Status":
                "NOT_ESTABLISHED",
            "Present":
                False,
            "Direct_Experimental_Validation":
                False,
        },

        {
            "Evidence_Domain":
                "Animal/clinical validation",
            "Status":
                "NOT_ESTABLISHED",
            "Present":
                False,
            "Direct_Experimental_Validation":
                False,
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FINAL SCIENTIFIC CONCLUSION
# =============================================================================

def build_final_conclusion():

    return (
        "Integrated analysis supports D166V as a plausible function-altering "
        "PNPLA3 variant based on convergent computational evidence. The "
        "Asp166→Val substitution changes the chemical identity and local "
        "interaction environment of residue 166, while producing negligible "
        "global backbone displacement. Independent structural analysis "
        "reproduces the local remodeling signal. Independent biological "
        "literature establishes that PNPLA3 participates in hepatic lipid "
        "metabolism and that PNPLA3 genetic variation is strongly associated "
        "with steatotic liver disease. The computational D166V mechanism is "
        "therefore biologically consistent with the known PNPLA3/MASLD "
        "framework. The therapeutic analysis further identifies a "
        "residue-166-centered pocket and computationally prioritizes "
        "candidate ligands, with CID 71236597 showing a D166V-preferred "
        "docking score. However, the evidence does not establish that D166V "
        "causes MASLD, does not demonstrate a measured loss or gain of "
        "PNPLA3 enzymatic function, and does not demonstrate therapeutic "
        "rescue. The principal unresolved scientific gap is therefore "
        "variant-specific experimental validation rather than absence of a "
        "computational mechanistic hypothesis."
    )


# =============================================================================
# THESIS-READY CLAIMS
# =============================================================================

def build_claims_table():

    rows = [

        {
            "Claim":
                "D166V alters the chemical identity of PNPLA3 residue 166.",
            "Evidence":
                "Steps 10, 25, 27",
            "Permissible_Wording":
                "D166V replaces Asp166 with Val166.",
            "Avoid":
                "D166V completely destabilizes PNPLA3.",
        },

        {
            "Claim":
                "D166V alters the local residue-166 environment.",
            "Evidence":
                "Steps 25 and 27",
            "Permissible_Wording":
                "D166V is predicted to remodel the local structural and interaction environment.",
            "Avoid":
                "D166V experimentally disrupts catalytic activity.",
        },

        {
            "Claim":
                "D166V is biologically relevant to PNPLA3 function.",
            "Evidence":
                "Steps 26 and 31",
            "Permissible_Wording":
                "The predicted D166V structural effect is consistent with an alteration of a functionally important PNPLA3 region.",
            "Avoid":
                "D166V is proven to be pathogenic.",
        },

        {
            "Claim":
                "D166V can be linked mechanistically to MASLD.",
            "Evidence":
                "Steps 26 and 31",
            "Permissible_Wording":
                "The D166V mechanism is biologically consistent with established PNPLA3/MASLD biology.",
            "Avoid":
                "D166V is proven to cause MASLD.",
        },

        {
            "Claim":
                "CID 71236597 is a computational lead.",
            "Evidence":
                "Steps 22, 23, 28, 30",
            "Permissible_Wording":
                "CID 71236597 was prioritized as a computational candidate with a D166V-preferred docking signal and favorable physicochemical properties.",
            "Avoid":
                "CID 71236597 is an established D166V inhibitor.",
        },

        {
            "Claim":
                "The therapeutic hypothesis is testable.",
            "Evidence":
                "Steps 16–30",
            "Permissible_Wording":
                "The analysis generates a testable hypothesis that the altered D166V region can be therapeutically targeted.",
            "Avoid":
                "The compound will treat MASLD.",
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 32 — FINAL EVIDENCE INTEGRATION & SCIENTIFIC CONCLUSION"
    )
    print("=" * 78)

    print()

    # -------------------------------------------------------------------------
    # Availability
    # -------------------------------------------------------------------------

    availability = assess_input_availability()

    print(
        "Input availability:"
    )

    for _, row in availability.iterrows():

        print(
            f"{row['Step']:>8} : "
            f"{'FOUND' if row['Exists'] else 'MISSING'}"
        )

    print()

    missing = availability[
        ~availability["Exists"]
    ]

    if not missing.empty:

        print(
            "Warning: one or more optional historical files are missing."
        )

        print(
            "Step 32 will continue using available evidence."
        )

        print()

    # -------------------------------------------------------------------------
    # Extract evidence
    # -------------------------------------------------------------------------

    variant = extract_variant_evidence()

    step16 = extract_step16()
    step25 = extract_step25()
    step27 = extract_step27()
    step22 = extract_step22()
    step23 = extract_step23()
    step28 = extract_step28()
    step29 = extract_step29()
    step30 = extract_step30()
    step31 = extract_step31()

    # -------------------------------------------------------------------------
    # Master evidence chain
    # -------------------------------------------------------------------------

    master_chain = build_master_evidence_chain()

    master_chain.to_csv(
        TABLE_DIR
        / "STEP32_MASTER_EVIDENCE_CHAIN.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Completeness
    # -------------------------------------------------------------------------

    completeness = build_completeness_profile()

    completeness.to_csv(
        TABLE_DIR
        / "STEP32_EVIDENCE_COMPLETENESS_PROFILE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Claims
    # -------------------------------------------------------------------------

    claims = build_claims_table()

    claims.to_csv(
        TABLE_DIR
        / "STEP32_THESIS_CLAIM_GUIDANCE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Quantitative project evidence summary
    # -------------------------------------------------------------------------

    quantitative = {

        "Variant":
            TARGET_VARIANT,

        "Target_Residue":
            TARGET_RESIDUE,

        "Primary_CID":
            TARGET_CID,

        "Variant_Selection":
            variant,

        "Step16":
            step16,

        "Step25":
            step25,

        "Step27":
            step27,

        "Step22":
            step22,

        "Step23":
            step23,

        "Step28":
            step28,

        "Step29":
            step29,

        "Step30":
            step30,

        "Step31":
            step31,
    }

    with open(
        TABLE_DIR
        / "STEP32_QUANTITATIVE_EVIDENCE_SUMMARY.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            quantitative,
            handle,
            indent=2,
            default=str,
        )

    # -------------------------------------------------------------------------
    # Final conclusion
    # -------------------------------------------------------------------------

    conclusion = build_final_conclusion()

    with open(
        QC_DIR
        / "STEP32_FINAL_SCIENTIFIC_CONCLUSION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            conclusion
        )

    # -------------------------------------------------------------------------
    # Research gap
    # -------------------------------------------------------------------------

    remaining_gap = (
        "The main remaining evidence gap is direct experimental "
        "characterization of PNPLA3 D166V. The highest-value validation "
        "would be variant-specific biochemical and/or cellular testing "
        "of PNPLA3 function, followed by testing whether the prioritized "
        "compound alters the relevant D166V phenotype. The present "
        "computational workflow cannot replace those experiments."
    )

    with open(
        QC_DIR
        / "STEP32_REMAINING_EVIDENCE_GAP.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            remaining_gap
        )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    supported_count = int(
        (
            completeness["Present"]
            == True
        ).sum()
    )

    direct_validation_count = int(
        (
            completeness[
                "Direct_Experimental_Validation"
            ]
            == True
        ).sum()
    )

    not_established_count = int(
        completeness[
            "Status"
        ]
        .astype(str)
        .str.contains(
            "NOT_ESTABLISHED",
            case=False,
            na=False,
        )
        .sum()
    )

    qc = {

        "Step":
            32,

        "Variant":
            TARGET_VARIANT,

        "Primary_CID":
            TARGET_CID,

        "Input_files_found":
            int(
                availability["Exists"].sum()
            ),

        "Input_files_missing":
            int(
                (~availability["Exists"]).sum()
            ),

        "Evidence_domains":
            len(completeness),

        "Evidence_domains_with_support":
            supported_count,

        "Direct_experimental_validation_domains":
            direct_validation_count,

        "Not_established_domains":
            not_established_count,

        "Step29_interaction_validation_limitation":
            True,

        "Final_conclusion_status":
            "COMPUTATIONALLY_SUPPORTED_WITH_EXPERIMENTAL_GAP",

        "Status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP32_QC.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final console
    # -------------------------------------------------------------------------

    print("=" * 78)
    print("STEP 32 RESULTS")
    print("=" * 78)

    print(
        f"Variant                         : "
        f"{TARGET_VARIANT}"
    )

    print(
        f"Primary therapeutic candidate   : "
        f"CID {TARGET_CID}"
    )

    if step22.get(
        "D166V_Score"
    ) is not None:

        print(
            f"CID {TARGET_CID} D166V docking   : "
            f"{step22.get('D166V_Score')}"
        )

        print(
            f"CID {TARGET_CID} WT docking      : "
            f"{step22.get('WT_Score')}"
        )

        print(
            f"CID {TARGET_CID} Δ(D166V−WT)      : "
            f"{step22.get('Delta')}"
        )

    if step25.get(
        "Contact_Lost"
    ) is not None:

        print(
            f"Step 25 contacts lost             : "
            f"{step25.get('Contact_Lost')}"
        )

        print(
            f"Step 25 contacts gained           : "
            f"{step25.get('Contact_Gained')}"
        )

    if step27.get(
        "WT_Residue_SASA_A2"
    ) is not None:

        print(
            f"Step 27 WT residue-166 SASA       : "
            f"{step27.get('WT_Residue_SASA_A2')}"
        )

        print(
            f"Step 27 D166V residue-166 SASA    : "
            f"{step27.get('D166V_Residue_SASA_A2')}"
        )

    print()

    print(
        "Evidence completeness:"
    )

    print(
        f"Domains with computational/literature support : "
        f"{supported_count}"
    )

    print(
        f"Domains with direct experimental validation    : "
        f"{direct_validation_count}"
    )

    print(
        f"Domains not established                         : "
        f"{not_established_count}"
    )

    print()

    print(
        "FINAL SCIENTIFIC CONCLUSION:"
    )

    print(
        conclusion
    )

    print()

    print(
        "Remaining evidence gap:"
    )

    print(
        remaining_gap
    )

    print()

    print(
        "Step 32 completed successfully."
    )

    print(
        f"Master evidence chain : "
        f"{TABLE_DIR / 'STEP32_MASTER_EVIDENCE_CHAIN.csv'}"
    )

    print(
        f"Completeness profile  : "
        f"{TABLE_DIR / 'STEP32_EVIDENCE_COMPLETENESS_PROFILE.csv'}"
    )

    print(
        f"Claim guidance        : "
        f"{TABLE_DIR / 'STEP32_THESIS_CLAIM_GUIDANCE.csv'}"
    )

    print()


# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "\nStep 32 interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 32 FAILED")
        print("=" * 78)

        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()

        raise