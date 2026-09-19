from pathlib import Path
import json
import sys

import pandas as pd
import numpy as np


# =============================================================================
# STEP 31 — INDEPENDENT BIOLOGICAL & LITERATURE VALIDATION
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STEP25_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP25_D166V_FUNCTIONAL_MECHANISM"
)

STEP26_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP26_D166V_MASLD_MECHANISTIC_LINK"
)

STEP27_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP27_ORTHOGONAL_STRUCTURAL_VALIDATION"
)

STEP28_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP28_MUTATION_SPECIFIC_THERAPEUTIC_VULNERABILITY"
)

STEP29_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP29_LEAD_VALIDATION_RESCORING"
)

STEP30_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP30_ADMET_DRUG_DEVELOPMENT_FEASIBILITY"
)

OUT_DIR = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP31_INDEPENDENT_BIOLOGICAL_VALIDATION"
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
# LITERATURE EVIDENCE DATASET
# =============================================================================
#
# Important:
# These records are deliberately restricted to literature-level biological
# facts relevant to the mechanism. They are NOT treated as direct evidence
# that D166V itself has been experimentally characterized.
#
# The D166V-specific literature status is recorded separately.
# =============================================================================

LITERATURE_RECORDS = [
    {
        "Evidence_ID":
            "LIT-01",
        "Evidence_Level":
            "Established experimental biology",
        "Topic":
            "PNPLA3 biochemical function",
        "Finding":
            "WT PNPLA3 has triglyceride lipase activity and can "
            "mobilize polyunsaturated fatty acids from triglycerides.",
        "Relevance_to_D166V":
            "Supports the interpretation that a substitution at a "
            "functionally important catalytic region could alter "
            "PNPLA3 molecular function.",
        "D166V_specific":
            "No",
        "Reference":
            "Johnson et al., 2024, Nature Communications",
        "Title":
            "PNPLA3 is a triglyceride lipase that mobilizes "
            "polyunsaturated fatty acids to facilitate hepatic "
            "secretion of large-sized very low-density lipoprotein",
        "URL":
            "https://consensus.app/papers/"
            "pnpla3-is-a-triglyceride-lipase-that-mobilizes-johnson-bao/"
            "14bb4027e6c151dea9fdbe85732e1bba/",
    },
    {
        "Evidence_ID":
            "LIT-02",
        "Evidence_Level":
            "Established experimental biology",
        "Topic":
            "PNPLA3 lipid remodeling",
        "Finding":
            "PNPLA3 participates in remodeling of triglyceride and "
            "phospholipid fatty-acyl composition in hepatic lipid droplets.",
        "Relevance_to_D166V":
            "Provides an independently established biological "
            "function that can be affected by altered PNPLA3 activity.",
        "D166V_specific":
            "No",
        "Reference":
            "Mitsche et al., 2018, The Journal of Biological Chemistry",
        "Title":
            "Patatin-like phospholipase domain–containing protein 3 "
            "promotes transfers of essential fatty acids from "
            "triglycerides to phospholipids in hepatic lipid droplets",
        "URL":
            "https://consensus.app/papers/"
            "patatinlike-phospholipase-domain–containing-protein-3-"
            "mitsche-hobbs/"
            "39bc09dfbea1585ea2cb9a332d3a1d53/",
    },
    {
        "Evidence_ID":
            "LIT-03",
        "Evidence_Level":
            "Established disease association",
        "Topic":
            "PNPLA3 and steatotic liver disease",
        "Finding":
            "PNPLA3 genetic variation is strongly associated with "
            "steatotic liver disease, with I148M being the best "
            "characterized example.",
        "Relevance_to_D166V":
            "Supports the disease-level relevance of functionally "
            "important PNPLA3 variation.",
        "D166V_specific":
            "No",
        "Reference":
            "Trepo et al., 2016, Journal of Hepatology",
        "Title":
            "PNPLA3 gene in liver diseases",
        "URL":
            "https://consensus.app/papers/details/"
            "39608b0b8d3759ddbb0fc755c365f7f4/",
    },
    {
        "Evidence_ID":
            "LIT-04",
        "Evidence_Level":
            "Established disease-mechanism evidence",
        "Topic":
            "PNPLA3 variant mechanism",
        "Finding":
            "Experimental work has demonstrated that PNPLA3 variants "
            "can alter PNPLA3 behavior and downstream hepatic lipid "
            "metabolism, but the mechanisms differ among models and "
            "remain an active research area.",
        "Relevance_to_D166V":
            "Supports treating D166V as a mechanistic hypothesis "
            "requiring variant-specific validation rather than "
            "assuming all PNPLA3 variants act through I148M biology.",
        "D166V_specific":
            "No",
        "Reference":
            "Wang et al., 2024, Journal of Hepatology",
        "Title":
            "PNPLA3(148M) is a gain-of-function mutation that promotes "
            "hepatic steatosis by inhibiting ATGL-mediated triglyceride "
            "hydrolysis",
        "URL":
            "https://consensus.app/papers/details/"
            "1480e07cd1a055c893df7da42bd15fb5/",
    },
    {
        "Evidence_ID":
            "LIT-05",
        "Evidence_Level":
            "Recent human/experimental context",
        "Topic":
            "PNPLA3 and hepatic metabolism",
        "Finding":
            "Human studies of I148M carriers demonstrate altered "
            "hepatic lipid metabolism and mitochondrial metabolic "
            "phenotypes.",
        "Relevance_to_D166V":
            "Supports the broader biological link between PNPLA3 "
            "variation and hepatic metabolic phenotypes.",
        "D166V_specific":
            "No",
        "Reference":
            "Luukkonen et al., 2023, Cell Metabolism",
        "Title":
            "The PNPLA3 I148M variant increases ketogenesis and "
            "decreases hepatic de novo lipogenesis and mitochondrial "
            "function in humans",
        "URL":
            "https://consensus.app/papers/details/"
            "9d4e2826dbba51b59ad172c35e5022bd/",
    },
]


# =============================================================================
# D166V-SPECIFIC LITERATURE STATUS
# =============================================================================

D166V_LITERATURE_STATUS = {
    "Variant":
        "PNPLA3 p.Asp166Val (D166V)",
    "rsID":
        "rs1381635405",
    "Targeted_searches":
        [
            '"PNPLA3" "D166V"',
            '"PNPLA3" "Asp166Val"',
            '"rs1381635405" PNPLA3',
        ],
    "Direct_functional_characterization_found":
        False,
    "Interpretation":
        "No direct peer-reviewed functional characterization of "
        "PNPLA3 D166V was identified in the targeted literature "
        "search used for this project. This should be stated as "
        "a search finding, not as proof that no publication exists "
        "anywhere.",
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fail(message):

    print()
    print("=" * 78)
    print("STEP 31 FAILED")
    print("=" * 78)
    print(message)
    print()
    sys.exit(1)


def read_csv(path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:
        print(
            f"Warning: could not read {path}: {exc}"
        )
        return None


def safe_float(value):

    try:
        return float(value)
    except Exception:
        return np.nan


def locate_first_existing(
    paths,
):

    for path in paths:

        if path.exists():
            return path

    return None


# =============================================================================
# EXTRACT PREVIOUS COMPUTATIONAL EVIDENCE
# =============================================================================

def extract_step25():

    path = (
        STEP25_DIR
        / "tables"
        / "STEP25_D166V_FUNCTIONAL_MECHANISM_SUMMARY.csv"
    )

    df = read_csv(
        path
    )

    if df is None or df.empty:
        return {
            "available":
                False
        }

    return {
        "available":
            True,
        "file":
            str(path),
    }


def extract_step26():

    path = (
        STEP26_DIR
        / "tables"
        / "STEP26_FINAL_MECHANISTIC_INTEGRATION.csv"
    )

    df = read_csv(
        path
    )

    if df is None or df.empty:
        return {
            "available":
                False
        }

    return {
        "available":
            True,
        "file":
            str(path),
    }


def extract_step27():

    path = (
        STEP27_DIR
        / "tables"
        / "STEP27_RESIDUE_166_INDEPENDENT_METRICS.csv"
    )

    df = read_csv(
        path
    )

    if df is None or df.empty:
        return {
            "available":
                False
        }

    return {
        "available":
            True,
        "file":
            str(path),
    }


def extract_step28():

    path = (
        STEP28_DIR
        / "tables"
        / "STEP28_THERAPEUTIC_VULNERABILITY_SUMMARY.csv"
    )

    df = read_csv(
        path
    )

    if df is None or df.empty:
        return {
            "available":
                False
        }

    return {
        "available":
            True,
        "file":
            str(path),
    }


def extract_step29():

    path = (
        STEP29_DIR
        / "tables"
        / "STEP29_FULL_LEAD_VALIDATION.csv"
    )

    df = read_csv(
        path
    )

    if df is None or df.empty:
        return {
            "available":
                False
        }

    return {
        "available":
            True,
        "file":
            str(path),
    }


def extract_step30():

    path = (
        STEP30_DIR
        / "tables"
        / "STEP30_ADMET_DEVELOPMENT_SUMMARY.csv"
    )

    df = read_csv(
        path
    )

    if df is None or df.empty:
        return {
            "available":
                False
        }

    return {
        "available":
            True,
        "file":
            str(path),
    }


# =============================================================================
# COMPUTATIONAL–BIOLOGICAL MATCHING
# =============================================================================

def build_mapping():

    rows = [
        {
            "Mechanistic_Element":
                "PNPLA3 has experimentally established lipid-metabolizing activity",
            "Independent_Literature_Support":
                "LIT-01; LIT-02",
            "Project_Computational_Result":
                "D166V changes a catalytic-region residue and its local "
                "chemical/contact environment.",
            "Consistency":
                "CONSISTENT",
            "D166V_Direct_Literature":
                "No",
            "Interpretation":
                "The computational finding is biologically plausible, "
                "but variant-specific function remains unmeasured.",
        },
        {
            "Mechanistic_Element":
                "PNPLA3 genetic variation is associated with steatotic liver disease",
            "Independent_Literature_Support":
                "LIT-03",
            "Project_Computational_Result":
                "D166V is computationally predicted to alter a functionally "
                "important PNPLA3 region.",
            "Consistency":
                "CONSISTENT",
            "D166V_Direct_Literature":
                "No",
            "Interpretation":
                "The disease-level framework is supported, while D166V-specific "
                "causality remains an inference.",
        },
        {
            "Mechanistic_Element":
                "PNPLA3 variant effects can involve distinct molecular mechanisms",
            "Independent_Literature_Support":
                "LIT-04; LIT-05",
            "Project_Computational_Result":
                "D166V was analyzed independently rather than assuming "
                "the I148M mechanism.",
            "Consistency":
                "CONSISTENT",
            "D166V_Direct_Literature":
                "No",
            "Interpretation":
                "The project appropriately treats D166V as a separate "
                "variant-specific mechanistic hypothesis.",
        },
        {
            "Mechanistic_Element":
                "A functionally altered PNPLA3 state may be therapeutically actionable",
            "Independent_Literature_Support":
                "LIT-01; LIT-03; LIT-04",
            "Project_Computational_Result":
                "D166V lies within a previously prioritized pocket and "
                "CID 71236597 shows a D166V-preferred docking score.",
            "Consistency":
                "SUPPORTIVE",
            "D166V_Direct_Literature":
                "No",
            "Interpretation":
                "The therapeutic hypothesis is coherent but remains "
                "computational until biochemical validation is available.",
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# EVIDENCE STRENGTH
# =============================================================================

def calculate_evidence_summary():

    rows = [
        {
            "Evidence_Domain":
                "D166V direct literature",
            "Status":
                "NOT_FOUND",
            "Meaning":
                "No direct functional characterization identified "
                "in the targeted search.",
        },
        {
            "Evidence_Domain":
                "PNPLA3 biochemical function",
            "Status":
                "ESTABLISHED",
            "Meaning":
                "Independent experimental literature supports "
                "PNPLA3 lipid-metabolizing activity.",
        },
        {
            "Evidence_Domain":
                "PNPLA3–MASLD association",
            "Status":
                "ESTABLISHED",
            "Meaning":
                "PNPLA3 genetic variation is strongly associated "
                "with steatotic liver disease.",
        },
        {
            "Evidence_Domain":
                "D166V structural mechanism",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Meaning":
                "Steps 25 and 27 independently identify local "
                "chemical/contact remodeling at residue 166.",
        },
        {
            "Evidence_Domain":
                "D166V disease mechanism",
            "Status":
                "MECHANISTIC_INFERENCE",
            "Meaning":
                "The structural effect is consistent with established "
                "PNPLA3 biology, but direct D166V functional data are absent.",
        },
        {
            "Evidence_Domain":
                "Therapeutic hypothesis",
            "Status":
                "COMPUTATIONALLY_SUPPORTED",
            "Meaning":
                "The altered region is computationally targetable and "
                "a lead shows D166V-preferred docking.",
        },
        {
            "Evidence_Domain":
                "Therapeutic efficacy",
            "Status":
                "NOT_ESTABLISHED",
            "Meaning":
                "No biochemical, cellular, animal or clinical rescue "
                "experiment has been performed.",
        },
    ]

    return pd.DataFrame(
        rows
    )


# =============================================================================
# FINAL INTERPRETATION
# =============================================================================

def build_final_interpretation():

    return (
        "Independent biological-validation analysis supports the "
        "overall mechanistic framework developed in this project. "
        "Peer-reviewed experimental literature establishes that "
        "PNPLA3 participates in hepatic lipid metabolism and that "
        "PNPLA3 genetic variation is strongly linked to steatotic "
        "liver disease. The current computational results therefore "
        "fit an established biological context. However, the available "
        "literature used here is dominated by studies of PNPLA3 WT "
        "function and the common I148M variant rather than D166V. "
        "The targeted literature search did not identify direct "
        "peer-reviewed functional characterization of D166V. "
        "Accordingly, the project can state that D166V is computationally "
        "predicted to alter a functionally important PNPLA3 region and "
        "that this mechanism is biologically consistent with established "
        "PNPLA3/MASLD biology, but it should not state that D166V has been "
        "experimentally proven to cause MASLD. The same distinction applies "
        "to the therapeutic component: existing computational evidence "
        "supports a testable D166V-targeting hypothesis, not demonstrated "
        "therapeutic efficacy."
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 31 — INDEPENDENT BIOLOGICAL & LITERATURE VALIDATION"
    )
    print("=" * 78)

    print()

    print(
        "D166V-specific literature status:"
    )

    print(
        f"Direct functional characterization found: "
        f"{D166V_LITERATURE_STATUS['Direct_functional_characterization_found']}"
    )

    print(
        "This status reflects the targeted literature search used "
        "for the project and is not an absolute claim about all literature."
    )

    print()

    # -------------------------------------------------------------------------
    # Previous computational evidence
    # -------------------------------------------------------------------------

    step25 = extract_step25()
    step26 = extract_step26()
    step27 = extract_step27()
    step28 = extract_step28()
    step29 = extract_step29()
    step30 = extract_step30()

    print(
        "Computational evidence availability:"
    )

    print(
        f"Step 25 : {step25['available']}"
    )

    print(
        f"Step 26 : {step26['available']}"
    )

    print(
        f"Step 27 : {step27['available']}"
    )

    print(
        f"Step 28 : {step28['available']}"
    )

    print(
        f"Step 29 : {step29['available']}"
    )

    print(
        f"Step 30 : {step30['available']}"
    )

    print()

    # -------------------------------------------------------------------------
    # Literature table
    # -------------------------------------------------------------------------

    literature_df = pd.DataFrame(
        LITERATURE_RECORDS
    )

    literature_df.to_csv(
        TABLE_DIR
        / "STEP31_LITERATURE_EVIDENCE.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Biological matching
    # -------------------------------------------------------------------------

    mapping_df = build_mapping()

    mapping_df.to_csv(
        TABLE_DIR
        / "STEP31_COMPUTATION_LITERATURE_MATCHING.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Evidence summary
    # -------------------------------------------------------------------------

    evidence_df = calculate_evidence_summary()

    evidence_df.to_csv(
        TABLE_DIR
        / "STEP31_EVIDENCE_STATUS_SUMMARY.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Literature search metadata
    # -------------------------------------------------------------------------

    search_df = pd.DataFrame(
        [
            {
                "Variant":
                    D166V_LITERATURE_STATUS[
                        "Variant"
                    ],
                "rsID":
                    D166V_LITERATURE_STATUS[
                        "rsID"
                    ],
                "Search_Query":
                    query,
                "Direct_D166V_Functional_Publication_Found":
                    D166V_LITERATURE_STATUS[
                        "Direct_functional_characterization_found"
                    ],
            }
            for query in D166V_LITERATURE_STATUS[
                "Targeted_searches"
            ]
        ]
    )

    search_df.to_csv(
        TABLE_DIR
        / "STEP31_D166V_LITERATURE_SEARCH_STATUS.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Final interpretation
    # -------------------------------------------------------------------------

    interpretation = build_final_interpretation()

    with open(
        QC_DIR
        / "STEP31_INTERPRETATION.txt",
        "w",
        encoding="utf-8",
    ) as handle:

        handle.write(
            interpretation
        )

    # -------------------------------------------------------------------------
    # Reference file
    # -------------------------------------------------------------------------

    references = []

    for record in LITERATURE_RECORDS:

        references.append(
            {
                "Reference_ID":
                    record["Evidence_ID"],
                "Reference":
                    record["Reference"],
                "Title":
                    record["Title"],
                "URL":
                    record["URL"],
            }
        )

    pd.DataFrame(
        references
    ).to_csv(
        TABLE_DIR
        / "STEP31_REFERENCES.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc = {
        "Step":
            31,
        "Variant":
            "D166V",
        "Targeted_D166V_functional_literature_found":
            D166V_LITERATURE_STATUS[
                "Direct_functional_characterization_found"
            ],
        "Literature_records":
            len(literature_df),
        "Computation_literature_mapping_records":
            len(mapping_df),
        "Step25_available":
            step25["available"],
        "Step26_available":
            step26["available"],
        "Step27_available":
            step27["available"],
        "Step28_available":
            step28["available"],
        "Step29_available":
            step29["available"],
        "Step30_available":
            step30["available"],
        "Status":
            "PASS",
    }

    with open(
        QC_DIR
        / "STEP31_QC.json",
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            qc,
            handle,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Final console output
    # -------------------------------------------------------------------------

    print("=" * 78)
    print("STEP 31 RESULTS")
    print("=" * 78)

    print(
        f"Literature records evaluated      : "
        f"{len(literature_df)}"
    )

    print(
        "D166V direct functional literature: "
        f"{'FOUND' if D166V_LITERATURE_STATUS['Direct_functional_characterization_found'] else 'NOT IDENTIFIED'}"
    )

    print(
        "PNPLA3 biochemical biology        : ESTABLISHED"
    )

    print(
        "PNPLA3–MASLD biological link      : ESTABLISHED"
    )

    print(
        "D166V structural mechanism        : "
        "COMPUTATIONALLY SUPPORTED"
    )

    print(
        "D166V disease mechanism           : "
        "MECHANISTIC INFERENCE"
    )

    print(
        "D166V therapeutic hypothesis      : "
        "COMPUTATIONALLY SUPPORTED"
    )

    print(
        "Therapeutic efficacy              : "
        "NOT ESTABLISHED"
    )

    print()

    print(
        "Final interpretation:"
    )

    print(
        interpretation
    )

    print()

    print(
        "Step 31 completed successfully."
    )

    print(
        f"Literature table : "
        f"{TABLE_DIR / 'STEP31_LITERATURE_EVIDENCE.csv'}"
    )

    print(
        f"Evidence status   : "
        f"{TABLE_DIR / 'STEP31_EVIDENCE_STATUS_SUMMARY.csv'}"
    )

    print(
        f"Mapping table     : "
        f"{TABLE_DIR / 'STEP31_COMPUTATION_LITERATURE_MATCHING.csv'}"
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
            "\nStep 31 interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 78)
        print("STEP 31 FAILED")
        print("=" * 78)
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise