# =============================================================================
# STEP 18 — THERAPEUTIC COMPOUND LIBRARY
# =============================================================================
#
# Purpose:
#   Build a provenance-aware therapeutic compound library for the prioritized
#   PNPLA3 D166V-associated pocket identified in Steps 16–17.
#
# Scientific scope:
#   - Retrieve compounds from PubChem using text-search context.
#   - Preserve search provenance.
#   - Retrieve standardized chemical properties.
#   - Perform basic chemical-structure QC.
#   - Deduplicate compounds using InChIKey.
#   - Produce a screening-ready compound library.
#
# IMPORTANT:
#   This step DOES NOT claim that any compound inhibits PNPLA3.
#   Search-category membership is not equivalent to experimentally validated
#   target activity.
#
#   Step 18 prepares compounds for downstream:
#       Step 19 — Drug-likeness & chemical filtering
#       Step 20 — Pharmacophore / shape screening
#       Step 21 — Virtual screening
#       Step 22 — WT vs D166V differential docking
#
# =============================================================================

from pathlib import Path
import json
import time
import re
import sys

import pandas as pd
import numpy as np
import requests


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

RESULTS_ROOT = PROJECT_ROOT / "RESULTS"

STEP18_ROOT = RESULTS_ROOT / "STEP18_THERAPEUTIC_COMPOUND_LIBRARY"

TABLES_DIR = STEP18_ROOT / "tables"
QC_DIR = STEP18_ROOT / "QC"
REPORTS_DIR = STEP18_ROOT / "reports"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
QC_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Maximum number of PubChem compounds retrieved per search term.
MAX_CIDS_PER_TERM = 100

# HTTP settings.
REQUEST_TIMEOUT = 60
MAX_RETRIES = 3

# Search terms used for the initial therapeutic/context library.
SEARCH_TERMS = [
    ("PNPLA3", "PNPLA3_specific"),
    ("PNPLA3 inhibitor", "PNPLA3_specific"),
    ("PNPLA3 small molecule", "PNPLA3_specific"),
    ("patatin phospholipase inhibitor", "enzyme_context"),
    ("lipid droplet lipase inhibitor", "lipid_metabolism"),
    ("hepatic lipid metabolism small molecule", "hepatic_metabolism"),
    ("MASLD small molecule", "MASLD_context"),
]


# =============================================================================
# HTTP SESSION
# =============================================================================

SESSION = requests.Session()

SESSION.headers.update(
    {
        "User-Agent": (
            "MASLD-PNPLA3-Step18/1.3 "
            "(therapeutic-compound-library)"
        ),
        "Accept": "application/json",
    }
)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_float(value):
    """Convert a value to float when possible."""
    if value is None:
        return np.nan

    if isinstance(value, str):
        value = value.strip()

        if value == "":
            return np.nan

    try:
        return float(value)
    except Exception:
        return np.nan


def clean_text(value):
    """Return a clean string representation."""
    if value is None:
        return ""

    if isinstance(value, float) and np.isnan(value):
        return ""

    return str(value).strip()


def sanitize_filename(text):
    """Make a safe filename."""
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("_")


def recursive_find(filename_pattern, root):
    """
    Recursively search for a file below root.

    Returns the first matching file sorted by path.
    """
    matches = sorted(root.rglob(filename_pattern))

    if not matches:
        return None

    return matches[0]


# =============================================================================
# STEP 16 VALIDATION
# =============================================================================

def locate_step16_table():
    """Locate the Step 16 pocket prioritization table."""

    exact_candidates = [
        RESULTS_ROOT
        / "STEP16_WT_vs_D166V_POCKET_DISCOVERY"
        / "tables"
        / "Table_16_04_D166V_Pocket_Prioritization.csv"
    ]

    for path in exact_candidates:
        if path.exists():
            return path

    path = recursive_find(
        "Table_16_04_D166V_Pocket_Prioritization.csv",
        RESULTS_ROOT
    )

    return path


def load_step16_pocket():
    """Load and validate Step 16 Pocket 1."""

    step16_path = locate_step16_table()

    if step16_path is None:
        raise FileNotFoundError(
            "Step 16 pocket prioritization table was not found."
        )

    df = pd.read_csv(step16_path)

    required_columns = [
        "Priority_Rank",
        "D166V_Pocket_ID",
        "WT_Pocket_ID",
        "Match_Status",
        "Remodeling_Category",
        "Druggability",
        "Pocket_Score",
        "Volume_A3",
        "D166_Distance_A",
        "D166_Distance_Category",
        "D166_Directly_Lining",
        "Lining_Jaccard",
        "Mutation_Remodeling_Score",
        "Druggability_Normalized",
        "Pocket_Score_Normalized",
        "Volume_Suitability",
        "D166_Proximity_Score",
        "Prioritization_Score",
        "Prioritization_Percent",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise RuntimeError(
            "Step 16 table is missing required columns:\n"
            + "\n".join(missing)
        )

    pocket1 = df[
        pd.to_numeric(
            df["Priority_Rank"],
            errors="coerce"
        ) == 1
    ].copy()

    if pocket1.empty:
        raise RuntimeError(
            "Step 16 table does not contain Priority_Rank = 1."
        )

    pocket1 = pocket1.iloc[0]

    return step16_path, pocket1


# =============================================================================
# STEP 17 VALIDATION
# =============================================================================

def locate_step17_report():
    """Locate Step 17 interpretation report."""

    exact_candidates = [
        RESULTS_ROOT
        / "STEP17_FINAL_STRUCTURAL_REMODELING"
        / "STEP17_FINAL_INTERPRETATION.txt"
    ]

    for path in exact_candidates:
        if path.exists():
            return path

    path = recursive_find(
        "STEP17_FINAL_INTERPRETATION.txt",
        RESULTS_ROOT
    )

    return path


def validate_step17():
    """Validate that Step 17 interpretation exists."""

    step17_path = locate_step17_report()

    if step17_path is None:
        raise FileNotFoundError(
            "Step 17 interpretation report was not found."
        )

    text = step17_path.read_text(
        encoding="utf-8",
        errors="replace"
    )

    if "STEP 17" not in text.upper():
        print(
            "WARNING: Step 17 report exists but could not be "
            "strongly validated by its text."
        )

    return step17_path


# =============================================================================
# PUBCHEM TEXT SEARCH
# =============================================================================

def pubchem_text_search(search_term, category):
    """
    Search PubChem Compound database through NCBI E-utilities.

    This is a genuine text search, unlike:
        /compound/name/{term}/cids/JSON

    which performs chemical-name lookup rather than semantic text search.
    """

    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"

    params = {
        "db": "pccompound",
        "term": search_term,
        "retmode": "json",
        "retmax": MAX_CIDS_PER_TERM,
    }

    last_error = None

    for attempt in range(MAX_RETRIES):
        try:
            response = SESSION.get(
                url,
                params=params,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            result = data.get("esearchresult", {})

            count = int(
                result.get("count", 0)
            )

            ids = result.get("idlist", [])

            cids = []

            for cid in ids:
                try:
                    cids.append(int(cid))
                except Exception:
                    continue

            return {
                "Search_Term": search_term,
                "Search_Category": category,
                "PubChem_Total_Matches": count,
                "Retrieved_CIDs": cids,
                "Search_Status": "Success",
                "Search_Error": "",
            }

        except Exception as exc:
            last_error = exc

            if attempt < MAX_RETRIES - 1:
                wait_time = 2 ** attempt

                print(
                    f"    Search error: {exc}; "
                    f"retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

    return {
        "Search_Term": search_term,
        "Search_Category": category,
        "PubChem_Total_Matches": 0,
        "Retrieved_CIDs": [],
        "Search_Status": "Failed",
        "Search_Error": str(last_error),
    }


# =============================================================================
# PUBCHEM PROPERTY RETRIEVAL
# =============================================================================

def retrieve_single_compound(cid):
    """
    Retrieve PubChem properties for one CID.

    IMPORTANT:
        PubChem PUG REST uses 'Charge'.
        It does NOT use 'FormalCharge'.

    The returned record is standardized internally so that the project's
    output column remains 'FormalCharge'.
    """

    properties = (
        "MolecularFormula,"
        "MolecularWeight,"
        "CanonicalSMILES,"
        "IsomericSMILES,"
        "InChI,"
        "InChIKey,"
        "IUPACName,"
        "XLogP,"
        "TPSA,"
        "HBondDonorCount,"
        "HBondAcceptorCount,"
        "RotatableBondCount,"
        "HeavyAtomCount,"
        "Charge,"
        "Complexity,"
        "ExactMass,"
        "MonoisotopicMass"
    )

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/"
        f"compound/cid/{cid}/property/{properties}/JSON"
    )

    last_error = None

    for attempt in range(MAX_RETRIES):

        try:

            response = SESSION.get(
                url,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            if "PropertyTable" not in data:
                raise RuntimeError(
                    "PubChem response does not contain PropertyTable."
                )

            records = data["PropertyTable"].get(
                "Properties",
                []
            )

            if not records:
                raise RuntimeError(
                    "PubChem returned an empty property record."
                )

            record = records[0]

            # Standardize PubChem Charge into our project field.
            record["FormalCharge"] = record.get(
                "Charge",
                ""
            )

            return record

        except Exception as exc:

            last_error = exc

            if attempt < MAX_RETRIES - 1:

                wait_time = 2 ** attempt

                print(
                    f"    Request error: {exc}; "
                    f"retrying in {wait_time}s..."
                )

                time.sleep(wait_time)

    raise RuntimeError(str(last_error))


# =============================================================================
# BUILD SEARCH PROVENANCE
# =============================================================================

def build_search_provenance(search_results):
    """
    Build one row per CID/search-term/category combination.
    """

    rows = []

    for result in search_results:

        term = result["Search_Term"]
        category = result["Search_Category"]

        cids = result["Retrieved_CIDs"]

        if not cids:

            rows.append(
                {
                    "Search_Term": term,
                    "Search_Category": category,
                    "PubChem_Total_Matches": result[
                        "PubChem_Total_Matches"
                    ],
                    "CID": "",
                    "Search_Status": result[
                        "Search_Status"
                    ],
                    "Search_Error": result[
                        "Search_Error"
                    ],
                }
            )

            continue

        for cid in cids:

            rows.append(
                {
                    "Search_Term": term,
                    "Search_Category": category,
                    "PubChem_Total_Matches": result[
                        "PubChem_Total_Matches"
                    ],
                    "CID": cid,
                    "Search_Status": result[
                        "Search_Status"
                    ],
                    "Search_Error": result[
                        "Search_Error"
                    ],
                }
            )

    return pd.DataFrame(rows)


# =============================================================================
# PROPERTY NORMALIZATION
# =============================================================================

def normalize_property_record(record, cid):
    """
    Convert a PubChem property record into the project's standardized schema.
    """

    return {
        "CID": cid,

        "MolecularFormula": clean_text(
            record.get("MolecularFormula")
        ),

        "MolecularWeight": safe_float(
            record.get("MolecularWeight")
        ),

        "CanonicalSMILES": clean_text(
            record.get("ConnectivitySMILES")
            or record.get("CanonicalSMILES")
        ),

        "IsomericSMILES": clean_text(
            record.get("SMILES")
            or record.get("IsomericSMILES")
        ),

        "InChI": clean_text(
            record.get("InChI")
        ),

        "InChIKey": clean_text(
            record.get("InChIKey")
        ),

        "IUPACName": clean_text(
            record.get("IUPACName")
        ),

        "XLogP": safe_float(
            record.get("XLogP")
        ),

        "TPSA": safe_float(
            record.get("TPSA")
        ),

        "HBondDonorCount": safe_float(
            record.get("HBondDonorCount")
        ),

        "HBondAcceptorCount": safe_float(
            record.get("HBondAcceptorCount")
        ),

        "RotatableBondCount": safe_float(
            record.get("RotatableBondCount")
        ),

        "HeavyAtomCount": safe_float(
            record.get("HeavyAtomCount")
        ),

        "FormalCharge": safe_float(
            record.get("FormalCharge")
        ),

        "Complexity": safe_float(
            record.get("Complexity")
        ),

        "ExactMass": safe_float(
            record.get("ExactMass")
        ),

        "MonoisotopicMass": safe_float(
            record.get("MonoisotopicMass")
        ),
    }


# =============================================================================
# PROVENANCE ANNOTATION
# =============================================================================

def annotate_provenance(compound_df, provenance_df):
    """
    Add search-category provenance to the compound library.

    PNPLA3_Specific_Search_Hit means the compound was returned by the
    PNPLA3-specific PubChem search category.

    It DOES NOT mean experimentally validated PNPLA3 activity.
    """

    if compound_df.empty:
        return compound_df

    provenance_df = provenance_df.copy()

    provenance_df["CID"] = pd.to_numeric(
        provenance_df["CID"],
        errors="coerce"
    )

    valid_provenance = provenance_df.dropna(
        subset=["CID"]
    ).copy()

    if valid_provenance.empty:
        compound_df["Search_Terms"] = ""
        compound_df["Search_Categories"] = ""
        compound_df["PNPLA3_Specific_Search_Hit"] = False
        return compound_df

    term_map = (
        valid_provenance
        .groupby("CID")["Search_Term"]
        .apply(
            lambda x: "; ".join(
                sorted(set(str(v) for v in x))
            )
        )
        .to_dict()
    )

    category_map = (
        valid_provenance
        .groupby("CID")["Search_Category"]
        .apply(
            lambda x: "; ".join(
                sorted(set(str(v) for v in x))
            )
        )
        .to_dict()
    )

    pnpla3_cids = set(
        valid_provenance.loc[
            valid_provenance["Search_Category"]
            == "PNPLA3_specific",
            "CID"
        ]
        .astype(int)
        .tolist()
    )

    compound_df["CID_numeric"] = pd.to_numeric(
        compound_df["CID"],
        errors="coerce"
    )

    compound_df["Search_Terms"] = (
        compound_df["CID_numeric"]
        .map(term_map)
        .fillna("")
    )

    compound_df["Search_Categories"] = (
        compound_df["CID_numeric"]
        .map(category_map)
        .fillna("")
    )

    compound_df[
        "PNPLA3_Specific_Search_Hit"
    ] = compound_df["CID_numeric"].apply(
        lambda x: (
            int(x) in pnpla3_cids
            if pd.notna(x)
            else False
        )
    )

    compound_df.drop(
        columns=["CID_numeric"],
        inplace=True
    )

    return compound_df


# =============================================================================
# STRUCTURE QC
# =============================================================================

def perform_structure_qc(df):
    """
    Perform basic structure completeness QC.

    A compound passes structural QC when it contains:
        - Molecular formula
        - Canonical SMILES
        - InChIKey

    No claim about activity or drug-likeness is made here.
    """

    df = df.copy()

    df["Structure_QC_MolecularFormula"] = (
        df["MolecularFormula"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    df["Structure_QC_CanonicalSMILES"] = (
        df["CanonicalSMILES"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    df["Structure_QC_InChIKey"] = (
        df["InChIKey"]
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )

    df["Structure_QC_Pass"] = (
        df["Structure_QC_MolecularFormula"]
        & df["Structure_QC_CanonicalSMILES"]
        & df["Structure_QC_InChIKey"]
    )

    return df


# =============================================================================
# DUPLICATION
# =============================================================================

def deduplicate_by_inchikey(df):
    """
    Deduplicate structures using InChIKey.

    If InChIKey is missing, CID is retained as the fallback identifier.
    """

    if df.empty:
        return df

    df = df.copy()

    df["Deduplication_Key"] = df["InChIKey"].fillna("").astype(str)

    missing_key = (
        df["Deduplication_Key"]
        .str.strip()
        .eq("")
    )

    df.loc[
        missing_key,
        "Deduplication_Key"
    ] = (
        "CID_"
        + df.loc[
            missing_key,
            "CID"
        ].astype(str)
    )

    # Keep first occurrence because provenance is summarized separately.
    deduplicated = (
        df
        .drop_duplicates(
            subset=["Deduplication_Key"],
            keep="first"
        )
        .copy()
    )

    deduplicated.drop(
        columns=["Deduplication_Key"],
        inplace=True
    )

    return deduplicated


# =============================================================================
# COMPOUND SUMMARY
# =============================================================================

def build_source_summary(provenance_df):
    """Build search-source summary table."""

    if provenance_df.empty:

        return pd.DataFrame(
            columns=[
                "Search_Category",
                "Search_Term",
                "Total_Matches",
                "Retrieved_CIDs",
            ]
        )

    summary_rows = []

    grouped = provenance_df.groupby(
        [
            "Search_Category",
            "Search_Term",
        ],
        dropna=False
    )

    for (
        category,
        term
    ), group in grouped:

        cid_values = []

        for value in group["CID"]:

            if pd.isna(value):
                continue

            try:
                cid_values.append(int(value))
            except Exception:
                continue

        summary_rows.append(
            {
                "Search_Category": category,
                "Search_Term": term,
                "Total_Matches": (
                    group["PubChem_Total_Matches"]
                    .dropna()
                    .iloc[0]
                    if not group[
                        "PubChem_Total_Matches"
                    ].dropna().empty
                    else 0
                ),
                "Retrieved_CIDs": len(
                    set(cid_values)
                ),
            }
        )

    return pd.DataFrame(summary_rows)


# =============================================================================
# WRITE INTERPRETATION REPORT
# =============================================================================

def write_interpretation_report(
    step16_path,
    step17_path,
    pocket,
    raw_hit_count,
    unique_cids,
    successful_properties,
    failed_properties,
    final_count,
    qc_pass_count,
    screening_ready_count,
):
    """Write Step 18 scientific interpretation."""

    report_path = (
        REPORTS_DIR
        / "STEP18_INTERPRETATION.txt"
    )

    pocket_id = pocket["D166V_Pocket_ID"]

    report = []

    report.append(
        "STEP 18 — THERAPEUTIC COMPOUND LIBRARY"
    )
    report.append(
        "=" * 78
    )
    report.append("")

    report.append(
        "PROJECT"
    )
    report.append(
        "MASLD–PNPLA3 computational variant prioritization"
    )
    report.append("")

    report.append(
        "STRUCTURAL TARGET CONTEXT"
    )
    report.append(
        f"Step 16 prioritized D166V pocket : {pocket_id}"
    )
    report.append(
        f"WT pocket ID                     : "
        f"{pocket['WT_Pocket_ID']}"
    )
    report.append(
        f"Pocket match status              : "
        f"{pocket['Match_Status']}"
    )
    report.append(
        f"Druggability                     : "
        f"{pocket['Druggability']}"
    )
    report.append(
        f"Pocket volume (A3)               : "
        f"{pocket['Volume_A3']}"
    )
    report.append(
        f"D166 distance (A)                : "
        f"{pocket['D166_Distance_A']}"
    )
    report.append(
        f"D166 directly lining             : "
        f"{pocket['D166_Directly_Lining']}"
    )
    report.append(
        f"Prioritization score             : "
        f"{pocket['Prioritization_Score']}"
    )
    report.append("")

    report.append(
        "STEP 17 STRUCTURAL CONTEXT"
    )
    report.append(
        "Step 17 reported no appreciable "
        "backbone or pocket-level geometric "
        "remodeling between the available WT "
        "and D166V structural models."
    )
    report.append(
        "The modeled difference is therefore "
        "primarily represented by the Asp166-to-Val166 "
        "chemical and side-chain identity change."
    )
    report.append("")

    report.append(
        "COMPOUND RETRIEVAL"
    )
    report.append(
        f"Raw search hits                    : "
        f"{raw_hit_count}"
    )
    report.append(
        f"Unique PubChem CIDs                : "
        f"{unique_cids}"
    )
    report.append(
        f"Successful property retrievals     : "
        f"{successful_properties}"
    )
    report.append(
        f"Failed property retrievals         : "
        f"{failed_properties}"
    )
    report.append(
        f"Final unique compounds             : "
        f"{final_count}"
    )
    report.append("")

    report.append(
        "STRUCTURE QC"
    )
    report.append(
        f"Structure-QC passing compounds     : "
        f"{qc_pass_count}"
    )
    report.append(
        f"Screening-ready compounds          : "
        f"{screening_ready_count}"
    )
    report.append("")

    report.append(
        "SCIENTIFIC INTERPRETATION"
    )
    report.append(
        "This step establishes a chemically defined "
        "compound library for downstream computational "
        "screening against the prioritized PNPLA3 "
        "D166V-associated pocket."
    )
    report.append("")
    report.append(
        "The compounds were retrieved using PubChem "
        "text-search categories spanning PNPLA3-specific "
        "context, phospholipase/enzyme context, lipid "
        "metabolism, hepatic metabolism, and MASLD context."
    )
    report.append("")
    report.append(
        "Membership in a search category is a retrieval "
        "criterion only. It must not be interpreted as "
        "experimental evidence that a compound binds or "
        "inhibits PNPLA3."
    )
    report.append("")
    report.append(
        "No docking score, binding affinity, inhibition "
        "claim, therapeutic efficacy claim, or mutation-"
        "specific activity claim is generated in Step 18."
    )
    report.append("")
    report.append(
        "The library is intended to undergo chemical "
        "property and drug-likeness filtering in Step 19, "
        "followed by structure-based screening and "
        "differential WT/D166V evaluation in subsequent "
        "steps."
    )
    report.append("")

    report.append(
        "INPUTS"
    )
    report.append(
        f"Step 16 input: {step16_path}"
    )
    report.append(
        f"Step 17 input: {step17_path}"
    )
    report.append("")

    report.append(
        "OUTPUT DIRECTORY"
    )
    report.append(
        str(STEP18_ROOT)
    )
    report.append("")

    report_path.write_text(
        "\n".join(report),
        encoding="utf-8"
    )

    return report_path


# =============================================================================
# QC JSON
# =============================================================================

def write_qc_json(
    step16_path,
    step17_path,
    pocket,
    search_results,
    raw_hit_count,
    unique_cids,
    successful_cids,
    failed_cids,
    final_df,
):
    """Write machine-readable QC information."""

    qc_pass_count = int(
        final_df["Structure_QC_Pass"].sum()
    ) if not final_df.empty else 0

    qc = {
        "Step": 18,
        "Step_Name": "Therapeutic Compound Library",

        "Status": (
            "PASS"
            if (
                len(successful_cids) > 0
                and qc_pass_count > 0
            )
            else "FAIL"
        ),

        "Inputs": {
            "Step16_Pocket_Table": str(step16_path),
            "Step17_Interpretation": str(step17_path),
        },

        "Pocket_Context": {
            "D166V_Pocket_ID": clean_text(
                pocket["D166V_Pocket_ID"]
            ),
            "WT_Pocket_ID": clean_text(
                pocket["WT_Pocket_ID"]
            ),
            "Match_Status": clean_text(
                pocket["Match_Status"]
            ),
            "Druggability": safe_float(
                pocket["Druggability"]
            ),
            "Volume_A3": safe_float(
                pocket["Volume_A3"]
            ),
            "D166_Distance_A": safe_float(
                pocket["D166_Distance_A"]
            ),
            "D166_Directly_Lining": bool(
                pocket["D166_Directly_Lining"]
            ),
            "Prioritization_Score": safe_float(
                pocket["Prioritization_Score"]
            ),
        },

        "Search": {
            "Number_of_Search_Terms": len(
                search_results
            ),
            "Raw_Search_Hits": raw_hit_count,
            "Unique_PubChem_CIDs": unique_cids,
        },

        "Property_Retrieval": {
            "Successful": len(successful_cids),
            "Failed": len(failed_cids),
            "Successful_CIDs": successful_cids,
            "Failed_CIDs": failed_cids,
        },

        "Structure_QC": {
            "Final_Unique_Compounds": int(
                len(final_df)
            ),
            "Structure_QC_Pass": qc_pass_count,
            "Structure_QC_Fail": int(
                len(final_df) - qc_pass_count
            ),
        },

        "Scientific_Limitations": [
            "PubChem search-category membership is not equivalent to target activity.",
            "Step 18 does not establish PNPLA3 binding.",
            "Step 18 does not establish PNPLA3 inhibition.",
            "Step 18 does not establish therapeutic efficacy.",
            "Step 18 does not establish D166V-specific activity.",
            "Docking and downstream structural analyses are required.",
        ],
    }

    qc_path = QC_DIR / "STEP18_QC.json"

    qc_path.write_text(
        json.dumps(
            qc,
            indent=2,
            default=str
        ),
        encoding="utf-8"
    )

    return qc_path


# =============================================================================
# MANIFEST
# =============================================================================

def write_manifest(
    step16_path,
    step17_path,
    search_results,
    final_df,
):
    """Write reproducibility manifest."""

    manifest_path = (
        QC_DIR
        / "STEP18_MANIFEST.txt"
    )

    lines = []

    lines.append(
        "STEP 18 — REPRODUCIBILITY MANIFEST"
    )
    lines.append(
        "=" * 78
    )
    lines.append("")

    lines.append(
        f"Project root: {PROJECT_ROOT}"
    )

    lines.append(
        f"Step 16 input: {step16_path}"
    )

    lines.append(
        f"Step 17 input: {step17_path}"
    )

    lines.append("")

    lines.append(
        "PubChem search terms:"
    )

    for term, category in SEARCH_TERMS:

        lines.append(
            f"  [{category}] {term}"
        )

    lines.append("")

    lines.append(
        f"Maximum CIDs per term: {MAX_CIDS_PER_TERM}"
    )

    lines.append(
        "PubChem property retrieval: one CID per request"
    )

    lines.append(
        "Charge field mapped to project field FormalCharge"
    )

    lines.append(
        "Deduplication key: InChIKey"
    )

    lines.append(
        "Structure QC requirements:"
    )

    lines.append(
        "  - MolecularFormula"
    )

    lines.append(
        "  - CanonicalSMILES"
    )

    lines.append(
        "  - InChIKey"
    )

    lines.append("")

    lines.append(
        f"Final library size: {len(final_df)}"
    )

    if not final_df.empty:

        qc_count = int(
            final_df[
                "Structure_QC_Pass"
            ].sum()
        )

        lines.append(
            f"Structure-QC passing: {qc_count}"
        )

    lines.append("")

    lines.append(
        "Scientific note:"
    )

    lines.append(
        "Search retrieval is not evidence of biochemical "
        "activity or therapeutic efficacy."
    )

    lines.append("")

    manifest_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    return manifest_path


# =============================================================================
# MAIN
# =============================================================================

def main():

    print("=" * 78)
    print(
        "STEP 18 — THERAPEUTIC COMPOUND LIBRARY"
    )
    print("=" * 78)

    print(
        f"Project root : {PROJECT_ROOT}"
    )

    if not PROJECT_ROOT.exists():

        raise FileNotFoundError(
            f"Project root does not exist:\n{PROJECT_ROOT}"
        )

    # -------------------------------------------------------------------------
    # Validate previous steps
    # -------------------------------------------------------------------------

    print()
    print(
        "Validating previous structural steps..."
    )
    print()

    step16_path, pocket = load_step16_pocket()

    print(
        "Step 16 table found:"
    )
    print(
        f"  {step16_path}"
    )

    step17_path = validate_step17()

    print()
    print(
        "Step 17 report found:"
    )
    print(
        f"  {step17_path}"
    )

    print()
    print(
        "Previous steps validated successfully."
    )

    # -------------------------------------------------------------------------
    # Print Pocket 1
    # -------------------------------------------------------------------------

    print()
    print(
        "Step 16 Pocket 1:"
    )

    print(
        f"  D166V pocket ID      : "
        f"{pocket['D166V_Pocket_ID']}"
    )

    print(
        f"  WT pocket ID         : "
        f"{pocket['WT_Pocket_ID']}"
    )

    print(
        f"  Match status         : "
        f"{pocket['Match_Status']}"
    )

    print(
        f"  Druggability         : "
        f"{pocket['Druggability']}"
    )

    print(
        f"  Volume               : "
        f"{pocket['Volume_A3']} Å³"
    )

    print(
        f"  D166 distance        : "
        f"{pocket['D166_Distance_A']} Å"
    )

    print(
        f"  D166 directly lining : "
        f"{pocket['D166_Directly_Lining']}"
    )

    print(
        f"  Prioritization score : "
        f"{pocket['Prioritization_Score']}"
    )

    # -------------------------------------------------------------------------
    # PubChem search
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "PUBCHEM TEXT SEARCH"
    )
    print("=" * 78)

    search_results = []

    for search_term, category in SEARCH_TERMS:

        print()
        print(
            f"Searching: {search_term}"
        )

        print(
            f"Category : {category}"
        )

        result = pubchem_text_search(
            search_term,
            category
        )

        search_results.append(result)

        print(
            f"  PubChem total matches : "
            f"{result['PubChem_Total_Matches']}"
        )

        print(
            f"  Retrieved CIDs        : "
            f"{len(result['Retrieved_CIDs'])}"
        )

        if result["Search_Status"] == "Failed":

            print(
                f"  WARNING: {result['Search_Error']}"
            )

    # -------------------------------------------------------------------------
    # Search provenance
    # -------------------------------------------------------------------------

    provenance_df = build_search_provenance(
        search_results
    )

    provenance_path = (
        TABLES_DIR
        / "STEP18_SEARCH_PROVENANCE.csv"
    )

    provenance_df.to_csv(
        provenance_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # CID collection
    # -------------------------------------------------------------------------

    all_cids = []

    for result in search_results:

        all_cids.extend(
            result["Retrieved_CIDs"]
        )

    raw_hit_count = len(all_cids)

    unique_cids = sorted(
        set(all_cids)
    )

    print()
    print("=" * 78)
    print(
        "COMPOUND DEDUPLICATION"
    )
    print("=" * 78)

    print(
        f"Raw search hits     : {raw_hit_count}"
    )

    print(
        f"Unique PubChem CIDs : {len(unique_cids)}"
    )

    if not unique_cids:

        raise RuntimeError(
            "No PubChem CIDs were retrieved. "
            "Step 18 is NOT complete."
        )

    # -------------------------------------------------------------------------
    # Retrieve properties
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "RETRIEVING PUBCHEM COMPOUND PROPERTIES"
    )
    print("=" * 78)

    property_records = []

    successful_cids = []
    failed_cids = []

    failed_property_errors = {}

    total = len(unique_cids)

    for index, cid in enumerate(
        unique_cids,
        start=1
    ):

        print(
            f"  [{index}/{total}] CID {cid}"
        )

        try:

            record = retrieve_single_compound(
                cid
            )

            normalized = normalize_property_record(
                record,
                cid
            )

            property_records.append(
                normalized
            )

            successful_cids.append(
                cid
            )

            print(
                "    OK"
            )

        except Exception as exc:

            failed_cids.append(
                cid
            )

            failed_property_errors[
                str(cid)
            ] = str(exc)

            print(
                f"    FAILED: {exc}"
            )

        # Small delay to be polite to PubChem.
        time.sleep(0.15)

    print()
    print(
        f"Successful property retrievals : "
        f"{len(successful_cids)}"
    )

    print(
        f"Failed property retrievals     : "
        f"{len(failed_cids)}"
    )

    # -------------------------------------------------------------------------
    # Stop only if absolutely nothing was retrieved
    # -------------------------------------------------------------------------

    if not property_records:

        error_path = (
            QC_DIR
            / "STEP18_PROPERTY_RETRIEVAL_ERRORS.json"
        )

        error_path.write_text(
            json.dumps(
                failed_property_errors,
                indent=2
            ),
            encoding="utf-8"
        )

        raise RuntimeError(
            "\n"
            "PubChem CIDs were found, but no compound "
            "properties could be retrieved.\n"
            "Step 18 is NOT complete.\n"
            f"Error log: {error_path}"
        )

    # -------------------------------------------------------------------------
    # Build compound dataframe
    # -------------------------------------------------------------------------

    compound_df = pd.DataFrame(
        property_records
    )

    # -------------------------------------------------------------------------
    # Annotate provenance
    # -------------------------------------------------------------------------

    compound_df = annotate_provenance(
        compound_df,
        provenance_df
    )

    # -------------------------------------------------------------------------
    # Structure QC
    # -------------------------------------------------------------------------

    compound_df = perform_structure_qc(
        compound_df
    )

    # -------------------------------------------------------------------------
    # Deduplicate by InChIKey
    # -------------------------------------------------------------------------

    before_dedup = len(
        compound_df
    )

    compound_df = deduplicate_by_inchikey(
        compound_df
    )

    after_dedup = len(
        compound_df
    )

    print()
    print("=" * 78)
    print(
        "STRUCTURE QC AND DEDUPLICATION"
    )
    print("=" * 78)

    print(
        f"Retrieved compound records : "
        f"{before_dedup}"
    )

    print(
        f"Unique structures           : "
        f"{after_dedup}"
    )

    qc_pass_count = int(
        compound_df[
            "Structure_QC_Pass"
        ].sum()
    )

    qc_fail_count = (
        len(compound_df)
        - qc_pass_count
    )

    print(
        f"Structure-QC PASS           : "
        f"{qc_pass_count}"
    )

    print(
        f"Structure-QC FAIL           : "
        f"{qc_fail_count}"
    )

    # -------------------------------------------------------------------------
    # Screening-ready library
    # -------------------------------------------------------------------------

    screening_ready_df = (
        compound_df[
            compound_df["Structure_QC_Pass"]
        ]
        .copy()
    )

    # Sort PNPLA3-specific hits first, then CID.
    if not screening_ready_df.empty:

        screening_ready_df = (
            screening_ready_df
            .sort_values(
                by=[
                    "PNPLA3_Specific_Search_Hit",
                    "CID",
                ],
                ascending=[
                    False,
                    True,
                ],
            )
            .reset_index(drop=True)
        )

        screening_ready_df.insert(
            0,
            "Screening_Rank",
            range(
                1,
                len(screening_ready_df) + 1
            )
        )

    # -------------------------------------------------------------------------
    # Master library
    # -------------------------------------------------------------------------

    master_columns = [
        "CID",
        "IUPACName",
        "MolecularFormula",
        "MolecularWeight",
        "CanonicalSMILES",
        "IsomericSMILES",
        "InChI",
        "InChIKey",
        "XLogP",
        "TPSA",
        "HBondDonorCount",
        "HBondAcceptorCount",
        "RotatableBondCount",
        "HeavyAtomCount",
        "FormalCharge",
        "Complexity",
        "ExactMass",
        "MonoisotopicMass",
        "Search_Terms",
        "Search_Categories",
        "PNPLA3_Specific_Search_Hit",
        "Structure_QC_MolecularFormula",
        "Structure_QC_CanonicalSMILES",
        "Structure_QC_InChIKey",
        "Structure_QC_Pass",
    ]

    # Ensure all columns exist.
    for col in master_columns:

        if col not in compound_df.columns:

            compound_df[col] = ""

    compound_df = compound_df[
        master_columns
    ].copy()

    screening_columns = [
        col
        for col in master_columns
    ]

    if "Screening_Rank" in screening_ready_df.columns:

        screening_columns = [
            "Screening_Rank"
        ] + screening_columns

    for col in screening_columns:

        if col not in screening_ready_df.columns:

            screening_ready_df[col] = ""

    screening_ready_df = (
        screening_ready_df[
            screening_columns
        ]
        .copy()
    )

    # -------------------------------------------------------------------------
    # Save master library
    # -------------------------------------------------------------------------

    master_path = (
        TABLES_DIR
        / "STEP18_MASTER_COMPOUND_LIBRARY.csv"
    )

    compound_df.to_csv(
        master_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Save screening-ready library
    # -------------------------------------------------------------------------

    screening_path = (
        TABLES_DIR
        / "STEP18_SCREENING_READY_LIBRARY.csv"
    )

    screening_ready_df.to_csv(
        screening_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Source summary
    # -------------------------------------------------------------------------

    source_summary_df = build_source_summary(
        provenance_df
    )

    source_summary_path = (
        TABLES_DIR
        / "STEP18_SOURCE_SUMMARY.csv"
    )

    source_summary_df.to_csv(
        source_summary_path,
        index=False
    )

    # -------------------------------------------------------------------------
    # Property retrieval error log
    # -------------------------------------------------------------------------

    if failed_property_errors:

        error_path = (
            QC_DIR
            / "STEP18_PROPERTY_RETRIEVAL_ERRORS.json"
        )

        error_path.write_text(
            json.dumps(
                failed_property_errors,
                indent=2
            ),
            encoding="utf-8"
        )

    # -------------------------------------------------------------------------
    # QC
    # -------------------------------------------------------------------------

    qc_path = write_qc_json(
        step16_path=step16_path,
        step17_path=step17_path,
        pocket=pocket,
        search_results=search_results,
        raw_hit_count=raw_hit_count,
        unique_cids=len(unique_cids),
        successful_cids=successful_cids,
        failed_cids=failed_cids,
        final_df=compound_df,
    )

    # -------------------------------------------------------------------------
    # Manifest
    # -------------------------------------------------------------------------

    manifest_path = write_manifest(
        step16_path=step16_path,
        step17_path=step17_path,
        search_results=search_results,
        final_df=compound_df,
    )

    # -------------------------------------------------------------------------
    # Interpretation report
    # -------------------------------------------------------------------------

    report_path = write_interpretation_report(
        step16_path=step16_path,
        step17_path=step17_path,
        pocket=pocket,
        raw_hit_count=raw_hit_count,
        unique_cids=len(unique_cids),
        successful_properties=len(
            successful_cids
        ),
        failed_properties=len(
            failed_cids
        ),
        final_count=len(compound_df),
        qc_pass_count=qc_pass_count,
        screening_ready_count=len(
            screening_ready_df
        ),
    )

    # -------------------------------------------------------------------------
    # Final console summary
    # -------------------------------------------------------------------------

    print()
    print("=" * 78)
    print(
        "STEP 18 COMPLETED"
    )
    print("=" * 78)

    print()
    print(
        f"Raw search hits              : "
        f"{raw_hit_count}"
    )

    print(
        f"Unique PubChem CIDs         : "
        f"{len(unique_cids)}"
    )

    print(
        f"Properties retrieved        : "
        f"{len(successful_cids)}"
    )

    print(
        f"Property retrieval failures : "
        f"{len(failed_cids)}"
    )

    print(
        f"Unique compound structures  : "
        f"{len(compound_df)}"
    )

    print(
        f"Structure-QC PASS            : "
        f"{qc_pass_count}"
    )

    print(
        f"Screening-ready compounds   : "
        f"{len(screening_ready_df)}"
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  Master library:"
    )

    print(
        f"    {master_path}"
    )

    print(
        f"  Screening-ready library:"
    )

    print(
        f"    {screening_path}"
    )

    print(
        f"  Search provenance:"
    )

    print(
        f"    {provenance_path}"
    )

    print(
        f"  Source summary:"
    )

    print(
        f"    {source_summary_path}"
    )

    print(
        f"  QC:"
    )

    print(
        f"    {qc_path}"
    )

    print(
        f"  Manifest:"
    )

    print(
        f"    {manifest_path}"
    )

    print(
        f"  Interpretation report:"
    )

    print(
        f"    {report_path}"
    )

    print()
    print(
        "Scientific note:"
    )

    print(
        "The Step 18 library is a compound-retrieval "
        "resource only. Search hits do not establish "
        "PNPLA3 binding or inhibition."
    )

    print()
    print(
        "Proceed to Step 19 only after reviewing "
        "the actual Step 18 compound count and QC."
    )

    print("=" * 78)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "Step 18 interrupted by user."
        )
        sys.exit(1)

    except Exception as exc:

        print()
        print("=" * 78)
        print(
            "STEP 18 FAILED"
        )
        print("=" * 78)
        print()
        print(
            str(exc)
        )
        print()
        print(
            "Step 18 is NOT complete."
        )
        print("=" * 78)

        sys.exit(1)