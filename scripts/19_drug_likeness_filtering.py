# =============================================================================
# STEP 19 — DRUG-LIKENESS & CHEMICAL FILTERING
# =============================================================================
# Purpose:
#   Filter the validated Step 18 compound library using standard
#   physicochemical/drug-likeness criteria.
#
# Input:
#   RESULTS\STEP18_THERAPEUTIC_COMPOUND_LIBRARY\tables\
#       STEP18_SCREENING_READY_LIBRARY.csv
#
# Output:
#   RESULTS\STEP19_DRUG_LIKENESS_FILTERING\
#
# Scientific principle:
#   This step filters compounds by chemical properties only.
#   It does NOT imply PNPLA3 binding, inhibition, efficacy, or selectivity.
# =============================================================================

from pathlib import Path
import pandas as pd
import json
from datetime import datetime


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP18_THERAPEUTIC_COMPOUND_LIBRARY"
    / "tables"
    / "STEP18_SCREENING_READY_LIBRARY.csv"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "RESULTS"
    / "STEP19_DRUG_LIKENESS_FILTERING"
)

TABLE_DIR = OUTPUT_ROOT / "tables"
QC_DIR = OUTPUT_ROOT / "QC"
REPORT_DIR = OUTPUT_ROOT / "reports"

for directory in [TABLE_DIR, QC_DIR, REPORT_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. STANDARD DRUG-LIKENESS THRESHOLDS
# =============================================================================
# These are conventional Rule-of-Five-style limits.
#
# PASS:
#   MW       <= 500 Da
#   XLogP    <= 5
#   HBD      <= 5
#   HBA      <= 10
#   TPSA     <= 140 Å²
#   RotB     <= 10
#
# A compound must satisfy all available criteria to receive PASS.
# Missing required properties are NOT treated as PASS.
# =============================================================================

THRESHOLDS = {
    "MolecularWeight": 500.0,
    "XLogP": 5.0,
    "HBondDonorCount": 5,
    "HBondAcceptorCount": 10,
    "TPSA": 140.0,
    "RotatableBondCount": 10,
}


# =============================================================================
# 3. LOAD STEP 18 LIBRARY
# =============================================================================

print("=" * 78)
print("STEP 19 — DRUG-LIKENESS & CHEMICAL FILTERING")
print("=" * 78)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Input file   : {INPUT_FILE}")

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: Step 18 screening-ready library was not found:\n"
        f"{INPUT_FILE}\n\n"
        f"Run Step 18 successfully before Step 19."
    )

df = pd.read_csv(INPUT_FILE)

print(f"\nInput compounds : {len(df)}")
print(f"Input columns   : {len(df.columns)}")


# =============================================================================
# 4. COLUMN VALIDATION
# =============================================================================

required_columns = [
    "MolecularWeight",
    "XLogP",
    "TPSA",
    "HBondDonorCount",
    "HBondAcceptorCount",
    "RotatableBondCount",
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    print("\nAvailable columns:")
    for c in df.columns:
        print(f"  - {c}")

    raise ValueError(
        "\nERROR: Required Step 18 property columns are missing:\n"
        + "\n".join(f"  {x}" for x in missing)
    )


# =============================================================================
# 5. NUMERIC CONVERSION
# =============================================================================

for column in required_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")


# =============================================================================
# 6. INDIVIDUAL PROPERTY FILTERS
# =============================================================================

df["MW_PASS"] = (
    df["MolecularWeight"].notna()
    & (df["MolecularWeight"] <= THRESHOLDS["MolecularWeight"])
)

df["XLogP_PASS"] = (
    df["XLogP"].notna()
    & (df["XLogP"] <= THRESHOLDS["XLogP"])
)

df["HBD_PASS"] = (
    df["HBondDonorCount"].notna()
    & (df["HBondDonorCount"] <= THRESHOLDS["HBondDonorCount"])
)

df["HBA_PASS"] = (
    df["HBondAcceptorCount"].notna()
    & (df["HBondAcceptorCount"] <= THRESHOLDS["HBondAcceptorCount"])
)

df["TPSA_PASS"] = (
    df["TPSA"].notna()
    & (df["TPSA"] <= THRESHOLDS["TPSA"])
)

df["RotB_PASS"] = (
    df["RotatableBondCount"].notna()
    & (df["RotatableBondCount"] <= THRESHOLDS["RotatableBondCount"])
)


# =============================================================================
# 7. PROPERTY COMPLETENESS
# =============================================================================

df["Property_Completeness"] = (
    df[required_columns]
    .notna()
    .sum(axis=1)
    / len(required_columns)
    * 100
)

df["All_Required_Properties_Available"] = (
    df[required_columns].notna().all(axis=1)
)


# =============================================================================
# 8. COUNT PASSED CRITERIA
# =============================================================================

pass_columns = [
    "MW_PASS",
    "XLogP_PASS",
    "HBD_PASS",
    "HBA_PASS",
    "TPSA_PASS",
    "RotB_PASS",
]

df["Criteria_Passed"] = df[pass_columns].sum(axis=1)

df["Criteria_Total"] = len(pass_columns)

df["Criteria_Pass_Percent"] = (
    df["Criteria_Passed"]
    / df["Criteria_Total"]
    * 100
)


# =============================================================================
# 9. OVERALL DRUG-LIKENESS CLASSIFICATION
# =============================================================================

def classify(row):

    if not row["All_Required_Properties_Available"]:
        return "INCOMPLETE"

    if row["Criteria_Passed"] == len(pass_columns):
        return "PASS"

    if row["Criteria_Passed"] >= 4:
        return "BORDERLINE"

    return "FAIL"


df["Drug_Likeness_Class"] = df.apply(classify, axis=1)


# =============================================================================
# 10. FILTER REASON
# =============================================================================

def failure_reason(row):

    if row["Drug_Likeness_Class"] == "PASS":
        return "All criteria passed"

    reasons = []

    if not row["MW_PASS"]:
        reasons.append("MW>500")

    if not row["XLogP_PASS"]:
        reasons.append("XLogP>5")

    if not row["HBD_PASS"]:
        reasons.append("HBD>5")

    if not row["HBA_PASS"]:
        reasons.append("HBA>10")

    if not row["TPSA_PASS"]:
        reasons.append("TPSA>140")

    if not row["RotB_PASS"]:
        reasons.append("RotB>10")

    if not reasons:
        reasons.append("Missing property")

    return "; ".join(reasons)


df["Filter_Reason"] = df.apply(failure_reason, axis=1)


# =============================================================================
# 11. SCREENING-READY DECISION
# =============================================================================

df["Retain_for_Step20"] = (
    df["Drug_Likeness_Class"].isin(["PASS", "BORDERLINE"])
)


# =============================================================================
# 12. PRIORITIZATION WITHIN RETAINED SET
# =============================================================================
# This is NOT a biological score.
# It simply ranks compounds by the number of physicochemical criteria passed.
# =============================================================================

df["Chemical_Filter_Score"] = (
    df["Criteria_Pass_Percent"]
)

df = df.sort_values(
    by=[
        "Retain_for_Step20",
        "Criteria_Passed",
        "Chemical_Filter_Score",
    ],
    ascending=[False, False, False]
).reset_index(drop=True)

df["Step19_Rank"] = range(1, len(df) + 1)


# =============================================================================
# 13. CREATE SCREENING LIBRARY
# =============================================================================

screening_df = df[
    df["Retain_for_Step20"]
].copy()

screening_df = screening_df.reset_index(drop=True)

screening_df["Step19_Screening_Rank"] = (
    range(1, len(screening_df) + 1)
)


# =============================================================================
# 14. SAVE MASTER TABLE
# =============================================================================

master_file = TABLE_DIR / "STEP19_MASTER_DRUG_LIKENESS_TABLE.csv"

df.to_csv(
    master_file,
    index=False
)


# =============================================================================
# 15. SAVE FILTERED LIBRARY
# =============================================================================

screening_file = TABLE_DIR / "STEP19_SCREENING_READY_LIBRARY.csv"

screening_df.to_csv(
    screening_file,
    index=False
)


# =============================================================================
# 16. PROPERTY SUMMARY
# =============================================================================

property_summary = []

for column, label, threshold in [
    ("MolecularWeight", "MW", THRESHOLDS["MolecularWeight"]),
    ("XLogP", "XLogP", THRESHOLDS["XLogP"]),
    ("HBondDonorCount", "HBD", THRESHOLDS["HBondDonorCount"]),
    ("HBondAcceptorCount", "HBA", THRESHOLDS["HBondAcceptorCount"]),
    ("TPSA", "TPSA", THRESHOLDS["TPSA"]),
    ("RotatableBondCount", "RotatableBonds", THRESHOLDS["RotatableBondCount"]),
]:

    pass_column = {
        "MolecularWeight": "MW_PASS",
        "XLogP": "XLogP_PASS",
        "HBondDonorCount": "HBD_PASS",
        "HBondAcceptorCount": "HBA_PASS",
        "TPSA": "TPSA_PASS",
        "RotatableBondCount": "RotB_PASS",
    }[column]

    property_summary.append({
        "Property": label,
        "Threshold": threshold,
        "Available": int(df[column].notna().sum()),
        "Passed": int(df[pass_column].sum()),
        "Failed": int((~df[pass_column]).sum()),
    })

property_summary_df = pd.DataFrame(property_summary)

property_summary_file = (
    TABLE_DIR / "STEP19_PROPERTY_FILTER_SUMMARY.csv"
)

property_summary_df.to_csv(
    property_summary_file,
    index=False
)


# =============================================================================
# 17. CLASS SUMMARY
# =============================================================================

class_counts = (
    df["Drug_Likeness_Class"]
    .value_counts()
    .rename_axis("Drug_Likeness_Class")
    .reset_index(name="Compound_Count")
)

class_summary_file = (
    TABLE_DIR / "STEP19_CLASS_SUMMARY.csv"
)

class_counts.to_csv(
    class_summary_file,
    index=False
)


# =============================================================================
# 18. QC
# =============================================================================

qc = {
    "step": "Step 19 — Drug-Likeness & Chemical Filtering",
    "timestamp": datetime.now().isoformat(),

    "input_file": str(INPUT_FILE),

    "input_compounds": int(len(df)),

    "property_completeness_100_percent": int(
        (df["Property_Completeness"] == 100).sum()
    ),

    "pass_count": int(
        (df["Drug_Likeness_Class"] == "PASS").sum()
    ),

    "borderline_count": int(
        (df["Drug_Likeness_Class"] == "BORDERLINE").sum()
    ),

    "fail_count": int(
        (df["Drug_Likeness_Class"] == "FAIL").sum()
    ),

    "incomplete_count": int(
        (df["Drug_Likeness_Class"] == "INCOMPLETE").sum()
    ),

    "retained_for_step20": int(
        df["Retain_for_Step20"].sum()
    ),

    "thresholds": THRESHOLDS,

    "scientific_note": (
        "Drug-likeness filtering is a physicochemical screening step. "
        "Passing compounds are not inferred to bind, inhibit, or "
        "modulate PNPLA3."
    ),
}

qc_file = QC_DIR / "STEP19_QC.json"

with open(qc_file, "w", encoding="utf-8") as f:
    json.dump(qc, f, indent=2)


# =============================================================================
# 19. INTERPRETATION REPORT
# =============================================================================

report_file = REPORT_DIR / "STEP19_INTERPRETATION.txt"

report = f"""
STEP 19 — DRUG-LIKENESS & CHEMICAL FILTERING
============================================================

Input:
{INPUT_FILE}

Input compounds:
{len(df)}

Drug-likeness classification:

PASS:
{int((df["Drug_Likeness_Class"] == "PASS").sum())}

BORDERLINE:
{int((df["Drug_Likeness_Class"] == "BORDERLINE").sum())}

FAIL:
{int((df["Drug_Likeness_Class"] == "FAIL").sum())}

INCOMPLETE:
{int((df["Drug_Likeness_Class"] == "INCOMPLETE").sum())}

Compounds retained for Step 20:
{int(df["Retain_for_Step20"].sum())}

Criteria:
- Molecular weight <= 500 Da
- XLogP <= 5
- H-bond donors <= 5
- H-bond acceptors <= 10
- TPSA <= 140 Å²
- Rotatable bonds <= 10

IMPORTANT SCIENTIFIC INTERPRETATION
-----------------------------------
This step evaluates physicochemical drug-likeness only.

A PASS classification means that the compound satisfies the
defined chemical-property filters.

A BORDERLINE classification means that the compound satisfies
most, but not all, of the defined filters and is retained to
avoid prematurely excluding potentially useful candidates.

A FAIL classification means that the compound does not satisfy
enough of the defined chemical criteria for the present
screening workflow.

No compound is considered a PNPLA3 binder, inhibitor,
therapeutic agent, or D166V-selective molecule on the basis
of Step 19 alone.

Step 20 should use the retained chemical library for
pharmacophore/shape-based screening.
"""

with open(report_file, "w", encoding="utf-8") as f:
    f.write(report.strip())


# =============================================================================
# 20. CONSOLE OUTPUT
# =============================================================================

print("\n" + "=" * 78)
print("STEP 19 COMPLETED")
print("=" * 78)

print(f"\nInput compounds       : {len(df)}")
print(
    f"PASS                  : "
    f"{int((df['Drug_Likeness_Class'] == 'PASS').sum())}"
)
print(
    f"BORDERLINE            : "
    f"{int((df['Drug_Likeness_Class'] == 'BORDERLINE').sum())}"
)
print(
    f"FAIL                  : "
    f"{int((df['Drug_Likeness_Class'] == 'FAIL').sum())}"
)
print(
    f"INCOMPLETE            : "
    f"{int((df['Drug_Likeness_Class'] == 'INCOMPLETE').sum())}"
)
print(
    f"Retained for Step 20  : "
    f"{int(df['Retain_for_Step20'].sum())}"
)

print("\nOutputs:")
print(f"  Master table:")
print(f"    {master_file}")
print(f"  Screening library:")
print(f"    {screening_file}")
print(f"  Property summary:")
print(f"    {property_summary_file}")
print(f"  Class summary:")
print(f"    {class_summary_file}")
print(f"  QC:")
print(f"    {qc_file}")
print(f"  Interpretation:")
print(f"    {report_file}")

print("\nScientific note:")
print(
    "Step 19 evaluates chemical drug-likeness only; "
    "it does not establish PNPLA3 binding or inhibition."
)

print("=" * 78)