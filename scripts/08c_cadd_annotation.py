import pandas as pd
import requests
import time
from pathlib import Path


print("=" * 78)
print("STEP 9C — CADD ANNOTATION OF PNPLA3 CANDIDATE VARIANTS")
print("=" * 78)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

INPUT_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_step9_alphamissense_annotated_FINAL.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_cadd_annotation.csv"
)

FINAL_MASTER = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_step9_cadd_annotated_FINAL.csv"
)


# ============================================================
# SETTINGS
# ============================================================

EXPECTED_VARIANTS = 82

# CADD API
CADD_API = "https://cadd.gs.washington.edu/api/v1.7/GRCh38/score"

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.15


# ============================================================
# REQUIRED INPUT COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_chromosome(chrom):
    """
    Convert chromosome representation to the form expected by CADD.
    """
    chrom = str(chrom).strip()

    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]

    return chrom


def clean_allele(value):
    """
    Clean REF/ALT allele values.
    """
    if pd.isna(value):
        return ""

    return str(value).strip().upper()


def classify_cadd(phred):
    """
    Computational interpretation of CADD PHRED score.

    These categories are used only for computational prioritization
    and are NOT clinical classifications.
    """

    if pd.isna(phred):
        return "Not available"

    if phred >= 20:
        return "High deleteriousness"

    elif phred >= 15:
        return "Moderate deleteriousness"

    else:
        return "Lower deleteriousness"


def extract_cadd_values(response_json):
    """
    Extract CADD raw score and PHRED score from API responses.

    The API response format may vary slightly between versions,
    so several common representations are handled.
    """

    raw_score = None
    phred_score = None

    # --------------------------------------------------------
    # Case 1: dictionary containing direct fields
    # --------------------------------------------------------

    if isinstance(response_json, dict):

        possible_raw = [
            "RawScore",
            "rawScore",
            "raw_score",
            "CADD_RawScore",
            "cadd_raw_score",
        ]

        possible_phred = [
            "PHRED",
            "phred",
            "CADD_PHRED",
            "cadd_phred",
        ]

        for key in possible_raw:
            if key in response_json:
                raw_score = response_json[key]
                break

        for key in possible_phred:
            if key in response_json:
                phred_score = response_json[key]
                break

        # Sometimes API returns data inside a nested object
        for nested_key in ["data", "result", "variant", "scores"]:

            nested = response_json.get(nested_key)

            if isinstance(nested, dict):

                if raw_score is None:
                    for key in possible_raw:
                        if key in nested:
                            raw_score = nested[key]
                            break

                if phred_score is None:
                    for key in possible_phred:
                        if key in nested:
                            phred_score = nested[key]
                            break

    # --------------------------------------------------------
    # Case 2: list containing one dictionary
    # --------------------------------------------------------

    elif isinstance(response_json, list) and len(response_json) > 0:

        first = response_json[0]

        if isinstance(first, dict):

            possible_raw = [
                "RawScore",
                "rawScore",
                "raw_score",
                "CADD_RawScore",
                "cadd_raw_score",
            ]

            possible_phred = [
                "PHRED",
                "phred",
                "CADD_PHRED",
                "cadd_phred",
            ]

            for key in possible_raw:
                if key in first:
                    raw_score = first[key]
                    break

            for key in possible_phred:
                if key in first:
                    phred_score = first[key]
                    break

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    try:
        if raw_score is not None:
            raw_score = float(raw_score)
    except Exception:
        raw_score = None

    try:
        if phred_score is not None:
            phred_score = float(phred_score)
    except Exception:
        phred_score = None

    return raw_score, phred_score


# ============================================================
# LOAD STEP 9B MASTER
# ============================================================

print("\nInput file:")
print(INPUT_FILE)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nERROR: Step 9B final master not found:\n{INPUT_FILE}"
    )

print("\nReading validated Step 9B master...")

df = pd.read_csv(INPUT_FILE)

print(f"Records loaded: {len(df)}")


# ============================================================
# INPUT QC
# ============================================================

print("\n" + "-" * 78)
print("BASIC INPUT QC")
print("-" * 78)

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

print("Required columns: PASS")


if len(df) != EXPECTED_VARIANTS:
    raise ValueError(
        f"Expected {EXPECTED_VARIANTS} variants but found {len(df)}"
    )

print(f"Expected variant count ({EXPECTED_VARIANTS}): PASS")


if df["VariationID"].nunique() != EXPECTED_VARIANTS:
    raise ValueError(
        "VariationID uniqueness failed."
    )

print("VariationID uniqueness: PASS")


if df["Exact_Variant_Key"].nunique() != EXPECTED_VARIANTS:
    raise ValueError(
        "Exact genomic variant identity uniqueness failed."
    )

print("Exact genomic variant identity uniqueness: PASS")


# ============================================================
# VALIDATE GENOMIC IDENTITIES
# ============================================================

print("\n" + "-" * 78)
print("VALIDATING GRCh38 VARIANT IDENTITIES")
print("-" * 78)

for column in [
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
]:

    missing = df[column].isna().sum()

    if missing > 0:
        raise ValueError(
            f"{column} contains {missing} missing values."
        )

    print(f"{column}: PASS")


df["CADD_Chromosome"] = df["Chromosome"].apply(clean_chromosome)
df["CADD_Reference"] = df["Reference"].apply(clean_allele)
df["CADD_Alternate"] = df["Alternate"].apply(clean_allele)

df["CADD_Position"] = pd.to_numeric(
    df["Position_GRCh38"],
    errors="coerce"
)

if df["CADD_Position"].isna().any():
    raise ValueError(
        "Invalid GRCh38 positions detected."
    )

print("GRCh38 genomic identity validation: PASS")


# ============================================================
# INITIALIZE CADD COLUMNS
# ============================================================

df["CADD_RawScore"] = pd.NA
df["CADD_PHRED"] = pd.NA
df["CADD_Interpretation"] = pd.NA
df["CADD_Status"] = pd.NA
df["CADD_Source"] = "CADD v1.7 GRCh38 API"


# ============================================================
# QUERY CADD
# ============================================================

print("\n" + "-" * 78)
print("QUERYING CADD")
print("-" * 78)

print(f"Total variants to query: {len(df)}")
print(f"CADD endpoint: {CADD_API}")
print("\nThis may take a few minutes...\n")


session = requests.Session()

successful = 0
not_found = 0
errors = 0


for index, row in df.iterrows():

    chrom = row["CADD_Chromosome"]
    pos = int(row["CADD_Position"])
    ref = row["CADD_Reference"]
    alt = row["CADD_Alternate"]

    variation_id = row["VariationID"]
    protein_change = row["Protein_Change"]

    params = {
        "chrom": chrom,
        "pos": pos,
        "ref": ref,
        "alt": alt,
    }

    print(
        f"[{index + 1:02d}/{len(df)}] "
        f"VariationID={variation_id} | "
        f"{chrom}-{pos}-{ref}-{alt} | "
        f"{protein_change}"
    )

    try:

        response = session.get(
            CADD_API,
            params=params,
            timeout=REQUEST_TIMEOUT
        )

        if response.status_code != 200:

            print(
                f"    CADD HTTP status: {response.status_code}"
            )

            df.at[index, "CADD_Status"] = (
                f"HTTP_ERROR_{response.status_code}"
            )

            errors += 1

            time.sleep(SLEEP_SECONDS)
            continue

        try:
            result = response.json()

        except Exception:

            print("    ERROR: CADD response was not valid JSON.")

            df.at[index, "CADD_Status"] = "INVALID_RESPONSE"

            errors += 1

            time.sleep(SLEEP_SECONDS)
            continue


        raw_score, phred_score = extract_cadd_values(result)


        if phred_score is not None:

            df.at[index, "CADD_RawScore"] = raw_score
            df.at[index, "CADD_PHRED"] = phred_score

            df.at[index, "CADD_Interpretation"] = (
                classify_cadd(phred_score)
            )

            df.at[index, "CADD_Status"] = "EXACT_MATCH"

            successful += 1

            print(
                f"    CADD PHRED: {phred_score}"
            )

            if raw_score is not None:
                print(
                    f"    CADD RawScore: {raw_score}"
                )

        else:

            df.at[index, "CADD_Status"] = (
                "NOT_FOUND_IN_CADD"
            )

            not_found += 1

            print(
                "    CADD score not available."
            )

    except requests.exceptions.Timeout:

        df.at[index, "CADD_Status"] = "TIMEOUT"

        errors += 1

        print("    ERROR: Request timed out.")

    except requests.exceptions.RequestException as e:

        df.at[index, "CADD_Status"] = "REQUEST_ERROR"

        errors += 1

        print(
            f"    ERROR: {str(e)}"
        )

    except Exception as e:

        df.at[index, "CADD_Status"] = "ERROR"

        errors += 1

        print(
            f"    ERROR: {str(e)}"
        )

    time.sleep(SLEEP_SECONDS)


session.close()


# ============================================================
# CONVERT CADD NUMERIC COLUMNS
# ============================================================

df["CADD_RawScore"] = pd.to_numeric(
    df["CADD_RawScore"],
    errors="coerce"
)

df["CADD_PHRED"] = pd.to_numeric(
    df["CADD_PHRED"],
    errors="coerce"
)


# ============================================================
# CADD SCORE QC
# ============================================================

print("\n" + "-" * 78)
print("CADD SCORE QC")
print("-" * 78)

available_scores = df["CADD_PHRED"].notna().sum()

print(f"CADD scores available: {available_scores}")
print(f"CADD scores unavailable: {len(df) - available_scores}")

if available_scores > 0:

    min_phred = df["CADD_PHRED"].min()
    max_phred = df["CADD_PHRED"].max()

    print(
        f"CADD PHRED range: "
        f"{min_phred:.4f} – {max_phred:.4f}"
    )

    if (df["CADD_PHRED"] < 0).any():
        raise ValueError(
            "Invalid negative CADD PHRED score detected."
        )

    print("CADD PHRED range QC: PASS")

else:

    print(
        "\nWARNING: No CADD scores were obtained."
    )

    print(
        "The CADD API response should be inspected before "
        "considering Step 9C complete."
    )


# ============================================================
# CADD STATUS DISTRIBUTION
# ============================================================

print("\n" + "-" * 78)
print("CADD ANNOTATION STATUS")
print("-" * 78)

print(
    df["CADD_Status"]
    .value_counts(dropna=False)
)


# ============================================================
# COMPUTATIONAL INTERPRETATION DISTRIBUTION
# ============================================================

print("\n" + "-" * 78)
print("CADD COMPUTATIONAL INTERPRETATION")
print("-" * 78)

print(
    df["CADD_Interpretation"]
    .value_counts(dropna=False)
)


# ============================================================
# SAVE CLEAN CADD ANNOTATION TABLE
# ============================================================

cadd_columns = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "Exact_Variant_Key",
    "CADD_RawScore",
    "CADD_PHRED",
    "CADD_Interpretation",
    "CADD_Status",
    "CADD_Source",
]

cadd_df = df[cadd_columns].copy()

cadd_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nCADD annotation table written:")
print(OUTPUT_FILE)


# ============================================================
# REMOVE TEMPORARY COLUMNS
# ============================================================

df.drop(
    columns=[
        "CADD_Chromosome",
        "CADD_Reference",
        "CADD_Alternate",
        "CADD_Position",
    ],
    inplace=True
)


# ============================================================
# FINAL MASTER
# ============================================================

df.to_csv(
    FINAL_MASTER,
    index=False
)

print("\nFinal Step 9C master written:")
print(FINAL_MASTER)


# ============================================================
# RELOAD FOR FINAL QC
# ============================================================

final_df = pd.read_csv(FINAL_MASTER)


# ============================================================
# FINAL STEP 9C QC
# ============================================================

print("\n" + "=" * 78)
print("FINAL STEP 9C QUALITY CONTROL")
print("=" * 78)

print(
    f"\nPNPLA3 variants analyzed:       {len(final_df)}"
)

print(
    f"CADD exact matches:             "
    f"{(final_df['CADD_Status'] == 'EXACT_MATCH').sum()}"
)

print(
    f"CADD unavailable:                "
    f"{(final_df['CADD_Status'] == 'NOT_FOUND_IN_CADD').sum()}"
)

print(
    f"CADD errors:                     "
    f"{(~final_df['CADD_Status'].isin(['EXACT_MATCH', 'NOT_FOUND_IN_CADD'])).sum()}"
)

print(
    f"Unique VariationIDs:             "
    f"{final_df['VariationID'].nunique()}"
)

print(
    f"Unique exact variant identities: "
    f"{final_df['Exact_Variant_Key'].nunique()}"
)


# ============================================================
# RETENTION QC
# ============================================================

if len(final_df) != EXPECTED_VARIANTS:
    raise ValueError(
        "Final variant count changed."
    )

if final_df["VariationID"].nunique() != EXPECTED_VARIANTS:
    raise ValueError(
        "VariationID uniqueness failed."
    )

if final_df["Exact_Variant_Key"].nunique() != EXPECTED_VARIANTS:
    raise ValueError(
        "Exact genomic identity uniqueness failed."
    )

print("\n82/82 variants retained: PASS")


# ============================================================
# NO FALSE ZERO QC
# ============================================================

missing_cadd = final_df["CADD_PHRED"].isna()

if missing_cadd.any():

    # Missing CADD must remain missing, not zero.
    missing_status = final_df.loc[
        missing_cadd,
        "CADD_Status"
    ]

    invalid_missing = ~missing_status.isin(
        [
            "NOT_FOUND_IN_CADD",
            "HTTP_ERROR_404",
            "HTTP_ERROR_400",
            "TIMEOUT",
            "REQUEST_ERROR",
            "INVALID_RESPONSE",
            "ERROR",
        ]
    )

    if invalid_missing.any():

        raise ValueError(
            "Some missing CADD scores have invalid statuses."
        )

print(
    "Missing CADD values handled explicitly: PASS"
)


# ============================================================
# FINAL SUCCESS / WARNING
# ============================================================

if successful == EXPECTED_VARIANTS:

    print("\n" + "-" * 78)
    print("FINAL STEP 9C QC: ALL TESTS PASSED")
    print("-" * 78)

    print(
        "\nAll 82 PNPLA3 variants received CADD annotations."
    )

    print(
        "\nSTEP 9C SUCCESSFULLY COMPLETED"
    )

elif successful > 0:

    print("\n" + "-" * 78)
    print("STEP 9C PARTIALLY COMPLETED")
    print("-" * 78)

    print(
        f"\nCADD annotations obtained for "
        f"{successful}/{EXPECTED_VARIANTS} variants."
    )

    print(
        f"Unavailable/error records: "
        f"{EXPECTED_VARIANTS - successful}"
    )

    print(
        "\nDo NOT proceed to Step 9D yet."
    )

    print(
        "Inspect the CADD response/status distribution first."
    )

else:

    print("\n" + "-" * 78)
    print("STEP 9C NOT COMPLETED")
    print("-" * 78)

    print(
        "\nNo usable CADD scores were obtained."
    )

    print(
        "Do NOT treat missing CADD values as zero."
    )

    print(
        "Do NOT proceed to Step 9D until the CADD retrieval "
        "method is validated."
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 78)
print("STEP 9C SUMMARY")
print("=" * 78)

print(f"Total variants:          {len(final_df)}")
print(f"CADD exact matches:      {successful}")
print(f"CADD unavailable:        {not_found}")
print(f"CADD errors:             {errors}")

if available_scores > 0:

    print(
        f"Minimum CADD PHRED:      "
        f"{final_df['CADD_PHRED'].min():.4f}"
    )

    print(
        f"Maximum CADD PHRED:      "
        f"{final_df['CADD_PHRED'].max():.4f}"
    )

print("\nOutput:")
print(FINAL_MASTER)

print("\n" + "=" * 78)