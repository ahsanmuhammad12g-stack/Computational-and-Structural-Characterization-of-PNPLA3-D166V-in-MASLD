import pandas as pd
import requests
import time
from pathlib import Path


# ============================================================
# FINAL GNOMAD RELEASE CHECK
# STEP 8 — PNPLA3 POPULATION FREQUENCY ANNOTATION
#
# PURPOSE:
#   Resolve the 9 remaining PNPLA3 variants against gnomAD
#   using EXACT chromosome + position + REF + ALT identity.
#
# IMPORTANT:
#   A variant at the same position but with a different ALT
#   is NOT considered a match.
#
#   Example:
#       Requested: 22-43926992-G-A
#       Found:     22-43926992-G-T
#
#   These are DIFFERENT variants and must NOT be merged.
# ============================================================


print("=" * 78)
print("FINAL GNOMAD RELEASE CHECK — EXACT VARIANT IDENTITY")
print("=" * 78)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

MAPPING_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_exact_variant_mapping.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
)

OUTPUT_FILE = OUTPUT_DIR / "pnpla3_gnomad_final_release_check.csv"


# ============================================================
# SETTINGS
# ============================================================

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

# gnomAD releases to test.
#
# gnomad_r4 is known to work from the previous successful run.
#
# Older dataset identifiers are included as release checks.
# If a dataset is unsupported by the current API, the script
# records the exact API error rather than treating it as absence.
#
DATASETS = [
    "gnomad_r4",
    "gnomad_r3",
    "gnomad_r2_1",
]

REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 0.5


# ============================================================
# THE 9 REMAINING VARIANTS
# ============================================================

FAILED_IDS = {
    3421690,
    4140671,
    899624,
    899627,
    900762,
    4140672,
    3216047,
    902436,
    3308174,
}


# ============================================================
# GRAPHQL QUERY
# ============================================================

QUERY = """
query VariantQuery(
    $dataset: Dataset!
    $variantId: String!
) {
    variant(
        dataset: $dataset
        variantId: $variantId
    ) {
        variant_id
        rsids

        exome {
            ac
            an
            af
        }

        genome {
            ac
            an
            af
        }
    }
}
"""


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):
    """
    Convert pandas NaN/empty values into None.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def make_exact_variant_id(chromosome, position, reference, alternate):
    """
    Construct exact gnomAD-style variant ID.

    Identity is strictly:
        chromosome-position-REF-ALT
    """

    chromosome = str(chromosome).replace("chr", "").strip()
    position = str(position).strip()
    reference = str(reference).strip().upper()
    alternate = str(alternate).strip().upper()

    return f"{chromosome}-{position}-{reference}-{alternate}"


def normalize_variant_id(variant_id):
    """
    Normalize a variant ID for comparison.
    """

    if variant_id is None:
        return None

    parts = str(variant_id).strip().split("-")

    if len(parts) != 4:
        return str(variant_id).strip()

    chrom, pos, ref, alt = parts

    chrom = chrom.replace("chr", "")

    return f"{chrom}-{pos}-{ref.upper()}-{alt.upper()}"


def exact_identity_match(requested_id, returned_id):
    """
    STRICT identity check.

    All four components must match:
        chromosome
        position
        reference allele
        alternate allele
    """

    if requested_id is None or returned_id is None:
        return False

    return (
        normalize_variant_id(requested_id)
        == normalize_variant_id(returned_id)
    )


def extract_graphql_errors(response_json):
    """
    Extract complete GraphQL error messages.
    """

    errors = response_json.get("errors", [])

    if not errors:
        return ""

    messages = []

    for error in errors:
        message = error.get("message", "")
        if message:
            messages.append(message)

    return " | ".join(messages)


# ============================================================
# QUERY GNOMAD
# ============================================================

def query_gnomad(dataset, variant_id):

    variables = {
        "dataset": dataset,
        "variantId": variant_id,
    }

    try:

        response = requests.post(
            GNOMAD_API,
            json={
                "query": QUERY,
                "variables": variables,
            },
            timeout=REQUEST_TIMEOUT,
        )

    except requests.exceptions.RequestException as exc:

        return {
            "http_status": None,
            "variant": None,
            "error": f"REQUEST_ERROR: {exc}",
        }

    http_status = response.status_code

    try:
        data = response.json()

    except ValueError:

        return {
            "http_status": http_status,
            "variant": None,
            "error": "INVALID_JSON_RESPONSE",
        }

    errors = extract_graphql_errors(data)

    variant = (
        data.get("data", {})
        .get("variant")
    )

    return {
        "http_status": http_status,
        "variant": variant,
        "error": errors,
    }


# ============================================================
# LOAD MAPPING FILE
# ============================================================

print("\nLoading exact variant mapping...")

if not MAPPING_FILE.exists():

    raise FileNotFoundError(
        f"\nMapping file not found:\n{MAPPING_FILE}"
    )


df = pd.read_csv(MAPPING_FILE)

print(f"Mapping records loaded: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "VariationID",
    "Protein_Change",
    "Candidate_Group",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "RS# (dbSNP)",
    "gnomAD_Variant_ID",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "\nMissing required columns:\n"
        + "\n".join(missing_columns)
    )


# ============================================================
# SELECT ONLY THE 9 FAILED VARIANTS
# ============================================================

failed_df = df[
    df["VariationID"].astype(int).isin(FAILED_IDS)
].copy()

failed_df = failed_df.sort_values("VariationID")


print(
    f"Remaining variants selected for final release check: "
    f"{len(failed_df)}"
)


if len(failed_df) != 9:

    print("\nWARNING:")
    print(
        f"Expected 9 variants but found {len(failed_df)} "
        "in the mapping file."
    )


# ============================================================
# DISPLAY EXACT INPUT IDENTITIES
# ============================================================

print("\n" + "-" * 78)
print("AUTHORITATIVE VARIANT IDENTITIES")
print("-" * 78)

for _, row in failed_df.iterrows():

    exact_id = make_exact_variant_id(
        row["Chromosome"],
        row["Position_GRCh38"],
        row["Reference"],
        row["Alternate"],
    )

    print(
        f"{int(row['VariationID'])} | "
        f"{row['Protein_Change']} | "
        f"{exact_id} | "
        f"rsID={clean_value(row['RS# (dbSNP)'])}"
    )


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# MAIN RESOLUTION LOOP
# ============================================================

for _, row in failed_df.iterrows():

    variation_id = int(row["VariationID"])

    protein_change = clean_value(
        row["Protein_Change"]
    )

    chromosome = clean_value(
        row["Chromosome"]
    )

    position = clean_value(
        row["Position_GRCh38"]
    )

    reference = clean_value(
        row["Reference"]
    )

    alternate = clean_value(
        row["Alternate"]
    )

    rsid = clean_value(
        row["RS# (dbSNP)"]
    )

    # --------------------------------------------------------
    # AUTHORITATIVE EXACT VARIANT ID
    # --------------------------------------------------------

    exact_variant_id = make_exact_variant_id(
        chromosome,
        position,
        reference,
        alternate,
    )

    print("\n")
    print("=" * 78)
    print(f"VariationID: {variation_id}")
    print(f"Protein change: {protein_change}")
    print(f"Exact requested ID: {exact_variant_id}")
    print(f"Reference allele: {reference}")
    print(f"Alternate allele: {alternate}")
    print(f"dbSNP rsID: {rsid}")
    print("=" * 78)

    resolved = False

    # --------------------------------------------------------
    # TEST EACH GNOMAD RELEASE
    # --------------------------------------------------------

    for dataset in DATASETS:

        print(
            f"\nChecking dataset: {dataset}"
        )

        result = query_gnomad(
            dataset,
            exact_variant_id,
        )

        http_status = result["http_status"]
        variant = result["variant"]
        graphql_error = result["error"]

        print(
            f"HTTP status: {http_status}"
        )

        # ----------------------------------------------------
        # GRAPHQL ERROR
        # ----------------------------------------------------

        if graphql_error:

            print(
                f"GraphQL error: {graphql_error}"
            )

            results.append({
                "VariationID": variation_id,
                "Protein_Change": protein_change,
                "Chromosome": chromosome,
                "Position_GRCh38": position,
                "Reference": reference,
                "Alternate": alternate,
                "Exact_Requested_Variant_ID": exact_variant_id,
                "RSID": rsid,
                "Dataset": dataset,
                "Returned_Variant_ID": "",
                "Identity_Match": False,
                "Status": "GRAPHQL_ERROR",
                "GraphQL_Error": graphql_error,
                "Exome_AC": None,
                "Exome_AN": None,
                "Exome_AF": None,
                "Genome_AC": None,
                "Genome_AN": None,
                "Genome_AF": None,
            })

            time.sleep(SLEEP_SECONDS)

            continue

        # ----------------------------------------------------
        # NO VARIANT
        # ----------------------------------------------------

        if variant is None:

            print(
                "No variant returned."
            )

            results.append({
                "VariationID": variation_id,
                "Protein_Change": protein_change,
                "Chromosome": chromosome,
                "Position_GRCh38": position,
                "Reference": reference,
                "Alternate": alternate,
                "Exact_Requested_Variant_ID": exact_variant_id,
                "RSID": rsid,
                "Dataset": dataset,
                "Returned_Variant_ID": "",
                "Identity_Match": False,
                "Status": "NOT_FOUND",
                "GraphQL_Error": "",
                "Exome_AC": None,
                "Exome_AN": None,
                "Exome_AF": None,
                "Genome_AC": None,
                "Genome_AN": None,
                "Genome_AF": None,
            })

            time.sleep(SLEEP_SECONDS)

            continue

        # ----------------------------------------------------
        # VARIANT RETURNED
        # ----------------------------------------------------

        returned_variant_id = variant.get(
            "variant_id"
        )

        print(
            f"Returned variant ID: {returned_variant_id}"
        )

        # ----------------------------------------------------
        # STRICT IDENTITY CHECK
        # ----------------------------------------------------

        identity_match = exact_identity_match(
            exact_variant_id,
            returned_variant_id,
        )

        print(
            f"Exact identity match: {identity_match}"
        )

        # ----------------------------------------------------
        # IDENTITY MISMATCH
        # ----------------------------------------------------

        if not identity_match:

            print(
                "!!! IDENTITY MISMATCH — REJECTED !!!"
            )

            results.append({
                "VariationID": variation_id,
                "Protein_Change": protein_change,
                "Chromosome": chromosome,
                "Position_GRCh38": position,
                "Reference": reference,
                "Alternate": alternate,
                "Exact_Requested_Variant_ID": exact_variant_id,
                "RSID": rsid,
                "Dataset": dataset,
                "Returned_Variant_ID": returned_variant_id,
                "Identity_Match": False,
                "Status": "IDENTITY_MISMATCH",
                "GraphQL_Error": "",
                "Exome_AC": None,
                "Exome_AN": None,
                "Exome_AF": None,
                "Genome_AC": None,
                "Genome_AN": None,
                "Genome_AF": None,
            })

            time.sleep(SLEEP_SECONDS)

            continue

        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        print(
            "!!! EXACT VARIANT MATCH !!!"
        )

        exome = variant.get("exome") or {}
        genome = variant.get("genome") or {}

        exome_ac = exome.get("ac")
        exome_an = exome.get("an")
        exome_af = exome.get("af")

        genome_ac = genome.get("ac")
        genome_an = genome.get("an")
        genome_af = genome.get("af")

        print(
            f"Exome:  AC={exome_ac} "
            f"AN={exome_an} "
            f"AF={exome_af}"
        )

        print(
            f"Genome: AC={genome_ac} "
            f"AN={genome_an} "
            f"AF={genome_af}"
        )

        results.append({
            "VariationID": variation_id,
            "Protein_Change": protein_change,
            "Chromosome": chromosome,
            "Position_GRCh38": position,
            "Reference": reference,
            "Alternate": alternate,
            "Exact_Requested_Variant_ID": exact_variant_id,
            "RSID": rsid,
            "Dataset": dataset,
            "Returned_Variant_ID": returned_variant_id,
            "Identity_Match": True,
            "Status": "EXACT_MATCH",
            "GraphQL_Error": "",
            "Exome_AC": exome_ac,
            "Exome_AN": exome_an,
            "Exome_AF": exome_af,
            "Genome_AC": genome_ac,
            "Genome_AN": genome_an,
            "Genome_AF": genome_af,
        })

        resolved = True

        # ----------------------------------------------------
        # IMPORTANT:
        # Stop checking older releases once an exact match
        # has been found.
        # ----------------------------------------------------

        break

    # --------------------------------------------------------
    # FINAL STATUS IF NOTHING EXACTLY RESOLVED
    # --------------------------------------------------------

    if not resolved:

        print(
            "\nNo exact identity match found in tested releases."
        )

        # Add an explicit summary row.
        results.append({
            "VariationID": variation_id,
            "Protein_Change": protein_change,
            "Chromosome": chromosome,
            "Position_GRCh38": position,
            "Reference": reference,
            "Alternate": alternate,
            "Exact_Requested_Variant_ID": exact_variant_id,
            "RSID": rsid,
            "Dataset": "ALL_TESTED_RELEASES",
            "Returned_Variant_ID": "",
            "Identity_Match": False,
            "Status": "NOT_FOUND_IN_TESTED_RELEASES",
            "GraphQL_Error": "",
            "Exome_AC": None,
            "Exome_AN": None,
            "Exome_AF": None,
            "Genome_AC": None,
            "Genome_AN": None,
            "Genome_AF": None,
        })

    time.sleep(SLEEP_SECONDS)


# ============================================================
# SAVE FULL DIAGNOSTIC TABLE
# ============================================================

results_df = pd.DataFrame(results)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results_df.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# FINAL EXACT-MATCH TABLE
# ============================================================

exact_matches = results_df[
    results_df["Status"] == "EXACT_MATCH"
].copy()


final_unresolved = results_df[
    results_df["Status"] == "NOT_FOUND_IN_TESTED_RELEASES"
].copy()


identity_mismatches = results_df[
    results_df["Status"] == "IDENTITY_MISMATCH"
].copy()


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 78)
print("FINAL GNOMAD RELEASE CHECK SUMMARY")
print("=" * 78)

print(
    f"Variants investigated: {len(failed_df)}"
)

print(
    f"Exact variants resolved: {len(exact_matches)}"
)

print(
    f"Unresolved after release testing: {len(final_unresolved)}"
)

print(
    f"Identity mismatches encountered: "
    f"{len(identity_mismatches)}"
)


# ============================================================
# DISPLAY EXACT MATCHES
# ============================================================

if len(exact_matches) > 0:

    print("\n")
    print("-" * 78)
    print("EXACT GNOMAD MATCHES")
    print("-" * 78)

    display_columns = [
        "VariationID",
        "Protein_Change",
        "Exact_Requested_Variant_ID",
        "Dataset",
        "Exome_AF",
        "Genome_AF",
    ]

    print(
        exact_matches[display_columns]
        .to_string(index=False)
    )

else:

    print("\nNo exact gnomAD matches were found.")


# ============================================================
# DISPLAY UNRESOLVED VARIANTS
# ============================================================

if len(final_unresolved) > 0:

    print("\n")
    print("-" * 78)
    print("VARIANTS STILL UNRESOLVED")
    print("-" * 78)

    unresolved_columns = [
        "VariationID",
        "Protein_Change",
        "Exact_Requested_Variant_ID",
        "RSID",
        "Status",
    ]

    print(
        final_unresolved[unresolved_columns]
        .to_string(index=False)
    )


# ============================================================
# DISPLAY IDENTITY MISMATCHES
# ============================================================

if len(identity_mismatches) > 0:

    print("\n")
    print("-" * 78)
    print("IDENTITY MISMATCHES — NOT ACCEPTED")
    print("-" * 78)

    mismatch_columns = [
        "VariationID",
        "Exact_Requested_Variant_ID",
        "Returned_Variant_ID",
        "Dataset",
    ]

    print(
        identity_mismatches[mismatch_columns]
        .to_string(index=False)
    )


# ============================================================
# FINAL SCIENTIFIC INTERPRETATION
# ============================================================

print("\n")
print("=" * 78)
print("SCIENTIFIC INTERPRETATION")
print("=" * 78)

if len(final_unresolved) == 0:

    print(
        "All 9 previously unresolved variants have now been "
        "resolved by exact variant identity in at least one "
        "tested gnomAD release."
    )

elif len(exact_matches) > 0:

    print(
        f"{len(exact_matches)} variant(s) were resolved by exact "
        "identity across the tested gnomAD releases."
    )

    print(
        f"{len(final_unresolved)} variant(s) remain unresolved."
    )

else:

    print(
        "None of the 9 variants could be resolved by exact "
        "variant identity in the tested gnomAD releases."
    )

print(
    "\nIMPORTANT: Unresolved variants must NOT be assigned "
    "AF = 0."
)

print(
    "IMPORTANT: A different ALT allele at the same genomic "
    "position is NOT an equivalent variant."
)

print(
    "IMPORTANT: Exome AF and Genome AF are preserved separately."
)

print(
    f"\nFull results saved to:\n{OUTPUT_FILE}"
)

print("=" * 78)
print("STEP 8 RELEASE CHECK COMPLETE")
print("=" * 78)