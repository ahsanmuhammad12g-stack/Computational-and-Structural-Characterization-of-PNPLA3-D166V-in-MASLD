import pandas as pd
import requests
import time
from pathlib import Path


# ============================================================
# FINAL GNOMAD RELEASE CHECK — VERSION 07g
#
# STEP 8 — PNPLA3 POPULATION FREQUENCY ANNOTATION
#
# STRICT RULE:
#   Only an exact chromosome-position-REF-ALT match is accepted.
#
#   Example:
#       Requested: 22-43926992-G-A
#       Returned:  22-43926992-G-T
#
#   This is NOT a match.
#
# IMPORTANT:
#   This version uses dataset names directly inside the GraphQL
#   query because the previous version incorrectly declared the
#   variable type as "Dataset".
#
# ============================================================


print("=" * 78)
print("FINAL GNOMAD RELEASE CHECK — VERSION 07g")
print("EXACT VARIANT IDENTITY + RATE-LIMIT PROTECTION")
print("=" * 78)


# ============================================================
# PROJECT PATHS
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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "pnpla3_gnomad_final_release_check_07g.csv"
)


# ============================================================
# GNOMAD API
# ============================================================

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

# Test releases one at a time.
#
# gnomad_r4 is already known to be valid from the successful
# earlier 73/82 run.
#
DATASETS = [
    "gnomad_r4",
    "gnomad_r3",
    "gnomad_r2_1",
]


# ============================================================
# FAILED VARIANTS
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
# RATE LIMIT SETTINGS
# ============================================================

REQUEST_TIMEOUT = 30

# Initial delay between normal requests
BASE_SLEEP = 2

# Maximum retry attempts after HTTP 429
MAX_RETRIES = 5

# Maximum wait after repeated rate limiting
MAX_BACKOFF = 60


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_value(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value == "":
        return None

    return value


def make_exact_variant_id(
    chromosome,
    position,
    reference,
    alternate
):

    chromosome = str(chromosome).strip()

    chromosome = chromosome.replace("chr", "")

    position = str(position).strip()

    reference = str(reference).strip().upper()

    alternate = str(alternate).strip().upper()

    return (
        f"{chromosome}-"
        f"{position}-"
        f"{reference}-"
        f"{alternate}"
    )


def normalize_variant_id(variant_id):

    if variant_id is None:
        return None

    parts = str(variant_id).strip().split("-")

    if len(parts) != 4:
        return str(variant_id).strip()

    chrom, pos, ref, alt = parts

    chrom = chrom.replace("chr", "")

    return (
        f"{chrom}-"
        f"{pos}-"
        f"{ref.upper()}-"
        f"{alt.upper()}"
    )


def exact_identity_match(
    requested_id,
    returned_id
):

    if requested_id is None:
        return False

    if returned_id is None:
        return False

    return (
        normalize_variant_id(requested_id)
        ==
        normalize_variant_id(returned_id)
    )


def extract_errors(data):

    errors = data.get("errors", [])

    if not errors:
        return ""

    messages = []

    for error in errors:

        message = error.get(
            "message",
            ""
        )

        if message:
            messages.append(message)

    return " | ".join(messages)


# ============================================================
# BUILD QUERY
# ============================================================

def build_query(dataset):

    # IMPORTANT:
    # Dataset is inserted directly into the GraphQL query.
    #
    # This avoids the previous invalid:
    #
    #   $dataset: Dataset!
    #
    # which produced:
    #
    #   Unknown type "Dataset"
    #
    query = f"""
    query VariantQuery($variantId: String!) {{

        variant(
            dataset: {dataset},
            variantId: $variantId
        ) {{

            variant_id

            rsids

            exome {{
                ac
                an
                af
            }}

            genome {{
                ac
                an
                af
            }}
        }}
    }}
    """

    return query


# ============================================================
# GNOMAD QUERY WITH RETRIES
# ============================================================

def query_gnomad(
    dataset,
    variant_id
):

    query = build_query(dataset)

    variables = {
        "variantId": variant_id
    }

    for attempt in range(
        1,
        MAX_RETRIES + 1
    ):

        try:

            response = requests.post(
                GNOMAD_API,
                json={
                    "query": query,
                    "variables": variables
                },
                timeout=REQUEST_TIMEOUT
            )

        except requests.exceptions.RequestException as exc:

            print(
                f"Request exception "
                f"(attempt {attempt}/{MAX_RETRIES}): "
                f"{exc}"
            )

            if attempt < MAX_RETRIES:

                wait_time = min(
                    BASE_SLEEP * (2 ** (attempt - 1)),
                    MAX_BACKOFF
                )

                print(
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue

            return {
                "http_status": None,
                "variant": None,
                "error": f"REQUEST_ERROR: {exc}"
            }

        # ----------------------------------------------------
        # HTTP 429 RATE LIMIT
        # ----------------------------------------------------

        if response.status_code == 429:

            print(
                f"HTTP 429 — rate limited "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )

            if attempt < MAX_RETRIES:

                wait_time = min(
                    BASE_SLEEP * (2 ** attempt),
                    MAX_BACKOFF
                )

                print(
                    f"Waiting {wait_time} seconds "
                    f"before retry..."
                )

                time.sleep(wait_time)

                continue

            return {
                "http_status": 429,
                "variant": None,
                "error": "HTTP_429_RATE_LIMIT"
            }

        # ----------------------------------------------------
        # PARSE JSON
        # ----------------------------------------------------

        try:

            data = response.json()

        except ValueError:

            return {
                "http_status": response.status_code,
                "variant": None,
                "error": "INVALID_JSON_RESPONSE"
            }

        # ----------------------------------------------------
        # GRAPHQL ERRORS
        # ----------------------------------------------------

        errors = extract_errors(data)

        variant = (
            data
            .get("data", {})
            .get("variant")
        )

        return {
            "http_status": response.status_code,
            "variant": variant,
            "error": errors
        }

    return {
        "http_status": None,
        "variant": None,
        "error": "UNKNOWN_ERROR"
    }


# ============================================================
# LOAD MAPPING FILE
# ============================================================

print("\nLoading exact variant mapping...")

if not MAPPING_FILE.exists():

    raise FileNotFoundError(
        f"\nMapping file not found:\n{MAPPING_FILE}"
    )


df = pd.read_csv(
    MAPPING_FILE
)

print(
    f"Mapping records loaded: {len(df)}"
)


# ============================================================
# REQUIRED COLUMNS
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
    "gnomAD_Variant_ID"
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
# SELECT 9 FAILED VARIANTS
# ============================================================

failed_df = df[
    df["VariationID"]
    .astype(int)
    .isin(FAILED_IDS)
].copy()


failed_df = failed_df.sort_values(
    "VariationID"
)


print(
    f"Remaining variants selected: "
    f"{len(failed_df)}"
)


# ============================================================
# SHOW AUTHORITATIVE IDENTITIES
# ============================================================

print("\n")
print("-" * 78)
print("AUTHORITATIVE VARIANT IDENTITIES")
print("-" * 78)


for _, row in failed_df.iterrows():

    exact_id = make_exact_variant_id(
        row["Chromosome"],
        row["Position_GRCh38"],
        row["Reference"],
        row["Alternate"]
    )

    print(
        f"{int(row['VariationID'])} | "
        f"{row['Protein_Change']} | "
        f"{exact_id} | "
        f"rsID={clean_value(row['RS# (dbSNP)'])}"
    )


# ============================================================
# RESULTS
# ============================================================

results = []


# ============================================================
# MAIN LOOP
# ============================================================

for _, row in failed_df.iterrows():

    variation_id = int(
        row["VariationID"]
    )

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

    exact_variant_id = make_exact_variant_id(
        chromosome,
        position,
        reference,
        alternate
    )


    print("\n")
    print("=" * 78)
    print(
        f"VariationID: {variation_id}"
    )
    print(
        f"Protein change: {protein_change}"
    )
    print(
        f"Exact requested ID: "
        f"{exact_variant_id}"
    )
    print(
        f"Reference allele: "
        f"{reference}"
    )
    print(
        f"Alternate allele: "
        f"{alternate}"
    )
    print(
        f"dbSNP rsID: "
        f"{rsid}"
    )
    print("=" * 78)


    exact_found = False

    # --------------------------------------------------------
    # RELEASE LOOP
    # --------------------------------------------------------

    for dataset in DATASETS:

        print(
            f"\nChecking dataset: "
            f"{dataset}"
        )

        result = query_gnomad(
            dataset,
            exact_variant_id
        )

        http_status = result[
            "http_status"
        ]

        variant = result[
            "variant"
        ]

        graphql_error = result[
            "error"
        ]


        print(
            f"HTTP status: "
            f"{http_status}"
        )


        # ----------------------------------------------------
        # GRAPHQL ERROR
        # ----------------------------------------------------

        if graphql_error:

            print(
                f"GraphQL error: "
                f"{graphql_error}"
            )

            results.append({

                "VariationID":
                    variation_id,

                "Protein_Change":
                    protein_change,

                "Chromosome":
                    chromosome,

                "Position_GRCh38":
                    position,

                "Reference":
                    reference,

                "Alternate":
                    alternate,

                "Exact_Requested_Variant_ID":
                    exact_variant_id,

                "RSID":
                    rsid,

                "Dataset":
                    dataset,

                "Returned_Variant_ID":
                    "",

                "Identity_Match":
                    False,

                "Status":
                    "GRAPHQL_OR_API_ERROR",

                "GraphQL_Error":
                    graphql_error,

                "HTTP_Status":
                    http_status,

                "Exome_AC":
                    None,

                "Exome_AN":
                    None,

                "Exome_AF":
                    None,

                "Genome_AC":
                    None,

                "Genome_AN":
                    None,

                "Genome_AF":
                    None
            })

            # Slow down after an API error
            time.sleep(BASE_SLEEP)

            continue


        # ----------------------------------------------------
        # NO VARIANT
        # ----------------------------------------------------

        if variant is None:

            print(
                "Variant not found "
                "in this dataset."
            )

            results.append({

                "VariationID":
                    variation_id,

                "Protein_Change":
                    protein_change,

                "Chromosome":
                    chromosome,

                "Position_GRCh38":
                    position,

                "Reference":
                    reference,

                "Alternate":
                    alternate,

                "Exact_Requested_Variant_ID":
                    exact_variant_id,

                "RSID":
                    rsid,

                "Dataset":
                    dataset,

                "Returned_Variant_ID":
                    "",

                "Identity_Match":
                    False,

                "Status":
                    "NOT_FOUND",

                "GraphQL_Error":
                    "",

                "HTTP_Status":
                    http_status,

                "Exome_AC":
                    None,

                "Exome_AN":
                    None,

                "Exome_AF":
                    None,

                "Genome_AC":
                    None,

                "Genome_AN":
                    None,

                "Genome_AF":
                    None
            })

            time.sleep(BASE_SLEEP)

            continue


        # ----------------------------------------------------
        # VARIANT RETURNED
        # ----------------------------------------------------

        returned_variant_id = variant.get(
            "variant_id"
        )


        print(
            f"Returned variant ID: "
            f"{returned_variant_id}"
        )


        # ----------------------------------------------------
        # STRICT IDENTITY
        # ----------------------------------------------------

        identity_match = exact_identity_match(
            exact_variant_id,
            returned_variant_id
        )


        print(
            f"Exact identity match: "
            f"{identity_match}"
        )


        # ----------------------------------------------------
        # IDENTITY MISMATCH
        # ----------------------------------------------------

        if not identity_match:

            print(
                "!!! IDENTITY MISMATCH — "
                "REJECTED !!!"
            )

            results.append({

                "VariationID":
                    variation_id,

                "Protein_Change":
                    protein_change,

                "Chromosome":
                    chromosome,

                "Position_GRCh38":
                    position,

                "Reference":
                    reference,

                "Alternate":
                    alternate,

                "Exact_Requested_Variant_ID":
                    exact_variant_id,

                "RSID":
                    rsid,

                "Dataset":
                    dataset,

                "Returned_Variant_ID":
                    returned_variant_id,

                "Identity_Match":
                    False,

                "Status":
                    "IDENTITY_MISMATCH",

                "GraphQL_Error":
                    "",

                "HTTP_Status":
                    http_status,

                "Exome_AC":
                    None,

                "Exome_AN":
                    None,

                "Exome_AF":
                    None,

                "Genome_AC":
                    None,

                "Genome_AN":
                    None,

                "Genome_AF":
                    None
            })

            time.sleep(BASE_SLEEP)

            continue


        # ----------------------------------------------------
        # EXACT MATCH
        # ----------------------------------------------------

        print(
            "!!! EXACT VARIANT MATCH !!!"
        )


        exome = (
            variant.get("exome")
            or {}
        )

        genome = (
            variant.get("genome")
            or {}
        )


        exome_ac = exome.get(
            "ac"
        )

        exome_an = exome.get(
            "an"
        )

        exome_af = exome.get(
            "af"
        )


        genome_ac = genome.get(
            "ac"
        )

        genome_an = genome.get(
            "an"
        )

        genome_af = genome.get(
            "af"
        )


        print(
            f"Exome: "
            f"AC={exome_ac} "
            f"AN={exome_an} "
            f"AF={exome_af}"
        )

        print(
            f"Genome: "
            f"AC={genome_ac} "
            f"AN={genome_an} "
            f"AF={genome_af}"
        )


        results.append({

            "VariationID":
                variation_id,

            "Protein_Change":
                protein_change,

            "Chromosome":
                chromosome,

            "Position_GRCh38":
                position,

            "Reference":
                reference,

            "Alternate":
                alternate,

            "Exact_Requested_Variant_ID":
                exact_variant_id,

            "RSID":
                rsid,

            "Dataset":
                dataset,

            "Returned_Variant_ID":
                returned_variant_id,

            "Identity_Match":
                True,

            "Status":
                "EXACT_MATCH",

            "GraphQL_Error":
                "",

            "HTTP_Status":
                http_status,

            "Exome_AC":
                exome_ac,

            "Exome_AN":
                exome_an,

            "Exome_AF":
                exome_af,

            "Genome_AC":
                genome_ac,

            "Genome_AN":
                genome_an,

            "Genome_AF":
                genome_af
        })


        exact_found = True

        # Stop release testing after exact match
        break


    # --------------------------------------------------------
    # NO EXACT MATCH IN ANY RELEASE
    # --------------------------------------------------------

    if not exact_found:

        print(
            "\nNo exact match found across "
            "the tested releases."
        )


        results.append({

            "VariationID":
                variation_id,

            "Protein_Change":
                protein_change,

            "Chromosome":
                chromosome,

            "Position_GRCh38":
                position,

            "Reference":
                reference,

            "Alternate":
                alternate,

            "Exact_Requested_Variant_ID":
                exact_variant_id,

            "RSID":
                rsid,

            "Dataset":
                "ALL_TESTED_RELEASES",

            "Returned_Variant_ID":
                "",

            "Identity_Match":
                False,

            "Status":
                "NOT_FOUND_IN_TESTED_RELEASES",

            "GraphQL_Error":
                "",

            "HTTP_Status":
                "",

            "Exome_AC":
                None,

            "Exome_AN":
                None,

            "Exome_AF":
                None,

            "Genome_AC":
                None,

            "Genome_AN":
                None,

            "Genome_AF":
                None
        })


    # Delay before next variant
    print(
        f"\nWaiting {BASE_SLEEP} seconds "
        f"before next variant..."
    )

    time.sleep(BASE_SLEEP)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

exact_matches = results_df[
    results_df["Status"]
    == "EXACT_MATCH"
]


unresolved = results_df[
    results_df["Status"]
    == "NOT_FOUND_IN_TESTED_RELEASES"
]


mismatches = results_df[
    results_df["Status"]
    == "IDENTITY_MISMATCH"
]


api_errors = results_df[
    results_df["Status"]
    == "GRAPHQL_OR_API_ERROR"
]


print("\n")
print("=" * 78)
print("FINAL 07g SUMMARY")
print("=" * 78)


print(
    f"Variants investigated: "
    f"{len(failed_df)}"
)

print(
    f"Exact matches: "
    f"{len(exact_matches)}"
)

print(
    f"Unresolved: "
    f"{len(unresolved)}"
)

print(
    f"Identity mismatches: "
    f"{len(mismatches)}"
)

print(
    f"GraphQL/API errors: "
    f"{len(api_errors)}"
)


# ============================================================
# EXACT MATCHES
# ============================================================

if len(exact_matches) > 0:

    print("\n")
    print("-" * 78)
    print("EXACT MATCHES")
    print("-" * 78)

    cols = [
        "VariationID",
        "Protein_Change",
        "Exact_Requested_Variant_ID",
        "Dataset",
        "Exome_AF",
        "Genome_AF"
    ]

    print(
        exact_matches[cols]
        .to_string(index=False)
    )


# ============================================================
# UNRESOLVED
# ============================================================

if len(unresolved) > 0:

    print("\n")
    print("-" * 78)
    print("STILL UNRESOLVED")
    print("-" * 78)

    cols = [
        "VariationID",
        "Protein_Change",
        "Exact_Requested_Variant_ID",
        "RSID"
    ]

    print(
        unresolved[cols]
        .to_string(index=False)
    )


# ============================================================
# API ERRORS
# ============================================================

if len(api_errors) > 0:

    print("\n")
    print("-" * 78)
    print("API / GRAPHQL ERRORS")
    print("-" * 78)

    cols = [
        "VariationID",
        "Dataset",
        "HTTP_Status",
        "GraphQL_Error"
    ]

    print(
        api_errors[cols]
        .to_string(index=False)
    )


# ============================================================
# SCIENTIFIC INTERPRETATION
# ============================================================

print("\n")
print("=" * 78)
print("SCIENTIFIC INTERPRETATION")
print("=" * 78)


if len(api_errors) > 0:

    print(
        "Some release checks could not be completed because "
        "of GraphQL/API errors. These variants MUST NOT be "
        "classified as absent from gnomAD."
    )


if len(unresolved) > 0:

    print(
        f"{len(unresolved)} variant(s) remain unresolved "
        "after successfully completed exact-identity checks."
    )

    print(
        "Unresolved variants must NOT be assigned AF = 0."
    )


if len(exact_matches) > 0:

    print(
        f"{len(exact_matches)} variant(s) were resolved "
        "using exact chromosome-position-REF-ALT identity."
    )


print(
    "\nDifferent ALT alleles at the same genomic position "
    "are never accepted as equivalent variants."
)


print(
    "\nExome and Genome population frequencies are retained "
    "as separate measurements."
)


print(
    f"\nOutput file:\n{OUTPUT_FILE}"
)


print("\n")
print("=" * 78)
print("07g GNOMAD RELEASE CHECK COMPLETE")
print("=" * 78)