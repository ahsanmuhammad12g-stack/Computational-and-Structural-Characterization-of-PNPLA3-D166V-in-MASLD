import pandas as pd
import requests
import time
from pathlib import Path


print("=" * 70)
print("FINAL PNPLA3 GNOMAD POPULATION FREQUENCY ANNOTATION")
print("=" * 70)


# ============================================================
# FILE PATHS
# ============================================================

INPUT_FILE = Path(
    r"04_VARIANT ANALYSIS\annotations\pnpla3_exact_variant_mapping.csv"
)

OUTPUT_FILE = Path(
    r"04_VARIANT ANALYSIS\annotations\pnpla3_gnomad_final.csv"
)

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading exact PNPLA3 variant mapping...")

df = pd.read_csv(INPUT_FILE)

print(f"Total variants: {len(df)}")


# ============================================================
# GNOMAD API SETTINGS
# ============================================================

GNOMAD_URL = "https://gnomad.broadinstitute.org/api"

DATASET = "gnomad_r4"


# IMPORTANT:
# variantId MUST be supplied to the variant() field.
QUERY = """
query VariantQuery($variantId: String!, $dataset: DatasetId!) {
  variant(
    variantId: $variantId
    dataset: $dataset
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
# GNOMAD QUERY FUNCTION
# ============================================================

def query_gnomad(
    variant_id,
    max_retries=5
):

    variables = {
        "variantId": variant_id,
        "dataset": DATASET
    }

    payload = {
        "query": QUERY,
        "variables": variables
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "PNPLA3-MASLD-Variant-Analysis"
    }


    for attempt in range(1, max_retries + 1):

        try:

            response = requests.post(
                GNOMAD_URL,
                json=payload,
                headers=headers,
                timeout=30
            )


            # =================================================
            # HTTP 200
            # =================================================

            if response.status_code == 200:

                try:
                    data = response.json()

                except ValueError:

                    return {
                        "status": "Invalid JSON response",
                        "data": None,
                        "error": response.text[:500]
                    }


                # ---------------------------------------------
                # GRAPHQL ERRORS
                # ---------------------------------------------

                if "errors" in data:

                    errors = data.get("errors", [])

                    error_messages = []

                    for error in errors:

                        message = error.get(
                            "message",
                            "Unknown GraphQL error"
                        )

                        error_messages.append(message)


                    error_text = " | ".join(
                        error_messages
                    )


                    return {
                        "status": "GraphQL error",
                        "data": None,
                        "error": error_text
                    }


                # ---------------------------------------------
                # EXTRACT VARIANT
                # ---------------------------------------------

                variant_data = (
                    data
                    .get("data", {})
                    .get("variant")
                )


                # ---------------------------------------------
                # VARIANT NOT PRESENT
                # ---------------------------------------------

                if variant_data is None:

                    return {
                        "status": "Not found / no frequency",
                        "data": None,
                        "error": None
                    }


                # ---------------------------------------------
                # SUCCESS
                # ---------------------------------------------

                return {
                    "status": "Found",
                    "data": variant_data,
                    "error": None
                }


            # =================================================
            # RATE LIMIT 429
            # =================================================

            elif response.status_code == 429:

                if attempt < max_retries:

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if retry_after:

                        try:
                            wait_time = float(
                                retry_after
                            )

                        except ValueError:
                            wait_time = attempt * 10

                    else:

                        wait_time = attempt * 10


                    print(
                        f"   Rate limited (429). "
                        f"Waiting {wait_time:.0f} seconds..."
                    )

                    time.sleep(wait_time)

                    continue


                else:

                    return {
                        "status": "HTTP 429",
                        "data": None,
                        "error": "Rate limit exceeded after retries"
                    }


            # =================================================
            # BAD REQUEST 400
            # =================================================

            elif response.status_code == 400:

                try:

                    error_json = response.json()

                    error_text = str(
                        error_json
                    )

                except Exception:

                    error_text = response.text[:1000]


                return {
                    "status": "HTTP 400",
                    "data": None,
                    "error": error_text
                }


            # =================================================
            # SERVER ERRORS
            # =================================================

            elif response.status_code in [500, 502, 503, 504]:

                if attempt < max_retries:

                    wait_time = attempt * 10

                    print(
                        f"   Server error "
                        f"({response.status_code}). "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue


                return {
                    "status": f"HTTP {response.status_code}",
                    "data": None,
                    "error": response.text[:500]
                }


            # =================================================
            # OTHER HTTP ERRORS
            # =================================================

            else:

                return {
                    "status": f"HTTP {response.status_code}",
                    "data": None,
                    "error": response.text[:500]
                }


        # =====================================================
        # TIMEOUT
        # =====================================================

        except requests.exceptions.Timeout:

            if attempt < max_retries:

                wait_time = attempt * 5

                print(
                    f"   Timeout. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue


            return {
                "status": "Timeout",
                "data": None,
                "error": "Request timed out"
            }


        # =====================================================
        # CONNECTION ERROR
        # =====================================================

        except requests.exceptions.ConnectionError:

            if attempt < max_retries:

                wait_time = attempt * 5

                print(
                    f"   Connection error. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue


            return {
                "status": "Connection error",
                "data": None,
                "error": "Connection failed after retries"
            }


        # =====================================================
        # UNEXPECTED ERROR
        # =====================================================

        except Exception as e:

            return {
                "status": "Unexpected error",
                "data": None,
                "error": str(e)
            }


    return {
        "status": "Failed after retries",
        "data": None,
        "error": "Maximum retries exceeded"
    }


# ============================================================
# CREATE OUTPUT COLUMNS
# ============================================================

df["gnomAD_rsIDs"] = None

df["gnomAD_variant_id_returned"] = None

df["Exome_AC"] = None
df["Exome_AN"] = None
df["Exome_AF"] = None

df["Genome_AC"] = None
df["Genome_AN"] = None
df["Genome_AF"] = None

df["gnomAD_Status"] = None

df["gnomAD_Error"] = None


# ============================================================
# VALIDATE INPUT
# ============================================================

required_column = "gnomAD_Variant_ID"

if required_column not in df.columns:

    raise ValueError(
        f"Required column '{required_column}' "
        f"not found in input file."
    )


# ============================================================
# START ANNOTATION
# ============================================================

print("\nStarting gnomAD annotation...")
print(f"Dataset: {DATASET}")
print(f"API: {GNOMAD_URL}")
print("-" * 70)


total = len(df)


for index, row in df.iterrows():

    variant_id = row["gnomAD_Variant_ID"]


    # --------------------------------------------------------
    # CHECK VARIANT ID
    # --------------------------------------------------------

    if pd.isna(variant_id) or str(variant_id).strip() == "":

        df.at[index, "gnomAD_Status"] = (
            "Missing variant ID"
        )

        print(
            f"[{index + 1:>3}/{total}] "
            f"VariationID {row['VariationID']}"
        )

        print("   -> Missing variant ID")

        continue


    variant_id = str(
        variant_id
    ).strip()


    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print(
        f"[{index + 1:>3}/{total}] "
        f"VariationID {row['VariationID']} "
        f"({variant_id})"
    )


    # --------------------------------------------------------
    # QUERY GNOMAD
    # --------------------------------------------------------

    result = query_gnomad(
        variant_id
    )


    status = result["status"]

    df.at[index, "gnomAD_Status"] = status


    # Store API error if present
    if result.get("error"):

        df.at[index, "gnomAD_Error"] = (
            result["error"]
        )


    # ========================================================
    # SUCCESS
    # ========================================================

    if status == "Found":

        variant_data = result["data"]


        # ----------------------------------------------------
        # Returned variant ID
        # ----------------------------------------------------

        returned_variant_id = (
            variant_data.get(
                "variant_id"
            )
        )

        df.at[
            index,
            "gnomAD_variant_id_returned"
        ] = returned_variant_id


        # ----------------------------------------------------
        # rsIDs
        # ----------------------------------------------------

        rsids = variant_data.get(
            "rsids"
        )


        if rsids:

            df.at[
                index,
                "gnomAD_rsIDs"
            ] = ";".join(
                map(str, rsids)
            )


        # ----------------------------------------------------
        # EXOME
        # ----------------------------------------------------

        exome = variant_data.get(
            "exome"
        )


        if exome:

            df.at[
                index,
                "Exome_AC"
            ] = exome.get("ac")

            df.at[
                index,
                "Exome_AN"
            ] = exome.get("an")

            df.at[
                index,
                "Exome_AF"
            ] = exome.get("af")


        # ----------------------------------------------------
        # GENOME
        # ----------------------------------------------------

        genome = variant_data.get(
            "genome"
        )


        if genome:

            df.at[
                index,
                "Genome_AC"
            ] = genome.get("ac")

            df.at[
                index,
                "Genome_AN"
            ] = genome.get("an")

            df.at[
                index,
                "Genome_AF"
            ] = genome.get("af")


        # ----------------------------------------------------
        # DISPLAY RESULT
        # ----------------------------------------------------

        print(
            f"   -> Found "
            f"(Exome AF: "
            f"{df.at[index, 'Exome_AF']}, "
            f"Genome AF: "
            f"{df.at[index, 'Genome_AF']})"
        )


    # ========================================================
    # NOT FOUND
    # ========================================================

    elif status == "Not found / no frequency":

        print(
            "   -> Not found / no frequency"
        )


    # ========================================================
    # GRAPHQL ERROR
    # ========================================================

    elif status == "GraphQL error":

        print(
            "   -> GraphQL error"
        )

        if result.get("error"):

            print(
                f"      {result['error']}"
            )


    # ========================================================
    # OTHER ERROR
    # ========================================================

    else:

        print(
            f"   -> {status}"
        )


    # --------------------------------------------------------
    # DELAY
    # --------------------------------------------------------

    # Conservative delay to reduce rate limiting.
    time.sleep(2.0)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL GNOMAD POPULATION FREQUENCY SUMMARY")
print("=" * 70)


print(
    f"\nInput variants: {len(df)}"
)


# ============================================================
# STATUS DISTRIBUTION
# ============================================================

print("\nStatus distribution:")

print(
    df["gnomAD_Status"]
    .value_counts(dropna=False)
)


# ============================================================
# SUCCESS COUNT
# ============================================================

found_count = (
    df["gnomAD_Status"]
    == "Found"
).sum()


print(
    f"\nSuccessfully found: "
    f"{found_count}"
)


# ============================================================
# NOT FOUND COUNT
# ============================================================

not_found_count = (
    df["gnomAD_Status"]
    == "Not found / no frequency"
).sum()


print(
    f"Not found / no frequency: "
    f"{not_found_count}"
)


# ============================================================
# ERROR COUNT
# ============================================================

error_statuses = [
    "Found",
    "Not found / no frequency"
]


error_count = (
    ~df["gnomAD_Status"]
    .isin(error_statuses)
).sum()


print(
    f"Errors: {error_count}"
)


# ============================================================
# FREQUENCY SUMMARY
# ============================================================

print("\nFrequency availability:")


exome_available = (
    df["Exome_AF"]
    .notna()
).sum()


genome_available = (
    df["Genome_AF"]
    .notna()
).sum()


print(
    f"Variants with Exome AF: "
    f"{exome_available}"
)

print(
    f"Variants with Genome AF: "
    f"{genome_available}"
)


# ============================================================
# IMPORTANT VARIANT CHECK
# ============================================================

positive_control = df[
    df["gnomAD_Variant_ID"]
    == "22-43928847-C-G"
]


if len(positive_control) > 0:

    print(
        "\nPositive-control check "
        "(rs738409 / 22-43928847-C-G):"
    )

    print(
        positive_control[
            [
                "VariationID",
                "gnomAD_Variant_ID",
                "gnomAD_variant_id_returned",
                "gnomAD_rsIDs",
                "Exome_AC",
                "Exome_AN",
                "Exome_AF",
                "Genome_AC",
                "Genome_AN",
                "Genome_AF",
                "gnomAD_Status"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# FIRST 15 RESULTS
# ============================================================

print(
    "\nFirst 15 annotated variants:"
)


display_columns = [

    "VariationID",
    "Protein_Change",
    "Candidate_Group",

    "gnomAD_Variant_ID",
    "gnomAD_variant_id_returned",
    "gnomAD_rsIDs",

    "Exome_AC",
    "Exome_AN",
    "Exome_AF",

    "Genome_AC",
    "Genome_AN",
    "Genome_AF",

    "gnomAD_Status"
]


print(
    df[display_columns]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# OUTPUT
# ============================================================

print("\nOutput saved to:")

print(
    OUTPUT_FILE.resolve()
)


print(
    "\nPopulation frequency annotation completed."
)

print("=" * 70)