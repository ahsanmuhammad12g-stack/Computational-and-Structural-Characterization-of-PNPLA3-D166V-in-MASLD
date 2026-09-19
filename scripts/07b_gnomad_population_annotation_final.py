import pandas as pd
import requests
import time
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = Path(
    r"04_VARIANT ANALYSIS\pnpla3_candidate_master.csv"
)

OUTPUT_DIR = Path(
    r"04_VARIANT ANALYSIS\annotations"
)

OUTPUT_FILE = OUTPUT_DIR / "pnpla3_gnomad_population_frequency_final.csv"


GNOMAD_API = "https://gnomad.broadinstitute.org/api"

HEADERS = {
    "Content-Type": "application/json"
}


# ============================================================
# GNOMAD GRAPHQL QUERY
# ============================================================

QUERY = """
query VariantQuery($variantId: String!, $datasetId: DatasetId!) {
  variant(variantId: $variantId, dataset: $datasetId) {
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
# FUNCTION: QUERY GNOMAD WITH RETRIES
# ============================================================

def query_gnomad(variant_id, max_retries=5):

    variables = {
        "variantId": variant_id,
        "datasetId": "gnomad_r4"
    }

    payload = {
        "query": QUERY,
        "variables": variables
    }

    for attempt in range(1, max_retries + 1):

        try:

            response = requests.post(
                GNOMAD_API,
                json=payload,
                headers=HEADERS,
                timeout=30
            )

            # Rate limiting
            if response.status_code == 429:

                wait_time = attempt * 5

                print(
                    f"      Rate limited. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

                continue


            # Other HTTP errors
            if response.status_code != 200:

                return {
                    "status": f"HTTP {response.status_code}",
                    "data": None
                }


            result = response.json()


            # GraphQL error
            if "errors" in result:

                return {
                    "status": "GraphQL error",
                    "data": None
                }


            variant_data = result.get(
                "data",
                {}
            ).get(
                "variant"
            )


            if variant_data is None:

                return {
                    "status": "Not found",
                    "data": None
                }


            return {
                "status": "Found",
                "data": variant_data
            }


        except requests.exceptions.Timeout:

            print(
                f"      Timeout (attempt {attempt}/{max_retries})"
            )

            time.sleep(attempt * 3)


        except requests.exceptions.RequestException as e:

            print(
                f"      Connection error "
                f"(attempt {attempt}/{max_retries})"
            )

            time.sleep(attempt * 3)


    return {
        "status": "Failed after retries",
        "data": None
    }


# ============================================================
# FUNCTION: GET ALLELES FROM ENSEMBL VARIATION API
# ============================================================

def get_ensembl_variant_info(rsid):

    if pd.isna(rsid) or str(rsid) in ["-1", "", "nan"]:

        return None


    rsid = str(rsid)

    if not rsid.startswith("rs"):

        rsid = "rs" + rsid


    url = (
        "https://rest.ensembl.org/"
        f"variation/human/{rsid}"
    )


    headers = {
        "Content-Type": "application/json"
    }


    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=30
        )


        if response.status_code != 200:

            return None


        data = response.json()


        mappings = data.get("mappings", [])


        for mapping in mappings:

            if (
                mapping.get("assembly_name") == "GRCh38"
                and mapping.get("seq_region_name") == "22"
            ):

                return mapping


    except requests.exceptions.RequestException:

        return None


    return None


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PNPLA3 FINAL GNOMAD POPULATION FREQUENCY ANNOTATION")
print("=" * 70)

print("\nLoading candidate variants...")

df = pd.read_csv(INPUT_FILE)

print(f"Total candidate variants: {len(df)}")


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# RESULT STORAGE
# ============================================================

results = []


# ============================================================
# PROCESS EACH VARIANT
# ============================================================

print("\nStarting annotation...\n")


for index, row in df.iterrows():

    number = index + 1
    total = len(df)

    variation_id = row["VariationID"]

    chromosome = str(row["Chromosome"])

    position = int(row["Start"])

    rsid_raw = row["RS# (dbSNP)"]


    # --------------------------------------------------------
    # DISPLAY VARIANT
    # --------------------------------------------------------

    if (
        pd.isna(rsid_raw)
        or str(rsid_raw) in ["-1", "nan"]
    ):

        rsid_display = "No rsID"

    else:

        rsid_display = f"rs{rsid_raw}"


    print(
        f"[{number:3}/{total}] "
        f"VariationID {variation_id} "
        f"({rsid_display})"
    )


    # --------------------------------------------------------
    # GET ALLELE INFORMATION
    # --------------------------------------------------------

    mapping = None


    if rsid_display != "No rsID":

        mapping = get_ensembl_variant_info(
            rsid_raw
        )


    # --------------------------------------------------------
    # IF RSID AVAILABLE AND ENSEMBL RETURNS ALLELES
    # --------------------------------------------------------

    if mapping:

        allele_string = mapping.get(
            "allele_string"
        )

        print(
            f"   Ensembl allele string: "
            f"{allele_string}"
        )


        alleles = allele_string.split("/")

        ref = alleles[0]

        alternates = alleles[1:]


        variant_found = False


        for alt in alternates:

            variant_identifier = (
                f"{chromosome}-"
                f"{position}-"
                f"{ref}-"
                f"{alt}"
            )


            print(
                f"   Testing {variant_identifier}"
            )


            response = query_gnomad(
                variant_identifier
            )


            time.sleep(1.5)


            if response["status"] == "Found":

                data = response["data"]

                exome = data.get("exome") or {}

                genome = data.get("genome") or {}


                exome_af = exome.get("af")

                genome_af = genome.get("af")


                # A variant may exist but have no frequency
                # Continue testing other alternate alleles.

                if (
                    exome_af is not None
                    or genome_af is not None
                ):

                    print(
                        "   -> Found population frequency"
                    )

                    results.append({

                        "VariationID":
                            variation_id,

                        "Name":
                            row["Name"],

                        "Protein_Change":
                            row["Protein_Change"],

                        "Candidate_Group":
                            row["Candidate_Group"],

                        "Chromosome":
                            chromosome,

                        "Position_GRCh38":
                            position,

                        "rsID":
                            rsid_display,

                        "Reference":
                            ref,

                        "Alternate":
                            alt,

                        "gnomAD_Variant_ID":
                            variant_identifier,

                        "Exome_AC":
                            exome.get("ac"),

                        "Exome_AN":
                            exome.get("an"),

                        "Exome_AF":
                            exome_af,

                        "Genome_AC":
                            genome.get("ac"),

                        "Genome_AN":
                            genome.get("an"),

                        "Genome_AF":
                            genome_af,

                        "gnomAD_Status":
                            "Found"
                    })


                    variant_found = True

                    break


        if not variant_found:

            results.append({

                "VariationID":
                    variation_id,

                "Name":
                    row["Name"],

                "Protein_Change":
                    row["Protein_Change"],

                "Candidate_Group":
                    row["Candidate_Group"],

                "Chromosome":
                    chromosome,

                "Position_GRCh38":
                    position,

                "rsID":
                    rsid_display,

                "Reference":
                    None,

                "Alternate":
                    None,

                "gnomAD_Variant_ID":
                    None,

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
                    None,

                "gnomAD_Status":
                    "Not found / no frequency"
            })


    # --------------------------------------------------------
    # NO RSID AVAILABLE
    # --------------------------------------------------------

    else:

        print(
            "   No rsID available."
        )

        print(
            "   Marked for coordinate-based follow-up."
        )


        results.append({

            "VariationID":
                variation_id,

            "Name":
                row["Name"],

            "Protein_Change":
                row["Protein_Change"],

            "Candidate_Group":
                row["Candidate_Group"],

            "Chromosome":
                chromosome,

            "Position_GRCh38":
                position,

            "rsID":
                "No rsID",

            "Reference":
                None,

            "Alternate":
                None,

            "gnomAD_Variant_ID":
                None,

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
                None,

            "gnomAD_Status":
                "Requires coordinate mapping"
        })


# ============================================================
# CREATE OUTPUT
# ============================================================

results_df = pd.DataFrame(results)


results_df.to_csv(
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

print(
    f"Output variants: {len(results_df)}"
)


print("\nStatus distribution:")

print(
    results_df[
        "gnomAD_Status"
    ].value_counts()
)


print(
    f"\nOutput saved to:\n{OUTPUT_FILE}"
)


print("\nPopulation frequency annotation completed.")