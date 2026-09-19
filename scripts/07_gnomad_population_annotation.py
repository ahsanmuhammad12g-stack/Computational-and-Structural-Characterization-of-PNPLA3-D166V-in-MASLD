import pandas as pd
import requests
import time
from pathlib import Path


# ==================================================
# PROJECT PATHS
# ==================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_DIR
    / "04_VARIANT ANALYSIS"
    / "pnpla3_candidate_master.csv"
)

OUTPUT_DIR = (
    PROJECT_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
)

OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "pnpla3_gnomad_population_frequency.csv"
)


# ==================================================
# API SETTINGS
# ==================================================

ENSEMBL_URL = "https://rest.ensembl.org/variation/human"

GNOMAD_URL = "https://gnomad.broadinstitute.org/api"

HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json"
}


# ==================================================
# GNOMAD GRAPHQL QUERY
# ==================================================

GNOMAD_QUERY = """
query VariantSearch($variantId: String!, $dataset: DatasetId!) {
  variant(variantId: $variantId, dataset: $dataset) {
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


# ==================================================
# LOAD DATA
# ==================================================

print("=" * 70)
print("PNPLA3 GNOMAD POPULATION FREQUENCY ANNOTATION")
print("=" * 70)

print("\nLoading candidate variants...")

df = pd.read_csv(INPUT_FILE)

print(f"Total candidate variants: {len(df)}")


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def clean_rsid(value):

    if pd.isna(value):
        return None

    value = str(value).strip()

    if value in ["", "-", "-1", "nan", "None"]:
        return None

    try:
        value = str(int(float(value)))
    except Exception:
        pass

    if value.startswith("rs"):
        return value

    return f"rs{value}"


def get_ensembl_mapping(rsid):

    url = f"{ENSEMBL_URL}/{rsid}"

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code != 200:
            return None, f"Ensembl HTTP {response.status_code}"

        data = response.json()

        mappings = data.get("mappings", [])

        for mapping in mappings:

            if (
                mapping.get("assembly_name") == "GRCh38"
                and mapping.get("seq_region_name") == "22"
            ):
                return mapping, "OK"

        return None, "No GRCh38 chr22 mapping"

    except requests.exceptions.Timeout:
        return None, "Ensembl timeout"

    except requests.exceptions.RequestException as e:
        return None, f"Ensembl error: {str(e)}"


def query_gnomad(variant_id):

    variables = {
        "variantId": variant_id,
        "dataset": "gnomad_r4"
    }

    payload = {
        "query": GNOMAD_QUERY,
        "variables": variables
    }

    try:

        response = requests.post(
            GNOMAD_URL,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            return None, f"gnomAD HTTP {response.status_code}"

        data = response.json()

        if "errors" in data:
            return None, "gnomAD GraphQL error"

        variant = (
            data
            .get("data", {})
            .get("variant")
        )

        if variant is None:
            return None, "Variant not found"

        return variant, "OK"

    except requests.exceptions.Timeout:
        return None, "gnomAD timeout"

    except requests.exceptions.RequestException as e:
        return None, f"gnomAD error: {str(e)}"


# ==================================================
# PROCESS VARIANTS
# ==================================================

results = []

print("\nStarting annotation...\n")

for index, row in df.iterrows():

    number = index + 1
    total = len(df)

    variation_id = row.get("VariationID")

    rsid = clean_rsid(
        row.get("RS# (dbSNP)")
    )

    print(
        f"[{number:3}/{total}] "
        f"VariationID {variation_id} "
        f"({rsid if rsid else 'No rsID'})"
    )

    result = {
        "VariationID": variation_id,
        "Protein_Change": row.get("Protein_Change"),
        "Name": row.get("Name"),
        "ClinicalSignificance": row.get(
            "ClinicalSignificance"
        ),
        "Chromosome": row.get("Chromosome"),
        "Position": row.get("Start"),
        "rsID": rsid,
        "Ensembl_Status": None,
        "Ensembl_Allele_String": None,
        "gnomAD_Variant_ID": None,
        "gnomAD_Status": None,
        "Exome_AC": None,
        "Exome_AN": None,
        "Exome_AF": None,
        "Genome_AC": None,
        "Genome_AN": None,
        "Genome_AF": None
    }


    # ----------------------------------------------
    # rsID required for Ensembl allele lookup
    # ----------------------------------------------

    if rsid is None:

        result["Ensembl_Status"] = "No rsID"
        result["gnomAD_Status"] = "Not queried"

        results.append(result)

        print("   -> No rsID")

        continue


    # ----------------------------------------------
    # Get genomic mapping from Ensembl
    # ----------------------------------------------

    mapping, ensembl_status = get_ensembl_mapping(
        rsid
    )

    result["Ensembl_Status"] = ensembl_status


    if mapping is None:

        result["gnomAD_Status"] = "Not queried"

        results.append(result)

        print(f"   -> {ensembl_status}")

        continue


    allele_string = mapping.get(
        "allele_string"
    )

    result[
        "Ensembl_Allele_String"
    ] = allele_string


    # ----------------------------------------------
    # Extract alleles
    # ----------------------------------------------

    alleles = allele_string.split("/")

    reference = alleles[0]

    alternate_alleles = alleles[1:]


    # ----------------------------------------------
    # Query each alternate allele
    # ----------------------------------------------

    found_variant = False

    for alternate in alternate_alleles:

        variant_id = (
            f"22-"
            f"{mapping['start']}-"
            f"{reference}-"
            f"{alternate}"
        )

        print(
            f"   Testing {variant_id}"
        )

        gnomad_data, gnomad_status = (
            query_gnomad(variant_id)
        )


        if gnomad_data is not None:

            result[
                "gnomAD_Variant_ID"
            ] = gnomad_data.get(
                "variant_id"
            )

            result[
                "gnomAD_Status"
            ] = "Found"

            exome = (
                gnomad_data.get("exome")
                or {}
            )

            genome = (
                gnomad_data.get("genome")
                or {}
            )


            result["Exome_AC"] = exome.get("ac")
            result["Exome_AN"] = exome.get("an")
            result["Exome_AF"] = exome.get("af")

            result["Genome_AC"] = genome.get("ac")
            result["Genome_AN"] = genome.get("an")
            result["Genome_AF"] = genome.get("af")

            found_variant = True

            print(
                f"   -> Found "
                f"(Exome AF: {result['Exome_AF']})"
            )

            break


        else:

            result[
                "gnomAD_Status"
            ] = gnomad_status


    if not found_variant:

        print(
            f"   -> {result['gnomAD_Status']}"
        )


    results.append(result)


    # ----------------------------------------------
    # Be polite to APIs
    # ----------------------------------------------

    time.sleep(0.2)


# ==================================================
# SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ==================================================
# SUMMARY
# ==================================================

print("\n" + "=" * 70)
print("GNOMAD POPULATION FREQUENCY SUMMARY")
print("=" * 70)

print(f"\nInput variants: {len(results_df)}")

print(
    "\nSuccessfully found in gnomAD:",
    (results_df["gnomAD_Status"] == "Found").sum()
)

print(
    "\nVariants without rsID:",
    results_df["rsID"].isna().sum()
)

print("\nStatus distribution:")

print(
    results_df["gnomAD_Status"]
    .value_counts(dropna=False)
)


print("\nOutput saved to:")

print(OUTPUT_FILE)

print("\nPopulation frequency annotation completed.")