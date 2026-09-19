import pandas as pd
import requests
import time
from pathlib import Path


print("=" * 80)
print("SECOND-LEVEL RESOLUTION OF UNRESOLVED PNPLA3 GNOMAD VARIANTS")
print("=" * 80)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_exact_variant_mapping.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
    / "pnpla3_failed_gnomad_resolution.csv"
)


# ============================================================
# APIs
# ============================================================

GNOMAD_API = "https://gnomad.broadinstitute.org/api"

ENSEMBL_VARIATION_API = (
    "https://rest.ensembl.org/variation/human"
)

ENSEMBL_REGION_API = (
    "https://rest.ensembl.org/overlap/region/human"
)


# ============================================================
# VARIANTS THAT REMAINED UNRESOLVED IN GNOMAD R4
# ============================================================

FAILED_VARIANTS = [
    3421690,
    4140671,
    899624,
    899627,
    900762,
    4140672,
    3216047,
    902436,
    3308174,
]


# ============================================================
# GNOMAD GRAPHQL QUERY
# ============================================================

GNOMAD_QUERY = """
query VariantQuery($variantId: String!) {
    variant(
        dataset: gnomad_r4,
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
# READ INPUT
# ============================================================

print("\nReading mapping file:")
print(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

print(f"Total mapping rows: {len(df)}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

REQUIRED_COLUMNS = [
    "VariationID",
    "Protein_Change",
    "Chromosome",
    "Position_GRCh38",
    "Reference",
    "Alternate",
    "RS# (dbSNP)",
    "gnomAD_Variant_ID",
]

missing = [
    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns
]

if missing:
    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing)
    )


# ============================================================
# SELECT THE 9 FAILED VARIANTS
# ============================================================

failed_df = df[
    df["VariationID"].isin(FAILED_VARIANTS)
].copy()

failed_df = failed_df.drop_duplicates(
    subset=["VariationID"]
)

print(
    f"Failed variants found in mapping file: "
    f"{len(failed_df)}/{len(FAILED_VARIANTS)}"
)


# ============================================================
# GNOMAD QUERY FUNCTION
# ============================================================

def query_gnomad(variant_id):

    payload = {
        "query": GNOMAD_QUERY,
        "variables": {
            "variantId": variant_id
        }
    }

    try:

        response = requests.post(
            GNOMAD_API,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:

            return {
                "status": f"HTTP {response.status_code}",
                "variant": None,
                "error": response.text[:1000]
            }

        data = response.json()

        if "errors" in data:

            return {
                "status": "GraphQL_ERROR",
                "variant": None,
                "error": str(data["errors"])
            }

        variant = (
            data
            .get("data", {})
            .get("variant")
        )

        if variant is None:

            return {
                "status": "NOT_FOUND",
                "variant": None,
                "error": None
            }

        return {
            "status": "FOUND",
            "variant": variant,
            "error": None
        }

    except Exception as e:

        return {
            "status": "REQUEST_ERROR",
            "variant": None,
            "error": str(e)
        }


# ============================================================
# ENSEMBL RSID LOOKUP
# ============================================================

def query_ensembl_rsid(rsid):

    if pd.isna(rsid):
        return None

    rsid = str(rsid).strip()

    if rsid in ["", "-1", "nan", "None"]:
        return None

    if not rsid.startswith("rs"):
        rsid = "rs" + rsid

    url = f"{ENSEMBL_VARIATION_API}/{rsid}"

    headers = {
        "Content-Type": "application/json"
    }

    try:

        response = requests.get(
            url,
            headers=headers,
            timeout=60
        )

        if response.status_code != 200:
            return None

        return response.json()

    except Exception:
        return None


# ============================================================
# EXTRACT ENSEMBL GRCh38 MAPPINGS
# ============================================================

def extract_grch38_mappings(
    variation_data,
    chromosome,
    position,
    reference,
    alternate
):

    mappings = []

    if not variation_data:
        return mappings

    for mapping in variation_data.get(
        "mappings", []
    ):

        assembly = mapping.get(
            "assembly_name"
        )

        if assembly != "GRCh38":
            continue

        seq_region = str(
            mapping.get("seq_region_name", "")
        )

        start = mapping.get("start")

        allele_string = mapping.get(
            "allele_string",
            ""
        )

        strand = mapping.get(
            "strand"
        )

        # Only retain chromosome 22 mappings
        if seq_region not in [
            str(chromosome),
            f"chr{chromosome}"
        ]:
            continue

        mappings.append({
            "Ensembl_Chromosome": seq_region,
            "Ensembl_Position": start,
            "Ensembl_Allele_String": allele_string,
            "Ensembl_Strand": strand
        })

    return mappings


# ============================================================
# ENSEMBL POSITION LOOKUP
# FOR VARIANTS WITHOUT AN rsID
# ============================================================

def query_ensembl_region(
    chromosome,
    position
):

    region = (
        f"{chromosome}:{position}-{position}"
    )

    url = f"{ENSEMBL_REGION_API}/{region}"

    params = {
        "feature": "variation"
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=60
        )

        if response.status_code != 200:
            return []

        return response.json()

    except Exception:
        return []


# ============================================================
# MAIN RESOLUTION
# ============================================================

results = []

print("\n")
print("=" * 80)
print("BEGINNING SECOND-LEVEL RESOLUTION")
print("=" * 80)


for _, row in failed_df.iterrows():

    variation_id = int(
        row["VariationID"]
    )

    chromosome = str(
        row["Chromosome"]
    ).replace(
        "chr",
        ""
    )

    position = int(
        row["Position_GRCh38"]
    )

    reference = str(
        row["Reference"]
    ).strip().upper()

    alternate = str(
        row["Alternate"]
    ).strip().upper()

    protein_change = str(
        row["Protein_Change"]
    )

    rsid = row["RS# (dbSNP)"]

    original_gnomad_id = str(
        row["gnomAD_Variant_ID"]
    )

    exact_variant_id = (
        f"{chromosome}-"
        f"{position}-"
        f"{reference}-"
        f"{alternate}"
    )

    print("\n" + "-" * 80)
    print(f"VariationID : {variation_id}")
    print(f"Protein     : {protein_change}")
    print(
        f"GRCh38      : "
        f"{chromosome}-{position}-"
        f"{reference}-{alternate}"
    )
    print(f"rsID        : {rsid}")
    print(
        f"Original ID : "
        f"{original_gnomad_id}"
    )

    # --------------------------------------------------------
    # STEP 1 — EXACT GNOMAD QUERY
    # --------------------------------------------------------

    print("\n[1] Exact gnomAD query...")

    exact_result = query_gnomad(
        exact_variant_id
    )

    print(
        "Result:",
        exact_result["status"]
    )

    if exact_result["status"] == "FOUND":

        variant = exact_result["variant"]

        exome = variant.get(
            "exome"
        ) or {}

        genome = variant.get(
            "genome"
        ) or {}

        results.append({

            "VariationID": variation_id,
            "Protein_Change": protein_change,

            "Chromosome": chromosome,
            "Position_GRCh38": position,
            "Reference": reference,
            "Alternate": alternate,

            "Input_rsID": (
                str(rsid)
                if not pd.isna(rsid)
                else ""
            ),

            "Tested_gnomAD_ID":
                exact_variant_id,

            "Resolved_gnomAD_ID":
                variant.get(
                    "variant_id"
                ),

            "Resolved_rsIDs":
                ";".join(
                    variant.get(
                        "rsids"
                    ) or []
                ),

            "Exome_AC":
                exome.get("ac"),

            "Exome_AN":
                exome.get("an"),

            "Exome_AF":
                exome.get("af"),

            "Genome_AC":
                genome.get("ac"),

            "Genome_AN":
                genome.get("an"),

            "Genome_AF":
                genome.get("af"),

            "Resolution_Method":
                "Exact GRCh38 coordinate",

            "Final_Status":
                "FOUND",

            "Error": None
        })

        time.sleep(3)
        continue

    # --------------------------------------------------------
    # STEP 2 — RSID THROUGH ENSEMBL
    # --------------------------------------------------------

    ensembl_data = None

    if (
        not pd.isna(rsid)
        and str(rsid).strip()
        not in ["", "-1", "nan"]
    ):

        print(
            "\n[2] Querying Ensembl using rsID..."
        )

        ensembl_data = query_ensembl_rsid(
            rsid
        )

        if ensembl_data:

            print(
                "Ensembl rsID lookup: SUCCESS"
            )

        else:

            print(
                "Ensembl rsID lookup: NOT FOUND"
            )

    else:

        print(
            "\n[2] No rsID available."
        )

    # --------------------------------------------------------
    # EXTRACT GRCh38 MAPPINGS
    # --------------------------------------------------------

    mappings = extract_grch38_mappings(
        ensembl_data,
        chromosome,
        position,
        reference,
        alternate
    )

    print(
        f"GRCh38 Ensembl mappings: "
        f"{len(mappings)}"
    )

    # --------------------------------------------------------
    # TRY GNOMAD IDS GENERATED FROM ENSEMBL
    # --------------------------------------------------------

    found = False

    for mapping in mappings:

        ens_chr = str(
            mapping[
                "Ensembl_Chromosome"
            ]
        ).replace(
            "chr",
            ""
        )

        ens_pos = mapping[
            "Ensembl_Position"
        ]

        allele_string = mapping[
            "Ensembl_Allele_String"
        ]

        if not allele_string:
            continue

        alleles = allele_string.split("/")

        if len(alleles) != 2:
            continue

        ens_ref = alleles[0].upper()
        ens_alt = alleles[1].upper()

        # Only try if mapping is at expected position
        if (
            ens_chr != chromosome
            or ens_pos != position
        ):
            continue

        candidate_ids = [
            (
                f"{ens_chr}-{ens_pos}-"
                f"{ens_ref}-{ens_alt}"
            ),
            (
                f"{ens_chr}-{ens_pos}-"
                f"{ens_alt}-{ens_ref}"
            )
        ]

        for candidate_id in candidate_ids:

            print(
                "Trying Ensembl-derived ID:",
                candidate_id
            )

            candidate_result = query_gnomad(
                candidate_id
            )

            if (
                candidate_result["status"]
                == "FOUND"
            ):

                variant = (
                    candidate_result[
                        "variant"
                    ]
                )

                exome = (
                    variant.get(
                        "exome"
                    ) or {}
                )

                genome = (
                    variant.get(
                        "genome"
                    ) or {}
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

                    "Input_rsID":
                        str(rsid)
                        if not pd.isna(rsid)
                        else "",

                    "Tested_gnomAD_ID":
                        exact_variant_id,

                    "Resolved_gnomAD_ID":
                        variant.get(
                            "variant_id"
                        ),

                    "Resolved_rsIDs":
                        ";".join(
                            variant.get(
                                "rsids"
                            ) or []
                        ),

                    "Exome_AC":
                        exome.get("ac"),

                    "Exome_AN":
                        exome.get("an"),

                    "Exome_AF":
                        exome.get("af"),

                    "Genome_AC":
                        genome.get("ac"),

                    "Genome_AN":
                        genome.get("an"),

                    "Genome_AF":
                        genome.get("af"),

                    "Resolution_Method":
                        "Ensembl rsID mapping",

                    "Final_Status":
                        "FOUND",

                    "Error": None
                })

                found = True
                break

            time.sleep(3)

        if found:
            break

    if found:
        continue

    # --------------------------------------------------------
    # STEP 3 — POSITION-BASED ENSEMBL SEARCH
    # --------------------------------------------------------

    print(
        "\n[3] Position-based Ensembl search..."
    )

    region_variants = query_ensembl_region(
        chromosome,
        position
    )

    print(
        f"Ensembl variants at position: "
        f"{len(region_variants)}"
    )

    position_candidates = []

    for item in region_variants:

        start = item.get("start")

        if start != position:
            continue

        alleles = item.get(
            "alleles",
            []
        )

        if isinstance(alleles, list):

            for allele in alleles:

                if isinstance(
                    allele,
                    str
                ):
                    position_candidates.append(
                        allele.upper()
                    )

    position_candidates = list(
        dict.fromkeys(
            position_candidates
        )
    )

    print(
        "Alleles detected:",
        position_candidates
    )

    # --------------------------------------------------------
    # TRY CANDIDATE ALLELES
    # --------------------------------------------------------

    for candidate_alt in position_candidates:

        if candidate_alt in [
            reference,
            alternate
        ]:
            continue

        candidate_id = (
            f"{chromosome}-"
            f"{position}-"
            f"{reference}-"
            f"{candidate_alt}"
        )

        print(
            "Trying position-derived ID:",
            candidate_id
        )

        candidate_result = query_gnomad(
            candidate_id
        )

        if (
            candidate_result["status"]
            == "FOUND"
        ):

            variant = (
                candidate_result["variant"]
            )

            exome = (
                variant.get("exome")
            ) or {}

            genome = (
                variant.get("genome")
            ) or {}

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

                "Input_rsID":
                    str(rsid)
                    if not pd.isna(rsid)
                    else "",

                "Tested_gnomAD_ID":
                    exact_variant_id,

                "Resolved_gnomAD_ID":
                    variant.get(
                        "variant_id"
                    ),

                "Resolved_rsIDs":
                    ";".join(
                        variant.get(
                            "rsids"
                        ) or []
                    ),

                "Exome_AC":
                    exome.get("ac"),

                "Exome_AN":
                    exome.get("an"),

                "Exome_AF":
                    exome.get("af"),

                "Genome_AC":
                    genome.get("ac"),

                "Genome_AN":
                    genome.get("an"),

                "Genome_AF":
                    genome.get("af"),

                "Resolution_Method":
                    "Ensembl position search",

                "Final_Status":
                    "FOUND",

                "Error": None
            })

            found = True
            break

        time.sleep(3)

    if found:
        continue

    # --------------------------------------------------------
    # FINAL UNRESOLVED STATUS
    # --------------------------------------------------------

    print(
        "\nNo alternate gnomAD representation found."
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

        "Input_rsID":
            str(rsid)
            if not pd.isna(rsid)
            else "",

        "Tested_gnomAD_ID":
            exact_variant_id,

        "Resolved_gnomAD_ID":
            "",

        "Resolved_rsIDs":
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
            None,

        "Resolution_Method":
            "Exact + Ensembl resolution attempted",

        "Final_Status":
            "UNRESOLVED_IN_GNOMAD_R4",

        "Error":
            exact_result.get("error")
    })

    time.sleep(3)


# ============================================================
# SAVE RESULTS
# ============================================================

result_df = pd.DataFrame(results)

result_df = result_df.sort_values(
    "VariationID"
)

result_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 80)
print("SECOND-LEVEL GNOMAD RESOLUTION SUMMARY")
print("=" * 80)

print(
    f"Variants investigated : "
    f"{len(FAILED_VARIANTS)}"
)

print(
    f"Results generated     : "
    f"{len(result_df)}"
)

print(
    f"Found                 : "
    f"{(
        result_df['Final_Status']
        == 'FOUND'
    ).sum()}"
)

print(
    f"Still unresolved      : "
    f"{(
        result_df['Final_Status']
        == 'UNRESOLVED_IN_GNOMAD_R4'
    ).sum()}"
)


print("\nFINAL RESULTS:\n")

display_columns = [
    "VariationID",
    "Protein_Change",
    "Tested_gnomAD_ID",
    "Resolved_gnomAD_ID",
    "Resolved_rsIDs",
    "Exome_AF",
    "Genome_AF",
    "Resolution_Method",
    "Final_Status"
]

print(
    result_df[
        display_columns
    ].to_string(index=False)
)


print("\nOutput saved to:")
print(OUTPUT_FILE)

print("\nResolution completed.")