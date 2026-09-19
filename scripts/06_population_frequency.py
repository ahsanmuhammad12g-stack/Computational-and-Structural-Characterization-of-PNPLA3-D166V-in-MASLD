import pandas as pd
import requests
import time
from pathlib import Path

# ============================================================
# PNPLA3 POPULATION FREQUENCY ANNOTATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "pnpla3_candidate_master.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "04_VARIANT ANALYSIS"
    / "annotations"
)

OUTPUT_FILE = OUTPUT_DIR / "pnpla3_population_frequency.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

ENSEMBL_URL = "https://rest.ensembl.org/variation/human/{}"

HEADERS = {
    "Accept": "application/json"
}

REQUEST_DELAY = 0.15


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PNPLA3 POPULATION FREQUENCY ANNOTATION")
print("=" * 70)

print("\nLoading candidate master file...")

df = pd.read_csv(INPUT_FILE)

print(f"Total candidate variants: {len(df)}")


# ============================================================
# CHECK RSID COLUMN
# ============================================================

if "RS# (dbSNP)" not in df.columns:
    raise ValueError(
        "Column 'RS# (dbSNP)' was not found in the candidate master file."
    )


# ============================================================
# OUTPUT COLUMNS
# ============================================================

df["EnsemblStatus"] = ""
df["PopulationFrequency"] = ""
df["FrequencyPopulation"] = ""
df["FrequencyAllele"] = ""


# ============================================================
# FUNCTION: QUERY ENSEMBL
# ============================================================

def query_ensembl(rsid):

    url = ENSEMBL_URL.format(rsid)

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        if response.status_code != 200:
            return None, f"HTTP {response.status_code}"

        try:
            data = response.json()
        except Exception:
            return None, "Invalid JSON"

        return data, "OK"

    except requests.exceptions.Timeout:
        return None, "Timeout"

    except requests.exceptions.ConnectionError:
        return None, "Connection error"

    except requests.exceptions.RequestException as e:
        return None, f"Request error: {str(e)}"


# ============================================================
# FUNCTION: EXTRACT POPULATION FREQUENCIES
# ============================================================

def extract_population_frequency(data):

    populations = data.get("populations", [])

    if not populations:
        return None, None, None

    frequencies = []

    for pop in populations:

        population_name = pop.get("population")

        frequency = pop.get("frequency")

        allele = pop.get("allele")

        if frequency is None:
            continue

        try:
            frequency = float(frequency)
        except (TypeError, ValueError):
            continue

        frequencies.append(
            (
                population_name,
                frequency,
                allele
            )
        )

    if not frequencies:
        return None, None, None

    # --------------------------------------------------------
    # Prefer large reference/population datasets when present
    # --------------------------------------------------------

    preferred_keywords = [
        "gnomAD",
        "1000GENOMES",
        "1000 Genomes",
        "TOPMED"
    ]

    for keyword in preferred_keywords:

        for population, frequency, allele in frequencies:

            if population and keyword.lower() in population.lower():

                return frequency, population, allele

    # --------------------------------------------------------
    # Otherwise return highest available population frequency
    # --------------------------------------------------------

    best = max(
        frequencies,
        key=lambda x: x[1]
    )

    return best[1], best[0], best[2]


# ============================================================
# PROCESS VARIANTS
# ============================================================

print("\nQuerying Ensembl variation API...")
print("-" * 70)

success = 0
no_rsid = 0
api_error = 0
no_frequency = 0


for index, row in df.iterrows():

    rsid = row["RS# (dbSNP)"]

    # --------------------------------------------------------
    # Clean rsID
    # --------------------------------------------------------

    if pd.isna(rsid):

        df.at[index, "EnsemblStatus"] = "No rsID"
        no_rsid += 1

        continue

    rsid = str(rsid).strip()

    if rsid in ["", "-", "-1", "nan", "NA", "na"]:

        df.at[index, "EnsemblStatus"] = "No rsID"
        no_rsid += 1

        continue

    # --------------------------------------------------------
    # Make sure rs prefix exists
    # --------------------------------------------------------

    if not rsid.lower().startswith("rs"):

        rsid = "rs" + rsid

    print(
        f"[{index + 1:>3}/{len(df)}] "
        f"{rsid} ... ",
        end=""
    )

    # --------------------------------------------------------
    # Query Ensembl
    # --------------------------------------------------------

    data, status = query_ensembl(rsid)

    if status != "OK":

        df.at[index, "EnsemblStatus"] = status

        print(status)

        api_error += 1

        time.sleep(REQUEST_DELAY)

        continue

    # --------------------------------------------------------
    # Extract frequency
    # --------------------------------------------------------

    frequency, population, allele = (
        extract_population_frequency(data)
    )

    if frequency is None:

        df.at[index, "EnsemblStatus"] = (
            "Found - no population frequency"
        )

        print("Found - no population frequency")

        no_frequency += 1

    else:

        df.at[index, "EnsemblStatus"] = "Success"

        df.at[index, "PopulationFrequency"] = frequency
        df.at[index, "FrequencyPopulation"] = population
        df.at[index, "FrequencyAllele"] = allele

        print(
            f"{frequency:.6g} "
            f"({population})"
        )

        success += 1

    time.sleep(REQUEST_DELAY)


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("POPULATION FREQUENCY ANNOTATION SUMMARY")
print("=" * 70)

print(f"\nInput variants: {len(df)}")
print(f"Successful frequency annotations: {success}")
print(f"No rsID: {no_rsid}")
print(f"No population frequency: {no_frequency}")
print(f"API errors: {api_error}")

print("\nStatus distribution:")
print(
    df["EnsemblStatus"]
    .value_counts(dropna=False)
)

print("\nOutput:")
print(OUTPUT_FILE)

print("\nDone.")