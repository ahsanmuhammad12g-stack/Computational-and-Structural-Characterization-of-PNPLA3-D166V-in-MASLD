import pandas as pd
import requests
import re
from pathlib import Path


# =============================================================================
# STEP 10 — PNPLA3 WILD-TYPE AND MUTANT STRUCTURAL
# PREPARATION AND COMPARATIVE STRUCTURAL ANALYSIS
# =============================================================================


print("=" * 78)
print("STEP 10 — PNPLA3 WILD-TYPE AND MUTANT STRUCTURAL")
print("PREPARATION AND COMPARATIVE STRUCTURAL ANALYSIS")
print("=" * 78)


# =============================================================================
# PROJECT PATHS
# =============================================================================

PROJECT_DIR = Path(
    r"C:\Users\Asif Computer\Documents\Bioinformatics projects\MASLD_PNPLA3"
)

ANNOTATION_DIR = PROJECT_DIR / "04_VARIANT ANALYSIS" / "annotations"

STRUCTURE_DIR = PROJECT_DIR / "05_STRUCTURAL_ANALYSIS"

STRUCTURE_DIR.mkdir(parents=True, exist_ok=True)


INPUT_FILE = (
    ANNOTATION_DIR /
    "pnpla3_final_mutant_candidate_selection.csv"
)


WT_PDB_FILE = (
    STRUCTURE_DIR /
    "PNPLA3_WT_AlphaFold.pdb"
)


SEQUENCE_FILE = (
    STRUCTURE_DIR /
    "PNPLA3_WT_sequence.fasta"
)


METADATA_FILE = (
    STRUCTURE_DIR /
    "PNPLA3_structure_metadata.csv"
)


SUMMARY_FILE = (
    STRUCTURE_DIR /
    "pnpla3_structural_preparation_summary.csv"
)


# =============================================================================
# CONSTANTS
# =============================================================================

UNIPROT_ID = "Q9Y3V1"

PROTEIN_NAME = "Patatin-like phospholipase domain-containing protein 3"

GENE_NAME = "PNPLA3"

EXPECTED_PROTEIN_LENGTH = 481

EXPECTED_MUTATION = "p.Asp166Val"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def fail(message):

    print("\nERROR:")
    print(message)

    raise SystemExit(1)


# =============================================================================
# ALPHAFOLD STRUCTURE RETRIEVAL
# =============================================================================


def download_alphafold_structure(uniprot_id, output_file):

    """
    Robust AlphaFold structure retrieval.

    Strategy:
    1. Reuse existing validated structure if present.
    2. Try AlphaFold API.
    3. If API unavailable, try direct PDB URLs across known model versions.
    4. Validate downloaded PDB content.
    """

    headers = {
        "User-Agent": "Mozilla/5.0"
    }


    # -------------------------------------------------------------------------
    # CHECK EXISTING FILE
    # -------------------------------------------------------------------------

    if output_file.exists():

        file_size = output_file.stat().st_size

        if file_size > 1000:

            print("\nExisting AlphaFold structure detected.")
            print(f"File: {output_file}")
            print(f"File size: {file_size:,} bytes")
            print("Existing structure reuse: PASS")

            return {
                "source": "Existing local AlphaFold structure",
                "url": "Local file",
                "model_version": "Previously downloaded"
            }

        else:

            print(
                "\nExisting structure file appears invalid."
            )

            print("Removing invalid file.")

            output_file.unlink()


    # -------------------------------------------------------------------------
    # STRATEGY 1 — ALPHAFOLD API
    # -------------------------------------------------------------------------

    print("\nTrying AlphaFold API retrieval...")

    api_url = (
        f"https://alphafold.ebi.ac.uk/api/prediction/"
        f"{uniprot_id}"
    )

    print(f"API: {api_url}")


    try:

        response = requests.get(
            api_url,
            headers=headers,
            timeout=60
        )

        print(
            f"AlphaFold API HTTP status: "
            f"{response.status_code}"
        )


        if response.status_code == 200:

            api_data = response.json()


            if isinstance(api_data, list) and len(api_data) > 0:

                metadata = api_data[0]

                pdb_url = metadata.get("pdbUrl")


                if pdb_url:

                    print(
                        "AlphaFold API PDB URL obtained."
                    )

                    download_response = requests.get(
                        pdb_url,
                        headers=headers,
                        timeout=120
                    )

                    print(
                        f"PDB download HTTP status: "
                        f"{download_response.status_code}"
                    )


                    if (
                        download_response.status_code == 200
                        and len(download_response.content) > 1000
                    ):

                        with open(
                            output_file,
                            "wb"
                        ) as f:

                            f.write(
                                download_response.content
                            )


                        print(
                            "AlphaFold structure download: PASS"
                        )

                        return {
                            "source": "AlphaFold API",
                            "url": pdb_url,
                            "model_version": metadata.get(
                                "latestVersion",
                                "Unknown"
                            )
                        }


    except Exception as e:

        print(
            f"AlphaFold API retrieval failed: {e}"
        )


    print(
        "\nAlphaFold API retrieval unavailable."
    )

    print(
        "Switching to direct AlphaFold structure retrieval..."
    )


    # -------------------------------------------------------------------------
    # STRATEGY 2 — DIRECT ALPHAFOLD FILE RETRIEVAL
    # -------------------------------------------------------------------------

    print("\nTesting AlphaFold model URLs...")


    model_versions = [

        "v4",
        "v5",
        "v6",
        "v3",
        "v2"

    ]


    for version in model_versions:

        pdb_url = (
            f"https://alphafold.ebi.ac.uk/files/"
            f"AF-{uniprot_id}-F1-model_{version}.pdb"
        )


        print(
            f"\nTrying model version: {version}"
        )

        print(pdb_url)


        try:

            response = requests.get(
                pdb_url,
                headers=headers,
                timeout=120
            )


            print(
                f"HTTP status: {response.status_code}"
            )


            if response.status_code != 200:

                continue


            content = response.content


            if len(content) < 1000:

                print(
                    "Response too small to be a valid PDB."
                )

                continue


            text_content = content.decode(
                "utf-8",
                errors="ignore"
            )


            if (
                "ATOM" not in text_content
                and "MODEL" not in text_content
            ):

                print(
                    "Downloaded content does not appear "
                    "to be a valid PDB."
                )

                continue


            with open(
                output_file,
                "wb"
            ) as f:

                f.write(content)


            print(
                "\nValid AlphaFold structure found."
            )

            print(
                "Direct structure download: PASS"
            )

            print(
                f"Model version: {version}"
            )

            print(
                f"Saved to: {output_file}"
            )


            return {
                "source": "AlphaFold direct file server",
                "url": pdb_url,
                "model_version": version
            }


        except Exception as e:

            print(
                f"Connection error: {e}"
            )

            continue


    # -------------------------------------------------------------------------
    # FAILURE
    # -------------------------------------------------------------------------

    fail(
        "Unable to retrieve PNPLA3 AlphaFold structure.\n\n"
        "Both retrieval strategies failed:\n"
        "1. AlphaFold API\n"
        "2. Direct AlphaFold model file URLs\n\n"
        "Do not continue structural analysis until "
        "a valid wild-type structure is obtained."
    )


# =============================================================================
# EXTRACT SEQUENCE FROM PDB
# =============================================================================


def extract_sequence_from_pdb(pdb_file):

    amino_acids = {

        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V"

    }


    residues = []

    seen_residues = set()


    with open(
        pdb_file,
        "r"
    ) as f:


        for line in f:


            if not line.startswith("ATOM"):

                continue


            residue_name = line[17:20].strip()

            chain_id = line[21].strip()

            residue_number = line[22:26].strip()

            insertion_code = line[26].strip()


            residue_key = (

                chain_id,
                residue_number,
                insertion_code

            )


            if residue_key in seen_residues:

                continue


            seen_residues.add(residue_key)


            if residue_name in amino_acids:


                residues.append({

                    "Chain": chain_id,

                    "Residue_Number":
                        int(residue_number),

                    "Residue_3Letter":
                        residue_name,

                    "Residue_1Letter":
                        amino_acids[residue_name]

                })


    return pd.DataFrame(residues)


# =============================================================================
# EXTRACT PLDDT VALUES
# =============================================================================


def get_plddt_statistics(pdb_file):

    plddt_values = []

    seen_residues = set()


    with open(
        pdb_file,
        "r"
    ) as f:


        for line in f:


            if not line.startswith("ATOM"):

                continue


            chain = line[21].strip()

            residue = line[22:26].strip()


            key = (

                chain,
                residue

            )


            if key in seen_residues:

                continue


            seen_residues.add(key)


            try:

                plddt = float(
                    line[60:66].strip()
                )

                plddt_values.append(plddt)


            except Exception:

                continue


    if len(plddt_values) == 0:

        return None


    series = pd.Series(plddt_values)


    return {

        "Residue_Count":
            len(series),

        "Mean_pLDDT":
            series.mean(),

        "Median_pLDDT":
            series.median(),

        "Minimum_pLDDT":
            series.min(),

        "Maximum_pLDDT":
            series.max()

    }


# =============================================================================
# INTERPRET PLDDT
# =============================================================================


def interpret_plddt(score):

    if pd.isna(score):

        return "Unavailable"


    if score >= 90:

        return "Very high confidence"


    elif score >= 70:

        return "Confident"


    elif score >= 50:

        return "Low confidence"


    else:

        return "Very low confidence"


# =============================================================================
# STEP 1 — CHECK INPUT
# =============================================================================


print("\n" + "-" * 78)
print("CHECKING INPUT FILES")
print("-" * 78)


if not INPUT_FILE.exists():

    fail(
        "Final Step 9F candidate selection file "
        "not found:\n"
        f"{INPUT_FILE}"
    )


print(
    "Final Step 9F candidate selection file: PASS"
)


# =============================================================================
# STEP 2 — READ INPUT
# =============================================================================


print("\n" + "-" * 78)
print("READING FINAL MUTANT CANDIDATE SELECTION")
print("-" * 78)


df = pd.read_csv(INPUT_FILE)


print(
    f"Records loaded: {len(df)}"
)


required_columns = [

    "VariationID",

    "Protein_Change",

    "Protein_Position",

    "Final_Mutant_Rank",

    "Final_Mutant_Selection_Score",

    "Final_Evidence_Category",

    "Structural_Analysis_Recommendation"

]


missing_columns = [

    column
    for column in required_columns
    if column not in df.columns

]


if missing_columns:

    fail(
        "Required Step 9F columns missing:\n"
        + "\n".join(
            f" - {x}"
            for x in missing_columns
        )
    )


print(
    "Required Step 9F columns: PASS"
)


if len(df) == 9:

    print(
        "Expected candidate count (9): PASS"
    )

else:

    print(
        f"WARNING: Expected 9 candidates, "
        f"found {len(df)}"
    )


# =============================================================================
# STEP 3 — IDENTIFY PRIMARY MUTANT
# =============================================================================


print("\n" + "-" * 78)
print("IDENTIFYING PRIMARY PNPLA3 MUTANT CANDIDATE")
print("-" * 78)


primary_df = df[

    df[
        "Structural_Analysis_Recommendation"
    ].astype(str).str.strip().str.lower()

    ==

    "primary mutant candidate"

].copy()


if len(primary_df) != 1:

    fail(
        "Exactly one primary mutant candidate "
        "was expected.\n"
        f"Found: {len(primary_df)}"
    )


primary = primary_df.iloc[0]


print(
    "\nPRIMARY MUTANT SELECTED FOR STRUCTURAL ANALYSIS\n"
)


print(
    f"VariationID: "
    f"{primary['VariationID']}"
)

print(
    f"Protein change: "
    f"{primary['Protein_Change']}"
)

print(
    f"Final rank: "
    f"{primary['Final_Mutant_Rank']}"
)

print(
    f"Final selection score: "
    f"{primary['Final_Mutant_Selection_Score']}"
)

print(
    f"Evidence category: "
    f"{primary['Final_Evidence_Category']}"
)


if primary["Protein_Change"] == EXPECTED_MUTATION:

    print(
        f"Expected primary mutation "
        f"({EXPECTED_MUTATION}): PASS"
    )

else:

    fail(
        "Unexpected primary mutation selected."
    )


# =============================================================================
# STEP 4 — DOWNLOAD WILD-TYPE STRUCTURE
# =============================================================================


print("\n" + "-" * 78)
print("OBTAINING PNPLA3 WILD-TYPE STRUCTURE")
print("-" * 78)


print(
    "\nUsing robust multi-strategy AlphaFold retrieval."
)


structure_info = download_alphafold_structure(

    UNIPROT_ID,

    WT_PDB_FILE

)


# =============================================================================
# STEP 5 — STRUCTURE QC
# =============================================================================


print("\n" + "-" * 78)
print("WILD-TYPE STRUCTURE QUALITY CONTROL")
print("-" * 78)


if not WT_PDB_FILE.exists():

    fail(
        "Wild-type structure file is missing."
    )


file_size = WT_PDB_FILE.stat().st_size


print(
    f"PDB file size: "
    f"{file_size:,} bytes"
)


if file_size < 1000:

    fail(
        "PDB file appears invalid."
    )


print(
    "Structure file presence: PASS"
)

print(
    "Structure file size QC: PASS"
)


# =============================================================================
# STEP 6 — EXTRACT SEQUENCE
# =============================================================================


print("\n" + "-" * 78)
print("PROTEIN SEQUENCE VALIDATION")
print("-" * 78)


residue_df = extract_sequence_from_pdb(

    WT_PDB_FILE

)


if residue_df.empty:

    fail(
        "No amino-acid residues could be extracted "
        "from the PDB."
    )


sequence = "".join(

    residue_df[
        "Residue_1Letter"
    ].tolist()

)


protein_length = len(sequence)


print(
    f"Protein residues extracted: "
    f"{protein_length}"
)


if protein_length == EXPECTED_PROTEIN_LENGTH:

    print(
        f"Expected PNPLA3 length "
        f"({EXPECTED_PROTEIN_LENGTH} aa): PASS"
    )

else:

    print(
        "WARNING: Protein length differs from "
        "expected UniProt length."
    )


# =============================================================================
# SAVE FASTA
# =============================================================================


with open(
    SEQUENCE_FILE,
    "w"
) as f:


    f.write(

        f">PNPLA3_WT|UniProt={UNIPROT_ID}\n"

    )


    for i in range(

        0,

        len(sequence),

        60

    ):

        f.write(

            sequence[i:i + 60]

            + "\n"

        )


print(
    "\nWild-type FASTA written:"
)

print(
    SEQUENCE_FILE
)


# =============================================================================
# STEP 7 — MUTATION VALIDATION
# =============================================================================


print("\n" + "-" * 78)
print("PRIMARY MUTATION POSITION VALIDATION")
print("-" * 78)


mutation_match = re.match(

    r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})",

    primary["Protein_Change"]

)


if not mutation_match:

    fail(
        "Unable to parse protein mutation."
    )


wt_three = mutation_match.group(1)

mutation_position = int(

    mutation_match.group(2)

)

mut_three = mutation_match.group(3)


three_to_one = {

    "Ala": "A",
    "Arg": "R",
    "Asn": "N",
    "Asp": "D",
    "Cys": "C",
    "Gln": "Q",
    "Glu": "E",
    "Gly": "G",
    "His": "H",
    "Ile": "I",
    "Leu": "L",
    "Lys": "K",
    "Met": "M",
    "Phe": "F",
    "Pro": "P",
    "Ser": "S",
    "Thr": "T",
    "Trp": "W",
    "Tyr": "Y",
    "Val": "V"

}


expected_wt = three_to_one[wt_three]

expected_mutant = three_to_one[mut_three]


print(
    f"Wild-type residue: "
    f"{wt_three} ({expected_wt})"
)

print(
    f"Mutation position: "
    f"{mutation_position}"
)

print(
    f"Mutant residue: "
    f"{mut_three} ({expected_mutant})"
)


mutation_site = residue_df[

    residue_df[
        "Residue_Number"
    ]

    ==

    mutation_position

]


if len(mutation_site) != 1:

    fail(
        "Mutation position could not be uniquely "
        "identified in the structure."
    )


observed_wt = mutation_site.iloc[0][

    "Residue_1Letter"

]


print(
    f"Observed residue at position "
    f"{mutation_position}: "
    f"{observed_wt}"
)


if observed_wt == expected_wt:

    print(
        "Wild-type residue validation: PASS"
    )

else:

    fail(
        f"Mutation-site residue mismatch.\n"
        f"Expected: {expected_wt}\n"
        f"Observed: {observed_wt}"
    )


# =============================================================================
# STEP 8 — ALPHAFOLD CONFIDENCE ANALYSIS
# =============================================================================


print("\n" + "-" * 78)
print("ALPHAFOLD STRUCTURAL CONFIDENCE ANALYSIS")
print("-" * 78)


plddt_stats = get_plddt_statistics(

    WT_PDB_FILE

)


mutation_plddt = pd.NA


if plddt_stats is not None:


    print(
        f"Residues with pLDDT: "
        f"{plddt_stats['Residue_Count']}"
    )

    print(
        f"Mean pLDDT: "
        f"{plddt_stats['Mean_pLDDT']:.2f}"
    )

    print(
        f"Median pLDDT: "
        f"{plddt_stats['Median_pLDDT']:.2f}"
    )

    print(
        f"Minimum pLDDT: "
        f"{plddt_stats['Minimum_pLDDT']:.2f}"
    )

    print(
        f"Maximum pLDDT: "
        f"{plddt_stats['Maximum_pLDDT']:.2f}"
    )


    with open(
        WT_PDB_FILE,
        "r"
    ) as f:


        for line in f:


            if not line.startswith("ATOM"):

                continue


            residue_number = line[22:26].strip()


            if residue_number != str(
                mutation_position
            ):

                continue


            try:

                mutation_plddt = float(

                    line[60:66].strip()

                )

                break


            except Exception:

                continue


    if not pd.isna(mutation_plddt):

        print(
            f"\npLDDT at mutation position "
            f"{mutation_position}: "
            f"{mutation_plddt:.2f}"
        )

        print(
            f"Local confidence: "
            f"{interpret_plddt(mutation_plddt)}"
        )


# =============================================================================
# STEP 9 — SAVE METADATA
# =============================================================================


print("\n" + "-" * 78)
print("SAVING STRUCTURAL METADATA")
print("-" * 78)


metadata_df = pd.DataFrame([{

    "Protein":
        PROTEIN_NAME,

    "Gene":
        GENE_NAME,

    "UniProt_ID":
        UNIPROT_ID,

    "Structure_Source":
        structure_info["source"],

    "Structure_URL":
        structure_info["url"],

    "AlphaFold_Model_Version":
        structure_info["model_version"],

    "Structure_File":
        str(WT_PDB_FILE),

    "Protein_Length":
        protein_length,

    "Mutation":
        primary["Protein_Change"],

    "Mutation_Position":
        mutation_position,

    "Mutation_Site_pLDDT":
        mutation_plddt

}])


metadata_df.to_csv(

    METADATA_FILE,

    index=False

)


print(
    "Structural metadata saved:"
)

print(
    METADATA_FILE
)


# =============================================================================
# STEP 10 — STRUCTURAL PREPARATION SUMMARY
# =============================================================================


print("\n" + "-" * 78)
print("GENERATING STRUCTURAL PREPARATION SUMMARY")
print("-" * 78)


summary_df = pd.DataFrame([{

    "Protein":
        "PNPLA3",

    "UniProt_ID":
        UNIPROT_ID,

    "Wild_Type_Structure_Status":
        "Prepared and validated",

    "Wild_Type_Structure_File":
        str(WT_PDB_FILE),

    "Protein_Length":
        protein_length,

    "Primary_Mutation":
        primary["Protein_Change"],

    "Mutation_Position":
        mutation_position,

    "Wild_Type_Residue":
        expected_wt,

    "Mutant_Residue":
        expected_mutant,

    "Mutation_Site_pLDDT":
        mutation_plddt,

    "Mutation_Site_Confidence":
        interpret_plddt(mutation_plddt),

    "Mutant_Model_Status":
        "Pending computational mutant modelling",

    "Comparative_Analysis_Status":
        "Wild-type preparation completed"

}])


summary_df.to_csv(

    SUMMARY_FILE,

    index=False

)


print(
    "Structural preparation summary saved:"
)

print(
    SUMMARY_FILE
)


# =============================================================================
# FINAL QUALITY CONTROL
# =============================================================================


print("\n" + "=" * 78)
print("FINAL STEP 10 QUALITY CONTROL")
print("=" * 78)


checks = {

    "Input file available":
        INPUT_FILE.exists(),

    "Single primary candidate":
        len(primary_df) == 1,

    "Expected mutation selected":
        primary["Protein_Change"]
        == EXPECTED_MUTATION,

    "Wild-type structure available":
        WT_PDB_FILE.exists(),

    "Valid structure file":
        WT_PDB_FILE.stat().st_size > 1000,

    "Protein sequence extracted":
        protein_length > 0,

    "Mutation position validated":
        observed_wt == expected_wt,

    "Metadata file generated":
        METADATA_FILE.exists(),

    "Summary file generated":
        SUMMARY_FILE.exists()

}


all_pass = True


for check, status in checks.items():


    result = "PASS" if status else "FAIL"


    print(
        f"{check}: {result}"
    )


    if not status:

        all_pass = False


# =============================================================================
# FINAL SUMMARY
# =============================================================================


print("\n" + "=" * 78)
print("STEP 10 STRUCTURAL PREPARATION SUMMARY")
print("=" * 78)


print(
    f"\nProtein: PNPLA3"
)

print(
    f"UniProt ID: {UNIPROT_ID}"
)

print(
    f"Primary mutant: "
    f"{primary['Protein_Change']}"
)

print(
    f"Mutation position: "
    f"{mutation_position}"
)

print(
    f"Validated residue change: "
    f"{observed_wt} → {expected_mutant}"
)

print(
    f"Wild-type structure source: "
    f"{structure_info['source']}"
)

print(
    f"Protein residues extracted: "
    f"{protein_length}"
)


if not pd.isna(mutation_plddt):

    print(
        f"Mutation-site pLDDT: "
        f"{mutation_plddt:.2f}"
    )


if all_pass:


    print("\n" + "=" * 78)
    print("STEP 10 SUCCESSFULLY COMPLETED")
    print("=" * 78)


    print(
        "\nWild-type PNPLA3 structure has been "
        "retrieved and validated."
    )


    print(
        "\nNext scientific stage:"
    )


    print(
        "STEP 11 — Integrated mutant structure modelling "
        "and comparative WT-versus-D166V structural analysis."
    )


else:


    print("\n" + "=" * 78)
    print("STEP 10 NOT COMPLETED")
    print("=" * 78)


    fail(
        "One or more structural quality-control checks failed."
    )


print("=" * 78)