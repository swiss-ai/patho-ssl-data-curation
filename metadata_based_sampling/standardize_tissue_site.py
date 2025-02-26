"""
Description: This script standardizes GTEx and TCGA tissue/ organ site values, based on a pre-defined list of standardized values and fuzzy matching.
"""


from typing import Dict, List, Optional

import pandas as pd
from fuzzywuzzy import process


def map_organ_sites(
    uncleaned_list: List[str], primary_list: List[str], threshold: int = 80
) -> Dict[str, Optional[str]]:
    """Map uncleaned organ site list to the primary organ site list using fuzzy matching."""
    matches: Dict[str, Optional[str]] = {}
    unmatched: List[str] = []
    for orig_item in uncleaned_list:
        match, score = process.extractOne(orig_item, primary_list)
        if score >= threshold:
            matches[orig_item] = match
        else:
            print(orig_item)
            unmatched.append(orig_item)
    return matches, unmatched


if __name__ == "__main__":
    ## File paths and configs ################################################################################
    data_path = "."
    standardized_tissue_values_txt = (
        f"{data_path}/tissue_standardization_files/standardized_tissue_values.txt"
    )
    metadata_csvs = [
        f"{data_path}/tissue_standardization_files/TCGA_clinical.tsv",
        f"{data_path}/tissue_standardization_files/GTEx Portal.csv",
    ]
    tcga_case_metadata_csv = f"{data_path}/tissue_standardization_files/TCGA_case_metadata.tsv"

    mapping_csv = f"{data_path}/tissue_standardization_files/organ_mappings.csv"
    tissue_priors_csv = f"{data_path}/tissue_standardization_files/tissue_priors.csv"
    col_name = {"tcga": "tissue_or_organ_of_origin", "gtex": "Tissue"}
    #################################################################################################

    with open(standardized_tissue_values_txt, "r") as f:
        STANDARDIZED_TISSUE_VALUES = f.readlines()[0].split(",")

    tissue_priors_df = pd.read_csv(tissue_priors_csv)
    orig_tissue_values = pd.Series(dtype=str)

    # get all unique tissue values from metadata files
    for metadata_csv in metadata_csvs:
        df = pd.read_csv(
            metadata_csv, sep="\t" if metadata_csv.endswith("tsv") else ","
        )
        tissue_col = col_name["tcga"] if "TCGA" in metadata_csv else col_name["gtex"]
        tissue_value = df[tissue_col]
        orig_tissue_values = pd.concat((orig_tissue_values, tissue_value))

    # remove duplicates and replace tissue values with cleaned values
    orig_tissue_values = orig_tissue_values.drop_duplicates().reset_index(drop=True)
    tissue_priors = tissue_priors_df.set_index("uncleaned")["cleaned"].to_dict()
    orig_tissue_values = orig_tissue_values.replace(tissue_priors)
    mapping, unmatched = map_organ_sites(
        orig_tissue_values.tolist(), STANDARDIZED_TISSUE_VALUES
    )
    assert len(unmatched) == 0, f"Unmatched tissue values: {unmatched}"
    final_mapping = {**mapping, **tissue_priors}

    # add clean tissue values to metadata files
    for metadata_csv in metadata_csvs:
        df = pd.read_csv(
            metadata_csv, sep="\t" if metadata_csv.endswith("tsv") else ","
        )
        df = df.rename(columns={"case_submitter_id": "Case ID", "Tissue Sample ID": "slide_id"})
        if "tisue" not in df.columns:
            tissue_col = (
                col_name["tcga"] if "TCGA" in metadata_csv else col_name["gtex"]
            )
            df["tissue"] = df[tissue_col].replace(final_mapping)
            if "TCGA" in metadata_csv:
                # for TCGA add tissue and disease information to case metadata
                case_df = pd.read_csv(tcga_case_metadata_csv, sep="\t")
                df["primary_diagnosis"] = (
                    df["primary_diagnosis"]
                    .str.replace(", NOS", "")
                    .str.replace("'--", "")
                    .str.replace("not reported", "Unknown")
                    .replace(r"^\s*$", "Unknown", regex=True)
                )
                merged_df = pd.merge(
                    case_df, df[["Case ID", "tissue", "primary_diagnosis"]], on="Case ID"
                )
                merged_df["tissue_diagnosis"] = merged_df["tissue"] + "_" + merged_df["primary_diagnosis"]
                merged_df["slide_id"] = merged_df["File Name"].str.split(".").str[0]
                merged_df[["slide_id", "tissue", "primary_diagnosis", "tissue_diagnosis"]].to_csv("standardized_tissue_metadata_TCGA.csv",index=False)
            else:
                df["tissue_diagnosis"] = df["tissue"]
                df[["slide_id", "tissue", "tissue_diagnosis"]].to_csv("standardized_tissue_metadata_GTEx.csv",index=False)


