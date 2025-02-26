"""
This script samples tiles based on the tissue_diagnosis metadata (TCGA: tissue+primar_diagnosis, GTEx: tissue only). 
"""

import os
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_and_merge_metadata(tissue_metadata_csvs: List[str]) -> pd.DataFrame:
    """Load metadata from multiple datasets and merge them based on slide_id"""
    tissue_metadata_df = pd.concat(
        [
            pd.read_csv(csv)[["slide_id", "tissue_diagnosis", "num_tiles"]]
            for csv in tissue_metadata_csvs
        ]
    )
    slide_tile_ids_df = pd.DataFrame({
        'slide_id': np.repeat(tissue_metadata_df['slide_id'].values, tissue_metadata_df['num_tiles'].values),
        'tissue_diagnosis': np.repeat(tissue_metadata_df['tissue_diagnosis'].values, tissue_metadata_df['num_tiles'].values),
        'tile_idx': np.hstack([np.arange(n) for n in tissue_metadata_df['num_tiles'].values])
    })
    return slide_tile_ids_df


def calculate_sampling_weights(slide_tile_ids_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate sampling weights based on the number of tiles per tissue diagnosis"""
    slide_tile_ids_df["sampling_weight"] = slide_tile_ids_df["tissue_diagnosis"].map(
        slide_tile_ids_df.groupby("tissue_diagnosis").size().to_dict()
    )
    slide_tile_ids_df["sampling_weight"] = (
        slide_tile_ids_df["sampling_weight"].max()
        / slide_tile_ids_df["sampling_weight"]
    )
    return slide_tile_ids_df


def create_stats_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Create a dataframe with statistics about the number of slides and tiles per tissue diagnosis"""
    if "num_tiles" in df.columns:
        info_df = (
            df.groupby("tissue_diagnosis")
            .agg({"slide_id": "nunique", "num_tiles": "sum"})
            .reset_index()
        )
    else:
        info_df = (
            df.groupby("tissue_diagnosis").agg({"slide_id": "nunique"}).reset_index()
        )
        num_tiles_df = (
            df.groupby("tissue_diagnosis").size().reset_index(name="num_tiles")
        )
        info_df = info_df.merge(num_tiles_df, on="tissue_diagnosis", how="left")
    info_df = info_df.rename(columns={"slide_id": "num_slides"})
    return info_df


def visualize_sampled_stats(merged_info: pd.DataFrame, output_path: str) -> None:
    """Visualize the distribution of the sampled data vs full dataset"""
    for col in ["num_slides", "num_tiles"]:
        plt.figure(figsize=(36, 10))
        df = merged_info.sort_values(by=f"{col}_full", ascending=False)
        df[f"{col}_remaining"] = df[f"{col}_full"] - df[f"{col}_sampled"]
        plt.bar(
            df["tissue_diagnosis"],
            df[f"{col}_sampled"],
            label="Sampled subset",
            color="#6fa3ef",
        )
        plt.bar(
            df["tissue_diagnosis"],
            df[f"{col}_remaining"],
            bottom=df[f"{col}_sampled"],
            label="Full set",
            color="#37598a",
        )
        plt.xlabel("Tissue and Primary Diagnosis")
        plt.ylabel(f"Number of {'Tiles (Log Scale)' if 'tile' in col else 'Slides'}")
        if col == "num_tiles":
            plt.yscale("log")
        plt.title("Samples per Tissue and Primary Diagnosis: Subset vs Full Dataset")
        plt.xticks(rotation=80, ha="right")
        plt.margins(x=0)
        plt.legend()
        plt.tight_layout(pad=3.0)
        plt.savefig(f"{output_path}_{col}.png")


if __name__ == "__main__":
    ###### Paths and sampling configs ##########################################################
    data_dir = "./metadata_csvs"
    output_dir = "./sampled_data"
    datasets = ["TCGA", "GTEx"]
    tissue_metadata_csvs = [
        f"{data_dir}/{dataset}_tissue_metadata.csv"
        for dataset in datasets
    ]

    full_dataset_size = 350000000
    sample_percentage = 0.1  # % of the data
    random_seed = 25
    target_size = int(full_dataset_size * sample_percentage)
    verbose=True
    ##########################################################################################
    # Sample data based on tissue_diagnosis
    if not os.path.exists(
        f"{data_dir}/sampled_N={target_size}_seed={random_seed}.csv"
    ):
        os.makedirs(output_dir, exist_ok=True)
        slide_tile_ids_df = load_and_merge_metadata(tissue_metadata_csvs)
        slide_tile_ids_df = calculate_sampling_weights(slide_tile_ids_df)

        sampled_df = slide_tile_ids_df.sample(
            n=target_size, random_state=random_seed, weights="sampling_weight"
        )
        print("Sampled data shape:", sampled_df.shape)
        sampled_df[["slide_id", "tile_idx", "tissue_diagnosis"]].to_csv(
            f"{output_dir}/sampled_N={target_size}_seed={random_seed}.csv",
            index=False,
        )
        print("Sampled data saved to:", f"{output_dir}/sampled_N={target_size}_seed={random_seed}.csv")
    else:
        print("Sampled data already exists. Loading from file:", f"{output_dir}/sampled_N={target_size}_seed={random_seed}.csv")
        sampled_df = pd.read_csv(
            f"{output_dir}/sampled_N={target_size}_seed={random_seed}.csv"
        )

    if verbose:
        # log statistics of the sampled data
        info_df = create_stats_csv(sampled_df)
        info_df.to_csv(
            f"{output_dir}/sampled_N={target_size}_seed={random_seed}_stats.csv",
            index=False,
        )

        # visualize the distribution of the sampled data vs full dataset
        tissue_metadata_df = pd.concat(
            [
                pd.read_csv(csv)[["slide_id", "num_tiles", "tissue_diagnosis"]]
                for csv in tissue_metadata_csvs
            ]
        )
        full_info_df = create_stats_csv(tissue_metadata_df)
        full_info_df.to_csv(
            f"{output_dir}/full_set_stats.csv", index=False
        )
        merged_info = info_df.merge(
            full_info_df, on="tissue_diagnosis", how="left", suffixes=("_sampled", "_full")
        )

        visualize_sampled_stats(
            merged_info,
            f"{output_dir}/sampled_N={target_size}_seed={random_seed}",
        )
