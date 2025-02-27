import os
import random
import argparse
import time
import tempfile
from collections import defaultdict


import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle



def save_cluster2tile_sampler(cluster2tile_sampler, output_path):
    """
    Saves the entire cluster2tile_sampler dictionary to a pickle file safely.
    :param cluster2tile_sampler: Dictionary of TileSampler objects for each cluster.
    :param output_path: Path where the pickle file will be saved.
    """
    # time
    start = time.time()
    print(f"Saving cluster2tile_sampler to {output_path}")

    # Create a temporary file in the same directory as output_path
    temp_file = tempfile.NamedTemporaryFile(
        delete=False, dir=os.path.dirname(output_path)
    )

    try:
        # Write to the temporary file first
        with open(temp_file.name, "wb") as f:
            pickle.dump(cluster2tile_sampler, f)

        # Rename the temporary file to the final output path (atomic operation)
        os.rename(temp_file.name, output_path)

        print(f"Saving cluster2tile_sampler took {time.time() - start:.4f}s")
        print(f"cluster2tile_sampler saved to {output_path}")

    except Exception as e:
        # Clean up the temporary file in case of an error
        if os.path.exists(temp_file.name):
            os.remove(temp_file.name)
        print(f"Error saving the file: {e}")


def get_cluster2tile_sampler(input_csv_path, output_dir, weight_fn, h_level):
    """
    Load or build the cluster2tile_sampler dictionary.
    :param input_csv_path: Path to the input CSV file.
    :param output_dir: Directory where the cluster2tile_sampler will be saved.
    :param weight_fn: Weight function for the tile sampling counts.
    :param h_level: Hierarchical level.
    :return: Dictionary of TileSampler objects for each cluster.
    """

    # go up one level and set the output path
    output_path = os.path.join(output_dir, "cluster2tile_sampler.pkl")

    if os.path.exists(output_path):
        print(f"Found cluster2tile_sampler at {output_path}. Loading...")
        with open(output_path, "rb") as f:
            cluster2tile_sampler = pickle.load(f)

    else:
        print("cluster2tile_sampler not found. Building...")

        df = pd.read_csv(input_csv_path)

        # Preprocess pd csv
        # Track progress using tqdm and create cluster2tiles dictionary
        cluster2tiles = {}

        if h_level == "tissue":
            grouping_var = "tissue_diagnosis"
        else:
            grouping_var = f"level_{h_level}"

        for cluster, group in tqdm(
            df.groupby(grouping_var), desc="Building cluster2tiles"
        ):
            slide_ids = group["slide_id"].values
            tile_idxs = group["tile_idx"].values.astype(int).tolist()
            cluster2tiles[cluster] = list(zip(slide_ids, tile_idxs))

        # Build tile sampler per cluster
        cluster2tile_sampler = {}
        for cluster_id in tqdm(cluster2tiles, desc="Building cluster2tile_sampler"):
            this_cluster_tiles = cluster2tiles[cluster_id]
            cluster2tile_sampler[cluster_id] = TileSampler(
                this_cluster_tiles, weight_fn=weight_fn
            )

        save_cluster2tile_sampler(cluster2tile_sampler, output_path=output_path)

    return cluster2tile_sampler


class TileSampler:
    def __init__(self, tiles, weight_fn="inverse"):
        """
        Initialize the TileSampler with the number of tiles.
        :param tiles: List of tuples (slide_id, tile_id).
        """
        # Build self slides and tiles
        self.tiles = np.array(tiles)
        self.slides = self.tiles[:, 0]
        self.num_tiles = len(tiles)
        self.tile_counts = np.zeros(
            self.num_tiles, dtype=int
        )  # Tracks the count for each tile

        self.slide_to_idxs = defaultdict(list)
        for idx, slide_id in tqdm(
            enumerate(self.slides), desc="Building slide_to_idxs"
        ):
            self.slide_to_idxs[slide_id].append(idx)

        self.idxs = np.arange(self.num_tiles)
        self.weight_fn = weight_fn

    def _sample_from_idxs(self, idxs, num_to_sample):
        weights = self.tile_counts[idxs] <= np.min(self.tile_counts[idxs])
        total_weight = weights.sum()
        if total_weight == 0:
            # Assign equal probabilities if all weights are zero
            # sometimes all weights become 0, e.g., underflow of repetitive application of exponential
            probs = np.ones(len(idxs)) / len(idxs)
        else:
            probs = weights / total_weight

        sampled_tile_idxs = np.random.choice(idxs, size=num_to_sample, p=probs)
        self.tile_counts[sampled_tile_idxs] += 1
        return sampled_tile_idxs

    def sample_tiles(self, num_to_sample):
        """
        Samples tiles, prioritizing less observed ones, from all idxs
        """
        sampled_idxs = self._sample_from_idxs(self.idxs, num_to_sample)
        return self.tiles[sampled_idxs], self.slides[sampled_idxs]

    def sample_tiles_from_slides(self, num_to_sample, open_slides):
        """
        Samples tiles from a set of slides, prioritizing less observed ones.
        """
        # Retrieve valid tile indices for the given slides
        all_valid_idxs = []
        for slide in open_slides:
            if slide in self.slide_to_idxs:
                all_valid_idxs.extend(self.slide_to_idxs[slide])
        all_valid_idxs = np.array(all_valid_idxs)

        if len(all_valid_idxs) > 0:
            sampled_idxs = self._sample_from_idxs(all_valid_idxs, num_to_sample)
        else:
            # If no valid tiles are found, sample from all tiles
            sampled_idxs = self._sample_from_idxs(self.idxs, num_to_sample)

        return self.tiles[sampled_idxs], self.slides[sampled_idxs]


def create_batch(
    target_num_tiles_per_cluster,
    open_slides,
    cluster2tile_sampler,
    max_slides_per_epoch,
):
    batch_tiles = []
    cluster_ids = list(cluster2tile_sampler.keys())

    random.shuffle(cluster_ids)
    for cluster_idx, cluster_id in enumerate(cluster_ids):
        # along with the suffle of ids, this allow to vary the number of tiles per cluster in the batch (there will be differences of 1 tile as the batch size is not divisible by the number of clusters)
        target_cluster_num_tiles = target_num_tiles_per_cluster[cluster_idx]
        tile_sampler = cluster2tile_sampler[cluster_id]

        if len(open_slides) < max_slides_per_epoch:
            # Extract tiles
            cluster_tiles, cluster_slides = tile_sampler.sample_tiles(
                target_cluster_num_tiles
            )
        else:
            cluster_tiles, cluster_slides = tile_sampler.sample_tiles_from_slides(
                target_cluster_num_tiles, open_slides
            )

        # Reformat according to the batch format
        cluster_tiles = [
            {"slide_id": slide_id, "ind": tile_idx}
            for slide_id, tile_idx in cluster_tiles
        ]
        batch_tiles.extend(cluster_tiles)

        # update open slides
        open_slides.update(cluster_slides.tolist())

    return batch_tiles, open_slides



def assign_num_tiles_per_cluster_in_batch(batch_size,num_clusters):
    target_num_tiles_per_cluster = np.array([batch_size//num_clusters]*num_clusters)
    tmp = np.zeros(num_clusters, dtype=int)
    tmp[:batch_size%num_clusters] = 1
    target_num_tiles_per_cluster += tmp
    assert sum(target_num_tiles_per_cluster) == batch_size

    return target_num_tiles_per_cluster

def main(args):
    # Constants
    BS = 2048
    NUM_TILES = 3.5e8  # total
    NUM_SAMPLES_PER_EPOCH = 3.5e5  # copy from curated dataset implementation
    # number of clusters: level 3 = 350, level 4 = 62

    # parse arguments
    username = args.username
    h_level = args.h_level
    data_percentage = args.data_percentage
    max_slides_per_epoch = args.max_slides_per_epoch
    weight_fn = args.weight_fn
    curation_mode = args.curation_mode
    freq_saving = args.freq_saving

    # seed everything
    np.random.seed(0)
    random.seed(0)

    # intitialization
    num_selected_tiles = int(NUM_TILES * data_percentage / 100)
    num_epochs = int(NUM_TILES / NUM_SAMPLES_PER_EPOCH)
    num_epochs_to_cover_selected_tiles = num_selected_tiles / NUM_SAMPLES_PER_EPOCH
    input_csv_path = f""
    assert input_csv_path != "", "Please provide the input CSV path"
    output_dir = f""
    assert output_dir != "", "Please provide the output directory"
    os.makedirs(output_dir, exist_ok=True)
    cluster2tile_sampler = get_cluster2tile_sampler(
        input_csv_path,
        output_dir,
        weight_fn,
        h_level,
    )
    num_clusters = len(cluster2tile_sampler.keys())

    # Some slides will have 1 more tile than others because batch-size is not divisible by the number of clusters. But we will vary which slides do.
    target_num_tiles_per_cluster = assign_num_tiles_per_cluster_in_batch(
        BS, num_clusters
    )
    num_batches = int(NUM_SAMPLES_PER_EPOCH // BS)

    # last batch
    last_batch_remaining_samples = int(NUM_SAMPLES_PER_EPOCH % BS)
    last_batch_num_tiles_per_cluster = assign_num_tiles_per_cluster_in_batch(
        last_batch_remaining_samples, num_clusters
    )

    # now we distribute the tiles per batches
    for epoch in range(num_epochs):

        # check if epoch was not yet written
        if os.path.exists(f"{output_dir}/schedule_{epoch:03d}.csv"):
            print(f"Epoch {epoch} already exists. Skipping...")
            continue

        epoch_tiles = []
        open_slides = set([])
        for _ in tqdm(range(num_batches), desc=f"Generating batches of epoch: {epoch}"):
            # for _ in range(num_batches):
            batch_tiles, open_slides = create_batch(
                target_num_tiles_per_cluster,
                open_slides,
                cluster2tile_sampler,
                max_slides_per_epoch,
            )

            assert (
                len(batch_tiles) == BS
            ), f"Batch has {len(batch_tiles)} samples instead of {BS}"
            epoch_tiles.extend(batch_tiles)

        # Last batch: less samples than the batch size
        batch_tiles, open_slides = create_batch(
            last_batch_num_tiles_per_cluster,
            open_slides,
            cluster2tile_sampler,
            max_slides_per_epoch,
        )
        epoch_tiles.extend(batch_tiles)

        print(f"Epoch {epoch} opened {len(open_slides)} slides")

        assert (
            len(epoch_tiles) == NUM_SAMPLES_PER_EPOCH
        ), f"Epoch {epoch} has {len(epoch_tiles)} samples instead of {NUM_SAMPLES_PER_EPOCH}"

        # Save tile sampler
        if epoch % freq_saving == 0:
            print(f"Saving cluster2tile_sampler at epoch {epoch}...")
            save_cluster2tile_sampler(
                cluster2tile_sampler,
                output_path=f"{output_dir}/cluster2tile_sampler.pkl",
            )

        # Write the schedules
        print(f"Writing epoch {epoch} in {output_dir}...")
        # Schedule itself
        df_epoch_tiles = pd.DataFrame(epoch_tiles)
        df_epoch_tiles.to_csv(f"{output_dir}/schedule_{epoch:03d}.csv", index=False)
        # Schedule list
        schedule_list_path = f"{output_dir}/schedule_list.csv"
        schedule_list_df = pd.DataFrame(
            {"epoch": [epoch], "schedule_path": [f"schedule_{epoch:03}.csv"]}
        )
        schedule_list_df.to_csv(
            schedule_list_path,
            mode="a",
            header=not os.path.exists(schedule_list_path),
            index=False,
        )


def int_or_str(value):
    try:
        return int(value)  # Try converting to int
    except ValueError:
        return value  # If it fails, return as str


def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate schedules stratified.")
    parser.add_argument("--username", type=str, default="anonymous", help="Username")
    parser.add_argument(
        "--h_level", type=int_or_str, default=4, help="Hierarchical level"
    )
    parser.add_argument(
        "--data_percentage", type=int, default=10, help="Percentage of data to use"
    )
    parser.add_argument(
        "--max_slides_per_epoch", type=int, default=800, help="Max slides per epoch"
    )
    parser.add_argument(
        "--weight_fn",
        type=str,
        default="adaptive_step",
        help="Weight function for the tile sampling counts",
    )
    parser.add_argument(
        "--curation_mode",
        type=str,
        default="higher_compress",
        help="Curation mode of the hierarchical scheme",
    )
    parser.add_argument(
        "--freq_saving",
        type=int,
        default=1,
        help="Frequency of saving the cluster2tile_sampler",
    )

    args = parser.parse_args()
    # Print all arguments
    print("Running with hyperparameters:")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    return args


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
