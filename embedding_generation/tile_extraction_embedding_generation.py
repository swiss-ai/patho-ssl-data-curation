import logging
import os

import numpy as np
import openslide
import pandas as pd
import timm
import torch
from huggingface_hub import login
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


class WSITileDataset(Dataset):
    """Dataset for extracting and transforming WSI tiles from coordinates for a single slide."""

    def __init__(
        self,
        slide_id: str,
        tile_coords: pd.DataFrame,
        slide_dir: str,
        transform: transforms.Compose,
        tile_um: int,
        out_px: int,
    ):
        self.slide_id = slide_id
        self.tile_coords = tile_coords
        self.slide_path = f"{slide_dir}/{slide_id}.svs"
        self.transform = transform
        self.tile_um = tile_um
        self.out_px = out_px
        self.slide = openslide.OpenSlide(str(self.slide_path))

        assert (
            "openslide.mpp-x" in self.slide.properties
            and "openslide.mpp-y" in self.slide.properties
        ), f"Missing MPP metadata in slide {slide_id}"
        self.mpp_x = float(self.slide.properties["openslide.mpp-x"])
        self.mpp_y = float(self.slide.properties["openslide.mpp-y"])
        self.px_width = int(round(self.tile_um / self.mpp_x))
        self.px_height = int(round(self.tile_um / self.mpp_y))

    def __len__(self) -> int:
        return len(self.tile_coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row = self.tile_coords.iloc[idx]
        tile = self.slide.read_region(
            (int(row["tile_x"]), int(row["tile_y"])), 0, (self.px_width, self.px_height)
        ).convert("RGB")
        return self.transform(tile)


if __name__ == "__main__":
    ############ Configuration ######################################
    csv_path = "/path-to/clustering_result/clustering_t1.csv"  # {clustering_t1.csv, clustering_t2.csv}
    slide_dir = "path-to-slide-dir"  # directory containing the WSI files
    output_dir = "path-to-output-dir"  # output directory where features will be saved
    os.makedirs(output_dir, exist_ok=True)

    batch_size = 16
    tile_um = 112  # 112 microns, 224px at 0.5um/pixel
    out_px = 224  # 224px
    uni_model_weights = "/capstor/scratch/cscs/lschoenp/models/uni/pytorch_model.bin"
    ################################################################

    # Load UNI model
    login()  # login to Hugging Face Hub, you need to set up your token
    if uni_model_weights is None:
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        )
    else:
        model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=False,
            init_values=1e-5,
            dynamic_img_size=True,
        )
        state_dict = torch.load(uni_model_weights, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)

    config = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    logging.info(f"Loaded UNI model to device {device}")

    # Load tile information
    df = pd.read_csv(csv_path)[["slide_id", "tile_x", "tile_y"]]

    # Extract features for each slide
    for slide_id in df["slide_id"].unique():
        if os.path.exists(f"{output_dir}/{slide_id}.npy"):
            logging.info(f"Features for slide {slide_id} already exist, skipping.")
            continue

        logging.info(f"Processing slide {slide_id}")
        slide_tiles = df[df["slide_id"] == slide_id]
        dataset = WSITileDataset(
            slide_id, slide_tiles, slide_dir, transform, tile_um, out_px
        )
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)
        all_features = []
        with torch.inference_mode():
            for batch in tqdm(loader, desc="Extract features", total=len(loader)):
                feats = model(batch.to(device))
                all_features.append(feats.cpu().numpy())
        features = np.concatenate(all_features, axis=0)
        np.save(f"{output_dir}/{slide_id}.npy", features)
        logging.info(f"Saved features to {output_dir}/{slide_id}.npy")
