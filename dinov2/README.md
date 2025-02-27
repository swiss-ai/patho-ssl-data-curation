# Pathology Foundation Model Training Code

This repository contains code for training a pathology foundation model based on the dinov2 framework with custom modifications to handle whole-slide pathology images. The code efficiently processes large pathology datasets by breaking whole-slide images into tiles and organizing training into pseudo epochs using schedule files.

---

## Overview

The training pipeline is designed to:

- Load whole-slide image data stored in HDF5 files.
- Use a custom dataset class (`training_tile_dataset`) to retrieve image tiles.
- Organize training into pseudo epochs, where each epoch uses a specific schedule file to determine which tiles to process.
- Map whole-slide images to their corresponding HDF5 files via a metadata CSV.

---

## Dataset Structure and Workflow

### CSV Files

1. **Metadata CSV (`metadata.csv`):**  
   Maps each whole-slide image (identified by `slide_id`) to its corresponding HDF5 file path.  
   **Example:**
   ```csv
   slide_id,slide_path
   slide1,/data/slide1.h5
   slide2,/data/slide2.h5
   slide3,/data/slide3.h5
   ...
   ```
2. **Schedule List CSV (`schedule_list.csv`):**  
   Contains the mapping between epoch numbers and their corresponding schedule file paths.
   **Example:**
   ```csv
    epoch,schedule_path
    0,schedule_000.csv
    1,schedule_001.csv
    2,schedule_002.csv
   ...
   ```

The above CSV files are required and should be specified in `./data/datasets/path_dataset.py`:

3. **Schedule CSV (e.g.,`schedule_000.csv`):**  
   For each pseudo epoch, this file specifies which tile (by `ind`) from a given `slide_id` should be used.  
   **Example:**
   ```csv
    slide_id,ind
    slide1,0
    slide2,5
    slide3,2
   ...
   ```

## Custom Dataset Class: `training_tile_dataset`

### Initialization
The dataset class loads the metadata and the schedule list. It accepts parameters for tile size, transformation functions, and the schedule path to use.

### Setting Data for an Epoch (`set_data(epoch)`)
For a given epoch, the dataset:
- Loads the corresponding schedule CSV.
- Opens the required HDF5 files based on the slide IDs present in the schedule.
- Logs the number of tiles loaded for that epoch.

### Tile Retrieval (`__getitem__`)
For a given index:
- The schedule CSV row is used to determine which slide and tile index to load.
- The tile is read from the HDF5 file and reconstructed as a PIL Image with a fixed size (currently 224×224).
- Any specified transformations are applied before the image is returned.

## Configuration File

The YAML file (`vitl14.yaml`) defines model parameters, dataset paths, and training hyperparameters. It also requires the following inputs:

- **schedule_path:** Directory containing all schedule files (e.g., `schedule_000.csv`, `schedule_001.csv`, etc.) for the curated (or full) dataset.
- **data_len:** The length (number of rows) in each schedule file, which represents the number of tiles in that pseudo epoch.

## Running the Training

To initiate training, execute the following command:

```bash
python3 ./run/train/train.py --config-file=./configs/train/vitl14.yaml --output-dir=./path/to/output
```
