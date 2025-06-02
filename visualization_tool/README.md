## UMAP interactive Visualizer
This was built using [Marimo](https://github.com/marimo-team/marimo) and is based on the following work [Histomorphological Phenotype Learning](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/tree/master)
We provide the UMAP coordinates for N=2M curated tiles, and further provide the tile images for a subset of N=500k tiles (due to the large file size for 2M tiles) at our [hugging face repository](https://huggingface.co/datasets/swiss-ai/patho-ssl-data-curation/tree/main/visualization_tool). To visualize all 2M tiles, please extract the missing tiles. The slides can be downloaded from the [TCGA](https://portal.gdc.cancer.gov/) and [GTEx](https://www.gtexportal.org/home/histologyPage) websites, for tile extraction from the WSIs we recommend [openslide](https://openslide.org/api/python/). The slide ids and tiles `(x,y)` coordinates are specified in [umap_metadata.csv](umap_metadata.csv), tile coordinates are specified at the highest available pyramid level of the slide, the tiles are of size `224px X 224px` at 20x magnification (`=112um X 112um`). 

## Dependencies
* matplotlib
* numpy
* marimo
* pandas
* altair
* sns

## Prerequisite for tile visualization
To visualize the tiles, make sure you have downloaded the tiles as tar files (`tiles_0000.tar`-`tiles_0009.tar`) from our [hugging face repository](https://huggingface.co/datasets/swiss-ai/patho-ssl-data-curation/tree/main/visualization_tool), as well as the metadata for the 500k tiles from [metadata.csv](https://huggingface.co/datasets/swiss-ai/patho-ssl-data-curation/blob/main/visualization_tool/metadata.csv). The script [visualization_tool.py](visualization_tool.py) expects the tiles and tar files to be located at ``./visualization_tool`. The paths can also be adjusted within the first cell of the script:
```python
@app.cell
def _():
    ### Set paths ###########################
    tar_dir = "./visualization_tool"  # <--- adapt path here if desired
    csv_path = './visualization_tool/metadata.csv' # <--- adapt path here if desired
    return csv_path, tar_dir
```

## Run the code
1. You can edit the code by running `marimo edit visualization_tool.py`. Run the app with `marimo run visualization_tool.py`. 
2. Now you should be able to access the marimo app via [http://localhost:2718](http://localhost:2718/)


https://github.com/user-attachments/assets/892a9f93-7619-4719-8e74-a5c51053a710

