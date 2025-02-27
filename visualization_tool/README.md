## UMAP interactive Visualizer
This was built using [Marimo](https://github.com/marimo-team/marimo) and is based on the following work [Histomorphological Phenotype Learning](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/tree/master)
The file [umap_metadata.csv](umap_metadata.csv) is a sample of N=100k tiles (due to size limits), upon acceptance we will extend this to the full N=2M curated tiles.

## Dependencies
* matplotlib
* numpy
* marimo
* pandas
* altair
* sns

## Prerequisite for tile visualization
To visualize the tiles, make sure you have all slides and respective tiles available. The slides can be downloaded from the [TCGA](https://portal.gdc.cancer.gov/) and [GTEx](https://www.gtexportal.org/home/histologyPage) websites, for tile extraction from the WSIs we recommend [openslide](https://openslide.org/api/python/). The slide ids and tiles `(x,y)` coordinates are specified in [umap_metadata.csv](umap_metadata.csv), tile coordinates are specified at the highest available pyramid level of the slide, the tiles are of size `224px X 224px` at 20x magnification (`=112um X 112um`). 
Once tiles are available, please adjust the code to load the tiles from file in [visualization_tool.py](visualization_tool/visualization_tool.py). 
```python
    def load_image(slide_id, tile_x_coords, tile_y_coords):
        """Slide id and tile coordinates."""
        ## Please add your code here loading the tiles from file 
        ## This could be done using openslide to read the tiles from the WSI
        image_data = np.zeros((100, 100, 3))  # Empty black image
        return image_data
```

## Run the code
1. You can edit the code by running `marimo edit tile_visualizer_umap.py`. Run the app with `marimo run tile_visualizer_umap.py`. 
2. Now you should be able to access the marimo app via [http://localhost:2718](http://localhost:2718/)


https://github.com/user-attachments/assets/892a9f93-7619-4719-8e74-a5c51053a710

