import marimo

__generated_with = "0.10.5"
app = marimo.App(layout_file="layouts/visualization_tool.grid.json")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import marimo as mo
    import pandas as pd
    import altair as alt
    import seaborn as sns
    import random
    import os
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    from distinct_colors import distinct_colors_100, distinct_colors_34
    CORE_COLS = ["UMAP0", "UMAP1", "slide_id","tile_x","tile_y"]
    random.seed(42)
    random.shuffle(distinct_colors_34)
    random.shuffle(distinct_colors_100)
    return (
        CORE_COLS,
        ProcessPoolExecutor,
        alt,
        distinct_colors_100,
        distinct_colors_34,
        mo,
        np,
        os,
        partial,
        pd,
        plt,
        random,
        sns,
    )


@app.cell
def _(mo):
    mo.md(
        f"""
        # **Hierachical Cluster UMAP Visualization**

        **Hierarchical K-Means Clustering Results and Tile Representation Explorer.**
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        f"""
        ### **Citation**

        **Building on Github Repository: [https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning)**

        **[Quiros A.C.<sup>+</sup>, Coudray N.<sup>+</sup>, Yeaton A., Yang X., Liu B., Chiriboga L., Karimkhan A., Narula N., Moore D.A., Park C.Y., Pass H., Moreira A.L., Le Quesne J.<sup>\*</sup>, Tsirigos A.<sup>\*</sup>, and Yuan K.<sup>\*</sup> Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unlabeled, unannotated pathology slides. 2024](https://arxiv.org/abs/2205.01931)**
    """
    )
    return


@app.cell
def _(__file__, os, pd, sample_size_slider):
    csv_path = 'umap_metadata.csv'
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data = pd.read_csv(csv_path)
    data["source site"] = "TCGA-"+data["slide_id"].str.split('-').str[1]
    data.loc[data["dataset"]=="GTEx","source site"] = "GTEx"
    data = data.sample(sample_size_slider.value)
    return csv_path, data, dir_path


@app.cell
def _(mo):
    sample_size_slider = mo.ui.slider(start=50000, stop=100000, step=10000, value=100000, label='Number of samples')
    sample_size_slider
    return (sample_size_slider,)


@app.cell
def _(cols, data, mo):
    _filtered_cols = [_c for _c in cols if _c not in ["UMAP0", "UMAP1","tile_x","tile_y", "slide_id", "dataset"]]
    _filtered_cols_display = []
    _filter_values = []
    for _c in _filtered_cols:
        _options = data[_c].drop_duplicates().sort_values().astype("str")
        if len(_options) <= 100000:
            _col_filter=mo.ui.multiselect(_options, value=_options)
            _filter_values.append(_col_filter)
            _filtered_cols_display.append(_c)

    filtered_cols = _filtered_cols_display
    array = mo.ui.array(_filter_values)
    grid = mo.vstack([mo.md(f"""
        **Select subset of data**"""), mo.hstack([mo.md(_c) for _c in filtered_cols]), array.hstack()])
    grid
    return array, filtered_cols, grid


@app.cell
def _(CORE_COLS, data, mo):
    cols = data.columns
    dropdown_cols = [c for c in cols if c not in CORE_COLS]

    dropdown = mo.ui.dropdown(
      options=dropdown_cols,
      value=[_c for _c in dropdown_cols if 'level' in _c][-1],
      label='Visualize labels for'
    )
    dropdown
    return cols, dropdown, dropdown_cols


@app.cell
def _(mo):
    controls = mo.ui.radio(["Selection", "Pan & Zoom"], value="Selection")
    return (controls,)


@app.cell
def _(mo):
    slider = mo.ui.slider(start=4, stop=50, step=2, label='Point size')
    slider
    return (slider,)


@app.cell
def _(array, data, filtered_cols, np):
    def filter_data(values=None):
        if values is None:
            return data
        _mask = True
        for _v, _c in zip(values, filtered_cols):
            if "level" in _c:
                _v = np.array(_v).astype("int")
            _current_mask = data[_c].isin(_v)
            if "nan" in _v:
                _current_mask = data[_c].isin(_v) | ~data[_c].notna()
            _mask &= _current_mask
        return data[_mask]

    data_tab = filter_data(array.value)
    return data_tab, filter_data


@app.cell
def _(
    alt,
    controls,
    data_tab,
    distinct_colors_100,
    distinct_colors_34,
    dropdown,
    mo,
    slider,
):
    selection = dropdown.value

    # Scatter points
    if 'level' in selection:
        colorscale = alt.Scale(range=distinct_colors_100)
        labeling = alt.Color('%s:N' %selection, scale=colorscale)
    elif 'tissue' in selection:
        colorscale = alt.Scale(range=distinct_colors_34)
        labeling = alt.Color('%s:N' %selection, scale=colorscale)
    elif 'dataset' in selection:
        labeling = alt.Color('%s:N' %selection)
    else:
        labeling = alt.Color('%s' %selection)

    alt_chart = alt.Chart(data_tab, width=550, height=550).mark_circle(size=slider.value).encode(
    x=alt.X("UMAP0:Q"),
    y=alt.Y("UMAP1:Q"),
    color=labeling
    )
    chart = mo.ui.altair_chart(
            alt_chart.interactive() if controls.value == "Pan & Zoom" else alt_chart,
            chart_selection=controls.value == "Selection",
            legend_selection=controls.value == "Selection", 
        )
    mo.vstack([chart, controls])
    return alt_chart, chart, colorscale, labeling, selection


@app.cell
def _(chart, mo):
    table = mo.ui.table(chart.value, page_size=5)
    return (table,)


@app.cell
def _(chart, mo, random, show_images, table):
    mo.stop(not len(chart.value))

    if not len(table.value):
        _indices = random.sample(range(len(chart.value)), k=15)
        _slide_ids = list(chart.value['slide_id'].iloc[_indices])
        _tile_x_coords = list(chart.value['tile_x'].iloc[_indices])
        _tile_y_coords = list(chart.value['tile_y'].iloc[_indices])
        _selected_images = show_images(_slide_ids, _tile_x_coords, _tile_y_coords)
    else:
        _slide_ids = list(table.value['slide_id'])
        _selected_images = show_images(_slide_ids, list(table.value['tile_x']), list(table.value['tile_y']))
    #    **Data Selected:**
    mo.md(
        f"""
        {mo.as_html(_selected_images)}
        {table}
        """
    )
    return


@app.cell
def _(ProcessPoolExecutor, np, plt):
    def load_image(slide_id, tile_x_coords, tile_y_coords):
        """Slide id and tile coordinates."""
        ## Please add your code here loading the tiles from file 
        ## This could be done using openslide to read the tiles from the WSI
        image_data = np.zeros((100, 100, 3))  # Empty black image
        return image_data

    def show_images(slide_ids, tile_x_coords, tile_y_coords):
        """Show images."""
        fig, axes = plt.subplots(3, 5, figsize=(6, 4))
        axes = axes.flatten()

        with ProcessPoolExecutor() as executor:
            images = list(executor.map(load_image, slide_ids, tile_x_coords, tile_y_coords))

        for i, (image_data, slide_id, tile_x, tile_y, ax) in enumerate(zip(images, slide_ids, tile_x_coords, tile_y_coords, axes)):
            ax.imshow(image_data)
            ax.axis('off')
            #, (x,y)=({tile_x},{tile_y})
            ax.set_title(f"{'-'.join(slide_id.split('-')[:3])}", fontsize=6)

        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        for ax in axes:
            ax.set_yticks([])
            ax.set_xticks([])

        plt.tight_layout()
        return fig
    return load_image, show_images


if __name__ == "__main__":
    app.run()
