# Revisiting Automatic Data Curation for Vision Foundation Models in Digital Pathology
[HuggingFace Repo](https://huggingface.co/datasets/swiss-ai/patho-ssl-data-curation/tree/main)| [Paper](https://arxiv.org/abs/2503.18709)

**Abstract** Vision foundation models (FMs) are accelerating the devel- opment of digital pathology algorithms and transforming biomedical research. These models learn, in a self-supervised manner, to represent histological features in highly heterogeneous tiles extracted from whole-slide images (WSIs) of real-world patient samples. The performance of these FMs is significantly influenced by the size, diversity, and balance of the pre-training data. However, data selection has been primarily guided by expert knowledge at the WSI level, focusing on factors such as disease classification and tissue types, while largely overlooking the granular details available at the tile level. In this paper, we investigate the potential of unsupervised automatic data curation at the tile-level, taking into account 350 million tiles. Specifically, we apply hierarchical clustering trees to pre-extracted tile embeddings, allowing us to sample balanced datasets uniformly across the embedding space of the pretrained FM. We further identify these datasets are subject to a trade-off between size and balance, potentially compromising the quality of representations learned by FMs, and propose tailored batch sampling strategies to mitigate this effect. We demonstrate the effectiveness of our method through improved performance on a diverse range of clinically relevant downstream tasks.

# Code and Data
We provide code and data for:
* **Data Curation**: Based on the code from [https://github.com/facebookresearch/ssl-data-curation](https://github.com/facebookresearch/ssl-data-curation), with our added configuration files [orig-ssl-data-curation/configs](https://github.com/facebookresearch/ssl-data-curation/tree/63b3073db596d2fddf9eeb83112cbcedbda81419/configs). We ran the clustering on 350M tiles from [TCGA](https://portal.gdc.cancer.gov/) and [GTEx](https://www.gtexportal.org/home/histologyPage) whole slide images (WSIs) embedded with the foundation model [UNI](https://huggingface.co/MahmoodLab/UNI). However, due to the large size of the full tile embeddings (~670GB), we are not able to provide the embeddings file. We provide a pipeline for tile extraction and UNI feature embedding in [tile_extraction_embedding_generation.py](embedding_generation/tile_extraction_embedding_generation.py), which requires the downloaded TCGA and GTEx WSIs. It consists of the following steps:
	* Tile extraction: The slides can be downloaded from the TCGA and GTEx websites. The slide ids and tiles (x,y) coordinates are specified in `clustering_results/clustering_{t1,t2}.csv`, tile coordinates are specified at the highest available pyramid level of the slide, the tiles are of size 224px X 224px at 20x magnification (=112um X 112um).
	* Feature embedding: Generate the feature embedding using the UNI model on the extracted tiles.
Combine all the embeddings into a large numpy file. With it available, you can run the data curation by following the instructions from the Meta repository for k-means clustering and sampling from the tree.
* **Clustering results**: Our clustering results are available as csvs files at our [HuggingFace Repo](https://huggingface.co/datasets/swiss-ai/patho-ssl-data-curation/tree/main). Structure of `clustering_results/clustering_{t1,t2}.csv`:

  	| slide\_id                                         | tile\_x | tile\_y | level\_1 | level\_2 | level\_3 | level\_4 |
	| ------------------------------------------------- | ------- | ------- | -------- | -------- | -------- | -------- |
	| TCGA-22-1017-01Z-00-DX1.9562FE79-A261-42D3-B39... | 32406   | 10621   | 1301309  | 17404    | 2        | 24       |
	| TCGA-22-1017-01Z-00-DX1.9562FE79-A261-42D3-B39... | 32850   | 10621   | 3481104  | 17557    | 343      | 8        |
	| TCGA-22-1017-01Z-00-DX1.9562FE79-A261-42D3-B39... | 30630   | 11064   | 2269415  | 34147    | 2        | 24       |
	| TCGA-22-1017-01Z-00-DX1.9562FE79-A261-42D3-B39... | 31074   | 11064   | 3352403  | 3486     | 2        | 24       |
	| TCGA-22-1017-01Z-00-DX1.9562FE79-A261-42D3-B39... | 31519   | 11064   | 3352388  | 11187    | 2        | 24       |

  **slide\_id**: Unique identifier for the slide image, **tile\_x, tile\_y**: Coordinates of the tile within the slide at level 0, the highest pyramid level. The tiles are of size `224px X 224px`at 20x magnification (=`112um X 112um`), **level\_1 to level\_4**: Hierarchical cluster labels the tile is associated with.
* **Metadata-based sampling**: [metadata_based_sampling](metadata_based_sampling).
* **Batch stratification based on curated trees**: [batch_stratification](batch_stratification).
* **Vision self-supervised training based on DINOv2**: [dinov2](dinov2).
* **Visualization Tool**: [visualization_tool](visualization_tool).

# License
Please cite our publication, if you use the provided code or data.
```
@misc{chen2025revisitingautomaticdatacuration,
      title={Revisiting Automatic Data Curation for Vision Foundation Models in Digital Pathology}, 
      author={Boqi Chen and Cédric Vincent-Cuaz and Lydia A. Schoenpflug and Manuel Madeira and Lisa Fournier and Vaishnavi Subramanian and Sonali Andani and Samuel Ruiperez-Campillo and Julia E. Vogt and Raphaëlle Luisier and Dorina Thanou and Viktor H. Koelzer and Pascal Frossard and Gabriele Campanella and Gunnar Rätsch},
      year={2025},
      eprint={2503.18709},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.18709}, 
}
```

# Citations
We build our work on the following public repositories:

-  [ssl-data-curation](https://github.com/facebookresearch/ssl-data-curation):
```
@article{vo2024automatic,
  title={Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach},
  author={Vo, Huy V. and Khalidov, Vasil and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Smetanin, Nikita and Szafraniec, Marc and Touvron, Hugo and Couprie, Camille and Oquab, Maxime and Joulin, Armand and Jégou, Hervé and Labatut, Patrick and Bojanowski, Piotr},
  journal={arXiv:2405.15613},
  year={2024},
}
```
- [Histomorphological-Phenotype-Learning](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/tree/master):
```
@article{QuirosCoudray2024,
	author = {Claudio Quiros, Adalberto and Coudray, Nicolas and Yeaton, Anna and Yang, Xinyu and Liu, Bojing and Le, Hortense and Chiriboga, Luis and Karimkhan, Afreen and Narula, Navneet and Moore, David A. and Park, Christopher Y. and Pass, Harvey and Moreira, Andre L. and Le Quesne, John and Tsirigos, Aristotelis and Yuan, Ke},
	journal = {Nature Communications},
	number = {1},
	pages = {4596},
	title = {Mapping the landscape of histomorphological cancer phenotypes using self-supervised learning on unannotated pathology slides},
	volume = {15},
	year = {2024}}
}
```

- [dinov2](https://github.com/facebookresearch/dinov2):
```
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
@misc{darcet2023vitneedreg,
  title={Vision Transformers Need Registers},
  author={Darcet, Timothée and Oquab, Maxime and Mairal, Julien and Bojanowski, Piotr},
  journal={arXiv:2309.16588},
  year={2023}
}
```
- [UNI](https://huggingface.co/MahmoodLab/UNI):
```
@article{chen2024uni,
  title={Towards a General-Purpose Foundation Model for Computational Pathology},
  author={Chen, Richard J and Ding, Tong and Lu, Ming Y and Williamson, Drew FK and Jaume, Guillaume and Chen, Bowen and Zhang, Andrew and Shao, Daniel and Song, Andrew H and Shaban, Muhammad and others},
  journal={Nature Medicine},
  publisher={Nature Publishing Group},
  year={2024}
}
```
