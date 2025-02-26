# Revisiting Automatic Data Curation for Vision Foundation Models in Digital Pathology

**Abstract** Vision foundation models (FMs) are accelerating the devel- opment of digital pathology algorithms and transforming biomedical research. These models learn, in a self-supervised manner, to represent histological features in highly heterogeneous tiles extracted from whole-slide images (WSIs) of real-world patient samples. The performance of these FMs is significantly influenced by the size, diversity, and balance of the pre-training data. However, data selection has been primarily guided by expert knowledge at the WSI level, focusing on factors such as disease classification and tissue types, while largely overlooking the granular details available at the tile level. In this paper, we investigate the potential of unsupervised automatic data curation at the tile-level, taking into account 350 million tiles. Specifically, we apply hierarchical clustering trees to pre-extracted tile embeddings, allowing us to sample balanced datasets uniformly across the embedding space of the pretrained FM. We further identify these datasets are subject to a trade-off between size and balance, potentially compromising the quality of representations learned by FMs, and propose tailored batch sampling strategies to mitigate this effect. We demonstrate the effectiveness of our method through improved performance on a diverse range of clinically relevant downstream tasks.

We provide code for:
* Data Curation using the code from [https://github.com/facebookresearch/ssl-data-curation](https://github.com/facebookresearch/ssl-data-curation), with our added configuration files [orig-ssl-data-curation](https://github.com/facebookresearch/ssl-data-curation/tree/63b3073db596d2fddf9eeb83112cbcedbda81419). Please follow the instructions from the Meta repository for k-means clustering and sampling from the tree.
* Metadata-based sampling in [metadata_based_sampling](metadata_based_sampling).
* Running our visualization tool in [visualization_tool](visualization_tool).


# Citations
We build our work on the public repositories [ssl-data-curation](https://github.com/facebookresearch/ssl-data-curation):
```
@article{vo2024automatic,
  title={Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach},
  author={Vo, Huy V. and Khalidov, Vasil and Darcet, Timoth{\'e}e and Moutakanni, Th{\'e}o and Smetanin, Nikita and Szafraniec, Marc and Touvron, Hugo and Couprie, Camille and Oquab, Maxime and Joulin, Armand and Jégou, Hervé and Labatut, Patrick and Bojanowski, Piotr},
  journal={arXiv:2405.15613},
  year={2024},
}
```
And [Histomorphological-Phenotype-Learning](https://github.com/AdalbertoCq/Histomorphological-Phenotype-Learning/tree/master):
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
