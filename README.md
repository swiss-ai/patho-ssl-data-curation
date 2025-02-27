# Revisiting Automatic Data Curation for Vision Foundation Models in Digital Pathology

**Abstract** Vision foundation models (FMs) are accelerating the devel- opment of digital pathology algorithms and transforming biomedical research. These models learn, in a self-supervised manner, to represent histological features in highly heterogeneous tiles extracted from whole-slide images (WSIs) of real-world patient samples. The performance of these FMs is significantly influenced by the size, diversity, and balance of the pre-training data. However, data selection has been primarily guided by expert knowledge at the WSI level, focusing on factors such as disease classification and tissue types, while largely overlooking the granular details available at the tile level. In this paper, we investigate the potential of unsupervised automatic data curation at the tile-level, taking into account 350 million tiles. Specifically, we apply hierarchical clustering trees to pre-extracted tile embeddings, allowing us to sample balanced datasets uniformly across the embedding space of the pretrained FM. We further identify these datasets are subject to a trade-off between size and balance, potentially compromising the quality of representations learned by FMs, and propose tailored batch sampling strategies to mitigate this effect. We demonstrate the effectiveness of our method through improved performance on a diverse range of clinically relevant downstream tasks.

We provide code for:
* Data Curation using the code from [https://github.com/facebookresearch/ssl-data-curation](https://github.com/facebookresearch/ssl-data-curation), with our added configuration files [orig-ssl-data-curation/configs](https://github.com/facebookresearch/ssl-data-curation/tree/63b3073db596d2fddf9eeb83112cbcedbda81419/configs). We ran the clustering on 350M tiles embedded with the foundation model UNI. However, due to the large size of the full tile embeddings (~670GB), we will provide the embeddings file on acceptance. It is however possible to trace back the 350M tiles from TCGA and GTEx. We describe how to do this in [README.md]()
  Please follow the instructions from the Meta repository for k-means clustering and sampling from the tree. 
* Metadata-based sampling in [metadata_based_sampling](metadata_based_sampling).
* Batch stratification based on curated trees in [batch_stratification](batch_stratification).
* Vision self-supervised training based on DINOv2 in [dinov2](dinov2).
* Running our visualization tool in [visualization_tool](visualization_tool).


# Citations
We build our work on the following public repositories"

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
