# Metadata Based Sampling

This folder contains scripts for metadata-based sampling from the TCGA and GTEx datasets. Below is a description of each file, its purpose, and how to use it.

#### Dependencies
* pandas
* numpy
* matplotlib
* fuzzywuzzy


## Files

### `metadata_based_sampling.py`
This script is used to sample data based on tissue and primary diagnosis. It loads metadata from multiple datasets, calculates sampling weights, samples the data, and visualizes the distribution of the sampled data versus the full dataset.

#### Usage
```bash
cd metadata_based_sampling
```
Set the sampling parameters in the script:
```python
    sample_percentage = 0.1  # % of the data
    random_seed = 25
```
Run the script:
```bash
python metadata_based_sampling.py
```

### `standardize_tissue_site.py`
This script standardizes tissue site names using fuzzy matching, as well as a csv with prior information for hard to map tissue sites. It maps uncleaned organ site names to a primary list of standardized names, updates metadata files with the cleaned names, and saves the mappings to a CSV file.

#### List of standardized tissue sites:
`Adipose,Adrenal Gland,Artery,Bladder,Brain,Breast,Cervix,Colon,Esophagus,Eye and Adnexa,Fallopian Tube,Heart,Kidney,Liver,Lung,Lymphatic System,Muscle,Nerve,Oral Cavity and Pharynx,Other,Ovary,Pancreas,Pituitary,Prostate,Skin,Skeletal System,Small Intestine,Spleen,Stomach,Testis,Thyroid,Uterus,Unknown,Vagina`

#### Usage
```bash
cd metadata_based_sampling
```
```bash
python standardize_tissue_site.py
```



