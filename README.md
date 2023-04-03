# TSClustering: A Comprehensive Library with Classic, Machine Learning, and Deep Learning Methods

TSClustering is a new open, end-to-end benchmark suite to ease the evaluation of univariate time-series clustering methods. Our library contains 80 time-series clustering methods belonging to 9 popular classes. 

## Data

To ease reproducibility, we share our results over an established benchmarks:

* The UCR Univariate Archive, which contains 128 univariate time-series datasets. 
  * Download all 128 preprocessed datasets [here](https://www.thedatum.org/datasets/UCR2022_DATASETS.zip).

For the preprocessing steps check [here](https://github.com/thedatumorg/UCRArchiveFixes).


## Usage

```
$ python main.py --dataset ucr_uea --data_path ./data/UCR2018/ --results_path ./experiment.json --model k_shape --start 1 --end 1
```
```
'rand_score': 0.6553072625698324
'adjusted_rand_score': 0.28193671569162176
'normalized_mutual_information': 0.4241221530777422
'sub_dataset_name': 'BME'
'distance_timing': None
'cluster_timing': 8.67257952690124
```


## Results

Server Specifications: 4  Dual Intel(R) Xeon(R) Silver 4116 (12-core with 2-way SMT), 2.10 GHz, 196GB RAM; Ubuntu Linux 18.04.3 LTS

GPU Specifications: NVIDIA GeForce RTX 2080 GPU, 32GB memory.

<p align="center">
    <img src="https://github.com/johnpaparrizos/TSClusteringEval/blob/main/docs/images/cd-plots.png">
</p>

