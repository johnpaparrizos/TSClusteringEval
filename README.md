# Time-Series Clustering: A Comprehensive Study of Data Mining, Machine Learning, and Deep Learning Methods

Clustering is one of the most popular time-series tasks because it enables unsupervised data exploration and often serves as a subroutine or 
preprocessing step for other tasks. Despite being the subject of active research for decades across disciplines, only limited efforts focused on 
benchmarking clustering methods for time series. Therefore, we comprehensively evaluate 80 time-series clustering methods spanning 9 different 
classes from the data mining, machine learning, and deep learning literature.

## Data

We conduct our evaluation using the UCR Time-Series Archive, the largest collection of class-labeled time series datasets. 
The archive consists of a collection of 128 datasets sourced from different sensor readings while performing diverse tasks from multiple 
domains. All datasets in the archive span between 40 to 24000 time-series and have lengths varying from 15 to 2844. Datasets are z-normalized, 
and each time-series in the dataset belongs to only one class. There is a small subset of datasets in the archive containing missing values and 
varying lengths. We employ linear interpolation to fill the missing values and resample shorter time series to reach the longest time series 
in each dataset.

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




## Methods

We have implemented 80 methods from 9 classes of time-series clustering methods proposed for univariate time series. The following table 
lists the methods considered:

### <span style='color:Tomato'>Partitional Clustering</span>
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:------------------------------------|
|𝑘-AVG              |ED                                 |                 [1]          |
|𝑘-Shape              |SBD                                 |          [3]                 |
|𝑘-SC              |STID                                 |            [5]               |
|𝑘-DBA              |DTW                                 |           [4]                |
|PAM             |MSM                                 |           [2]                 |
|PAM              |TWED                                 |       [2]                     |
|PAM              |ERP                                 |        [2]                    |
|PAM              |SBD                                 |         [2]                   |
|PAM              |SWALE                                 |       [2]                     |
|PAM              |DTW                                 |           [2]                 |
|PAM              |EDR                                 |        [2]                    |
|PAM              |LCSS                                 |       [2]                     |
|PAM              |ED                                 |          [2]                  |
### <span style='color:Tomato'>Kernel Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:------------------|
| KKM              | SINK                                 |     [6]              |
| KKM              | GAK                                 |       [6]            |
| KKM              | KDTW                                 |      [6]             |
| KKM              | RBF                                 |      [6]             |
| SC              | SINK                                 |        [7]           |
| SC              | GAK                                 |        [7]           |
| SC              | KDTW                                 |      [7]             |
| SC              | RBF                                 |       [7]            |
### <span style='color:Tomato'>Density Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:------------------------|
| DBSCAN              | ED                                 |         [8]             |
| DBSCAN              | SBD                                 |       [8]               |
| DBSCAN              | MSM                                 |       [8]               |
| DP              | ED                                 |        [10]              |
| DP              | SBD                                 |       [10]               |
| DP              | MSM                                 |       [10]               |
| OPTICS              | ED                                 |         [9]             |
| OPTICS              | SBD                                 |         [9]             |
| OPTICS              | MSM                                 |         [9]             |
### <span style='color:Tomato'>Hierarchical Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-------------------|
| AGG              | ED                                 |         [11]           |
| AGG              | SBD                                 |       [11]             |
| AGG              | MSM                                 |       [11]             |
| BIRCH              | -                                 |        [12]            |
### <span style='color:Tomato'>Distribution Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-----------------------|
| AP              | ED                                 |      [13]         |
| AP              | SBD                                 |      [13]         |
| AP              | MSM                                 |      [13]         |
| GMM              | -                                 |      [14]         |
### <span style='color:Tomato'>Shapelet Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:----------------------|
| UShapelet              | -                                 |       [15]                  |
| LDPS              | -                                 |              [16]           |
| USLM             | -                                 |           [17]               |
### <span style='color:Tomato'>Semi-Supervised Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-------------------|
| FeatTS              | -                                 |        [18]             |
| SS-DTW             | -                                 |         [19]             |
### <span style='color:Tomato'>Model and Feature Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:----------------------------|
| 𝑘-AVG             | AR-COEFF                                 |                 [20]              |
| 𝑘-AVG             | AR-PVAL                                 |               [22]                |
| 𝑘-AVG             | LPCC                                 |             [21]                  |
| 𝑘-AVG             | CATCH22                                 |          [23]                     |
| 𝑘-AVG             | ES-COEFF                                 |           [22]                    |
### <span style='color:Tomato'>Deep Learning Clustering</span> 
| <span style="background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold"> Clustering Method </span>  | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Distance Measure / Feature Vector </span> | <span style='background-color:DarkSlateGray;color:LavenderBlush;font-weight:bold'> Reference  </span> |
|:------------------|:----------------------------------|:-------------------------------------|
| IDEC            | -                                 |               [27]                |
| DEC            | -                                 |             [26]                  |
| DTC           | -                                 |              [29]                  |
| DTCR            | -                                 |                [28]               |
| SOM-VAE            | -                                 |               [30]                |
| DEPICT           | -                                 |             [31]                   |
| SDCN            | -                                 |                 [32]              |
| ClusterGAN            | -                                 |           [34]                    |
| VADE            | -                                 |              [33]                 |
| DCN            | -                                 |              [25]                 |


## References

[1] MacQueen, J. "Some methods for classiﬁcation and analysis of multivariate observations." In Proc. 5th Berkeley Symposium on Math., Stat., and Prob, p. 281. 1965.
<br>
[2] Kaufman, Leonard, and Peter J. Rousseeuw. Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
<br>
[3] Paparrizos, John, and Luis Gravano. "k-shape: Efficient and accurate clustering of time series." In Proceedings of the 2015 ACM SIGMOD international conference on management of data, pp. 1855-1870. 2015.
<br>
(4) Petitjean, François, Alain Ketterlin, and Pierre Gançarski. "A global averaging method for dynamic time warping, with applications to clustering." Pattern recognition 44, no. 3 (2011): 678-693.
<br>
[5] Yang, Jaewon, and Jure Leskovec. "Patterns of temporal variation in online media." In Proceedings of the fourth ACM international conference on Web search and data mining, pp. 177-186. 2011.
<br>
[6] Dhillon, Inderjit S., Yuqiang Guan, and Brian Kulis. "Kernel k-means: spectral clustering and normalized cuts." In Proceedings of the tenth ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 551-556. 2004.
<br>
[7] Ng, Andrew, Michael Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems 14 (2001).
<br>
[8] Ester, Martin, Hans-Peter Kriegel, Jörg Sander, and Xiaowei Xu. "A density-based algorithm for discovering clusters in large spatial databases with noise." In kdd, vol. 96, no. 34, pp. 226-231. 1996.
<br>
[9] Ankerst, Mihael, Markus M. Breunig, Hans-Peter Kriegel, and Jörg Sander. "OPTICS: Ordering points to identify the clustering structure." ACM Sigmod record 28, no. 2 (1999): 49-60.
<br>
[10] Rodriguez, Alex, and Alessandro Laio. "Clustering by fast search and find of density peaks." science 344, no. 6191 (2014): 1492-1496.
<br>
[11] Kaufman, Leonard, and Peter J. Rousseeuw. Finding groups in data: an introduction to cluster analysis. John Wiley & Sons, 2009.
<br>
[12] Zhang, Tian, Raghu Ramakrishnan, and Miron Livny. "BIRCH: an efficient data clustering method for very large databases." ACM sigmod record 25, no. 2 (1996): 103-114.
<br>
[13] Frey, Brendan J., and Delbert Dueck. "Clustering by passing messages between data points." science 315, no. 5814 (2007): 972-976.
<br>
[14] Dempster, Arthur P., Nan M. Laird, and Donald B. Rubin. "Maximum likelihood from incomplete data via the EM algorithm." Journal of the royal statistical society: series B (methodological) 39, no. 1 (1977): 1-22.
<br>
[15] Zakaria, Jesin, Abdullah Mueen, and Eamonn Keogh. "Clustering time series using unsupervised-shapelets." In 2012 IEEE 12th International Conference on Data Mining, pp. 785-794. IEEE, 2012.
<br>
[16] Lods, Arnaud, Simon Malinowski, Romain Tavenard, and Laurent Amsaleg. "Learning DTW-preserving shapelets." In Advances in Intelligent Data Analysis XVI: 16th International Symposium, IDA 2017, London, UK, October 26–28, 2017, Proceedings 16, pp. 198-209. springer International Publishing, 2017.
<br>
[17] Zhang, Qin, Jia Wu, Hong Yang, Yingjie Tian, and Chengqi Zhang. "Unsupervised feature learning from time series." In IJCAI, pp. 2322-2328. 2016.
<br>
[18] Tiano, Donato, Angela Bonifati, and Raymond Ng. "FeatTS: Feature-based Time Series Clustering." In Proceedings of the 2021 International Conference on Management of Data, pp. 2784-2788. 2021.
<br>
[19] Dau, Hoang Anh, Nurjahan Begum, and Eamonn Keogh. "Semi-supervision dramatically improves time series clustering under dynamic time warping." In Proceedings of the 25th ACM International on Conference on Information and Knowledge Management, pp. 999-1008. 2016.
<br>
[20] Piccolo, Domenico. "A distance measure for classifying ARIMA models." Journal of time series analysis 11, no. 2 (1990): 153-164.
<br>
[21] Kalpakis, Konstantinos, Dhiral Gada, and Vasundhara Puttagunta. "Distance measures for effective clustering of ARIMA time-series." In Proceedings 2001 IEEE international conference on data mining, pp. 273-280. IEEE, 2001.
<br>
[22] Maharaj, Elizabeth Ann. "Cluster of Time Series." Journal of Classification 17, no. 2 (2000).
<br>
[23] Lubba, Carl H., Sarab S. Sethi, Philip Knaute, Simon R. Schultz, Ben D. Fulcher, and Nick S. Jones. "catch22: CAnonical Time-series CHaracteristics: Selected through highly comparative time-series analysis." Data Mining and Knowledge Discovery 33, no. 6 (2019): 1821-1852.
<br>
[24] Fulcher, Ben D., and Nick S. Jones. "hctsa: A computational framework for automated time-series phenotyping using massive feature extraction." Cell systems 5, no. 5 (2017): 527-531.
<br>
[25] Yang, Bo, Xiao Fu, Nicholas D. Sidiropoulos, and Mingyi Hong. "Towards k-means-friendly spaces: Simultaneous deep learning and clustering." In international conference on machine learning, pp. 3861-3870. PMLR, 2017.
<br>
[26] Xie, Junyuan, Ross Girshick, and Ali Farhadi. "Unsupervised deep embedding for clustering analysis." In International conference on machine learning, pp. 478-487. PMLR, 2016.
<br>
[27] Guo, Xifeng, Long Gao, Xinwang Liu, and Jianping Yin. "Improved deep embedded clustering with local structure preservation." In Ijcai, pp. 1753-1759. 2017.
<br>
[28] Ma, Qianli, Jiawei Zheng, Sen Li, and Gary W. Cottrell. "Learning representations for time series clustering." Advances in neural information processing systems 32 (2019).
<br>
[29] Madiraju, Naveen Sai. "Deep temporal clustering: Fully unsupervised learning of time-domain features." PhD diss., Arizona State University, 2018.
<br>
[30] Fortuin, Vincent, Matthias Hüser, Francesco Locatello, Heiko Strathmann, and Gunnar Rätsch. "Som-vae: Interpretable discrete representation learning on time series." arXiv preprint arXiv:1806.02199 (2018).
<br>
[31] Ghasedi Dizaji, Kamran, Amirhossein Herandi, Cheng Deng, Weidong Cai, and Heng Huang. "Deep clustering via joint convolutional autoencoder embedding and relative entropy minimization." In Proceedings of the IEEE international conference on computer vision, pp. 5736-5745. 2017.
<br>
[32] Bo, Deyu, Xiao Wang, Chuan Shi, Meiqi Zhu, Emiao Lu, and Peng Cui. "Structural deep clustering network." In Proceedings of the web conference 2020, pp. 1400-1410. 2020.
<br>
[33] Jiang, Zhuxi, Yin Zheng, Huachun Tan, Bangsheng Tang, and Hanning Zhou. "Variational deep embedding: A generative approach to clustering." CoRR, abs/1611.05148 1 (2016).
<br>
[34] Ghasedi, Kamran, Xiaoqian Wang, Cheng Deng, and Heng Huang. "Balanced self-paced learning for generative adversarial clustering network." In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4391-4400. 2019.

