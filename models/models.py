import os
import time
import numpy as np
from collections import Counter


class ClusterModel:
     """
    A class representing a clustering model for time series data.
    Attributes:
        ts (numpy.ndarray): The time series data.
        labels (numpy.ndarray): The ground truth labels for the time series data.
        dm (DistanceMeasure): The distance measure object used for calculating distances between time series.
        nclusters (int): The number of clusters to create.
        linkage (str): The linkage method used for hierarchical clustering.
        threshold_metric (str): The metric used for determining the threshold in density-based clustering.
        param (tuple): Additional parameters for specific clustering algorithms.
        clustering_loss (str): The type of clustering loss used.
        ar_coeff_transforms (str): The type of autoregressive coefficient transforms used.
        precomputed (str): Indicates whether the distance matrix is precomputed or not.
        precomputed_dist_path (str): The path to the precomputed distance matrix.
        sub_dataset_name (str): The name of the sub-dataset.
        estimator: The clustering estimator object.
        inertia: The inertia value of the clustering model.
        pretrain_loss: The pretraining loss value.
        train_loss: The training loss value.
        init_ri: The initial Rand Index value.
        final_ri: The final Rand Index value.
        final_ari: The final Adjusted Rand Index value.
        final_nmi: The final Normalized Mutual Information value.
    """
    def __init__(self, ts, labels, dm, nclusters, linkage, threshold_metric, param, clustering_loss='none', ar_coeff_transforms='lpcc', precomputed='False', precomputed_dist_path=None, sub_dataset_name=None):
        self.ts = ts
        self.dm = dm
        self.labels = labels
        self.nclusters = nclusters
        self.linkage = linkage
        self.threshold_metric = threshold_metric
        self.param = param
        self.clustering_loss = clustering_loss
        self.ar_coeff_transforms = ar_coeff_transforms
        self.precomputed = precomputed
        self.precomputed_dist_path = precomputed_dist_path
        self.sub_dataset_name = sub_dataset_name
        self.estimator = None
        self.inertia = None
        self.pretrain_loss = None
        self.train_loss = None
        self.init_ri = None
        self.final_ri = None
        self.final_ari = None
        self.final_nmi = None


    def dist_to_aff(self, dist_mat, sigma=1):
        return np.exp(-np.power(dist_mat, 2) / (2 * (dist_mat.mean() ** 2)))


    # Checked - Precomputed: True and False
    def kmeans(self, distance_measure):
        if distance_measure != 'euclidean':
            raise ValueError('kmeans only supports euclidean distance')
        else:
            if self.precomputed == 'False':
                dist_time_start = time.time()
                dist_mat = getattr(self.dm, distance_measure)(self.ts)
                dist_timing = time.time() - dist_time_start
            else:
                dist_mat = np.nan_to_num(np.loadtxt(os.path.join(self.precomputed_dist_path, distance_measure+'.txt')))
                dist_mat[dist_mat < 0] = 0
                dist_timing = None

        from .kmeans import kMeans
        
        cluster_time_start = time.time()
        predictions, _ = kMeans(self.ts, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    # Checked - No Precomputed Support
    def k_shape(self, distance_measure):
        from kshape.core import kshape
        ts = np.expand_dims(self.ts, axis=2)
        
        cluster_time_start = time.time()
        kshape_model = kshape(ts, self.nclusters, centroid_init='zero', max_iter=100)
        cluster_timing = time.time() - cluster_time_start
        
        predictions = np.zeros(ts.shape[0])
        for i in range(self.nclusters):
                predictions[kshape_model[i][1]] = i

        return predictions, None, cluster_timing


    # Checked - No Precomputed Support
    def km_dba(self, distance_measure): 
        from .kdba import kDBA
        
        cluster_time_start = time.time()
        predictions, _ = kDBA(self.ts, self.nclusters)
        cluster_timing = time.time() - cluster_time_start 

        return predictions, None, cluster_timing
 

    # Checked - No Precomputed Support
    def ksc(self, distance_measure):  
        from .pyksc import ksc
        
        cluster_time_start = time.time() 
        _, predictions, _, _ = ksc.ksc(self.ts+0.0001, self.nclusters, 100, 1)
        cluster_timing = time.time() - cluster_time_start 
        
        return predictions, None, cluster_timing


    # Checked - Precomputed: True and False; Euclidean, Lorentzian, and SBD
    def kmediods(self, distance_measure):
        if self.precomputed == 'False':
            dist_time_start = time.time()
            dist_mat = getattr(self.dm, distance_measure)(self.ts)
            dist_timing = time.time() - dist_time_start
        else:
            dist_mat = np.nan_to_num(np.loadtxt(os.path.join(self.precomputed_dist_path, distance_measure+'.txt')))
            dist_mat[dist_mat < 0] = 0
            dist_timing = None

        from .partitioningaroundmedoids import PartitioningAroundMedoids
        
        cluster_time_start = time.time()
        predictions = PartitioningAroundMedoids(self.nclusters, dist_mat)
        cluster_timing = time.time() - cluster_time_start

        return predictions, dist_timing, cluster_timing

 
    # Checked
    def kernel_kmeans(self, distance_measure):
        if self.precomputed == 'False':
            if distance_measure == 'rbf':
                dist_time_start = time.time()
                from sklearn.metrics.pairwise import pairwise_kernels
                dist_mat = pairwise_kernels(self.ts, metric=distance_measure)
                dist_timing = time.time() - dist_time_start
            else:
                dist_time_start = time.time()
                distance_measure, gamma = distance_measure.split('_')[0], int(distance_measure.split('_')[1])
                dist_mat = getattr(self.dm, distance_measure)(self.ts, gamma=gamma)
                dist_timing = time.time() - dist_time_start
        else:
            if 'sink' in distance_measure:
                dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                        self.precomputed_dist_path, distance_measure+'.txt')))
                dist_mat[dist_mat < 0] = 0
                dist_timing = None
            else:
                dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
                dist_mat[dist_mat < 0] = 0
                dist_timing = None

        from .kernelkmeans import KernelKMeans
        
        cluster_time_start = time.time()
        predictions = KernelKMeans(n_clusters=self.nclusters).fit(dist_mat, distance_measure).labels_
        cluster_timing = time.time() - cluster_time_start

        return predictions, dist_timing, cluster_timing


    # Checked
    def spectralclustering(self, distance_measure):
        if self.precomputed == 'False':
            if distance_measure=='rbf':
                dist_time_start = time.time()
                from sklearn.metrics.pairwise import pairwise_kernels
                dist_mat = pairwise_kernels(self.ts, metric=distance_measure)
                aff_mat = self.dist_to_aff(dist_mat)
                dist_timing = time.time() - dist_time_start
            elif 'sink' in distance_measure:
                dist_time_start = time.time()
                distance_measure, gamma = distance_measure.split('_')[0], int(distance_measure.split('_')[1])
                dist_mat = getattr(self.dm, distance_measure)(self.ts, gamma=gamma)
                dist_timing = time.time() - dist_time_start
                aff_mat = dist_mat
        else:
            if 'sink' in distance_measure:
                dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
                dist_mat[dist_mat < 0] = 0
                dist_timing = None
                aff_mat = dist_mat
            else:
                dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
                dist_mat[dist_mat < 0] = 0
                dist_timing = None
                aff_mat = dist_mat

        from sklearn.cluster import SpectralClustering     
        
        cluster_time_start = time.time()
        predictions = SpectralClustering(n_clusters=self.nclusters, affinity="precomputed", n_jobs=1).fit(aff_mat).labels_
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    # Checked
    def dbscan(self, distance_measure):
        if self.precomputed == 'False':
            dist_time_start = time.time()
            dist_mat = getattr(self.dm, distance_measure)(self.ts)
            dist_timing = time.time() - dist_time_start
        else:
            dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
            dist_mat[dist_mat < 0] = 0
            dist_timing = None

        from sklearn.cluster import DBSCAN
        
        cluster_time_start = time.time()
        predictions = DBSCAN(
            eps=self.param[0], min_samples=self.param[1], metric="precomputed").fit(dist_mat).labels_
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    # Checked
    def density_peaks(self, distance_measure):
        if self.precomputed == 'False':
            dist_time_start = time.time()
            dist_mat = getattr(self.dm, distance_measure)(self.ts)
            dist_timing = time.time() - dist_time_start
        else:
            dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
            dist_mat[dist_mat < 0] = 0
            dist_timing = None

        from .densitypeaks import DensityPeakCluster
        
        cluster_time_start = time.time()
        predictions = DensityPeakCluster(
            threshold_metric=self.threshold_metric).fit(dist_mat)
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    # Checked
    def optics(self, distance_measure):
        if self.precomputed == 'False':
            dist_time_start = time.time()
            dist_mat = getattr(self.dm, distance_measure)(self.ts)
            dist_timing = time.time() - dist_time_start
        else:
            dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
            dist_mat[dist_mat < 0] = 0
            dist_timing = None

        from sklearn.cluster import OPTICS
        
        cluster_time_start = time.time()
        predictions = OPTICS(
            min_samples=3, metric="precomputed").fit(dist_mat).labels_
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    # Checked
    def birch(self, distance_measure):
        from sklearn.cluster import Birch
        cluster_time_start = time.time()
        predictions = Birch(n_clusters=self.nclusters).fit(self.ts).labels_
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, None, cluster_timing


    # Checked
    def agglomerative(self, distance_measure):
        if self.precomputed == 'False':
            dist_time_start = time.time()
            dist_mat = getattr(self.dm, distance_measure)(self.ts)
            dist_timing = time.time() - dist_time_start
        else:
            dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
            dist_mat[dist_mat < 0] = 0
            dist_timing = None

        from sklearn.cluster import AgglomerativeClustering

        cluster_time_start = time.time()
        predictions = AgglomerativeClustering(
            n_clusters=self.nclusters, affinity="precomputed", linkage=self.linkage).fit(dist_mat).labels_
        cluster_timing = time.time() - cluster_time_start

        return predictions, dist_timing, cluster_timing


    # Checked
    def affinity_propagation(self, distance_measure):
        if self.precomputed == 'False':
            dist_time_start = time.time()
            dist_mat = getattr(self.dm, distance_measure)(self.ts)
            dist_timing = time.time() - dist_time_start
        else:
            dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, distance_measure+'.txt')))
            dist_mat[dist_mat < 0] = 0
            dist_timing = None
            aff_mat = self.dist_to_aff(dist_mat)

        from sklearn.cluster import AffinityPropagation

        cluster_time_start = time.time()
        predictions = AffinityPropagation(
            affinity="precomputed").fit(aff_mat).labels_
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    # Checked
    def gaussian_mixture(self, distance_measure):
        from sklearn.mixture import GaussianMixture

        cluster_time_start = time.time()
        predictions = GaussianMixture(
            n_components=self.nclusters, covariance_type="full", random_state=0).fit(self.ts).predict(self.ts)
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, None, cluster_timing


    # Checked
    def UShapelet(self, distance_measure):
        from .UShapeletPythonCode.RunManyClusters_Fast import UShapelet
        
        cluster_time_start = time.time()
        predictions = UShapelet(self.ts, self.labels.copy())
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, None, cluster_timing


    def LDPS(self, distance_measure):
        from .LDPS.main import ldps

        cluster_time_start = time.time()
        predictions = ldps(np.expand_dims(self.ts, axis=2), self.nclusters)
        cluster_timing = time.time() - cluster_time_start

        return predictions, None, cluster_timing

               
    def find_best_order(self, ts):
        from statsmodels.tsa.ar_model import AutoReg
        orders = [1, 2, 3, 4, 5]
        votes = []
        for series in ts:
            min_aic, best_min_order = np.float('inf'), None
            for order in orders:
                res = AutoReg(series, lags=order, old_names=True).fit()
                if min_aic > res.aic:
                    min_aic = res.aic
                    best_min_order = order
            votes.append(best_min_order)
        votes_counter = Counter(votes)
        return votes_counter.most_common(1)[0][0]


    # Checked
    def ar_coeff(self, distance_measure):
        from statsmodels.tsa.ar_model import AutoReg
        dist_time_start = time.time()
        ar_order = self.find_best_order(self.ts)
        coeff_features = np.zeros((self.ts.shape[0], ar_order+1))
        for i, series in enumerate(self.ts):
            res = AutoReg(series, lags=ar_order, old_names=True).fit()
            coeff_features[i, :] = res.params

        dist_mat = np.nan_to_num(
            getattr(self.dm, 'euclidean')(coeff_features))
        dist_timing = time.time() - dist_time_start

        from .partitioningaroundmedoids import PartitioningAroundMedoids

        cluster_time_start = time.time()
        predictions = PartitioningAroundMedoids(self.nclusters, dist_mat)
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    # Checked
    def ar_pval(self, distance_measure):
        from statsmodels.tsa.ar_model import AutoReg
        ar_order = self.find_best_order(self.ts)
        coeff_features = np.zeros((self.ts.shape[0], ar_order+1))
        for i, series in enumerate(self.ts):
            res = AutoReg(series, lags=ar_order, old_names=True).fit()
            coeff_features[i, :] = res.pvalues

        dist_time_start = time.time()
        dist_mat = np.nan_to_num(
            getattr(self.dm, 'euclidean')(coeff_features))
        dist_timing = time.time() - dist_time_start

        from .partitioningaroundmedoids import PartitioningAroundMedoids
        
        cluster_time_start = time.time()
        predictions = PartitioningAroundMedoids(self.nclusters, dist_mat)
        cluster_timing = time.time() - cluster_time_start
        
        return predictions, dist_timing, cluster_timing


    def lpcc(self, coef):
        ar_order = len(coef) - 1
        lpcc_coeffs = [-coef[0]]
        for n in range(1, ar_order + 1):
            upbound = (ar_order + 1 if n > ar_order else n)
            lpcc_coef = -np.sum(i * lpcc_coeffs[i] * coef[n - i - 1]
                                for i in range(1, upbound)) * 1. / upbound
            lpcc_coef -= coef[n - 1] if n <= len(coef) else 0
            lpcc_coeffs.append(lpcc_coef)
        return np.array(lpcc_coeffs)


    # Checked
    def ar_lpcc(self, distance_measure):
        from statsmodels.tsa.ar_model import AutoReg
        ar_order = self.find_best_order(self.ts)
        coeff_features = np.zeros((self.ts.shape[0], ar_order+1))
        for i, series in enumerate(self.ts):
            res = AutoReg(series, lags=ar_order, old_names=True).fit()
            if self.ar_coeff_transforms == 'lpcc':
                coeff_features[i, :] = self.lpcc(res.params)

        dist_time_start = time.time()
        dist_mat = np.nan_to_num(
            getattr(self.dm, 'euclidean')(coeff_features))
        dist_timing = time.time() - dist_time_start

        from .partitioningaroundmedoids import PartitioningAroundMedoids
        cluster_time_start = time.time()
        predictions = PartitioningAroundMedoids(self.nclusters, dist_mat)
        cluster_timing = time.time() - cluster_time_start

        return predictions, dist_timing, cluster_timing


    def catch22_features(self, ts):
        import catch22
        features = np.zeros((ts.shape[0], 22))
        for i in range(ts.shape[0]):
            catchOut = catch22.catch22_all(ts[i])
            features[i, :] = catchOut['values']
        return features

    
    # Checked
    def catch22(self, distance_measure):
        if distance_measure != 'euclidean':
            raise ValueError(
                "kmeans only supports euclidean distance")
        else:
            if self.precomputed == 'False':
                dist_time_start = time.time()
                self.ts = np.nan_to_num(self.catch22_features(self.ts))
                dist_mat = getattr(self.dm, 'euclidean')(self.ts)
                dist_timing = time.time() - dist_time_start
            else:
                dist_mat = np.nan_to_num(np.loadtxt(os.path.join(
                    self.precomputed_dist_path, 'catch22.txt')))
                dist_mat[dist_mat < 0] = 0
                dist_timing = None

            from .partitioningaroundmedoids import PartitioningAroundMedoids
            cluster_time_start = time.time()
            predictions = PartitioningAroundMedoids(self.nclusters, dist_mat)
            cluster_timing = time.time() - cluster_time_start
            return predictions, dist_timing, cluster_timing


    # Checked
    def exponential_smoothing(self, distance_measure):
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        dist_time_start = time.time()
        cluster_time_start = time.time()
        coeff_features = np.zeros((self.ts.shape[0], 3))
        for i, series in enumerate(self.ts):
            fit1 = ExponentialSmoothing(series, seasonal_periods=4, trend='add',
                                        seasonal='add', use_boxcox=False, initialization_method="heuristic").fit()
            coeff_features[i, :] = np.nan_to_num(np.array(
                [fit1.params['smoothing_level'], fit1.params['smoothing_trend'], fit1.params['damping_trend']]))

        dist_mat = getattr(self.dm, 'euclidean')(coeff_features)
        dist_timing = time.time() - dist_time_start

        from .partitioningaroundmedoids import PartitioningAroundMedoids
        cluster_time_start = time.time()
        predictions = PartitioningAroundMedoids(self.nclusters, dist_mat)
        cluster_timing = time.time() - cluster_time_start
        return predictions, dist_timing, cluster_timing

    
    # Checked
    def DCN(self, distance_measure):
        from .DCN_keras.DCN import dcn_clustering
        cluster_time_start = time.time()
        self.estimator, predictions = dcn_clustering(
            self.ts, self.labels, self.nclusters, self.param)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def DEC(self, distance_measure):
        from .IDEC.DEC import dec_clustering
        cluster_time_start = time.time()
        self.estimator, self.inertia, predictions = dec_clustering(
            self.ts, self.labels, self.nclusters, self.param)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def IDEC(self, distance_measure):
        from .IDEC.IDEC import idec_clustering
        from .IDEC.DEC import dec_clustering
        
        cluster_time_start = time.time()
        self.estimator, self.inertia, predictions = idec_clustering(
            self.ts, self.labels, self.nclusters, self.param)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing
   

    # Checked
    def DTCR(self, distance_measure):
        from .DTCR.main import dtcr_clustering
        cluster_time_start = time.time()
        self.estimator, predictions = dtcr_clustering(
            self.ts, self.labels, self.nclusters, self.param)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def SOM_VAE(self, distance_measure):
        from .SOM_VAE.main import som_vae
        cluster_time_start = time.time()
        predictions = som_vae(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def DTC(self, distance_measure):
        from .DTC.main import dtc
        cluster_time_start = time.time()
        predictions = dtc(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def DTSCNRV(self, distance_measure):
        from .DTSCNRV.main import dtscnrv
        cluster_time_start = time.time()
        print(self.clustering_loss)
        predictions = dtscnrv(self.ts, self.labels, self.nclusters, self.clustering_loss)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def DEPICT(self, distance_measure):
        from .DEPICT import depict
        cluster_time_start = time.time()
        predictions = depict(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    def VADE(self, distance_measure):
        from .VADE.main import vade
        cluster_time_start = time.time()
        predictions = vade(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def SDCN(self, distance_measure):
        from .SDCN.main import sdcn
        cluster_time_start = time.time()
        predictions = sdcn(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def MOMENT(self, distance_measure):
        from .MOMENT.main import moment
        cluster_time_start = time.time()
        predictions = moment(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def OFA(self, distance_measure):
        from .OFA.main import ofa
        cluster_time_start = time.time()
        predictions = ofa(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing


    # Checked
    def CHRONOS(self, distance_measure):
        from .CHRONOS.main import chronos
        cluster_time_start = time.time()
        predictions = chronos(self.ts, self.labels, self.nclusters)
        cluster_timing = time.time() - cluster_time_start
        return predictions, None, cluster_timing
