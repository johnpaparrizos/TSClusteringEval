import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics.pairwise import pairwise_kernels


class KernelKMeans(BaseEstimator, ClusterMixin):
    """
    Kernel K-means
    
    Reference
    ---------
    Kernel k-means, Spectral Clustering and Normalized Cuts.
    Inderjit S. Dhillon, Yuqiang Guan, Brian Kulis.
    KDD 2004.
    """
    def __init__(self, n_clusters=3, gamma=0.01, max_iter=100):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = 1e-3

    def fit(self, dist_mat, distance_measure):
        n_samples = dist_mat.shape[0]

        if distance_measure == 'euclidean':
            K = np.exp(-self.gamma*(dist_mat ** 2))

        else:
            K = dist_mat

        sw = np.ones(n_samples)
        self.sample_weight_ = sw

        rs = check_random_state(9)
        self.labels_ = rs.randint(self.n_clusters, size=n_samples)

        dist = np.zeros((n_samples, self.n_clusters))
        self.within_distances_ = np.zeros(self.n_clusters)

        for it in range(self.max_iter):
            dist.fill(0)
            self._compute_dist(K, dist, self.within_distances_,
                               update_within=True)
            labels_old = self.labels_
            self.labels_ = dist.argmin(axis=1)

            # Compute the number of samples whose cluster did not change 
            # since last iteration.
            n_same = np.sum((self.labels_ - labels_old) == 0)
            if 1 - float(n_same) / n_samples < self.tol:
                break

        self.X_fit_ = dist_mat
        return self

    def _compute_dist(self, K, dist, within_distances, update_within):
        """Compute a n_samples x n_clusters distance matrix using the 
        kernel trick."""
        sw = self.sample_weight_

        for j in range(self.n_clusters):
            mask = self.labels_ == j

            if np.sum(mask) == 0:
                dist[:, j] = 0
            else:
                denom = sw[mask].sum()
                denomsq = denom * denom

                if update_within:
                    KK = K[mask][:, mask]
                    dist_j = np.sum(np.outer(sw[mask], sw[mask]) * KK / denomsq)
                    within_distances[j] = dist_j
                    dist[:, j] += dist_j
                else:
                    dist[:, j] += within_distances[j]

                dist[:, j] -= 2 * np.sum(sw[mask] * K[:, mask], axis=1) / denom
