import numpy as np


class KMedoids():
    def __init__(self, num_clusters, max_T=100):
        self.num_clusters = num_clusters
        self.max_T = max_T
        self.num_iter = 0
        self.medoids = None
        self._labels = None

    def _init(self, DM):
        self.medoids = np.random.choice(range(DM.shape[0]), 
            self.num_clusters, replace=False)

    def _find_labels(self, DM):
        labels = self.medoids[np.argmin(DM[:,self.medoids], axis=1)]
        labels[self.medoids] = self.medoids # ensure medoids are labelled to themselves
        return labels

    def _find_medoids(self, labels, DM):
        new_medoids = np.array([-1] * self.num_clusters)
        for medoid in self.medoids:
            cluster = np.where(labels == medoid)[0] # datapoints assigned to this medoid
            mask = np.ones(DM.shape)
            mask[np.ix_(cluster, cluster)] = 0 # unmask distances between points in this cluster
            masked_distances = np.ma.masked_array(DM, mask=mask, fill_value=np.inf)
            costs = masked_distances.sum(axis=1)
            new_medoids[self.medoids == medoid] = costs.argmin(axis=0, fill_value=np.inf)
        return new_medoids

    def _relabel(self):
        label_dict = {v: k for k, v in enumerate(self.medoids)}
        for i in range(len(self._labels)):
            self._labels[i] = label_dict[self._labels[i]]

    def fit(self, X):
        self._init(X)
        num_iter = 0
        while 1:
            new_labels = self._find_labels(X)
            # Check convergence after re-assignment
            if np.array_equal(new_labels, self._labels):
                self.num_iter = num_iter
                self._relabel()
                return self._labels

            new_medoids = self._find_medoids(new_labels, X)
            num_iter += 1
            self._labels = new_labels
            self.medoids = new_medoids

            if num_iter == self.max_T:
                self.num_iter = num_iter
                self._relabel()
                return self._labels
